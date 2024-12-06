import netket as nk
import numpy as np
import json
from math import pi
import jax
import jax.numpy as jnp

L = 2
# Build square lattice with nearest and next-nearest neighbor edges
lattice = nk.graph.Square(L, max_neighbor_order=2)
hi = nk.hilbert.Spin(s=1 / 2, total_sz=0, N=lattice.n_nodes, inverted_ordering=False)
# Heisenberg with coupling J=1.0 for nearest neighbors
# and J=0.5 for next-nearest neighbors
H = nk.operator.Heisenberg(hilbert=hi, graph=lattice, J=[1.0, 0.5])



input = hi.random_state(size=64, key=jax.random.PRNGKey(1))
input = jnp.array(input)
print(input)
print(input.shape)


# Print model architecture and parameter count
def print_model_info(model):
    """Prints the model architecture and parameter count."""
    key = jax.random.PRNGKey(0)
    params = model.init(key, input)
    print("Model Architecture:")
    print(model)
    total_params = sum(jnp.prod(jnp.array(param.shape)) for param in jax.tree_util.tree_leaves(params))
    print(f"Total Trainable Parameters: {total_params}")

# Find an approximate ground state
machine = nk.models.GCNN(
    symmetries=lattice,
    parity=1,
    layers=4,
    features=4,
    param_dtype=complex,
)

print_model_info(machine)

n_chains = 128
n_samples = 4096
n_discard_per_chain = 10  # Number of samples to discard per chain
chunk_size = 4096

sampler = nk.sampler.MetropolisExchange(hi, n_chains=1024, graph=lattice, d_max=L, dtype=jnp.int8)
vstate = nk.vqs.MCState(sampler=sampler,model=machine,n_samples=n_samples,n_discard_per_chain=n_discard_per_chain,chunk_size=chunk_size)

opt = nk.optimizer.Sgd(learning_rate=0.01)
sr = nk.optimizer.SR(diag_shift=1e-4, holomorphic=False)

# gs = nk.driver.VMC(H, opt, variational_state=vstate, preconditioner=sr)
# gs.run(n_iter=50, out="ground_state")

data = json.load(open("ground_state.log"))
print("Energy averaged over last ten steps:", np.mean(data["Energy"]["Mean"]["real"][-10:]))
print("Energy per site averaged over last ten steps:", np.mean(data["Energy"]["Mean"]["real"][-10:]) / (lattice.n_nodes))
print("Energy std over last ten steps:", np.std(data["Energy"]["Mean"]["real"][-10:]) / (lattice.n_nodes))

from scipy.sparse.linalg import eigsh

eig_vals, eig_vecs = eigsh(H.to_sparse(), k=2, which="SA")

print("eigenvalues with scipy sparse:", eig_vals)

E_gs = eig_vals[0]

import jax.numpy as jnp
import math
from flax import linen as nn
from typing import Optional

class LinearToSquare(nn.Module):
    def __call__(self, x):
        """
        Reshape the input to N x N shape for processing.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, linear_size).
                Assumes linear_size is a perfect square.

        Returns:
            torch.Tensor: Reshaped tensor of shape (batch_size, N, N), where N is computed.
        """
        batch_size, linear_size = x.shape
        # Compute N
        n = int(linear_size**0.5)
        assert n * n == linear_size, "Input size must be a perfect square."
        return jnp.expand_dims(x.reshape(batch_size, n, n), axis=-1)

class Patches(nn.Module):
    patch_size: int
    embed_dim: int

    def setup(self):
        self.conv = nn.Conv(
            features=self.embed_dim,
            kernel_size=(self.patch_size, self.patch_size),
            strides=(self.patch_size, self.patch_size),
            padding="VALID",
        )

    def __call__(self, images):
        patches = self.conv(images)
        b, h, w, c = patches.shape
        patches = jnp.reshape(patches, (b, h * w, c))
        return patches


class PatchEncoder(nn.Module):
    hidden_dim: int

    @nn.compact
    def __call__(self, x):
        assert x.ndim == 3
        n, seq_len, _ = x.shape
        # Hidden dim
        x = nn.Dense(self.hidden_dim)(x)
        # Add cls token
        cls = self.param("cls_token", nn.initializers.zeros, (1, 1, self.hidden_dim))
        cls = jnp.tile(cls, (n, 1, 1))
        x = jnp.concatenate([cls, x], axis=1)
        # Add position embedding
        pos_embed = self.param(
            "position_embedding",
            nn.initializers.normal(stddev=0.001),  # From BERT
            (1, seq_len + 1, self.hidden_dim),
        )
        # pos_embed = self.param(
        #     "position_embedding",
        #     nn.initializers.normal(stddev=0.01),  # From BERT
        #     (1, seq_len, self.hidden_dim),
        # )
        return x + pos_embed


class MultiHeadSelfAttention(nn.Module):
    hidden_dim: int
    n_heads: int

    def setup(self):
        self.q_net = nn.Dense(self.hidden_dim)
        self.k_net = nn.Dense(self.hidden_dim)
        self.v_net = nn.Dense(self.hidden_dim)

        self.proj_net = nn.Dense(self.hidden_dim)

    def __call__(self, x, train=True):
        B, T, C = x.shape  # batch_size, seq_length, hidden_dim
        N, Dh = self.n_heads, C // self.n_heads  # num_heads, head_dim
        q = self.q_net(x).reshape(B, T, N, Dh).transpose(0, 2, 1, 3)  # (B, N, T, D)
        k = self.k_net(x).reshape(B, T, N, Dh).transpose(0, 2, 1, 3)
        v = self.v_net(x).reshape(B, T, N, Dh).transpose(0, 2, 1, 3)

        # weights (B, N, T, T)
        weights = jnp.matmul(q, jnp.swapaxes(k, -2, -1)) / math.sqrt(Dh)
        normalized_weights = nn.softmax(weights, axis=-1)

        # attention (B, N, T, D)
        attention = jnp.matmul(normalized_weights, v)

        # gather heads
        attention = attention.transpose(0, 2, 1, 3).reshape(B, T, N * Dh)

        # project
        out = self.proj_net(attention)

        return out


class MLP(nn.Module):
    mlp_dim: int
    out_dim: Optional[int] = None

    @nn.compact
    def __call__(self, inputs, train=True):
        actual_out_dim = inputs.shape[-1] if self.out_dim is None else self.out_dim
        x = nn.Dense(features=self.mlp_dim)(inputs)
        x = nn.gelu(x)
        x = nn.Dense(features=actual_out_dim)(x)
        return x


class Transformer(nn.Module):
    embed_dim: int
    hidden_dim: int
    n_heads: int
    mlp_dim: int

    def setup(self):
        self.mha = MultiHeadSelfAttention(self.hidden_dim, self.n_heads)
        self.mlp = MLP(self.mlp_dim)
        self.layer_norm = nn.LayerNorm(epsilon=1e-6)


    def __call__(self, inputs, train=True):
        # Attention Block
        x = self.layer_norm(inputs)
        x = self.mha(x, train)
        x = inputs + x
        # MLP block
        y = self.layer_norm(x)
        y = self.mlp(y, train)

        return x + y


class ViT(nn.Module):
    patch_size: int
    embed_dim: int
    hidden_dim: int
    n_heads: int
    num_layers: int
    mlp_dim: int
    num_classes: int

    def setup(self):
        self.lin_to_square = LinearToSquare()
        self.patch_extracter = Patches(self.patch_size, self.embed_dim)
        self.patch_encoder = PatchEncoder(self.hidden_dim)
        self.transformer_blocks = [
            Transformer(
                self.embed_dim, self.hidden_dim, self.n_heads, self.mlp_dim
            )
            for _ in range(self.num_layers)
        ]
        self.mlp_head = MLP(self.mlp_dim)
        self.cls_head = nn.Dense(features=self.num_classes)

    def __call__(self, x, train=True):
        x = self.lin_to_square(x)  # Call the __call__ method of the instance
        x = self.patch_extracter(x)
        x = self.patch_encoder(x)
        for block in self.transformer_blocks:
            x = block(x, train)
        # MLP head
        x = x[:, 0]  # [CLS] token
        x = self.mlp_head(x, train)
        x = self.cls_head(x)
        x = jnp.squeeze(x)
        return x

class ComplexViT(nn.Module):
    # Parameters for the ViT model
    patch_size: int
    embed_dim: int
    hidden_dim: int
    n_heads: int
    num_layers: int
    mlp_dim: int

    def setup(self):
        # Instantiate two ViT models for real and imaginary parts
        self.vit_real = ViT(
            patch_size=self.patch_size,
            embed_dim=self.embed_dim,
            hidden_dim=self.hidden_dim,
            n_heads=self.n_heads,
            num_layers=self.num_layers,
            mlp_dim=self.mlp_dim,
            num_classes=1,
        )
        self.vit_imag = ViT(
            patch_size=self.patch_size,
            embed_dim=self.embed_dim,
            hidden_dim=self.hidden_dim,
            n_heads=self.n_heads,
            num_layers=self.num_layers,
            mlp_dim=self.mlp_dim,
            num_classes=1,
        )

    def __call__(self, x, train=True):
        # Compute real and imaginary parts
        real_part = self.vit_real(x, train)
        imag_part = self.vit_imag(x, train)
        log_psi = real_part + 1j * imag_part
        return log_psi


machine = ComplexViT(
    patch_size=1, embed_dim=16, hidden_dim=16,
    n_heads=2, num_layers=2, mlp_dim=32
)
print_model_info(machine)

n_chains = 128
n_samples = 4096
n_discard_per_chain = 10  # Number of samples to discard per chain
chunk_size = 4096

sampler = nk.sampler.MetropolisExchange(hi, n_chains=1024, graph=lattice, d_max=L, dtype=jnp.int8)
vstate = nk.vqs.MCState(sampler=sampler,model=machine,n_samples=n_samples,n_discard_per_chain=n_discard_per_chain,chunk_size=chunk_size)

import optax

lr = 0.1
opt = nk.optimizer.Sgd(learning_rate=lr)
clip = optax.clip_by_global_norm(max_norm=1/lr)
opt = optax.chain(optax.zero_nans(), clip, opt)
sr = nk.optimizer.SR(diag_shift=1e-4, holomorphic=False)

gs = nk.driver.VMC(H, opt, variational_state=vstate, preconditioner=sr)
gs.run(n_iter=100, out="vit_ground_state")

data = json.load(open("vit_ground_state.log"))
print("Energy averaged over last ten steps:", np.mean(data["Energy"]["Mean"]["real"][-10:]))
print("Energy per site averaged over last ten steps:", np.mean(data["Energy"]["Mean"]["real"][-10:]) / (lattice.n_nodes))
print("Energy std over last ten steps:", np.std(data["Energy"]["Mean"]["real"][-10:]) / (lattice.n_nodes))
