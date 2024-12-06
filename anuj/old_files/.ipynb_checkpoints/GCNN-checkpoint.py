import netket as nk
import numpy as np
import json
from math import pi
import jax
import jax.numpy as jnp

L = 4
# Build square lattice with nearest and next-nearest neighbor edges
lattice = nk.graph.Square(L, max_neighbor_order=2)
print(lattice.n_nodes)
hi = nk.hilbert.Spin(s=1/2, total_sz=0, N=lattice.n_nodes, inverted_ordering=False)
J2 = 0.5
H = nk.operator.Heisenberg(hilbert=hi, graph=lattice, J=[1.0, J2])

from scipy.sparse.linalg import eigsh

eig_vals, eig_vecs = eigsh(H.to_sparse(), k=2, which="SA")

print("eigenvalues with scipy sparse:", eig_vals)

E_gs = eig_vals[0]

input = hi.random_state(size=64, key=jax.random.PRNGKey(1))
input = jnp.array(input)
print(input)
print(input.shape)


# Print model architecture and parameter count
def print_model_info(model):
    """Prints the model architecture and parameter count."""
    key = jax.random.PRNGKey(0)
    params = model.init(key, input)
    dtypes = jax.tree_util.tree_map(lambda x: x.dtype, params)
    print(dtypes)
    print("Model Architecture:")
    print(model)
    total_params = sum(jnp.prod(jnp.array(param.shape)) for param in jax.tree_util.tree_leaves(params))
    print(f"Total Trainable Parameters: {total_params}")

# # Find an approximate ground state
# machine = nk.models.GCNN(
#     symmetries=lattice,
#     parity=1,
#     layers=4,
#     features=4,
#     param_dtype=complex,
# )

from flax import linen as nn
from typing import Any

# Define the custom ViT-based NQS model with factored attention
class ViTNQS(nn.Module):
    Lx: int  # Lattice size in x direction
    Ly: int  # Lattice size in y direction
    patch_size: int
    d_model: int  # Embedding dimension
    num_heads: int
    num_layers: int
    param_dtype: Any = jnp.complex64

    @nn.compact
    def __call__(self, σ):
        # σ: (batch_size, N_sites)
        # N_sites = Lx * Ly
        batch_size = σ.shape[0]
        N_sites = σ.shape[1]

        # Reshape σ to (batch_size, Lx, Ly)
        σ = σ.reshape((batch_size, self.Lx, self.Ly))

        # Extract patches: (batch_size, n_patches, patch_size, patch_size)
        patches = extract_patches(σ, self.patch_size)

        # Flatten patches: (batch_size, n_patches, patch_size * patch_size)
        patches = patches.reshape((batch_size, -1, self.patch_size * self.patch_size))

        # Linear embedding of patches into d-dimensional space
        patch_embedding = nn.Dense(self.d_model, use_bias=True, param_dtype=self.param_dtype, name='patch_embedding')
        x = patch_embedding(patches)  # (batch_size, n_patches, d_model)

        # Positional encoding
        n_patches = x.shape[1]
        pos_embedding = self.param('pos_embedding', nn.initializers.normal(stddev=0.05), (1, n_patches, self.d_model), self.param_dtype)
        x = x + pos_embedding

        # ViT encoder blocks
        for _ in range(self.num_layers):
            x = ViTEncoderBlock(d_model=self.d_model, num_heads=self.num_heads, param_dtype=self.param_dtype)(x)

        # Hidden representation z
        z = jnp.sum(x, axis=1)  # (batch_size, d_model)

        # Final mapping to complex logarithm of amplitude
        # Parameters w, b
        w = self.param('w', nn.initializers.normal(stddev=0.1), (self.d_model,), jnp.complex64)
        b = self.param('b', nn.initializers.normal(stddev=0.1), (self.d_model,), jnp.complex64)

        # Compute pre-activation: (batch_size, d_model)
        pre_activation = w * z + b

        # Apply g(·) = logcosh(·)
        g = lambda x: jnp.log(jnp.cosh(x))

        # Handle complex inputs in logcosh
        log_psi = jnp.sum(g(pre_activation), axis=-1)  # (batch_size,)
        #print(log_psi)
        return log_psi

def extract_patches(σ, patch_size):
    # σ: (batch_size, Lx, Ly)
    batch_size, Lx, Ly = σ.shape
    n_patches_x = Lx // patch_size
    n_patches_y = Ly // patch_size

    # Reshape to get patches
    σ = σ.reshape(batch_size, n_patches_x, patch_size, n_patches_y, patch_size)
    σ = σ.transpose(0, 1, 3, 2, 4)
    patches = σ.reshape(batch_size, n_patches_x * n_patches_y, patch_size, patch_size)
    return patches

class ViTEncoderBlock(nn.Module):
    d_model: int
    num_heads: int
    param_dtype: Any = jnp.complex64

    @nn.compact
    def __call__(self, x):
        # Pre-Layer Normalization
        y = nn.LayerNorm(dtype=self.param_dtype, param_dtype=self.param_dtype)(x)
        # Factored Multi-Head Attention
        y = FactoredMultiHeadAttention(num_heads=self.num_heads, d_model=self.d_model, param_dtype=self.param_dtype)(y)
        # Skip connection
        x = x + y

        # Pre-Layer Normalization
        y = nn.LayerNorm(dtype=self.param_dtype, param_dtype=self.param_dtype)(x)
        # Feedforward network
        y = nn.Dense(4*self.d_model, dtype=self.param_dtype, param_dtype=self.param_dtype)(y)
        #y = nn.Dense(self.d_model, dtype=self.param_dtype, param_dtype=self.param_dtype)(y)
        y = nn.gelu(y)
        y = nn.Dense(self.d_model, dtype=self.param_dtype, param_dtype=self.param_dtype)(y)
        # Skip connection
        x = x + y

        return x

class FactoredMultiHeadAttention(nn.Module):
    num_heads: int
    d_model: int
    param_dtype: Any = jnp.complex64

    @nn.compact
    def __call__(self, x):
        batch_size, n_patches, _ = x.shape
        head_dim = self.d_model // self.num_heads

        # Linear projection for V: Values
        V = nn.DenseGeneral((self.num_heads, head_dim), axis=-1, dtype=self.param_dtype, param_dtype=self.param_dtype, use_bias=False, name='V_proj')(x)
        V = V.transpose(0, 2, 1, 3)  # (batch_size, num_heads, n_patches, head_dim)

        # Compute attention weights αF_ij = p_{i-j}
        positions = jnp.arange(n_patches)
        relative_positions = positions[None, :] - positions[:, None]  # (n_patches, n_patches)

        # Positional biases p: (num_heads, 2 * n_patches - 1)
        p = self.param('p', nn.initializers.normal(stddev=0.02), (self.num_heads, 2 * n_patches - 1), self.param_dtype)

        # Map relative positions to indices
        relative_position_indices = relative_positions + n_patches - 1  # Shift indices to be >=0

        # Get attention weights: (num_heads, n_patches, n_patches)
        attention_weights = p[:, relative_position_indices]

        # Expand to include batch dimension: (batch_size, num_heads, n_patches, n_patches)
        attention_weights = jnp.broadcast_to(attention_weights[None, :, :, :], (batch_size, self.num_heads, n_patches, n_patches))

        # # Softmax over the key dimension
        # attention_weights = nn.softmax(attention_weights, axis=-1)

        # Compute attention output
        attention_output = jnp.matmul(attention_weights, V)  # (batch_size, num_heads, n_patches, head_dim)

        # Reshape and project back to d_model
        attention_output = attention_output.transpose(0, 2, 1, 3).reshape(batch_size, n_patches, self.d_model)
        output = nn.Dense(self.d_model, dtype=self.param_dtype, param_dtype=self.param_dtype, use_bias=False, name='output_proj')(attention_output)

        return output

# # Instantiate the custom ViT-based NQS model
# machine = ViTNQS(
#     Lx=L,
#     Ly=L,
#     patch_size=2,
#     d_model=60,
#     num_heads=10,
#     num_layers=4,
#     param_dtype=jnp.float32
# )

# machine = ViTNQS(
#     Lx=L,
#     Ly=L,
#     patch_size=2,
#     d_model=60,
#     num_heads=10,
#     num_layers=4,
#     param_dtype=jnp.complex64
# )

machine = ViTNQS(
    Lx=L,
    Ly=L,
    patch_size=2,
    d_model=60,
    num_heads=1,
    num_layers=1,
    param_dtype=jnp.complex64
)


print_model_info(machine)

n_chains = 128
n_samples = 4096
n_discard_per_chain = 10  # Number of samples to discard per chain
chunk_size = 4096

sampler = nk.sampler.MetropolisExchange(hi, n_chains=1024, graph=lattice, d_max=2, dtype=jnp.int8)
vstate = nk.vqs.MCState(sampler=sampler,model=machine,n_samples=n_samples,n_discard_per_chain=n_discard_per_chain,chunk_size=chunk_size)

import optax
lr = 0.005
clip = optax.clip_by_global_norm(max_norm=1.0/lr)
opt = nk.optimizer.Sgd(learning_rate=lr)
opt = optax.chain(optax.zero_nans(), clip, opt)

#diag_shift = 1e-2
diag_shift = 1e-2
sr = nk.optimizer.SR(diag_shift=diag_shift, holomorphic=False)
gs = nk.driver.VMC(H, opt, variational_state=vstate, preconditioner=sr)

import netket.experimental as nkx
def linear_solver_pinv_smooth(rcond=1e-6):
    return lambda A, b: nk.optimizer.solver.pinv_smooth(A, b, rtol=rcond, rtol_smooth=rcond)[0]
    
# gs = nkx.driver.VMC_SRt(H, opt, diag_shift=0, variational_state=vstate, jacobian_mode="complex", linear_solver_fn=linear_solver_pinv_smooth(rcond=1e-4))
# gs = nkx.driver.VMC_SRt(H, opt, diag_shift=diag_shift, variational_state=vstate, jacobian_mode="complex")

gs.run(n_iter=2500, out="ground_state")

data = json.load(open("ground_state.log"))
print("Energy averaged over last ten steps:", np.mean(data["Energy"]["Mean"]["real"][-10:]))
print("Energy per site averaged over last ten steps:", np.mean(data["Energy"]["Mean"]["real"][-10:]) / (lattice.n_nodes))
print("Energy std over last ten steps:", np.std(data["Energy"]["Mean"]["real"][-10:]) / (lattice.n_nodes))

