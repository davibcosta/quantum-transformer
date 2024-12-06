import os

import netket as nk
import numpy as np
import json
from math import pi
import jax
import flax
import jax.numpy as jnp
from scipy.sparse.linalg import eigsh
from flax import linen as nn
from typing import Any
import optax
import netket.experimental as nkx
import csv

N = 16
hi = nk.hilbert.Spin(s=1 / 2, N=N, inverted_ordering=False)

from netket.operator.spin import sigmax,sigmaz 

Gamma = -1
H = sum([Gamma*sigmax(hi,i) for i in range(N)])
g = 1
V = Gamma*g
H += sum([V*sigmaz(hi,i)*sigmaz(hi,(i+1)%N) for i in range(N)])

sparse_h = H.to_sparse()

eig_vals, eig_vecs = eigsh(sparse_h, k=2, which="SA")

print("eigenvalues with scipy sparse:", eig_vals)

E_gs = eig_vals[0]

input_batch = hi.random_state(size=64, key=jax.random.PRNGKey(0))
input_batch = jnp.array(input_batch)

def print_model_info(model):
    """Prints the model architecture and parameter count."""
    key = jax.random.PRNGKey(seed)
    params = model.init(key, input_batch)
    dtypes = jax.tree_util.tree_map(lambda x: x.dtype, params)
    print(dtypes)
    print("Model Architecture:")
    print(model)
    total_params = sum(jnp.prod(jnp.array(param.shape)) for param in jax.tree_util.tree_leaves(params))
    print(f"Total Trainable Parameters: {total_params}")

def linear_solver_pinv_smooth(rcond=1e-6):
    return lambda A, b: nk.optimizer.solver.pinv_smooth(A, b, rtol=rcond, rtol_smooth=rcond)[0]


seed = 0
rng = jax.random.PRNGKey(seed)
rng, model_rng, sampler_rng = jax.random.split(rng, 3)

class ViTNQS(nn.Module):
    Lx: int  # Lattice size in x direction (total sites for 1D input)
    patch_size: int
    d_model: int  # Embedding dimension
    num_heads: int
    num_layers: int
    param_dtype: Any = jnp.complex64

    @nn.compact
    def __call__(self, σ):
        # σ: (batch_size, N_sites)
        # N_sites = Lx
        batch_size = σ.shape[0]
        N_sites = σ.shape[1]

        # Extract patches: (batch_size, n_patches, patch_size)
        patches = extract_patches_1d(σ, self.patch_size)

        # Flatten patches: (batch_size, n_patches, patch_size)
        # No reshaping required beyond patch extraction

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
        return log_psi


def extract_patches_1d(σ, patch_size):
    """
    Extract patches for 1D input without reshaping.
    Args:
        σ: (batch_size, Lx), where Lx is the lattice size.
        patch_size: Size of each patch.
    Returns:
        patches: (batch_size, n_patches, patch_size), where n_patches = Lx // patch_size.
    """
    batch_size, Lx = σ.shape
    n_patches = Lx // patch_size

    # Truncate σ to ensure full patches
    σ = σ[:, :n_patches * patch_size]

    # Split into patches along the last axis
    patches = σ.reshape(batch_size, n_patches, patch_size)
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

machine = ViTNQS(
    Lx=N,
    patch_size=4,
    d_model=32,
    num_heads=2,
    num_layers=1,
    param_dtype=jnp.complex64
)

print_model_info(machine)


lrs = [0.1]
rconds = [1e-6]
diag_shifts = [1e-4]

n_chains = 32
n_samples = 4096
n_discard_per_chain = 0
chunk_size = 4096

iterations = 200
timeout = 1200
max_norm = 1.0
clip = optax.clip_by_global_norm(max_norm=max_norm)

for lr in lrs:
    for rcond in rconds:
        for diag_shift in diag_shifts:
            print(f"Running for lr={lr} and rcond={rcond}, diag_shift={diag_shift}")

            sampler = nk.sampler.MetropolisLocal(
                hi,
                n_chains=n_chains,
                dtype=jnp.int8
            )

            # Initialize variational state with fixed seed
            vstate_rng, rng = jax.random.split(rng)
            vstate = nk.vqs.MCState(
                sampler=sampler,
                model=machine,
                n_samples=n_samples,
                n_discard_per_chain=n_discard_per_chain,
                chunk_size=chunk_size,
                seed=vstate_rng  # Set variational state seed
            )
            
            # Define optimizer
            lr_schedule = optax.cosine_decay_schedule(init_value=lr, decay_steps=iterations)
            optimizer = optax.chain(
                optax.zero_nans(),
                clip,
                nk.optimizer.Sgd(learning_rate=lr_schedule)
            )
            
            gs = nkx.driver.VMC_SRt(
                H,
                optimizer,
                diag_shift=diag_shift,
                variational_state=vstate,
                jacobian_mode="complex",
                linear_solver_fn=linear_solver_pinv_smooth(rcond=rcond)
            )
            
            # Run optimization
            gs.run(
                n_iter=iterations,
                out=f"state_lr_{lr}_rcond_{rcond}_shift_{diag_shift}",
                callback=[
                    nk.callbacks.Timeout(timeout=timeout),
                    nk.callbacks.InvalidLossStopping(monitor="mean", patience=1)
                ]
            )
    
            # Save results
            data = json.load(open(f"state_lr_{lr}_rcond_{rcond}_shift_{diag_shift}.log"))