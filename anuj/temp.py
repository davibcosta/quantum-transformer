# Import necessary libraries
import netket.experimental as nkx
import functools
from functools import partial
import flax.linen as nn
import numpy as np
import jax.numpy as jnp
import flax
import optax
import csv
import numpy as np
import os
import netket.experimental as nkx
import sys 
from math import pi
import json 

os.environ["JAX_PLATFORM_NAME"] = "cuda"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import jax
# Check NetKet installation and print version
import netket as nk
print(f"NetKet version: {nk.__version__}")
# Print available JAX devices for the current process
print(jax.devices())


L = 4
# Build square lattice with nearest and next-nearest neighbor edges
lattice = nk.graph.Square(L, max_neighbor_order=2)
hi = nk.hilbert.Spin(s=1 / 2, N=lattice.n_nodes, inverted_ordering=False)
# Heisenberg with coupling J=1.0 for nearest neighbors
# and J=0.5 for next-nearest neighbors
H = nk.operator.Heisenberg(hilbert=hi, graph=lattice, J=[1.0, 0.51])

# sparse_ham = H.to_sparse()
# sparse_ham.shape

# from scipy.sparse.linalg import eigsh

# eig_vals, eig_vecs = eigsh(sparse_ham, k=2, which="SA")

# print("eigenvalues with scipy sparse:", eig_vals)

# E_gs = eig_vals[0]
# print("Ground state energy from Exact Diagonalization:", E_gs)

class FFN(nn.Module):
    alpha : int = 1
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(features=self.alpha * x.shape[-1], 
                     use_bias=True, 
                     param_dtype=np.complex128, 
                     kernel_init=nn.initializers.normal(stddev=0.01), 
                     bias_init=nn.initializers.normal(stddev=0.01)
                    )(x)
        x = nk.nn.log_cosh(x)
        x = jnp.sum(x, axis=-1)
        return x

model = FFN(alpha=2)
sampler = nk.sampler.MetropolisLocal(hi)
vstate = nk.vqs.MCState(sampler, model, n_samples=1024)

optimizer = nk.optimizer.Sgd(learning_rate=0.01)

sr = nk.optimizer.SR(diag_shift=1e-6, holomorphic=False)
# Notice the use, again of Stochastic Reconfiguration, which considerably improves the optimisation
gs = nk.driver.VMC(H, optimizer, variational_state=vstate,preconditioner=sr)

# log=nk.logging.RuntimeLog()
# gs.run(n_iter=100,out=log)

# ffn_energy=vstate.expect(H)
# error=abs((ffn_energy.mean-E_gs)/E_gs)
# print("Optimized energy and relative error: ",ffn_energy,error)

import flax.linen as nn
import jax
import jax.numpy as jnp


jax.config.update("jax_enable_x64", True)


def reshape_to_LxL(x: jnp.ndarray, L: int) -> jnp.ndarray:
    batch_size, flattened_size = x.shape
    assert flattened_size == L * L, f"Input size {flattened_size} must match L^2 = {L**2}"
    return x.reshape(batch_size, L, L).astype(jnp.complex128)


def extract_patches(x: jnp.ndarray, patch_size: int) -> jnp.ndarray:
    batch_size, L, _ = x.shape
    assert L % patch_size == 0, f"Lattice size {L} must be divisible by patch size {patch_size}."
    num_patches_per_dim = L // patch_size
    num_patches = num_patches_per_dim ** 2
    patch_dim = patch_size ** 2
    x = x.reshape(batch_size, num_patches_per_dim, patch_size, num_patches_per_dim, patch_size)
    x = x.transpose(0, 1, 3, 2, 4)
    return x.reshape(batch_size, num_patches, patch_dim).astype(jnp.complex128)


class PatchEmbedding(nn.Module):
    embed_dim: int

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = nn.Dense(features=self.embed_dim, use_bias=True, param_dtype=jnp.complex128, dtype=jnp.complex128)(x)
        return x


class MultiHeadSelfAttention(nn.Module):
    num_heads: int
    embed_dim: int

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        head_dim = self.embed_dim // self.num_heads
        assert self.embed_dim % self.num_heads == 0, "Embedding dimension must be divisible by number of heads"
        q = nn.Dense(self.embed_dim, use_bias=False, param_dtype=jnp.complex128, dtype=jnp.complex128)(x)
        k = nn.Dense(self.embed_dim, use_bias=False, param_dtype=jnp.complex128, dtype=jnp.complex128)(x)
        v = nn.Dense(self.embed_dim, use_bias=False, param_dtype=jnp.complex128, dtype=jnp.complex128)(x)
        def split_heads(tensor):
            batch_size, num_patches, _ = tensor.shape
            return tensor.reshape(batch_size, num_patches, self.num_heads, head_dim).transpose(0, 2, 1, 3)
        q = split_heads(q)
        k = split_heads(k)
        v = split_heads(v)
        scaling_factor = jnp.array(head_dim ** -0.5, dtype=jnp.complex128)
        attn_scores = jnp.einsum('bhqd,bhkd->bhqk', q, k) * scaling_factor
        attn_weights = nn.softmax(jnp.abs(attn_scores), axis=-1) * jnp.exp(1j * jnp.angle(attn_scores))
        attn_output = jnp.einsum('bhqk,bhkd->bhqd', attn_weights, v)
        attn_output = attn_output.transpose(0, 2, 1, 3).reshape(x.shape[0], x.shape[1], self.embed_dim)
        return nn.Dense(self.embed_dim, use_bias=False, param_dtype=jnp.complex128, dtype=jnp.complex128)(attn_output)


class FeedForwardNetwork(nn.Module):
    hidden_dim: int
    embed_dim: int

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = nn.Dense(self.hidden_dim, use_bias=True, param_dtype=jnp.complex128, dtype=jnp.complex128)(x)
        x = nk.nn.reim_selu(x)
        x = nn.Dense(self.embed_dim, use_bias=True, param_dtype=jnp.complex128, dtype=jnp.complex128)(x)
        return x


class TransformerEncoderBlock(nn.Module):
    num_heads: int
    embed_dim: int
    hidden_dim: int

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        attn_output = MultiHeadSelfAttention(self.num_heads, self.embed_dim)(x)
        x = x + attn_output
        ffn_output = FeedForwardNetwork(self.hidden_dim, self.embed_dim)(x)
        x = x + ffn_output
        return x


def fixed_sine_cosine_embeddings(num_patches, embed_dim):
    """
    Generate fixed sine-cosine positional embeddings.
    Args:
        num_patches: Number of patches.
        embed_dim: Embedding dimension.

    Returns:
        Positional embeddings of shape (num_patches, embed_dim).
    """
    position = jnp.arange(num_patches)[:, None]  # Shape: (num_patches, 1)
    div_term = jnp.exp(jnp.arange(0, embed_dim, 2) * -(jnp.log(10000.0) / embed_dim))
    pos_embed = jnp.zeros((num_patches, embed_dim), dtype=jnp.float32)
    pos_embed = pos_embed.at[:, 0::2].set(jnp.sin(position * div_term))
    pos_embed = pos_embed.at[:, 1::2].set(jnp.cos(position * div_term))
    return pos_embed.astype(jnp.complex128)


def fixed_relative_positional_embeddings(grid_size, embed_dim):
    """
    Generate fixed 2D relative positional embeddings.

    Args:
        grid_size: Tuple (H, W) representing the height and width of the grid.
        embed_dim: Embedding dimension.

    Returns:
        A tensor of shape (H, W, embed_dim) with fixed relative positional embeddings.
    """
    H, W = grid_size
    max_rel_dist = max(H, W)

    # Create relative distance matrices for rows and columns
    rel_row = jnp.arange(-max_rel_dist, max_rel_dist + 1)[:, None]  # Shape: (2*H-1, 1)
    rel_col = jnp.arange(-max_rel_dist, max_rel_dist + 1)[None, :]  # Shape: (1, 2*W-1)

    # Define scaling for sine-cosine encoding
    div_term = jnp.exp(jnp.arange(0, embed_dim // 2, 2) * -(jnp.log(10000.0) / (embed_dim // 2)))

    # Adjust shapes for broadcasting
    rel_row = rel_row[:, :, None]  # Shape: (2*H-1, 1, 1)
    rel_col = rel_col[:, :, None]  # Shape: (1, 2*W-1, 1)
    div_term = div_term[None, None, :]  # Shape: (1, 1, embed_dim // 2)

    # Compute sine-cosine embeddings for rows and columns
    row_embed = jnp.sin(rel_row * div_term)
    col_embed = jnp.cos(rel_col * div_term)

    # Combine row and column embeddings
    rel_embed = row_embed + col_embed  # Shape: (2*H-1, 2*W-1, embed_dim // 2)

    return rel_embed


class VisionTransformer(nn.Module):
    L: int
    num_layers: int
    num_heads: int
    embed_dim: int
    hidden_dim: int
    patch_size: int
    num_patches: int

    @nn.compact
    def __call__(self, x: jnp.ndarray, return_intermediate: bool = False) -> jnp.ndarray:
        """
        Vision Transformer forward pass with optional intermediate outputs.

        Args:
            x: Input tensor of shape (batch_size, L**2), dtype=complex.
            return_intermediate: Whether to return intermediate layer outputs.

        Returns:
            Output tensor or dictionary of intermediate outputs.
        """
        x = x.astype(jnp.complex128)
        x = reshape_to_LxL(x, self.L)
        patches = extract_patches(x, self.patch_size)
        x = PatchEmbedding(self.embed_dim)(patches)

        #pos_embed = self.param("pos_embed", nn.initializers.normal(stddev=0.01), (self.num_patches, self.embed_dim))
        #pos_embed = fixed_sine_cosine_embeddings(self.num_patches, self.embed_dim)
        #x += pos_embed.astype(jnp.complex128)

        grid_size = (self.L, self.L)
        rel_pos_embed = fixed_relative_positional_embeddings(grid_size, embed_dim)
        # Add these embeddings to the input patches
        x = x + rel_pos_embed[:grid_size[0], :grid_size[1], :]

        intermediate_outputs = {} if return_intermediate else None

        for i in range(self.num_layers):
            x = TransformerEncoderBlock(self.num_heads, self.embed_dim, self.hidden_dim)(x)
            if return_intermediate:
                intermediate_outputs[f"layer_{i}"] = x

        x = x.mean(axis=1)
        x = nn.Dense(1, param_dtype=jnp.complex128, dtype=jnp.complex128)(x)

        x = jnp.squeeze(x, axis=-1)
        if return_intermediate:
            intermediate_outputs["output"] = x
            return intermediate_outputs

        return x


def initialise_vstate(vstate, model, width, x, key, stddev_init, debug=False):
    """
    Initializes vstate parameters using LSUV (Layer-Sequential Unit Variance) scaling.

    Args:
        vstate: Initial state of the variational model.
        model: Flax model to apply.
        width: Scaling factor for target variance.
        x: Input data without a batch dimension.
        key: JAX random key.
        stddev_init: Initial standard deviation for parameter initialization.
        debug: Enable verbose debugging output.

    Returns:
        Updated layer parameter dictionary with initialized weights.
    """
    print("Initializing vstate via LSUV")

    # Save dictionary of parameter shapes and types
    params_vstate = {'params': flax.core.copy(vstate.parameters, {})}
    layer_shape_dict = jax.tree_util.tree_map(lambda x: x.shape, params_vstate)['params']
    layer_param_dict = params_vstate['params']

    if debug:
        print("Layer shape dict:", layer_shape_dict)

    # Get initial weights, intermediate outputs, and variances
    intermediate_outputs = model.apply(params_vstate, x, return_intermediate=True)
    variances = {
        layer_name: jnp.sum(jnp.var(output, axis=0)).item()
        for layer_name, output in intermediate_outputs.items()
        if layer_name in layer_shape_dict
    }

    # Target variance for each layer
    target_variance = 1.0 / width
    tolerance = 0.01 * target_variance

    # Iterate over each layer
    for layer_name, layer_variance in variances.items():
        if debug:
            print(f"Processing layer: {layer_name}")
        for param_name, param_shape in layer_shape_dict.get(layer_name, {}).items():
            param_dtype = layer_param_dict[layer_name][param_name].dtype
            std_found = False
            new_stddev = stddev_init

            # Initialize bias to zero
            if param_name == 'bias':
                layer_param_dict[layer_name][param_name] = nn.initializers.zeros(dtype=param_dtype)(key, param_shape)
                continue

            # Iteratively rescale weights
            iters = 0
            while not std_found:
                # Safeguard against NaN variances
                layer_variance = variances.get(layer_name, target_variance)
                if jnp.isnan(layer_variance) or layer_variance == 0:
                    layer_variance = target_variance

                # Compute new standard deviation
                scaling_factor = jnp.sqrt(target_variance).item() * new_stddev * (1.0 / jnp.sqrt(layer_variance).item())
                new_param = nn.initializers.normal(stddev=scaling_factor, dtype=param_dtype)(key, param_shape)

                # Update parameter and re-evaluate
                layer_param_dict[layer_name][param_name] = new_param
                key, subkey = jax.random.split(key)  # Ensure a new key for each iteration
                new_params = {'params': layer_param_dict}
                intermediate_outputs = model.apply(new_params, x, return_intermediate=True)

                # Update variances
                variances = {
                    layer: jnp.sum(jnp.var(output, axis=0)).item()
                    for layer, output in intermediate_outputs.items()
                }
                layer_variance = variances.get(layer_name, target_variance)

                if debug:
                    print(f"Layer: {layer_name}, Param: {param_name}, Iter: {iters}, Variance: {layer_variance}")

                # Check if variance is within tolerance
                if abs(layer_variance - target_variance) < tolerance:
                    std_found = True

                # Break after too many iterations
                iters += 1
                if iters > 10:
                    print(f"Warning: Variance initialization failed for {layer_name}/{param_name} after 10 iterations.")
                    std_found = True

    return layer_param_dict




patch_size = 2
num_patches = (L // patch_size) ** 2
num_layers = 2
num_heads = 1
embed_dim = 16
hidden_dim = 16
batch_size = 2048

x = hi.random_state(size=batch_size, key=jax.random.PRNGKey(0))
print(x.shape)

model = VisionTransformer(L=L, num_layers=num_layers, num_heads=num_heads, embed_dim=embed_dim, 
                           hidden_dim=hidden_dim, patch_size=patch_size, num_patches=num_patches)

random_params = model.init(jax.random.PRNGKey(0), x.astype(jnp.complex128))
random_output = model.apply(random_params, x)
print("Random Output shape:", random_output.shape)
print("Random Output:", random_output)


sampler = nk.sampler.MetropolisLocal(hi)
vstate = nk.vqs.MCState(sampler, model, n_samples=4096)

width = 1
key = jax.random.PRNGKey(0)
stddev_init = 0.01

params = initialise_vstate(vstate, model, width=width, x=x, key=key, stddev_init=stddev_init, debug=True)

post_init_params = {'params': flax.core.copy(vstate.parameters, {})}
post_init_output = model.apply(post_init_params, x.astype(jnp.complex128), return_intermediate=False)
print(post_init_output)

clip = optax.clip_by_global_norm(max_norm=1e-2)
optimizer = nk.optimizer.Sgd(learning_rate=0.01)
optimizer = optax.chain(optax.zero_nans(), clip, optimizer)

def linear_solver_pinv_smooth(rcond=1e-6):
    return lambda A, b: nk.optimizer.solver.pinv_smooth(A, b, rtol=rcond, rtol_smooth=rcond)[0]

solver = linear_solver_pinv_smooth(1e-8)

#sr = nk.optimizer.SR(diag_shift=1e-2, holomorphic=False)
# Notice the use, again of Stochastic Reconfiguration, which considerably improves the optimisation
#gs = nk.driver.VMC(H, optimizer, variational_state=vstate,preconditioner=sr)

gs = nkx.driver.VMC_SRt(H,optimizer,diag_shift=0.0,variational_state=vstate,jacobian_mode="complex",linear_solver_fn=solver)

log=nk.logging.RuntimeLog()
gs.run(n_iter=500,out=log)
