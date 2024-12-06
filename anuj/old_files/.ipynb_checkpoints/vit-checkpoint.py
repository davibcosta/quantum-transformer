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
import math

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

class FFN(nn.Module):
    alpha : int = 1
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(features=self.alpha * x.shape[-1], 
                     use_bias=True, 
                     param_dtype=jnp.complex128, 
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

def masked_softmax(X, valid_lens):
    """Perform softmax operation by masking elements on the last axis.

    Defined in :numref:sec_attention-scoring-functions"""
    # X: 3D tensor, valid_lens: 1D or 2D tensor
    def _sequence_mask(X, valid_len, value=0):
        maxlen = X.shape[1]
        mask = jnp.arange((maxlen),
                          dtype=jnp.float64)[None, :] < valid_len[:, None]
        return jnp.where(mask, X, value)

    if valid_lens is None:
        return nn.softmax(X, axis=-1)
    else:
        shape = X.shape
        if valid_lens.ndim == 1:
            valid_lens = jnp.repeat(valid_lens, shape[1])
        else:
            valid_lens = valid_lens.reshape(-1)
        # On the last axis, replace masked elements with a very large negative
        # value, whose exponentiation outputs 0
        X = _sequence_mask(X.reshape(-1, shape[-1]), valid_lens, value=-1e6)
        return nn.softmax(X.reshape(shape), axis=-1)


def reshape_to_LxL(x: jnp.ndarray, L: int) -> jnp.ndarray:
    batch_size, flattened_size = x.shape
    assert flattened_size == L * L, f"Input size {flattened_size} must match L^2 = {L**2}"
    return x.reshape(batch_size, L, L)

class PatchEmbedding(nn.Module):
    img_size: int = 8
    patch_size: int = 2
    num_hiddens: int = 32

    def setup(self):
        def _make_tuple(x):
            if not isinstance(x, (list, tuple)):
                return (x, x)
            return x
        img_size, patch_size = _make_tuple(self.img_size), _make_tuple(self.patch_size)
        self.num_patches = (img_size[0] // patch_size[0]) * (
            img_size[1] // patch_size[1])
        self.conv = nn.Conv(self.num_hiddens, kernel_size=patch_size,
                            strides=patch_size, padding='SAME', param_dtype=jnp.float64)

    def __call__(self, X):
        X = self.conv(X)
        return X.reshape((X.shape[0], -1, X.shape[3]))

class ViTMLP(nn.Module):
    mlp_num_hiddens: int
    mlp_num_outputs: int

    @nn.compact
    def __call__(self, x, training=False):
        x = nn.Dense(self.mlp_num_hiddens, param_dtype=jnp.float64)(x)
        x = nn.gelu(x)
        x = nn.Dense(self.mlp_num_outputs, param_dtype=jnp.float64)(x)
        return x

class DotProductAttention(nn.Module):
    @nn.compact
    def __call__(self, queries, keys, values, valid_lens=None, training=False):
        d = queries.shape[-1]
        scores = queries @ (keys.swapaxes(1, 2)) / jnp.sqrt(d)
        attention_weights = nn.softmax(scores, axis=-1)
        return attention_weights @ values, attention_weights

class MultiHeadAttention(nn.Module):
    num_hiddens: int
    num_heads: int
    bias: bool = False

    def setup(self):
        self.attention = DotProductAttention()
        self.W_q = nn.Dense(self.num_hiddens, use_bias=self.bias, param_dtype=jnp.float64)
        self.W_k = nn.Dense(self.num_hiddens, use_bias=self.bias, param_dtype=jnp.float64)
        self.W_v = nn.Dense(self.num_hiddens, use_bias=self.bias, param_dtype=jnp.float64)
        self.W_o = nn.Dense(self.num_hiddens, use_bias=self.bias, param_dtype=jnp.float64)

    @nn.compact
    def __call__(self, queries, keys, values, valid_lens, training=False):
        queries = self.transpose_qkv(self.W_q(queries))
        keys = self.transpose_qkv(self.W_k(keys))
        values = self.transpose_qkv(self.W_v(values))

        if valid_lens is not None:
            valid_lens = jnp.repeat(valid_lens, self.num_heads, axis=0)

        output, attention_weights = self.attention(queries, keys, values, valid_lens, training=training)
        output_concat = self.transpose_output(output)
        return self.W_o(output_concat), attention_weights

    def transpose_qkv(self, X):
        X = X.reshape((X.shape[0], X.shape[1], self.num_heads, -1))
        X = jnp.transpose(X, (0, 2, 1, 3))
        return X.reshape((-1, X.shape[2], X.shape[3]))

    def transpose_output(self, X):
        X = X.reshape((-1, self.num_heads, X.shape[1], X.shape[2]))
        X = jnp.transpose(X, (0, 2, 1, 3))
        return X.reshape((X.shape[0], X.shape[1], -1))

class ViTBlock(nn.Module):
    num_hiddens: int
    mlp_num_hiddens: int
    num_heads: int
    use_bias: bool = False

    def setup(self):
        self.attention = MultiHeadAttention(self.num_hiddens, self.num_heads, self.use_bias)
        self.mlp = ViTMLP(self.mlp_num_hiddens, self.num_hiddens)

    @nn.compact
    def __call__(self, X, valid_lens=None, training=False):
        X = X + self.attention(*([nn.LayerNorm(param_dtype=jnp.float64)(X)] * 3), valid_lens, training=training)[0]
        return X + self.mlp(nn.LayerNorm(param_dtype=jnp.float64)(X), training=training)

class ViT_FFN(nn.Module):
    img_size: int
    patch_size: int
    num_hiddens: int
    mlp_num_hiddens: int
    num_heads: int
    num_blks: int
    training: bool = False
    use_bias: bool = False
    alpha: int = 1  # Scaling factor for FFN head

    def setup(self):
        self.patch_embedding = PatchEmbedding(self.img_size, self.patch_size, self.num_hiddens)
        num_steps = self.patch_embedding.num_patches
        self.pos_embedding = self.param('pos_embed', nn.initializers.normal(stddev=0.01), 
                                        (1, num_steps, self.num_hiddens))
        self.blks = [ViTBlock(self.num_hiddens, self.mlp_num_hiddens, self.num_heads, self.use_bias)
                     for _ in range(self.num_blks)]
        #self.head = nn.Dense(self.alpha, param_dtype=jnp.float64)
        self.head = FFN(self.alpha)

    @nn.compact
    def __call__(self, X):
        X = reshape_to_LxL(X, self.img_size)
        X = jnp.expand_dims(X, -1)
        X = self.patch_embedding(X)
        X = X + self.pos_embedding
        for blk in self.blks:
            X = blk(X)
        X = jnp.mean(X, axis=1)
        print(X.shape)
        output = self.head(X)
        
        return output



# class PatchEmbedding(nn.Module):
#     img_size: int = 8
#     patch_size: int = 2
#     num_hiddens: int = 32

#     def setup(self):
#         def _make_tuple(x):
#             if not isinstance(x, (list, tuple)):
#                 return (x, x)
#             return x

#         img_size, patch_size = _make_tuple(self.img_size), _make_tuple(self.patch_size)
#         self.num_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])
#         self.conv = nn.Conv(
#             features=self.num_hiddens,
#             kernel_size=patch_size,
#             strides=patch_size,
#             padding="SAME",
#             param_dtype=jnp.complex128,
#             dtype=jnp.complex128,
#         )

#     def __call__(self, X):
#         X = self.conv(X)
#         return X.reshape((X.shape[0], -1, X.shape[3]))


# class ViTMLP(nn.Module):
#     mlp_num_hiddens: int
#     mlp_num_outputs: int

#     @nn.compact
#     def __call__(self, x, training=False):
#         x = nn.Dense(
#             features=self.mlp_num_hiddens,
#             param_dtype=jnp.complex128,
#             dtype=jnp.complex128,
#         )(x)
#         x = nn.gelu(x)
#         x = nn.Dense(
#             features=self.mlp_num_outputs,
#             param_dtype=jnp.complex128,
#             dtype=jnp.complex128,
#         )(x)
#         return x

# def masked_softmax(X, valid_lens):
#     """Perform softmax operation by masking elements on the last axis.

#     Defined in :numref:sec_attention-scoring-functions"""
#     # X: 3D tensor, valid_lens: 1D or 2D tensor
#     def _sequence_mask(X, valid_len, value=0):
#         maxlen = X.shape[1]
#         mask = jnp.arange((maxlen),
#                           dtype=jnp.complex128)[None, :] < valid_len[:, None]
#         return jnp.where(mask, X, value)

#     if valid_lens is None:
#         return nn.softmax(X, axis=-1)
#     else:
#         shape = X.shape
#         if valid_lens.ndim == 1:
#             valid_lens = jnp.repeat(valid_lens, shape[1])
#         else:
#             valid_lens = valid_lens.reshape(-1)
#         # On the last axis, replace masked elements with a very large negative
#         # value, whose exponentiation outputs 0
#         X = _sequence_mask(X.reshape(-1, shape[-1]), valid_lens, value=-1e6)
#         return nn.softmax(X.reshape(shape), axis=-1)


# class DotProductAttention(nn.Module):
#     @nn.compact
#     def __call__(self, queries, keys, values, valid_lens=None, training=False):
#         d = queries.shape[-1]
#         scores = queries @ keys.swapaxes(1, 2) / jnp.sqrt(d)
#         attention_weights = masked_softmax(scores, valid_lens)
#         return attention_weights @ values, attention_weights


# class MultiHeadAttention(nn.Module):
#     num_hiddens: int
#     num_heads: int
#     bias: bool = False

#     def setup(self):
#         self.attention = DotProductAttention()
#         self.W_q = nn.Dense(
#             features=self.num_hiddens,
#             use_bias=self.bias,
#             param_dtype=jnp.complex128,
#             dtype=jnp.complex128,
#         )
#         self.W_k = nn.Dense(
#             features=self.num_hiddens,
#             use_bias=self.bias,
#             param_dtype=jnp.complex128,
#             dtype=jnp.complex128,
#         )
#         self.W_v = nn.Dense(
#             features=self.num_hiddens,
#             use_bias=self.bias,
#             param_dtype=jnp.complex128,
#             dtype=jnp.complex128,
#         )
#         self.W_o = nn.Dense(
#             features=self.num_hiddens,
#             use_bias=self.bias,
#             param_dtype=jnp.complex128,
#             dtype=jnp.complex128,
#         )

#     @nn.compact
#     def __call__(self, queries, keys, values, valid_lens, training=False):
#         queries = self.transpose_qkv(self.W_q(queries))
#         keys = self.transpose_qkv(self.W_k(keys))
#         values = self.transpose_qkv(self.W_v(values))

#         if valid_lens is not None:
#             valid_lens = jnp.repeat(valid_lens, self.num_heads, axis=0)

#         output, attention_weights = self.attention(queries, keys, values, valid_lens, training=training)
#         output_concat = self.transpose_output(output)
#         return self.W_o(output_concat), attention_weights

#     def transpose_qkv(self, X):
#         X = X.reshape((X.shape[0], X.shape[1], self.num_heads, -1))
#         X = jnp.transpose(X, (0, 2, 1, 3))
#         return X.reshape((-1, X.shape[2], X.shape[3]))

#     def transpose_output(self, X):
#         X = X.reshape((-1, self.num_heads, X.shape[1], X.shape[2]))
#         X = jnp.transpose(X, (0, 2, 1, 3))
#         return X.reshape((X.shape[0], X.shape[1], -1))


# class ViTBlock(nn.Module):
#     num_hiddens: int
#     mlp_num_hiddens: int
#     num_heads: int
#     use_bias: bool = False

#     def setup(self):
#         self.attention = MultiHeadAttention(
#             self.num_hiddens,
#             self.num_heads,
#             self.use_bias,
#         )
#         self.mlp = ViTMLP(
#             self.mlp_num_hiddens,
#             self.num_hiddens,
#         )

#     @nn.compact
#     def __call__(self, X, valid_lens=None, training=False):
#         X = X + self.attention(*([nn.LayerNorm(dtype=jnp.complex128)(X)] * 3), valid_lens, training=training)[0]
#         return X + self.mlp(nn.LayerNorm(dtype=jnp.complex128)(X), training=training)


# class ViT_FFN(nn.Module):
#     img_size: int
#     patch_size: int
#     num_hiddens: int
#     mlp_num_hiddens: int
#     num_heads: int
#     num_blks: int
#     training: bool = False
#     use_bias: bool = False
#     alpha: int = 1

#     def setup(self):
#         self.patch_embedding = PatchEmbedding(self.img_size, self.patch_size, self.num_hiddens)
#         num_steps = self.patch_embedding.num_patches
#         self.pos_embedding = self.param(
#             "pos_embed",
#             nn.initializers.normal(stddev=0.01, dtype=jnp.complex128),
#             (1, num_steps, self.num_hiddens),
#         )
#         self.blks = [
#             ViTBlock(self.num_hiddens, self.mlp_num_hiddens, self.num_heads, self.use_bias)
#             for _ in range(self.num_blks)
#         ]
#         self.head = FFN(self.alpha)

#     @nn.compact
#     def __call__(self, X):
#         X = reshape_to_LxL(X, self.img_size)
#         X = jnp.expand_dims(X, -1)
#         X = self.patch_embedding(X)
#         X = X + self.pos_embedding
#         for blk in self.blks:
#             X = blk(X)
#         X = jnp.mean(X, axis=1)
#         output = self.head(X)
#         return output



img_size, patch_size = L, 2
num_hiddens, mlp_num_hiddens, num_heads, num_blks = 32, 64, 4, 1
emb_dropout, blk_dropout = 0.1, 0.1
alpha = 4

model = ViT_FFN(img_size, patch_size, num_hiddens, mlp_num_hiddens, num_heads,
            num_blks, emb_dropout, blk_dropout, alpha)

batch_size = 4096
x = hi.random_state(size=batch_size, key=jax.random.PRNGKey(0))
print(x.shape)

key = jax.random.PRNGKey(0)  # Generate a PRNG key

# Split the key for params and dropout RNG
key_params, key_dropout = jax.random.split(key)

# Initialize the model with PRNG key for params and dropout
random_params = model.init({'params': key_params, 'dropout': key_dropout}, x)

# Apply the model using the dropout key
random_output = model.apply(random_params, x, rngs={'dropout': key_dropout})

print("Random Output shape:", random_output.shape)
print("Random Output:", random_output)


from flax.core import unfreeze

# Assuming `random_params` is your initialized parameter tree
params = unfreeze(random_params)

# Recursively print the dtypes for each layer
def print_dtypes(tree, prefix=""):
    for key, value in tree.items():
        if isinstance(value, dict):  # Nested structure (e.g., layers inside blocks)
            print_dtypes(value, prefix=prefix + key + "/")
        else:
            print(f"{prefix}{key}: {value.dtype}")

print_dtypes(params)



sampler = nk.sampler.MetropolisLocal(hi)
vstate = nk.vqs.MCState(sampler, model, n_samples=4096)

# width = 1
# key = jax.random.PRNGKey(0)
# stddev_init = 0.01

# params = initialise_vstate(vstate, model, width=width, x=x, key=key, stddev_init=stddev_init, debug=True)

# post_init_params = {'params': flax.core.copy(vstate.parameters, {})}
# post_init_output = model.apply(post_init_params, x.astype(jnp.complex128), return_intermediate=False)
# print(post_init_output)

clip = optax.clip_by_global_norm(max_norm=1e-2)
optimizer = nk.optimizer.Sgd(learning_rate=0.01)
optimizer = optax.chain(optax.zero_nans(), clip, optimizer)

def linear_solver_pinv_smooth(rcond=1e-6):
    return lambda A, b: nk.optimizer.solver.pinv_smooth(A, b, rtol=rcond, rtol_smooth=rcond)[0]

solver = linear_solver_pinv_smooth(1e-6)

sr = nk.optimizer.SR(diag_shift=1e-6, holomorphic=False)
# Notice the use, again of Stochastic Reconfiguration, which considerably improves the optimisation
gs = nk.driver.VMC(H, optimizer, variational_state=vstate,preconditioner=sr)

# gs = nkx.driver.VMC_SRt(H,optimizer,diag_shift=0.0,variational_state=vstate,jacobian_mode="complex",linear_solver_fn=solver)

log=nk.logging.RuntimeLog()
#gs.run(n_iter=500,out=log)
