import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Any


class ViTNQS(nn.Module):
    Lx: int
    patch_size: int
    d_model: int
    num_heads: int
    num_layers: int
    param_dtype: Any = jnp.complex64

    @nn.compact
    def __call__(self, σ):
        batch_size = σ.shape[0]
        N_sites = σ.shape[1]

        patches = extract_patches_1d(σ, self.patch_size)
        patch_embedding = nn.Dense(self.d_model, use_bias=True, param_dtype=self.param_dtype, name='patch_embedding')
        x = patch_embedding(patches)

        n_patches = x.shape[1]
        pos_embedding = self.param('pos_embedding', nn.initializers.normal(stddev=0.05), (1, n_patches, self.d_model), self.param_dtype)
        x = x + pos_embedding

        for _ in range(self.num_layers):
            x = ViTEncoderBlock(d_model=self.d_model, num_heads=self.num_heads, param_dtype=self.param_dtype)(x)

        z = jnp.sum(x, axis=1)

        w = self.param('w', nn.initializers.normal(stddev=0.1), (self.d_model,), jnp.complex64)
        b = self.param('b', nn.initializers.normal(stddev=0.1), (self.d_model,), jnp.complex64)
        pre_activation = w * z + b

        g = lambda x: jnp.log(jnp.cosh(x))
        log_psi = jnp.sum(g(pre_activation), axis=-1)
        return log_psi


def extract_patches_1d(σ, patch_size):
    batch_size, Lx = σ.shape
    n_patches = Lx // patch_size
    σ = σ[:, :n_patches * patch_size]
    patches = σ.reshape(batch_size, n_patches, patch_size)
    return patches


class ViTEncoderBlock(nn.Module):
    d_model: int
    num_heads: int
    param_dtype: Any = jnp.complex64

    @nn.compact
    def __call__(self, x):
        y = nn.LayerNorm(dtype=self.param_dtype, param_dtype=self.param_dtype)(x)
        y = FactoredMultiHeadAttention(num_heads=self.num_heads, d_model=self.d_model, param_dtype=self.param_dtype)(y)
        x = x + y

        y = nn.LayerNorm(dtype=self.param_dtype, param_dtype=self.param_dtype)(x)
        y = nn.Dense(4 * self.d_model, dtype=self.param_dtype, param_dtype=self.param_dtype)(y)
        y = nn.gelu(y)
        y = nn.Dense(self.d_model, dtype=self.param_dtype, param_dtype=self.param_dtype)(y)
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

        V = nn.DenseGeneral((self.num_heads, head_dim), axis=-1, dtype=self.param_dtype, param_dtype=self.param_dtype, use_bias=False, name='V_proj')(x)
        V = V.transpose(0, 2, 1, 3)

        positions = jnp.arange(n_patches)
        relative_positions = positions[None, :] - positions[:, None]
        p = self.param('p', nn.initializers.normal(stddev=0.02), (self.num_heads, 2 * n_patches - 1), self.param_dtype)
        relative_position_indices = relative_positions + n_patches - 1
        attention_weights = p[:, relative_position_indices]
        attention_weights = jnp.broadcast_to(attention_weights[None, :, :, :], (batch_size, self.num_heads, n_patches, n_patches))

        attention_output = jnp.matmul(attention_weights, V)
        attention_output = attention_output.transpose(0, 2, 1, 3).reshape(batch_size, n_patches, self.d_model)
        output = nn.Dense(self.d_model, dtype=self.param_dtype, param_dtype=self.param_dtype, use_bias=False, name='output_proj')(attention_output)

        return output
