import jax
import jax.numpy as jnp
from flax import nnx

from src.config import ModelConfig


class _Block(nnx.Module):
    def __init__(self, dim, num_heads, mlp_ratio, rngs, dtype=None):
        self.norm1 = nnx.LayerNorm(dim, rngs=rngs)
        self.attn = nnx.MultiHeadAttention(
            num_heads=num_heads, in_features=dim, dtype=dtype, rngs=rngs
        )
        self.norm2 = nnx.LayerNorm(dim, rngs=rngs)
        self.mlp = nnx.Sequential(
            nnx.Linear(dim, dim * mlp_ratio, dtype=dtype, rngs=rngs),
            jax.nn.gelu,
            nnx.Linear(dim * mlp_ratio, dim, dtype=dtype, rngs=rngs),
        )

    def __call__(self, x):
        x = x + self.attn(self.norm1(x), decode=False)
        x = x + self.mlp(self.norm2(x))
        return x


class ViT(nnx.Module):
    def __init__(
        self,
        *,
        rngs,
        base_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 384,
        depth: int = 12,
        num_heads: int = 6,
        mlp_ratio: int = 4,
        dtype=None,
    ):
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self._base_grid = (base_size // patch_size, base_size // patch_size)

        self.patch_embed = nnx.Conv(
            in_channels,
            embed_dim,
            kernel_size=(patch_size, patch_size),
            strides=(patch_size, patch_size),
            padding="VALID",
            dtype=dtype,
            rngs=rngs,
        )
        n_patches = (base_size // patch_size) ** 2
        self.cls_token = nnx.Param(
            jax.random.normal(rngs.params(), (1, 1, embed_dim)) * 0.02
        )
        self.pos_embed = nnx.Param(
            jax.random.normal(rngs.params(), (1, n_patches + 1, embed_dim)) * 0.02
        )
        self.blocks = nnx.List(
            [_Block(embed_dim, num_heads, mlp_ratio, rngs, dtype) for _ in range(depth)]
        )
        self.norm = nnx.LayerNorm(embed_dim, rngs=rngs)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        b = x.shape[0]
        x = self.patch_embed(x)
        gh, gw = x.shape[1], x.shape[2]
        x = x.reshape(b, gh * gw, self.embed_dim)

        cls = jnp.broadcast_to(self.cls_token[...], (b, 1, self.embed_dim))
        x = jnp.concatenate([cls, x], axis=1)
        x = x + self._resize_pos_embed(gh, gw)

        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return x[:, 0]

    def _resize_pos_embed(self, gh: int, gw: int) -> jnp.ndarray:
        pe = self.pos_embed[...]
        if (gh, gw) == self._base_grid:
            return pe
        base_gh, base_gw = self._base_grid
        cls = pe[:, :1]
        grid = pe[:, 1:].reshape(1, base_gh, base_gw, -1)
        grid = jax.image.resize(
            grid, (1, gh, gw, self.embed_dim), method="bilinear", antialias=True
        )
        return jnp.concatenate([cls, grid.reshape(1, gh * gw, -1)], axis=1)


def vit_tiny(*, rngs, **kwargs) -> ViT:
    kwargs.setdefault("embed_dim", 192)
    kwargs.setdefault("depth", 12)
    kwargs.setdefault("num_heads", 3)
    return ViT(rngs=rngs, **kwargs)

def vit_small(*, rngs, **kwargs) -> ViT:
    kwargs.setdefault("embed_dim", 384)
    kwargs.setdefault("depth", 12)
    kwargs.setdefault("num_heads", 6)
    return ViT(rngs=rngs, **kwargs)

def vit_base(*, rngs, **kwargs) -> ViT:
    kwargs.setdefault("embed_dim", 768)
    kwargs.setdefault("depth", 12)
    kwargs.setdefault("num_heads", 12)
    return ViT(rngs=rngs, **kwargs)


def vit_large(*, rngs, **kwargs) -> ViT:
    kwargs.setdefault("embed_dim", 1024)
    kwargs.setdefault("depth", 24)
    kwargs.setdefault("num_heads", 16)
    return ViT(rngs=rngs, **kwargs)


def build_vit(cfg: ModelConfig, *, rngs, dtype=None) -> ViT:
    return ViT(
        rngs=rngs,
        base_size=cfg.base_size,
        patch_size=cfg.patch_size,
        in_channels=cfg.in_channels,
        embed_dim=cfg.embed_dim,
        depth=cfg.num_layers,
        num_heads=cfg.num_heads,
        mlp_ratio=cfg.mlp_ratio,
        dtype=dtype,
    )



