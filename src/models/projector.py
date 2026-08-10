import jax
import jax.numpy as jnp
from flax import nnx
from src.config import ModelConfig

class Projector(nnx.Module):
    def __init__(
        self,
        *,
        embed_dim: int,
        proj_embed: int,
        proj_hidden: int,
        proj_dim: int,
        rngs,
        dtype=None,
    ):
        self.entry = nnx.Linear(embed_dim, proj_embed, use_bias=True, dtype=dtype, rngs=rngs)
        self.fc1 = nnx.Linear(proj_embed, proj_hidden, use_bias=True, dtype=dtype, rngs=rngs)
        self.bn1 = nnx.BatchNorm(proj_hidden, momentum=0.1, rngs=rngs)
        self.fc2 = nnx.Linear(proj_hidden, proj_hidden, use_bias=True, dtype=dtype, rngs=rngs)
        self.bn2 = nnx.BatchNorm(proj_hidden, momentum=0.1, rngs=rngs)
        self.fc3 = nnx.Linear(proj_hidden, proj_dim, use_bias=True, dtype=dtype, rngs=rngs)

    def __call__(self, x: jnp.ndarray, use_running_average: bool = False) -> jnp.ndarray:
        x = self.entry(x)
        x = jax.nn.relu(self.bn1(self.fc1(x), use_running_average=use_running_average))
        x = jax.nn.relu(self.bn2(self.fc2(x), use_running_average=use_running_average))
        return self.fc3(x)


def build_projector(cfg: ModelConfig, *, rngs, dtype=None) -> Projector:
    return Projector(
        embed_dim=cfg.embed_dim,
        proj_embed=cfg.proj_embed,
        proj_hidden=cfg.proj_hidden,
        proj_dim=cfg.proj_dim,
        rngs=rngs,
        dtype=dtype,
    )


__all__ = ["Projector", "build_projector"]
