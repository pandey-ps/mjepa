from typing import NamedTuple

import jax
import jax.numpy as jnp
from flax import nnx

from src.config import Config
from src.losses.lejepa_loss import lejepa_loss
from src.models.projector import build_projector
from src.models.vit import build_vit


class LeJEPAOutput(NamedTuple):
    loss: jnp.ndarray
    embedding: jnp.ndarray
    inv_loss: jnp.ndarray
    sigreg_loss: jnp.ndarray


class LeJEPA(nnx.Module):
    def __init__(self, *, config: Config, rngs):
        dtype = jnp.bfloat16 if config.train.bf16 else None
        self.backbone = build_vit(config.model, rngs=rngs, dtype=dtype)
        self.projector = build_projector(config.model, rngs=rngs, dtype=dtype)
        self.lamb = config.train.lamb
        self.num_slices = config.train.num_slices
        self.n_points = config.train.n_points
        self.t_max = config.train.t_max
        self.embed_dim = config.model.embed_dim

    def train_forward(self, global_views, local_views, sigreg_key) -> LeJEPAOutput:
        g_features = jnp.concatenate([self.backbone(v) for v in global_views])
        l_features = jnp.concatenate([self.backbone(v) for v in local_views])
        all_features = jnp.concatenate([g_features, l_features])

        all_projected = self.projector(all_features)

        bs = global_views[0].shape[0]
        n_views = len(global_views) + len(local_views)
        all_projected = all_projected.reshape(n_views, bs, -1)

        loss, inv_loss, sigreg_loss = lejepa_loss(
            all_projected,
            len(global_views),
            sigreg_key,
            self.lamb,
            num_slices=self.num_slices,
            n_points=self.n_points,
            t_max=self.t_max,
        )

        return LeJEPAOutput(
            loss=loss,
            embedding=jax.lax.stop_gradient(g_features),
            inv_loss=inv_loss,
            sigreg_loss=sigreg_loss,
        )

    def eval_forward(self, images) -> jnp.ndarray:
        return self.backbone(images)


__all__ = ["LeJEPA", "LeJEPAOutput"]
