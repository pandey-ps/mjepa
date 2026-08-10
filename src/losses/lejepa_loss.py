import jax.numpy as jnp
from src.losses.sigreg import sigreg

def invariance_loss(all_projected: jnp.ndarray, n_global: int) -> jnp.ndarray:
    centers = all_projected[:n_global].mean(axis=0)
    return ((centers[None, ...] - all_projected) ** 2).mean()


def lejepa_loss(
    all_projected: jnp.ndarray,
    n_global: int,
    key,
    lamb: float,
    num_slices: int = 1024,
    n_points: int = 17,
    t_max: float = 3.0,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    inv_loss = invariance_loss(all_projected, n_global)
    pooled = all_projected.reshape(-1, all_projected.shape[-1])
    sigreg_loss = sigreg(pooled, key, num_slices=num_slices, t_max=t_max, n_points=n_points)
    loss = inv_loss + lamb * sigreg_loss
    return loss, inv_loss, sigreg_loss


__all__ = ["invariance_loss", "lejepa_loss"]
