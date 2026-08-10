import jax
import jax.numpy as jnp
from jax import Array


def epps_pulley_statistic(
    projections: Array,
    t_max: float = 3.0,
    n_points: int = 17,
) -> Array:
    projections = projections.astype(jnp.float32)
    n_samples = projections.shape[0]

    t = jnp.linspace(0.0, t_max, n_points)
    dt = t_max / (n_points - 1)
    phi = jnp.exp(-0.5 * t**2)
    weights = jnp.full((n_points,), 2.0 * dt)
    weights = weights.at[0].set(dt)
    weights = weights.at[-1].set(dt)
    weights = weights * phi

    x_t = projections[..., None] * t
    cos_mean = jnp.mean(jnp.cos(x_t), axis=0)
    sin_mean = jnp.mean(jnp.sin(x_t), axis=0)

    err = (cos_mean - phi) ** 2 + sin_mean**2
    statistic = (err @ weights) * n_samples
    return statistic


def sample_projection_matrix(
    dim: int,
    num_slices: int,
    key: jax.Array,
) -> Array:
    matrix = jax.random.normal(key, (dim, num_slices))
    matrix = matrix / jnp.linalg.norm(matrix, axis=0)
    return matrix


def sigreg(
    embeddings: Array,
    key: jax.Array,
    num_slices: int = 1024,
    t_max: float = 3.0,
    n_points: int = 17,
) -> Array:
    dim = embeddings.shape[-1]
    matrix = sample_projection_matrix(dim, num_slices, key)
    projections = embeddings @ matrix
    statistic = epps_pulley_statistic(projections, t_max=t_max, n_points=n_points)
    return jnp.mean(statistic)
