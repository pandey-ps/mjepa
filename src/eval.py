from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np
import optax

from src.config import Config
from src.data.augment import eval_view
from src.data.chexpert import LABEL_COLUMNS, ChexpertDataset, ValidBatchIterator
from src.models.lejepa import LeJEPA


def roc_auc(y, scores, mask):
    y = y[mask]
    scores = scores[mask]
    pos = scores[y == 1]
    neg = scores[y == 0]
    n_pos, n_neg = pos.shape[0], neg.shape[0]
    if n_pos == 0 or n_neg == 0:
        return jnp.asarray(float("nan"))
    diff = pos[:, None] - neg[None, :]
    more = jnp.sum(diff > 0)
    equal = jnp.sum(diff == 0)
    return (more + 0.5 * equal) / (n_pos * n_neg)


def extract_embeddings(
    config: Config,
    model: LeJEPA,
    valid_dataset,
    batch_size: Optional[int] = None,
    indices: Optional[np.ndarray] = None,
):
    fwd = jax.jit(lambda imgs: model.eval_forward(eval_view(imgs, config.data)))
    emb_list = []
    lab_list = []
    for images, labels in ValidBatchIterator(config, valid_dataset, batch_size=batch_size, indices=indices):
        emb = np.asarray(fwd(images).astype(jnp.float32))
        emb_list.append(emb)
        lab_list.append(labels)
    return np.concatenate(emb_list, axis=0), np.concatenate(lab_list, axis=0)


def fit_probe(
    emb,
    labels,
    *,
    steps=1000,
    lr=1e-2,
    weight_decay=1e-4,
    center=None,
    scale=None,
):
    if center is None:
        center = emb.mean(axis=0)
    if scale is None:
        scale = emb.std(axis=0) + 1e-6
    x = (emb - center) / scale
    x = jnp.asarray(x)
    labels = jnp.asarray(labels)
    mask = labels >= 0
    y = (labels == 1).astype(jnp.float32)
    n_out = labels.shape[1]
    d = x.shape[1]

    params = {"w": jnp.zeros((n_out, d)), "b": jnp.zeros((n_out,))}

    def loss_fn(p):
        logits = x @ p["w"].T + p["b"]
        bce = optax.sigmoid_binary_cross_entropy(logits, y)
        bce = jnp.where(mask, bce, 0.0)
        return bce.sum() / mask.sum() + weight_decay * jnp.sum(p["w"] ** 2)

    opt = optax.adam(lr)
    state = opt.init(params)
    grad_fn = jax.grad(loss_fn)
    for _ in range(steps):
        grads = grad_fn(params)
        updates, state = opt.update(grads, state, params)
        params = optax.apply_updates(params, updates)
    return params


def evaluate(
    config: Config,
    model: LeJEPA,
    *,
    valid_dataset: Optional[ChexpertDataset] = None,
    train_dataset: Optional[ChexpertDataset] = None,
    probe_steps: int = 1000,
    probe_samples: int = 2048,
    batch_size: Optional[int] = None,
):
    if valid_dataset is None:
        valid_dataset = ChexpertDataset(config, "valid")
    if train_dataset is None:
        train_dataset = ChexpertDataset(config, "train")

    val_emb, val_labels = extract_embeddings(
        config, model, valid_dataset, batch_size=batch_size
    )

    n_train = len(train_dataset)
    idx = None
    if probe_samples is not None and n_train > probe_samples:
        rng = np.random.default_rng(0)
        idx = rng.choice(n_train, probe_samples, replace=False)
    train_emb, train_labels = extract_embeddings(
        config, model, train_dataset, batch_size=batch_size, indices=idx
    )

    center = train_emb.mean(axis=0)
    scale = train_emb.std(axis=0) + 1e-6
    params = fit_probe(
        train_emb, train_labels, steps=probe_steps, center=center, scale=scale
    )

    x = (val_emb - center) / scale
    x = jnp.asarray(x)
    logits = x @ params["w"].T + params["b"]
    probs = jax.nn.sigmoid(logits)
    labels = jnp.asarray(val_labels)
    mask = labels >= 0

    aucs = [
        float(roc_auc(labels[:, i], probs[:, i], mask[:, i])) for i in range(labels.shape[1])
    ]
    names = LABEL_COLUMNS
    values = [a for a in aucs if not np.isnan(a)]
    metrics = {"mean_auc": float(np.mean(values)) if values else float("nan")}
    metrics.update({name: a for name, a in zip(names, aucs)})
    return metrics


__all__ = ["roc_auc", "extract_embeddings", "fit_probe", "evaluate"]
