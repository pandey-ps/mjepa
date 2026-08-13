import dataclasses
import math
import time
from pathlib import Path
from typing import Callable, Optional

import jax
import jax.numpy as jnp
import optax
from flax import nnx
from orbax.checkpoint import PyTreeCheckpointer

from src.config import Config, TrainConfig
from src.data.chexpert import ChexpertDataset, TrainIterator
from src.models.lejepa import LeJEPA


@dataclasses.dataclass
class TrainResult:
    final_step: int
    history: list


def build_model(config: Config, key) -> LeJEPA:
    return LeJEPA(config=config, rngs=nnx.Rngs(key))


def make_schedule(config: TrainConfig, steps_per_epoch: int, total_steps: int):
    warmup_steps = min(int(config.warmup_epochs * steps_per_epoch), total_steps)
    if warmup_steps >= total_steps:
        if warmup_steps <= 0:
            return optax.constant_schedule(config.lr)
        return optax.linear_schedule(
            config.lr * config.start_lr_factor, config.lr, warmup_steps
        )
    cosine = optax.cosine_decay_schedule(
        config.lr, total_steps - warmup_steps, alpha=config.lr_final_ratio
    )
    if warmup_steps <= 0:
        return cosine
    warmup = optax.linear_schedule(
        config.lr * config.start_lr_factor, config.lr, warmup_steps
    )
    return optax.join_schedules([warmup, cosine], [warmup_steps])


def make_optimizer(config: TrainConfig, schedule):
    tx = optax.adamw(
        learning_rate=schedule,
        weight_decay=config.weight_decay,
        b1=config.betas[0],
        b2=config.betas[1],
    )
    if config.grad_clip is not None:
        tx = optax.chain(optax.clip_by_global_norm(config.grad_clip), tx)
    return tx


def _state_values(state):
    out = {}
    for k, v in state.items():
        if isinstance(v, nnx.Variable):
            out[k] = v.value
        else:
            out[k] = _state_values(v)
    return out


def save_checkpoint(path, model, optimizer, step):
    path = Path(path).resolve()
    path.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "model": _state_values(nnx.state(model)),
        "optimizer": _state_values(nnx.state(optimizer)),
        "step": step,
    }
    PyTreeCheckpointer().save(str(path / f"step_{step:07d}"), ckpt, force=True)


def fix_int_keys(x):
    if isinstance(x, dict):
        return {
            int(k) if isinstance(k, str) and k.isdigit() else k: fix_int_keys(v)
            for k, v in x.items()
        }
    if isinstance(x, list):
        return [fix_int_keys(v) for v in x]
    return x


def _set_state(node, state):
    if isinstance(state, dict):
        for key, value in state.items():
            if isinstance(node, nnx.List) or isinstance(key, int):
                child = node[key]
            else:
                child = getattr(node, key)
            _set_state(child, value)
    else:
        node.value = state


def restore_checkpoint(path, model):
    ckpt = PyTreeCheckpointer().restore(str(Path(path).resolve()))
    _set_state(model, fix_int_keys(ckpt["model"]))
    return ckpt.get("step")


def train(
    config: Config,
    *,
    dataset: Optional[ChexpertDataset] = None,
    max_steps: Optional[int] = None,
    eval_callback: Optional[Callable[[LeJEPA, int], Optional[dict]]] = None,
    log_fn: Callable[[str], None] = print,
    key=None,
) -> TrainResult:
    if dataset is None:
        dataset = ChexpertDataset(config, "train")
    if key is None:
        key = jax.random.key(config.train.seed)
    model = build_model(config, key)

    steps_per_epoch = int(math.ceil(len(dataset) / config.data.batch_size))
    total_steps = (
        max_steps if max_steps is not None else int(config.train.epochs * steps_per_epoch)
    )

    schedule = make_schedule(config.train, steps_per_epoch, total_steps)
    optimizer = nnx.Optimizer(
        model, make_optimizer(config.train, schedule), wrt=nnx.Param
    )

    base_key = jax.random.key(config.train.seed)

    def make_step(base_key):
        @nnx.jit
        def step_fn(model, optimizer, gv, lv, step):
            sigreg_key = jax.random.fold_in(base_key, step)

            def loss_fn(m):
                out = m.train_forward(gv, lv, sigreg_key)
                return out.loss, (out.inv_loss, out.sigreg_loss)

            (loss, (inv, sigreg)), grads = nnx.value_and_grad(
                loss_fn, argnums=0, has_aux=True
            )(model)
            grads = jax.tree.map(lambda g: g.astype(jnp.float32), grads)
            optimizer.update(model, grads)
            return loss, inv, sigreg

        return step_fn

    step_fn = make_step(base_key)

    history = []
    step = 0
    running = {"loss": 0.0, "inv": 0.0, "sigreg": 0.0}
    t_start = time.time()
    checkpointer = PyTreeCheckpointer()
    ckpt_dir = Path(config.train.checkpoint_dir)

    for batch in TrainIterator(config, dataset, seed=config.train.seed):
        if step >= total_steps:
            break
        loss, inv, sigreg = step_fn(
            model,
            optimizer,
            batch["global"],
            batch["local"],
            jnp.asarray(step, jnp.int32),
        )
        running["loss"] += float(loss)
        running["inv"] += float(inv)
        running["sigreg"] += float(sigreg)

        if (step + 1) % config.train.log_every == 0:
            n = config.train.log_every
            entry = {
                "step": step + 1,
                "loss": running["loss"] / n,
                "inv": running["inv"] / n,
                "sigreg": running["sigreg"] / n,
                "lr": float(schedule(step)),
                "elapsed": time.time() - t_start,
            }
            history.append(entry)
            log_fn(
                f"step {entry['step']:>6d} | loss {entry['loss']:.4f} "
                f"| inv {entry['inv']:.4f} | sigreg {entry['sigreg']:.4f} "
                f"| lr {entry['lr']:.2e} | {entry['elapsed']:.1f}s"
            )
            running = {"loss": 0.0, "inv": 0.0, "sigreg": 0.0}

        if (step + 1) % config.train.save_every == 0:
            save_checkpoint(ckpt_dir, model, optimizer, step + 1)
            log_fn(f"checkpoint saved at step {step + 1}")

        if eval_callback is not None and (step + 1) % config.train.eval_every == 0:
            metrics = eval_callback(model, step + 1)
            if metrics:
                history.append({"step": step + 1, **metrics})
                log_fn(f"eval {metrics}")

        step += 1

    save_checkpoint(ckpt_dir, model, optimizer, step)
    return TrainResult(final_step=step, history=history)


__all__ = [
    "TrainResult",
    "build_model",
    "make_schedule",
    "make_optimizer",
    "save_checkpoint",
    "restore_checkpoint",
    "train",
]
