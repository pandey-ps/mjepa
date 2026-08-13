import argparse
import dataclasses
import importlib.util
import sys


def _coerce(val, current):
    if isinstance(current, bool):
        return val.lower() in ("true", "1", "yes", "on")
    if isinstance(current, int):
        return int(val)
    if isinstance(current, float):
        return float(val)
    if isinstance(current, str):
        return val
    if isinstance(current, tuple):
        parts = [v.strip() for v in val.split(",")]
        if len(parts) != len(current):
            raise ValueError(
                f"tuple override needs {len(current)} values, got {len(parts)}"
            )
        return tuple(type(elem)(v) for v, elem in zip(parts, current))
    raise ValueError(f"unsupported override type for {current!r}")


def apply_overrides(cfg, overrides):
    for ov in overrides:
        if "=" not in ov:
            raise ValueError(f"override must be key=value, got {ov!r}")
        key, _, val = ov.partition("=")
        section, _, name = key.partition(".")
        if not section or not name:
            raise ValueError(f"override must be SECTION.name=value, got {ov!r}")
        section_obj = getattr(cfg, section)
        if not hasattr(section_obj, name):
            raise KeyError(f"unknown config field {section}.{name}")
        new_val = _coerce(val, getattr(section_obj, name))
        cfg = dataclasses.replace(
            cfg, **{section: dataclasses.replace(section_obj, **{name: new_val})}
        )
    return cfg


def build_parser():
    parser = argparse.ArgumentParser(
        description="LeJEPA pretraining on CheXpert",
        epilog=(
            "Example: python -m src.main data.batch_size=64 "
            "model.embed_dim=384 train.epochs=30"
        ),
    )
    parser.add_argument(
        "overrides", nargs="*", help="SECTION.name=value dataclass overrides"
    )
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--checkpoint", default=None, help="step_NNNNNNN dir to load")
    return parser


def load_config_from(path):
    spec = importlib.util.spec_from_file_location("_user_config", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, "config"):
        raise ValueError(f"{path} must be a variable named 'config'")
    return mod.config


def main(argv=None):
    from src.config import Config

    args = build_parser().parse_args(argv)
    cfg = apply_overrides(Config(), args.overrides)

    if args.eval_only:
        from src.eval import evaluate
        from src.data.chexpert import ChexpertDataset
        from src.train import build_model

        import jax

        valid = ChexpertDataset(cfg, "valid")
        train = ChexpertDataset(cfg, "train")
        model = build_model(cfg, jax.random.key(cfg.train.seed))
        if args.checkpoint:
            from src.train import restore_checkpoint

            restore_checkpoint(args.checkpoint, model)
        metrics = evaluate(cfg, model, valid_dataset=valid, train_dataset=train)
        for k, v in metrics.items():
            print(f"{k}: {v:.4f}")
        return 0

    from src.eval import evaluate
    from src.data.chexpert import ChexpertDataset
    from src.train import train

    valid_ds = ChexpertDataset(cfg, "valid")
    train_ds = ChexpertDataset(cfg, "train")

    def eval_callback(model, step):
        if step % (cfg.train.eval_every * 10) == 0:
            return evaluate(cfg, model, valid_dataset=valid_ds, train_dataset=train_ds)
        return None

    result = train(cfg, max_steps=args.max_steps, eval_callback=eval_callback)
    print(f"training done at step {result.final_step}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
