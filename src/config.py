from dataclasses import dataclass, field

@dataclass(frozen=True)
class DataConfig:
    data_root: str = "data/chexpert"
    train_dir: str = "train"
    valid_dir: str = "valid"
    img_scale: int = 320
    image_size: int = 224
    local_size: int = 96
    global_views: int = 2
    local_views: int = 6
    scale_global: tuple[float, float] = (0.3, 1.0)
    scale_local: tuple[float, float] = (0.05, 0.3)
    flip_p: float = 0.5
    jitter_p: float = 0.8
    brightness: float = 0.4
    contrast: float = 0.4
    saturation: float = 0.2
    hue: float = 0.1
    grayscale_p: float = 0.2
    blur_kernel: int = 23
    blur_sigma: tuple[float, float] = (0.1, 2.0)
    blur_p: float = 0.5
    solarize_p: float = 0.2
    solarize_threshold: float = 128.0
    mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: tuple[float, float, float] = (0.229, 0.224, 0.225)
    batch_size: int = 128
    num_workers: int = 8
    frontal_only: bool = True


@dataclass(frozen=True)
class ModelConfig:
    patch_size: int = 16
    base_size: int = 224
    in_channels: int = 3
    embed_dim: int = 768
    num_layers: int = 12
    num_heads: int = 12
    mlp_ratio: int = 4
    proj_embed: int = 512
    proj_hidden: int = 2048
    proj_dim: int = 512


@dataclass(frozen=True)
class TrainConfig:
    lr: float = 4e-4
    weight_decay: float = 0.05
    betas: tuple[float, float] = (0.9, 0.999)
    warmup_epochs: int = 10
    start_lr_factor: float = 0.01
    epochs: int = 100
    lr_final_ratio: float = 1e-3
    lamb: float = 0.02
    num_slices: int = 1024
    n_points: int = 17
    t_max: float = 3.0
    bf16: bool = True
    seed: int = 0
    grad_clip: float | None = None
    log_every: int = 50
    save_every: int = 500
    eval_every: int = 2000
    checkpoint_dir: str = "checkpoints"
    num_gpus: int = 1


@dataclass(frozen=True)
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
