ViT encoder using LeJEPA (Invariance + SIGReg loss).

## Architecture
- ViT-B/16 (92.5M params; ViT-S/Tiny) 
- 3-layer MLP with BatchNorm (768 → 512 → 2048 → 512)
- 2 global (224², scale 0.3–1.0) + 6 local (96², scale 0.05–0.3) per image
- Invariance + λ·SIGReg (Epps-Pulley statistic on random projections)

## Install
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install "jax[cuda12]" -f https://jax.github.io/jax/releases.html
```

## Data
expected layout:
data/data_name/
  train/
  valid/
  train.csv
  valid.csv


## Train
```bash
python -m src.main data.batch_size=128 train.epochs=100 
```

#### Evaluate
```bash
python -m src.main --eval-only --checkpoint checkpoints/step_0000010
```

#### Note
All hyperparameters in `src/config.py` (`DataConfig`, `ModelConfig`, `TrainConfig`): override via CLI: `python -m src.main data.batch_size=64 model.embed_dim=384 train.epochs=30`
