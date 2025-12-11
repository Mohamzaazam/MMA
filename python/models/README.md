# Motion Models

Neural networks for motion prediction pre-training.

## Files

| File | Description |
|------|-------------|
| `motion_nn.py` | MotionNN model (MLP / Transformer) |
| `train_motion.py` | Training script with multi-step horizon |
| `eval_motion.py` | Evaluation and visualization |

## MotionNN

Motion predictor: `state_t → state_{t+1}`

**Modes:**
| Mode | Input | Output | Use Case |
|------|-------|--------|----------|
| `mlp` | (batch, dim) | (batch, dim) | Fast, single-frame |
| `encoder` | (batch, seq, dim) | (batch, dim) | Sequence context → next |
| `seq2seq` | (batch, seq, dim) | (batch, seq, dim) | Full sequence prediction |

```python
from python.models import MotionNN

# MLP mode
model = MotionNN(state_dim=112, mode='mlp')

# Transformer encoder mode  
model = MotionNN(state_dim=112, mode='encoder', seq_len=32)

# Autoregressive rollout
trajectory = model.rollout(initial_state, steps=60)
```

## Training

```bash
# Basic training
pixi run python python/models/train_motion.py \
    --extracted_dir data/extracted \
    --mode mlp \
    --epochs 100

# Multi-step horizon training (16-step lookahead)
pixi run python python/models/train_motion.py \
    --extracted_dir data/extracted \
    --mode mlp \
    --max_horizon 16 \
    --min_horizon 1 \
    --multistep_ratio 0.3 \
    --epochs 50
```

**Key arguments:**
| Argument | Default | Description |
|----------|---------|-------------|
| `--mode` | `mlp` | `mlp`, `encoder`, `seq2seq` |
| `--max_horizon` | `1` | Max lookahead (1=single-step) |
| `--min_horizon` | `1` | Min horizon for adaptive training |
| `--multistep_ratio` | `0.3` | Fraction of batches using multi-step |
| `--weighted_loss` | off | Separate pos/vel loss weighting |

## Evaluation

```bash
# Evaluate on test set
pixi run python python/models/eval_motion.py \
    --model nn/motion_model_best.pt \
    --split_file nn/split_info.json \
    --use_test_set

# View training metrics
pixi run tensorboard --logdir runs
```

## Outputs

- `nn/motion_model_best.pt` — Best checkpoint
- `nn/motion_normalizer.npz` — Normalization parameters  
- `nn/split_info.json` — Train/val/test split
- `runs/motion_model/` — TensorBoard logs
