# Motion Models

Neural network models for motion prediction pre-training.

## Files

| File | Description |
|------|-------------|
| `motion_nn.py` | MotionNN class with MLP and Transformer backends |
| `train_motion.py` | Training script with TensorBoard logging |

## MotionNN

Motion prediction model: `s_t → s_{t+1}`

**Modes:**
- `mlp`: 3-layer MLP, single-step prediction
- `transformer_reg`: Transformer encoder, sequence → next frame
- `transformer_ar`: Transformer, sequence → next sequence

```python
from python.models import MotionNN

# MLP mode
model = MotionNN(state_dim=112, mode='mlp')

# Transformer mode
model = MotionNN(state_dim=112, mode='transformer_reg', seq_len=16)

# Save/load
model.save('nn/motion_model.pt')
model = MotionNN.load('nn/motion_model.pt')
```

## Training

```bash
# Train on specific files (MLP)
pixi run python python/models/train_motion.py \
    --bvh_files data/motion/walk.bvh data/motion/run.bvh \
    --mode mlp \
    --epochs 100

# Train on CMU data with subject-based split (Transformer)
pixi run python python/models/train_motion.py \
    --bvh_dir data/cmu \
    --mode transformer_reg \
    --subject_split \
    --max_subjects 10 \
    --epochs 50

# View training metrics
pixi run tensorboard
```

**Key arguments:**
| Argument | Default | Description |
|----------|---------|-------------|
| `--mode` | `mlp` | `mlp`, `transformer_reg`, `transformer_ar` |
| `--subject_split` | off | Split by subject (prevents data leakage) |
| `--activity_split` | off | Activity-stratified split (ensures activity overlap) |
| `--max_subjects` | all | Limit subjects for testing |
| `--epochs` | 100 | Training epochs |
| `--verbose` | off | Show C++ parsing output (default: log to file) |

## Outputs

- `nn/motion_model.pt` — Final checkpoint
- `nn/motion_model_best.pt` — Best validation loss
- `nn/motion_normalizer.npz` — Normalization parameters
- `nn/training.log` — C++ parsing output
- `runs/motion_model/` — TensorBoard logs
