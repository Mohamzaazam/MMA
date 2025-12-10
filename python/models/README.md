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

# Train on CMU data with subject-disjoint activity split (recommended)
pixi run python python/models/train_motion.py \
    --bvh_dir data/cmu \
    --mode transformer_reg \
    --activity_split \
    --test_ratio 0.15 \
    --max_subjects 20 \
    --epochs 50

# View training metrics
pixi run tensorboard
```

**Key arguments:**
| Argument | Default | Description |
|----------|---------|-------------|
| `--mode` | `mlp` | `mlp`, `transformer_reg`, `transformer_ar` |
| `--subject_split` | off | Split by subject (prevents data leakage) |
| `--activity_split` | off | Subject-disjoint activity split (no subject overlap + activity coverage) |
| `--test_ratio` | `0.15` | Ratio of subjects held out for testing |
| `--save_split` | auto | Path to save train/val/test split JSON |
| `--max_subjects` | `10` | Limit subjects for testing |
| `--epochs` | 100 | Training epochs |
| `--verbose` | off | Show C++ parsing output (default: log to file) |

## Evaluation

```bash
# Evaluate on test set from training split (recommended)
pixi run python python/models/eval_motion.py \
    --model nn/motion_model_best.pt \
    --split_file nn/split_info.json \
    --use_test_set \
    --output_dir eval_test_results

# Evaluate on arbitrary BVH files
pixi run python python/models/eval_motion.py \
    --model nn/motion_model_best.pt \
    --bvh_dir data/cmu \
    --num_frames 500 \
    --output_dir eval_results
```

**Evaluation arguments:**
| Argument | Default | Description |
|----------|---------|-------------|
| `--split_file` | none | Load files from split JSON |
| `--use_test_set` | off | Evaluate on held-out test set |
| `--use_val_set` | off | Evaluate on validation set |
| `--num_frames` | `500` | Frames for trajectory visualization |
| `--num_samples` | `500` | Samples for statistical metrics |

**Outputs:**
- `metrics.json` — All computed metrics (MSE, MAE, R², per-component)
- `per_component_errors.png` — Bar chart of per-joint errors
- `rollout_horizon.png` — Error vs prediction horizon
- `trajectory.png` — Time series comparison
- `joint_angles.png` — Lower limb joint angle tracking
- `joint_scatter.png` — Prediction accuracy scatter
- `error_distribution.png` — MSE histogram

## Training Outputs

- `nn/motion_model.pt` — Final checkpoint
- `nn/motion_model_best.pt` — Best validation loss
- `nn/motion_normalizer.npz` — Normalization parameters
- `nn/split_info.json` — Train/val/test split (subjects and files)
- `nn/training.log` — C++ parsing output
- `runs/motion_model/` — TensorBoard logs (includes position/velocity/limb losses)

---

## RL vs MotionNN: Key Differences

| Aspect | RL (SimulationNN + MuscleNN) | Supervised (MotionNN) |
|--------|------------------------------|----------------------|
| **Predicts** | Position offsets → SPD → Torques → Activations | Next state directly |
| **Training signal** | Reward from physics simulation | MSE vs BVH ground truth |
| **Physics** | Uses DART simulation | No physics |
| **Muscles** | Yes (Hill model) | No |
| **Purpose** | Control policy for simulation | Motion prior for pre-training |

---

## Potential Uses for MotionNN

MotionNN learns **motion dynamics** from BVH data:

1. **Motion prior** — Regularize RL policy to stay close to natural motions
2. **Warm-start** — Initialize policy with motion knowledge
3. **Reference generator** — Replace BVH file lookup during training
4. **Model-based RL** — Use as world model for planning

