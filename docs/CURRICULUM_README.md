# Curriculum Learning for MASS Multimodal Training

## Overview

This update adds **Adaptive Curriculum Learning** to the multimodal motion training pipeline. The curriculum dynamically adjusts which motions get more training focus based on performance, leading to faster and more balanced learning.

## New Files

| File | Description |
|------|-------------|
| `curriculum_manager.py` | Core curriculum learning logic |
| `multimodal_env.py` | Updated environment manager with better stats tracking |
| `main_multimodal.py` | Updated training script with curriculum integration |
| `test_curriculum_integration.py` | Tests to verify the curriculum logic |

## Installation

Simply copy the new Python files to your `python/` directory:

```bash
cp curriculum_manager.py multimodal_env.py main_multimodal.py test_curriculum_integration.py /path/to/MASS/python/
```

## Usage

### Basic Training with Curriculum Learning

```bash
# Using default 'balanced' strategy
python python/main_multimodal.py \
    --motion_list data/motion_list.txt \
    --template data/metadata.txt \
    --slaves 4

# With specific curriculum strategy
python python/main_multimodal.py \
    --motion_list data/motion_list.txt \
    --template data/metadata.txt \
    --slaves 4 \
    --curriculum performance \
    --warmup 20 \
    --update_freq 5
```

### Available Curriculum Strategies

| Strategy | Description | Best For |
|----------|-------------|----------|
| `uniform` | Equal weights for all motions | Baseline comparison |
| `performance` | Higher weights for harder motions (lower returns) | When motions have varying difficulty |
| `progress` | Higher weights for motions showing less improvement | When some motions plateau |
| `balanced` | Combines performance + coverage guarantee | **Recommended default** |
| `ucb` | Upper Confidence Bound - exploration vs exploitation | Research/experimentation |

### Command Line Arguments

```
--curriculum    Curriculum strategy (default: balanced)
--warmup        Number of epochs before starting curriculum (default: 20)
--update_freq   How often to update weights (default: 5)
--slaves        Parallel environments per motion (default: 4)
```

## How It Works

### 1. Warmup Period
During the first N epochs (default: 20), all motions are trained with equal probability. This establishes baseline performance metrics.

### 2. Performance Tracking
The `CurriculumManager` tracks:
- Average return per motion (rolling window)
- Episode counts per motion
- Improvement over baseline
- Best performance seen

### 3. Weight Adjustment
After warmup, weights are adjusted every `update_freq` epochs:

```
Performance Strategy:
  weight[motion] ∝ exp(-normalized_return / temperature)
  
  Lower performing motions → Higher weights → More training
```

### 4. Weight Bounds
Weights are bounded between `min_weight=0.1` and `max_weight=3.0` to ensure:
- No motion is completely ignored
- No motion dominates training

## Example Training Progress

```
Epoch 0 (Warmup):
  walk            weight=1.000  avg_return=0.800
  run             weight=1.000  avg_return=0.500
  jump            weight=1.000  avg_return=0.300

Epoch 50 (After curriculum kicks in):
  walk            weight=0.42   avg_return=0.850  ← Easy, less focus
  run             weight=0.84   avg_return=0.600  ← Medium
  jump            weight=1.73   avg_return=0.400  ← Hard, more focus
```

## TensorBoard Logging

New metrics are logged to TensorBoard:

- `Curriculum/<motion>_Weight`: Current weight for each motion
- `PerMotion/<motion>_AvgReturn`: Average return per motion
- `PerMotion/<motion>_Episodes`: Episode count per motion

View with:
```bash
tensorboard --logdir=runs --host 0.0.0.0 --port 6006
```

## Testing

Before running full training, verify the curriculum logic:

```bash
cd python
python test_curriculum_integration.py
```

Expected output:
```
============================================================
ALL TESTS PASSED!
============================================================
```

## Configuration Tips

### For 6 motions (your motion_list.txt):
```bash
# Recommended settings
python python/main_multimodal.py \
    --motion_list data/motion_list.txt \
    --template data/metadata.txt \
    --slaves 4 \
    --curriculum balanced \
    --warmup 30 \
    --update_freq 10
```

### For faster convergence:
- Use `--curriculum performance` (more aggressive)
- Reduce `--warmup` to 10-15

### For more stable training:
- Use `--curriculum balanced` (default)
- Increase `--warmup` to 30-50
- Increase `--update_freq` to 10

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    main_multimodal.py                       │
│  ┌──────────────┐  ┌───────────────────┐                    │
│  │ PPO Trainer  │  │ CurriculumManager │                    │
│  │              │──│                   │                    │
│  │ - Train      │  │ - Track stats     │                    │
│  │ - Evaluate   │  │ - Update weights  │                    │
│  └──────────────┘  └───────────────────┘                    │
│         │                    │                              │
│         ▼                    │                              │
│  ┌──────────────────────────────────────┐                   │
│  │       MultimodalEnvManager           │                   │
│  │                                      │                   │
│  │  ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐│                   │
│  │  │pymss │ │pymss │ │pymss │ │pymss ││                   │
│  │  │walk  │ │run   │ │jump  │ │...   ││                   │
│  │  └──────┘ └──────┘ └──────┘ └──────┘│                   │
│  │      ↑ weights from curriculum ↑     │                   │
│  └──────────────────────────────────────┘                   │
└─────────────────────────────────────────────────────────────┘
```

## Next Steps (Future Improvements)

This curriculum learning is **Step 1** of improving multimodal training. Future steps could include:

1. **Motion Embedding**: Add motion identifier to state for explicit conditioning
2. **Dynamic BVH Switching**: C++ level motion switching (headers already provided)
3. **Motion Blending**: Smooth transitions between motions
4. **Per-motion Learning Rates**: Different LR for different difficulty levels

## Troubleshooting

### Weights not updating
- Check that you're past the warmup period
- Verify `--update_freq` is reasonable (5-10 is good)

### One motion dominating
- Reduce `max_weight` (default 3.0)
- Increase `temperature` for smoother distribution

### All motions getting same weight
- You may still be in warmup
- Returns may be too similar for differentiation
- Try `--curriculum performance` for more aggressive differentiation

## References

- Original MASS: "Scalable Muscle-actuated Human Simulation and Control" (SIGGRAPH 2019)
- Curriculum Learning: "Curriculum Learning" (Bengio et al., ICML 2009)
- UCB Strategy: Multi-Armed Bandit algorithms