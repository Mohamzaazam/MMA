# Python Utilities

Data processing and dataset utilities for motion prediction training.

## Modules

| Module | Description |
|--------|-------------|
| `datasets.py` | `BVHDataset` and `ExtractedBVHDataset` for PyTorch training |
| `normalization.py` | `StateNormalizer` for mean/std normalization |
| `splits.py` | Subject-disjoint train/val/test splitting |
| `state_extractor.py` | C++ pymss integration for DART state extraction |
| `extract_states.py` | Script to pre-extract BVH → NPZ files |

## Quick Start

### Dataset Loading

```python
from python.utils import ExtractedBVHDataset, StateNormalizer

# Load pre-extracted NPZ files (fast)
dataset = ExtractedBVHDataset(
    npz_files=['data/extracted/01_01.npz', ...],
    mode='mlp',        # or 'transformer', 'autoregressive'
    max_horizon=16,    # Multi-step targets (1=single-step)
)

# Access samples
x, y = dataset[0]  # x: state_t, y: state_t+1 (or horizon targets)
```

### Subject-Disjoint Split

```python
from python.utils import subject_disjoint_activity_split

train, val, test, info = subject_disjoint_activity_split(
    bvh_root='data/cmu',
    train_ratio=0.7,
    test_ratio=0.15,
)
```

### State Extraction

```bash
# Extract BVH files to NPZ (run once)
pixi run python python/utils/extract_states.py \
    --bvh_dir data/cmu \
    --output_dir data/extracted \
    --max_subjects 50
```

## File Formats

- **BVH** → Raw motion capture (requires C++ pymss)
- **NPZ** → Pre-extracted states (112D: 56 pos + 56 vel)
