# Python Utilities

Utilities for data processing, dataset creation, and mocap analysis.

## Modules

| Module | Description |
|--------|-------------|
| `bvh_dataset.py` | PyTorch Dataset for BVH motion data |
| `state_extractor.py` | C++ pymss integration for DART states |
| `verify_conversion.py` | BVH conversion integrity checking |
| `scrape_mocap_metadata.py` | CMU mocap metadata scraper |
| `truncate_mocap.py` | Truncate motion capture files |

## BVH Dataset Pipeline

For motion model pre-training:

```python
# Small datasets (in-memory)
from python.utils.bvh_dataset import BVHDataset, train_val_split

train_ds, val_ds = train_val_split('data/cmu', train_ratio=0.8)

# Large datasets (disk-based)
from python.utils.bvh_dataset import extract_to_disk, LazyBVHDataset

# Step 1: Extract to disk (once)
files, norm = extract_to_disk(bvh_files, 'data/extracted')

# Step 2: Lazy loading
dataset = LazyBVHDataset(files, normalizer_path='data/extracted/normalizer.npz')
```

## State Extraction

Extract DART-compatible states from BVH:

```python
from python.utils.state_extractor import BVHStateExtractor

extractor = BVHStateExtractor('data/metadata.txt', 'build')
states = extractor.extract_all_states()  # (N, 112) - 56 pos + 56 vel
```

## Verify Conversion

Check BVH file integrity after conversion:

```bash
pixi run python python/utils/verify_conversion.py data/cmu --source /mnt/e/database/cmu --fix
```
