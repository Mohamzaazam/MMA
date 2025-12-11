# Temporary / Deprecated Scripts

Utility scripts that are not part of the main pipeline. These may be useful for data preparation but are not actively maintained.

## Files

| File | Description |
|------|-------------|
| `downloader.py` | CMU Motion Capture Database downloader |
| `run_downloader.py` | CLI wrapper for downloader |
| `scrape_mocap_metadata.py` | Scrape CMU mocap trial descriptions |
| `truncate_mocap.py` | Truncate BVH/AMC files to frame range |
| `verify_conversion.py` | Check BVH conversion integrity |

## Usage

### Download CMU Data

```bash
# Download entire CMU dataset (~500MB)
pixi run python python/tmp/downloader.py -o /path/to/cmu

# Download specific subjects
pixi run python python/tmp/run_downloader.py --subjects 1,2,86
```

### Verify BVH Conversion

```bash
# Check for truncated or corrupted BVH files
pixi run python python/tmp/verify_conversion.py data/cmu \
    --source /mnt/e/database/cmu \
    --fix --reconvert
```

### Truncate Motion Files

```bash
# Keep only frames 100-500
pixi run python python/tmp/truncate_mocap.py input.bvh output.bvh \
    --start 100 --end 500
```

## Note

These scripts were moved from `python/utils/` to reduce clutter in the main utility module. They are standalone tools and do not depend on the main pipeline.