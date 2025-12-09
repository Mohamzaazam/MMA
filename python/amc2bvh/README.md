# ASF/AMC to BVH Converter

Tools for converting CMU Motion Capture data from ASF/AMC format to BVH format.

## Files

| File | Description |
|------|-------------|
| `standalone.py` | Core converter for single ASF/AMC to BVH conversion |
| `batch_amc2bvh.py` | Batch converter with parallel processing |
| `bvh_parser.py` | Pure Python BVH parser |
| `bvh_visualizer.py` | BVH motion visualization and analysis |


## Quick Start

### 1. Scrape Motion Metadata (Optional but Recommended)

Downloads motion descriptions from the CMU mocap website:

```bash
pixi run python python/amc2bvh/scrape_mocap_metadata.py /mnt/e/database/cmu
```

This creates:
- `motion_catalog.txt` - Master catalog of all motions (searchable)
- `motion_catalog.json` - JSON version for programmatic access
- `subjects/*/trials.txt` - Per-subject trial descriptions

### 2. Batch Convert to BVH

Convert all ASF/AMC files to BVH format:

```bash
pixi run python python/amc2bvh/batch_amc2bvh.py /mnt/e/database/cmu -o data/cmu --walk-bvh
```

### 3. Convert Specific Subjects

```bash
pixi run python python/amc2bvh/batch_amc2bvh.py /mnt/e/database/cmu -o data/cmu --walk-bvh --subjects 01,02,86
```

### 4. Single File Conversion

```bash
pixi run python python/amc2bvh/standalone.py skeleton.asf motion.amc -o output.bvh --walk-bvh
```

## Options

### Batch Converter Options

| Option | Description |
|--------|-------------|
| `-o, --output` | Output directory (required) |
| `-w, --workers` | Parallel workers (default: 4) |
| `--walk-bvh` | Use Maya-compatible format (recommended) |
| `--subjects` | Comma-separated subject IDs (e.g., 01,02,86) |
| `--force` | Force reconversion of existing files |
| `--dry-run` | Preview without converting |
| `-v, --verbose` | Verbose logging |

### Walk-BVH Format

The `--walk-bvh` flag applies these settings for Maya compatibility:
- ZXY rotation order
- CMU unit scaling (×5.64)
- Character1_* joint naming
- Collapsed intermediate joints (lhipjoint, rhipjoint, etc.)

## Output Structure

```
data/cmu/
├── 01/
│   ├── 01_01.bvh
│   ├── 01_02.bvh
│   └── trials.txt      # Motion descriptions (if scraped)
├── 02/
│   ├── 02_01.bvh
│   └── trials.txt
├── motion_catalog.txt  # Master catalog
└── conversion_report.json
```

## Example trials.txt

```
# Subject 02: various expressions and human behaviors
02_01: walk
02_02: walk
02_03: run/jog
02_04: jump, balance
02_05: punch/strike
```
