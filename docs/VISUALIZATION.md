# BVH Visualization Tool

CLI tool for analyzing and visualizing BVH motion capture files.

## Usage

```bash
pixi run python python/amc2bvh/bvh_visualizer.py <bvh_file> [options]
```

## Options

### Input/Output
| Option | Description |
|--------|-------------|
| `bvh_file` | Primary BVH file to analyze (required) |
| `-c, --compare FILE` | Second BVH file for comparison |
| `-o, --output-dir DIR` | Output directory (default: `data/outputs`) |
| `--no-save` | Display plots without saving |

### Joint Selection
| Option | Description |
|--------|-------------|
| `-j, --joints JOINT...` | Joints to analyze (partial matching supported) |
| `--list-joints` | List all available joints and exit |

### Smoothing Methods
| Option | Description |
|--------|-------------|
| `-s, --smoothing METHOD` | `none`, `moving_average`, `gaussian`, `savgol`, `exponential`, `butterworth`, `median` |
| `--window-size N` | Window size for moving_average/savgol/median (default: 5) |
| `--sigma N` | Sigma for Gaussian (default: 2.0) |
| `--poly-order N` | Polynomial order for Savitzky-Golay (default: 3) |
| `--alpha N` | Alpha for exponential (default: 0.3) |
| `--cutoff N` | Cutoff for Butterworth (default: 0.1) |
| `--order N` | Order for Butterworth (default: 4) |
| `--kernel-size N` | Kernel size for median (default: 5) |

### Visualization
| Option | Description |
|--------|-------------|
| `-v, --viz TYPE...` | Types: `all`, `hierarchy`, `overview`, `statistics`, `smoothing`, `velocity`, `rotations`, `heatmap`, `report` |
| `--channel CH` | `Xrotation`, `Yrotation`, `Zrotation` (default: Zrotation) |
| `--no-display` | Only save, don't display plots |
| `--report-only` | Generate only text report |
| `-q, --quiet` | Suppress progress output |

## Examples

```bash
# Basic usage
pixi run python python/amc2bvh/bvh_visualizer.py data/motion/29_01.bvh

# Compare two files with Gaussian smoothing
pixi run python python/amc2bvh/bvh_visualizer.py data/motion/29_01.bvh \
    --compare data/motion/walk.bvh -s gaussian --sigma 4

# Analyze specific joints
pixi run python python/amc2bvh/bvh_visualizer.py data/motion/29_01.bvh \
    --joints Hips Spine --viz rotations velocity

# List available joints
pixi run python python/amc2bvh/bvh_visualizer.py data/motion/29_01.bvh --list-joints

# Save only (headless)
pixi run python python/amc2bvh/bvh_visualizer.py data/motion/29_01.bvh --no-display -o results/
```