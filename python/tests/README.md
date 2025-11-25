# MASS Testing & Visualization Scripts

This directory contains incremental testing and visualization scripts for the MASS (Muscle-Actuated Skeletal System) framework.

## Quick Start

```bash
# 1. Navigate to project root
cd /path/to/MASS

# 2. Copy these scripts to python directory
cp -r MASS_test_scripts/python/* python/

# 3. Enter pixi shell
pixi shell

# 4. Run all tests
python python/run_all_tests.py
```

## Script Progression (Baby Steps)

The scripts are designed to be run incrementally, with each step building on the previous:

### Step 1: `test_01_model_inference.py`
**Test neural networks WITHOUT the simulation environment**

```bash
python python/test_01_model_inference.py
```

Tests:
- Import Model classes (SimulationNN, MuscleNN)
- Create model instances with typical dimensions
- Run forward passes with dummy data
- Find and validate pre-trained weight files
- Test batch inference

✅ Pass this before moving to Step 2.

### Step 2: `test_02_single_env.py`
**Test single-motion environment (standard MASS pipeline)**

```bash
python python/test_02_single_env.py

# Or with specific metadata:
python python/test_02_single_env.py data/metadata.txt --slaves 4
```

Tests:
- Import pymss module
- Create environment from metadata
- Reset environment (with/without RSI)
- Get states from environment
- Step simulation with random actions
- Muscle interface (torques, activations)
- Complete episode rollout
- Inference with trained model (if available)

✅ Pass this before moving to Step 3.

### Step 3: `test_03_multimodal_env.py`
**Test multi-modal environment with multiple motions**

```bash
python python/test_03_multimodal_env.py

# Or with specific files:
python python/test_03_multimodal_env.py --motion_list data/motion_list.txt --template data/metadata.txt --slaves 2
```

Tests:
- Import multimodal components
- Load motion configurations from motion_list.txt
- Create MultimodalEnvManager
- Reset across all motions
- Get states from all slaves
- Per-slave access (motion name, rewards)
- Step simulation
- Muscle interface
- Episode rollout with per-motion tracking
- Trained multimodal model inference (if available)

✅ Pass this to confirm both pipelines work.

### Step 4: `test_04_visualize.py`
**Helper for launching visualization**

```bash
# Interactive mode
python python/test_04_visualize.py

# Single-motion visualization
python python/test_04_visualize.py --mode single

# Multi-modal visualization (select motion)
python python/test_04_visualize.py --mode multi --motion walk

# Show all possible commands
python python/test_04_visualize.py --mode all
```

Generates the correct `./build/render` commands for visualization.

⚠️ Requires X server (VcXsrv on WSL).

## Evaluation Scripts

### `evaluate_models.py`
**Headless evaluation without visualization**

```bash
# Compare all available models
python python/evaluate_models.py --mode compare

# Evaluate single-motion model
python python/evaluate_models.py --mode single --episodes 10

# Evaluate multimodal model on all motions
python python/evaluate_models.py --mode multi --episodes 5

# Evaluate multimodal model on specific motion
python python/evaluate_models.py --mode multi --motion walk
```

This is useful for:
- Comparing model performance
- Running on headless servers
- Benchmarking different checkpoints

## Complete Test Runner

### `run_all_tests.py`
**Run all tests in sequence**

```bash
python python/run_all_tests.py
```

Runs:
1. test_01_model_inference.py
2. test_02_single_env.py
3. test_03_multimodal_env.py

Stops at first failure and provides a summary.

## Prerequisites

1. **Build the project**:
   ```bash
   pixi run build
   ```

2. **Train models (optional, for inference tests)**:
   ```bash
   # Single-motion training
   pixi run train
   
   # Multi-modal training
   pixi run train_multimodal
   ```

3. **Prepare motion list (for multimodal)**:
   ```bash
   python python/tmps/scan_motions.py data/motion
   # Edit data/motion_list.txt as needed
   ```

## Expected Output Structure

After training, you should have:

```
nn/
├── current.pt              # Single-motion policy (latest)
├── current_muscle.pt       # Single-motion muscle net (latest)
├── max.pt                  # Single-motion policy (best)
├── max_muscle.pt           # Single-motion muscle net (best)
├── multimodal_current.pt   # Multi-modal policy (latest)
├── multimodal_current_muscle.pt
├── multimodal_max.pt       # Multi-modal policy (best)
├── multimodal_max_muscle.pt
├── 1.pt                    # Checkpoint at epoch 100
├── 1_muscle.pt
└── ...
```

## Troubleshooting

### "Cannot import pymss"
```bash
pixi run build  # Rebuild the project
```

### "Metadata file not found"
Make sure you're running from the project root directory.

### "NaN values in states"
Some motions may cause instability. Run:
```bash
python python/tmps/test_motions_incrementally.py
```
to identify problematic motions.

### X server errors during visualization
On WSL:
```bash
export DISPLAY=:0
# Make sure VcXsrv is running
```

### CUDA out of memory
Reduce batch size or number of slaves:
```bash
python python/test_02_single_env.py --slaves 2
```

## Quick Reference

| Script | Purpose | Requires |
|--------|---------|----------|
| test_01_model_inference.py | Test NNs only | Model.py |
| test_02_single_env.py | Test single env | pymss, metadata.txt |
| test_03_multimodal_env.py | Test multimodal | pymss, motion_list.txt |
| test_04_visualize.py | Launch render | X server |
| evaluate_models.py | Compare models | Trained models |
| run_all_tests.py | Run all tests | All above |
