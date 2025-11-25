# Multimodal Motion Training for MASS

This extension enables training a single policy to imitate multiple motion clips (walking, running, etc.) using the MASS (Muscle-Actuated Skeletal System) framework.

## Overview

The multimodal training system works by:
1. **Scanning** available BVH motion files in your motion directory
2. **Creating** separate environment instances for each motion
3. **Training** a single policy across all motions simultaneously
4. **Tracking** per-motion performance for analysis and curriculum learning

## Quick Start

### Step 1: Scan Your Motion Files

First, scan your motion directory to understand what's available:

```bash
python python/scan_motions.py data/motion
```

This generates `data/motion_list.txt` with all detected BVH files.

### Step 2: Review and Edit Motion List

Edit `data/motion_list.txt` to:
- Remove unwanted motions
- Adjust cyclic flags (`true` for looping motions like walking)
- Verify paths are correct

Format:
```
# Motion list for MASS multimodal training
# Format: <relative_path> <cyclic: true/false>

/data/motion/walk.bvh true
/data/motion/run.bvh true
/data/motion/jump.bvh false
```

### Step 3: Run Multimodal Training

```bash
python python/main_multimodal.py \
    --motion_list data/motion_list.txt \
    --template data/metadata.txt \
    --slaves 4
```

Arguments:
- `--motion_list`: Path to your motion list file
- `--template`: Path to your existing metadata.txt (used as template)
- `--slaves`: Number of parallel environments per motion (default: 4)

### Step 4: Monitor Training

View training progress with TensorBoard:

```bash
tensorboard --logdir=runs --host 0.0.0.0 --port 6006
```

Metrics logged:
- `Loss/Actor`, `Loss/Critic`, `Loss/Muscle`: Training losses
- `Train/AvgReturn`: Overall average return
- `PerMotion/<motion_name>_AvgReturn`: Per-motion performance
- `PerMotion/<motion_name>_Episodes`: Episode count per motion

## File Structure

```
python/
├── scan_motions.py          # Utility to scan BVH files
├── multimodal_env.py        # MultimodalEnvManager class
├── main_multimodal.py       # Multimodal training script
├── test_motion_library.py   # Test for motion selection logic
└── test_multimodal_env.py   # Test for environment manager

core/
├── MotionLibrary.h          # C++ motion library header (future)
└── MotionLibrary.cpp        # C++ motion library impl (future)

data/
├── motion_list.txt          # Generated motion list
└── metadata_<motion>.txt    # Auto-generated per-motion metadata
```

## How It Works

### Architecture

```
                    ┌─────────────────────────────────────┐
                    │      MultimodalEnvManager          │
                    │                                     │
                    │  ┌──────────┐    ┌──────────┐     │
                    │  │ pymss    │    │ pymss    │     │
                    │  │ (walk)   │    │ (run)    │ ... │
                    │  │ 4 slaves │    │ 4 slaves │     │
                    │  └──────────┘    └──────────┘     │
                    │                                     │
                    │  Unified interface for training    │
                    └─────────────────────────────────────┘
                                    ↕
                    ┌─────────────────────────────────────┐
                    │           PPO Trainer               │
                    │                                     │
                    │   Single Policy (SimulationNN)      │
                    │   Single Muscle Net (MuscleNN)      │
                    │                                     │
                    │   Per-motion statistics tracking    │
                    └─────────────────────────────────────┘
```

### Key Design Decisions

1. **No Core C++ Modifications**: The multimodal system works by wrapping multiple `pymss` instances, preserving the original MASS codebase unchanged.

2. **Weighted Sampling**: Motions can be weighted for curriculum learning (e.g., start with easier motions).

3. **Per-Motion Tracking**: Performance is tracked separately for each motion to identify which motions need more training.

## Advanced Usage

### Curriculum Learning

Adjust motion weights during training to focus on harder motions:

```python
# In main_multimodal.py, after initialization:
ppo.env.set_motion_weights({
    'walk': 1.0,   # Easy - normal weight
    'run': 2.0,    # Harder - double weight
    'jump': 3.0    # Hardest - triple weight
})
```

### Adding Custom Motions

1. Place your BVH file in `data/motion/`
2. Run `scan_motions.py` to regenerate the motion list
3. Edit cyclic flag if needed
4. Ensure your BVH's skeleton matches the expected hierarchy

### Skeleton Compatibility

Use the BVH analyzer to check skeleton compatibility:

```bash
python python/bvh_hierarchy_analyzer.py
```

This shows the joint mapping between different BVH formats.

## Troubleshooting

### Motion not loading
- Check BVH file path in motion_list.txt
- Verify skeleton hierarchy matches (use bvh_hierarchy_analyzer.py)
- Check cyclic flag is appropriate for the motion type

### Per-motion performance varies wildly
- This is normal initially - the policy is learning all motions
- Consider using weighted sampling to balance training
- Check if some motions have incompatible timing/scale

### Out of memory
- Reduce `--slaves` parameter
- Reduce number of motions loaded simultaneously
- Use gradient checkpointing (requires code modification)

## Future Improvements

1. **Dynamic BVH Switching**: C++ MotionLibrary class for runtime motion switching (headers provided)
2. **Motion Embedding**: Add motion identifier to state for better generalization
3. **Automatic Curriculum**: Adaptive weighting based on per-motion performance
4. **Motion Blending**: Smooth transitions between different motions

## References

- Original MASS paper: "Scalable Muscle-actuated Human Simulation and Control" (SIGGRAPH 2019)
- PPO: "Proximal Policy Optimization Algorithms" (Schulman et al., 2017)