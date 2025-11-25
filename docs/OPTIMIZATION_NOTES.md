# MASS Simulation Speed Optimizations

## Overview

This document describes the incremental optimizations made to improve simulation speed without modifying the core simulation logic.

## Changes Made

### 1. CMakeLists.txt - Compiler Optimizations

**Impact: HIGH (20-50% speedup expected)**

| Change | Description |
|--------|-------------|
| `-O3` | Maximum optimization level |
| `-march=native` | Uses CPU-specific instructions (SSE, AVX, etc.) |
| `EIGEN_NO_DEBUG` | Disables Eigen assertion checks |
| `EIGEN_FAST_MATH` | Enables Eigen's fast math mode |

**Optional flags** (can be enabled via cmake options):
- `-DENABLE_FAST_MATH=ON`: Enables `-ffast-math` (less precise but faster)
- `-DENABLE_LTO=ON`: Link-time optimization (better inlining across files)

### 2. EnvManager.cpp - OpenMP Improvements

**Impact: MEDIUM (10-30% speedup for parallel environments)**

| Change | Description |
|--------|-------------|
| Dynamic thread count | Matches threads to available cores |
| `schedule(dynamic)` | Better load balancing when environments terminate at different times |
| `schedule(static)` | Lower overhead for uniform workloads |
| Parallelized more functions | `Resets()`, `IsEndOfEpisodes()`, `GetStates()`, `SetActions()`, `GetRewards()` |
| Pre-computed offsets | Enables parallel tuple copying in `ComputeMuscleTuples()` |
| Pre-allocated matrices | Avoids repeated memory allocations |

### 3. Muscle.cpp/h - Memory Optimization

**Impact: MEDIUM (10-20% speedup in muscle simulation)**

| Change | Description |
|--------|-------------|
| Pre-allocated cache matrices | `mCachedJt`, `mCachedA`, `mCachedP`, `mCachedJtA_reduced` |
| Inlined anchor position update | Reduces function call overhead in `Update()` |
| Cached Jacobian transpose | Returns const reference instead of copy |
| `InitializeCaches()` | One-time allocation after muscle setup |

### 4. pixi.toml - Build Configuration

**New build commands:**
- `pixi run build` - Standard Release build (recommended)
- `pixi run build-debug` - Debug build with symbols
- `pixi run build-lto` - Release with Link-Time Optimization
- `pixi run build-fast` - Maximum speed (LTO + fast-math)

## How to Test

### Step 1: Clean and rebuild
```bash
pixi run clean
pixi run build
```

### Step 2: Run training and compare
```bash
# Time a short training run
time pixi run train
```

### Step 3: Enable more aggressive optimizations (optional)
```bash
# Try LTO build (longer compile, potentially faster runtime)
pixi run build-lto

# Try fast-math build (fastest, slightly less precise)
pixi run build-fast
```

## Profiling

To identify remaining bottlenecks:
```bash
pixi run train-profile
# Then analyze with:
python -c "import pstats; p = pstats.Stats('profile.prof'); p.sort_stats('cumtime').print_stats(20)"
```

## What's NOT Changed

The following core logic remains unchanged:
- DART physics simulation algorithms
- Muscle dynamics equations (g_al, g_pl, g_t functions)
- PPO training algorithm
- Reward computation
- State/action space definitions
- BVH motion data processing

## Expected Performance Gains

| Optimization Level | Expected Speedup |
|--------------------|------------------|
| Release (-O3) | 2-3x vs Debug |
| + march=native | +10-20% |
| + OpenMP improvements | +10-30% |
| + Memory pre-allocation | +10-20% |
| + LTO | +5-10% |
| + fast-math | +5-15% |

**Total potential improvement: 2-5x faster than unoptimized Debug build**

## Future Optimization Opportunities

If more speed is needed, consider (in order of impact):
1. GPU-accelerated physics (requires DART GPU support)
2. Reducing simulation Hz (currently 600, try 450 or 300)
3. Batched neural network inference
4. SIMD-optimized muscle force computation
5. Custom memory allocator for DART