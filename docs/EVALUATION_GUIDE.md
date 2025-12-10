# MotionNN Evaluation Guide

This document explains the evaluation metrics and visualizations used to assess MotionNN motion prediction performance.

## Single-Step vs Rollout Predictions

### Single-Step Prediction

**What it measures:** Given state $s_t$, how well does the model predict $s_{t+1}$?

```
Ground truth:  s_0 → s_1 → s_2 → s_3 → ...
Prediction:         ↑      ↑      ↑
                  p_1    p_2    p_3  (each uses GT input)
```

- Each prediction is made from the **ground truth** previous state
- Error is measured against ground truth at each step
- **Doesn't reveal error accumulation** over time
- Useful for measuring instantaneous prediction quality

### Rollout Prediction

**What it measures:** Starting from $s_0$, how well does the model predict a full trajectory by feeding its own predictions back in?

```
Ground truth:  s_0 → s_1 → s_2 → s_3 → ...
Rollout:       s_0 → p_1 → p_2 → p_3 → ...
                     ↓     ↓
               (predictions feed into next step, not GT)
```

- Errors **compound** because each prediction uses potentially incorrect input
- Reveals model stability and drift over time
- **More realistic** for actual deployment scenarios

### Which is More Insightful for Motion?

**Rollout is significantly more important** for motion data:

| Aspect | Single-Step | Rollout |
|--------|-------------|---------|
| **Real-world use** | Unrealistic | Matches deployment |
| **Stability** | Hides instability | Exposes drift, oscillations |
| **Biomechanics** | May look good | Reveals implausible motions |
| **Long-term dynamics** | Ignores | Tests if model learned dynamics |

### Typical Failure Modes (only visible in rollout)

1. **Mean regression** — predictions drift toward average pose
2. **High-frequency oscillation** — joints wobbling
3. **Accumulating drift** — character moves away from reference
4. **Pose collapse** — limbs interpenetrate or collapse

---

## Evaluation Plots Explained

### trajectory.png

**What it shows:** Time-series comparison of root position (X, Y, Z) and root velocity (VelX).

- **Blue line** = Ground truth from BVH data
- **Red dashed line** = Model prediction (single-step)
- **X-axis** = Frame number (time)
- **Y-axis** = Position (meters) or velocity (m/s)

**How to interpret:**
- Close overlap = good prediction
- Large gaps = poor prediction for that component
- Oscillations in prediction but not GT = instability
- Prediction lagging behind GT = the model is averaging

**Why root position matters:** Root position determines where the character is in space. Errors here cause the character to drift or teleport.

---

### joint_angles.png

**What it shows:** Lower limb joint angles over time for each joint and axis.

**Layout:** 6 rows (joints) × 3 columns (axes)
- **Rows:** Right Hip, Right Knee, Right Ankle, Left Hip, Left Knee, Left Ankle
- **Columns:** Flexion (X), Abduction (Y), Rotation (Z)

**Per-subplot:**
- **Blue line** = Ground truth angle
- **Red dashed line** = Predicted angle
- **Gray shading** = Error region between GT and prediction
- **MSE label** = Mean squared error for that joint-axis

**How to interpret:**
- **Good prediction:** Blue and red overlap closely, minimal gray shading
- **Knee flexion** typically has largest magnitude (walking involves bending knees)
- **Hip rotation** often has small magnitude but high prediction error
- Compare left vs right legs — should be similar (symmetric)

**Common patterns:**
- High MSE on rotation axes = model struggles with fine rotational dynamics
- Prediction smoother than GT = model averaging / not capturing details
- Phase shift = prediction is delayed relative to GT

---

### rollout_horizon.png

**What it shows:** How prediction error grows with rollout horizon (1, 5, 10, 30, 60 steps).

**Interpretation:**
- **Linear growth** = errors accumulate steadily (acceptable)
- **Sublinear growth** = model is stable, errors don't explode (good)
- **Exponential growth** = model is unstable, predictions diverge quickly (bad)

**Your results:** 1-step MSE = 0.06, 60-step MSE = 1.93 (32× increase over 60 steps)

This ~32× increase over 60 frames suggests the model is reasonably stable for short horizons but accumulates significant error over longer rollouts.

---

### per_component_errors.png

**What it shows:** Bar chart of MSE for each body component.

**How to interpret:**
- **Longer bars** = higher error = harder to predict
- Compare **position** vs **velocity** — velocity is typically harder
- Compare **limbs** — which body parts are most challenging?

---

## Recommended Evaluation Workflow

```bash
# 1. Train with test hold-out
pixi run python python/models/train_motion.py \
    --bvh_dir data/cmu \
    --activity_split \
    --test_ratio 0.15 \
    --epochs 50

# 2. Evaluate on held-out test set
pixi run python python/models/eval_motion.py \
    --model nn/motion_model_best.pt \
    --split_file nn/split_info.json \
    --use_test_set \
    --num_frames 500 \
    --output_dir eval_test_results

# 3. Check key metrics in metrics.json
cat eval_test_results/metrics.json | jq '.summary'
```

**Key metrics to monitor:**
- `total_mse` — Overall prediction quality
- `r2` — Coefficient of determination (closer to 1 = better)
- `position_mse` vs `velocity_mse` — Usually velocity is harder
- Rollout horizon plot — Is it linear or exponential?
