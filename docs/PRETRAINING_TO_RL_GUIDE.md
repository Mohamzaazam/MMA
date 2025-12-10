# Motion Learning Pipeline: Pre-training to RL

This document outlines the complete pipeline from supervised motion pre-training to reinforcement learning with muscle control.

---

## 1. Evaluation Fundamentals

### Single-Step vs Rollout Predictions

| Type | Description | Use Case |
|------|-------------|----------|
| **Single-step** | Predict $s_{t+1}$ from ground truth $s_t$ | Measures instantaneous accuracy |
| **Rollout** | Chain predictions: $s_0 → p_1 → p_2 → ...$ | Reveals stability & drift |

**For motion data: Rollout is more important** because:
- Matches deployment conditions (no ground truth at inference)
- Exposes instability, oscillations, pose collapse
- Tests if model learned actual dynamics

### Key Metrics to Monitor

| Metric | Good Value | Interpretation |
|--------|------------|----------------|
| Position MSE | < 0.001 | Where is the character? |
| Velocity MSE | < 0.5 | How is it moving? (harder) |
| R² | > 0.8 | Explained variance |
| Rollout growth | Sublinear | Stability over time |

---

## 2. Current Architecture

```
┌─────────────────────────────────────────────────────────┐
│  SimulationNN (Policy)                                  │
│  State → MLP(256-256) → Gaussian(τ_des)                 │
└─────────────────────────────────────────────────────────┘
                           ↓ Desired torques
┌─────────────────────────────────────────────────────────┐
│  SPD Controller                                         │
│  τ_des + current pose → actual τ                        │
└─────────────────────────────────────────────────────────┘
                           ↓ Torques
┌─────────────────────────────────────────────────────────┐
│  MuscleNN                                               │
│  (JtA, τ) → MLP(1024-512-512) → Activations            │
└─────────────────────────────────────────────────────────┘
```

**SPD is valuable because:**
- Provides stable torque computation from position targets
- Handles compliance and damping naturally
- More interpretable than direct torque prediction
- Well-tested in biomechanics literature

---

## 3. Pre-training Strategy (Current Focus)

### Goal
Train MotionNN to predict next-state from current-state using diverse BVH data.

### Pipeline
```
BVH Files → BVHDataset → MotionNN (Transformer) → State predictions
                              ↓
                    Encoder weights transfer to RL
```

### Recommended Improvements

#### A. Loss Function
```python
# Current: Equal weight on all dimensions
loss = F.mse_loss(pred, target)

# Better: Weight position higher than velocity
pos_weight, vel_weight = 1.0, 0.1
pos_loss = F.mse_loss(pred[:, :56], target[:, :56])
vel_loss = F.mse_loss(pred[:, 56:], target[:, 56:])
loss = pos_weight * pos_loss + vel_weight * vel_loss
```

#### B. Multi-step Training
```python
# Occasionally do rollout during training
if random.random() < 0.2:  # 20% of batches
    state = x
    total_loss = 0
    for step in range(5):
        pred = model(state)
        total_loss += criterion(pred, targets[step])
        state = pred.detach()
    loss = total_loss / 5
```

#### C. Smoothness Regularization
```python
# Penalize jerky predictions
accel = pred[2:] - 2*pred[1:-1] + pred[:-2]
smoothness_loss = 0.01 * torch.mean(accel ** 2)
loss = mse_loss + smoothness_loss
```

---

## 4. Handling Limited Compute

### Problem
Full CMU dataset is too large to load at once.

### Solutions

#### Option 1: Streaming Dataset (Recommended)
```python
class StreamingBVHDataset(Dataset):
    def __init__(self, bvh_files, max_files_in_memory=10):
        self.all_files = bvh_files
        self.max_files = max_files_in_memory
        self.current_files = []
        self.current_data = []
        self._refresh_subset()
    
    def _refresh_subset(self):
        # Sample random subset of files
        self.current_files = random.sample(self.all_files, self.max_files)
        self.current_data = load_files(self.current_files)
    
    def __getitem__(self, idx):
        # Periodically refresh data
        if idx % 10000 == 0:
            self._refresh_subset()
        return self.current_data[idx % len(self.current_data)]
```

#### Option 2: Curriculum Learning
```
Epoch 1-10:   Train on walk data only (small subset)
Epoch 11-20:  Add run data
Epoch 21-30:  Add jump data
...
```

#### Option 3: DataLoader with `num_workers`
```python
# Already using this - ensure workers > 0
DataLoader(dataset, batch_size=256, num_workers=4, prefetch_factor=2)
```

#### Option 4: Smaller Sequence Length
```bash
# Reduce seq_len for transformers
--seq_len 64  # instead of 128
```

---

## 5. RL Integration (Future)

### Transfer Strategy
```python
class SimulationNNWithPrior(nn.Module):
    def __init__(self, pretrained_motion_encoder):
        self.encoder = pretrained_motion_encoder  # From MotionNN
        self.encoder.requires_grad_(False)  # Freeze initially
        
        self.policy_head = nn.Linear(hidden_dim, num_actions)
        self.value_head = nn.Linear(hidden_dim, 1)
```

### Reward Design (No Reference Tracking)
```python
reward = (
    w_vel * velocity_reward +      # Match target velocity
    w_heading * heading_reward +   # Match target heading
    w_energy * energy_penalty +    # Minimize CoT
    w_smooth * smoothness_reward   # No jerky motion
)
# NO explicit pose tracking!
```

### SPD in RL Pipeline
```
Policy → Target Pose → SPD → τ_des → MuscleNN → Activations
```

SPD remains valuable:
- Converts pose targets to torques with compliance
- More stable than direct torque prediction
- Natural interface between kinematic and dynamic control

---

## 6. Action Items (Priority Order)

### Immediate (Pre-training Focus)
- [ ] Implement weighted loss (position > velocity)
- [ ] Add multi-step training (20% of batches)
- [ ] Implement streaming dataset for memory efficiency
- [ ] Train on subset, evaluate rollout stability

### Short-term (When pre-training works)
- [ ] Scale to full CMU dataset with streaming
- [ ] Evaluate on held-out test subjects
- [ ] Verify rollout horizon < 10× single-step error at 60 steps

### Medium-term (RL Integration)
- [ ] Extract encoder from MotionNN
- [ ] Create SimulationNNWithPrior class
- [ ] Design task-only reward (velocity, heading, energy)
- [ ] Test on single motion before multimodal

### Long-term (Multimodal)
- [ ] Add mode conditioning to policy
- [ ] Train hierarchical policy (mode selector + motion generator)
- [ ] Evaluate on diverse locomotion tasks

---

## 7. Do NOT Distract Yourself With

❌ Complex reward shaping before pre-training works  
❌ Multimodal training before single-motion succeeds  
❌ Full dataset before streaming is implemented  
❌ RL fine-tuning before encoder quality is verified  
❌ Adversarial training (AMP) before simpler approach works  
❌ Fancy architectures before MLP baseline is strong  

**Focus sequence:** Pre-training → Streaming → RL transfer → Multimodal

---

## 8. Data Preprocessing (ASF/AMC → BVH)

### When to Apply Smoothing

| Smoothing Location | Effect | Recommendation |
|-------------------|--------|----------------|
| **During conversion** | Removes sensor noise | ✅ OK if data has artifacts |
| **During training** | Creates artificial dynamics | ❌ Avoid |
| **Post-prediction** | Hides model deficiencies | ❌ Avoid |

### Butterworth Filter Settings
```python
from scipy.signal import butter, filtfilt

def smooth_motion(data, cutoff_hz=6, sample_rate=120, order=4):
    """Gentle low-pass filter for mocap data."""
    nyquist = sample_rate / 2
    normalized_cutoff = cutoff_hz / nyquist
    b, a = butter(order, normalized_cutoff, btype='low')
    return filtfilt(b, a, data, axis=0)
```

**Recommended:** 6-10 Hz cutoff for walking/running mocap data.  
**Do NOT smooth:** Quick foot strikes, directional changes (these are real dynamics).

---

## 9. RL Reward Strategies

### Option A: Reference Tracking (Current MASS)
```python
# DeepMimic-style: tracks specific reference motion
reward = (
    w_pose * pose_reward +          # Match reference pose
    w_vel * velocity_reward +       # Match reference velocity
    w_ee * end_effector_reward +    # Match reference hands/feet
    w_com * com_reward              # Match reference center of mass
)
```
**Use for:** Single motion imitation, verifying pipeline works.  
**Problem:** Cannot generalize to multimodal, requires reference at deployment.

### Option B: Task-Only Reward (Goal for Multimodal)
```python
# No reference tracking, just task objectives
reward = (
    w_target_vel * velocity_toward_goal +  # Achieve target speed
    w_heading * heading_toward_goal +      # Face target direction
    w_alive * alive_bonus +                # Don't fall
    w_energy * -energy_cost +              # Minimize CoT
    w_smooth * -jerk_penalty               # Smooth motion
)
```
**Use for:** Multimodal deployment, reference-free operation.  
**Challenge:** May produce unnatural motion without motion prior.

### Recommended Transition
```
Phase 1: Train with full reference tracking (verify pipeline)
Phase 2: Reduce reference weight (w_pose: 1.0 → 0.5 → 0.1)
Phase 3: Remove reference, keep task-only + motion prior regularization
```

---

## 10. On Velocity Weight (0.1) in Pre-training

### The Concern
> "Won't reducing velocity weight to 0.1 hurt learning?"

### Analysis

**Why velocity is harder:**
- Velocity = derivative of position → small position changes = large velocity changes
- Higher variance in velocity targets → dominates loss if equal weight
- Model may sacrifice position accuracy to reduce velocity loss

**Why 0.1 weight is reasonable:**
- Position accuracy matters more for pose quality
- Velocity can be derived from position sequence if needed
- Prevents velocity noise from dominating learning

**Alternative approach if concerned:**
```python
# Adaptive weighting based on variance
pos_var = target[:, :56].var()
vel_var = target[:, 56:].var()
vel_weight = pos_var / vel_var  # Auto-balance by variance
```

### Recommendation
Start with `pos_weight=1.0, vel_weight=0.1`. If rollout predictions drift (position OK but character glides), increase velocity weight to 0.3-0.5.

**Monitor:** Root velocity in trajectory plots. If prediction lags significantly behind GT, velocity weight is too low.

---

## 11. Quick Reference

### Pre-training Command
```bash
pixi run python python/models/train_motion.py \
    --bvh_dir data/cmu \
    --activity_split \
    --test_ratio 0.15 \
    --mode transformer_reg \
    --epochs 100
```

### Evaluation Command
```bash
pixi run python python/models/eval_motion.py \
    --model nn/motion_model_best.pt \
    --split_file nn/split_info.json \
    --use_test_set \
    --num_frames 500
```

### Key Files
| File | Purpose |
|------|---------|
| `train_motion.py` | Pre-training script |
| `eval_motion.py` | Evaluation with rollout analysis |
| `Model.py` | SimulationNN, MuscleNN for RL |
| `PPO.py` | RL training loop |
