# MASS Testing vs Training Pipeline Analysis

## Quick Summary

**Key Finding**: The current architecture is **intentionally designed** to use BVH reference motion during both training AND testing. This is NOT a bug - it's the DeepMimic-style motion imitation approach.

## Architecture Overview

```
┌──────────────────────────────────────────────────────────────────────────┐
│                        MASS Control Pipeline                              │
│                  (Same for TRAINING and TESTING)                          │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                           │
│   ┌─────────────┐                                                         │
│   │   State     │ = [body_positions, body_velocities, PHASE(φ)]          │
│   │ (from env)  │                                                         │
│   └──────┬──────┘                                                         │
│          │                                                                │
│          ▼                                                                │
│   ┌─────────────┐                                                         │
│   │  Policy NN  │  Output: action (RESIDUAL correction)                  │
│   │(SimulationNN)│  NOT absolute joint targets!                          │
│   └──────┬──────┘                                                         │
│          │ action                                                         │
│          ▼                                                                │
│   ┌─────────────────────────────────────────────────────────┐            │
│   │               Environment.SetAction()                    │            │
│   │                                                          │            │
│   │   1. target_pos = BVH.GetMotion(current_time)  ◄─────────┤ "Ground   │
│   │   2. final_target = target_pos + (action * 0.1)          │  Truth"   │
│   │                                                          │            │
│   └──────┬──────────────────────────────────────────────────┘            │
│          │ final_target                                                   │
│          ▼                                                                │
│   ┌─────────────┐                                                         │
│   │SPD Controller│  Computes desired torques to track final_target       │
│   └──────┬──────┘                                                         │
│          │ desired_torques                                                │
│          ▼                                                                │
│   ┌─────────────┐                                                         │
│   │  Muscle NN  │  Output: muscle_activations                            │
│   │ (MuscleNN)  │                                                         │
│   └──────┬──────┘                                                         │
│          │ activations                                                    │
│          ▼                                                                │
│   ┌─────────────┐                                                         │
│   │  Simulation │  Physics simulation with muscle forces                 │
│   │   (DART)    │                                                         │
│   └─────────────┘                                                         │
│                                                                           │
└──────────────────────────────────────────────────────────────────────────┘
```

## Where is "Ground Truth" Used?

### 1. Target Position Generation (`Environment.cpp::SetAction`)
```cpp
void Environment::SetAction(const Eigen::VectorXd& a)
{
    mAction = a*0.1;  // Scale action (residual)
    
    double t = mWorld->getTime();
    
    // Get target from BVH reference motion
    auto pv = mCharacter->GetTargetPosAndVel(t, 1.0/mControlHz);
    mTargetPositions = pv.first;   // ← BVH "ground truth"
    mTargetVelocities = pv.second;
    // ...
}
```

### 2. SPD Controller (`Environment.cpp::GetDesiredTorques`)
```cpp
Eigen::VectorXd Environment::GetDesiredTorques()
{
    Eigen::VectorXd p_des = mTargetPositions;
    // Add the policy's residual correction
    p_des.tail(mTargetPositions.rows()-mRootJointDof) += mAction;
    
    // Compute torques to track the combined target
    mDesiredTorque = mCharacter->GetSPDForces(p_des);
    return mDesiredTorque.tail(...);
}
```

### 3. Reward Computation (`Environment.cpp::GetReward`)
```cpp
double Environment::GetReward()
{
    // Compare current pose to BVH target
    Eigen::VectorXd p_diff = skel->getPositionDifferences(mTargetPositions, cur_pos);
    // ...
    double r_q = exp_of_squared(p_diff, 2.0);  // Pose reward
    // ...
}
```

## Why This Design?

This is the **DeepMimic** approach to motion imitation:

1. **Learning Residuals**: The policy learns WHEN and HOW MUCH to deviate from the reference, not the entire motion from scratch.

2. **Sample Efficiency**: Much easier to learn small corrections than entire locomotion behaviors.

3. **Motion Quality**: Guarantees motion stays close to natural human motion.

4. **Phase Tracking**: The state includes phase (φ), telling the network where in the motion cycle we are.

## Training vs Testing: Key Differences

| Aspect | Training | Testing |
|--------|----------|---------|
| Action selection | Sample from distribution | Use mean (deterministic) |
| Random initialization | Yes (RSI) | Yes (RSI) |
| BVH reference | Used ✓ | Used ✓ |
| Reward computation | Yes | Optional (for metrics) |
| Noise (exploration) | Yes | No |

## If You Want to Remove BVH Reference

If you want a **true absolute policy** (not residual), you would need to:

1. **Modify `Environment.cpp`**:
   - Change `GetDesiredTorques()` to NOT add BVH targets
   - Use action directly as target positions

2. **Modify state representation**:
   - Remove phase (φ) from state
   - Possibly add other task-relevant information

3. **Retrain from scratch**:
   - The network architecture would be the same
   - But learned behavior would be completely different

**Warning**: This is a significant architectural change that would require retraining and likely much longer training time.

## Files to Understand

1. **`render/Window.cpp`** - C++ testing (visualization)
   - `Step()` - Main control loop
   - `GetActionFromNN()` - Policy inference
   - `GetActivationFromNN()` - Muscle inference

2. **`python/main.py`** - Training script
   - `GenerateTransitions()` - Training data collection
   - Same pipeline as testing!

3. **`core/Environment.cpp`** - Core simulation
   - `SetAction()` - Where BVH targets are fetched
   - `GetDesiredTorques()` - Where residual is added
   - `GetReward()` - Where reward is computed

4. **`python/Model.py`** - Neural networks
   - `SimulationNN` - Policy network
   - `MuscleNN` - Muscle activation network