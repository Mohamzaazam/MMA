---
description: Coding Agent Instructions for Robotics, AI, Reinforcement Learning, Control Systems, Biomechanics Simulations project
---

## Developer Profile

**Background**: Robotics, AI, Reinforcement Learning, Control Systems, Biomechanics Simulations  
**Project**: MASS (Musculoskeletal Animation with Soft Skills) - DeepMimic-style motion imitation system  
**Current Phase**: Transitioning from BVH-dependent to autonomous MotionNN-based control

## Core Project Understanding

### Architecture Overview
This is a **residual learning system** for motion imitation:
- **Not**: Generate motions from scratch
- **Is**: Learn to track reference motions robustly with physics-based muscle simulation
- **Key Component**: Policy learns small corrections (residuals) to reference BVH motion
- **Pipeline**: State → Policy NN → Residual → BVH Target + Residual → SPD Controller → Muscle NN → Physics Sim

### Technology Stack
- **Simulation**: DART physics engine (C++)
- **Training**: PPO (Proximal Policy Optimization) in Python
- **Neural Networks**: PyTorch
- **Motion Data**: BVH (Biovision Hierarchy) files
- **Build System**: CMake
- **Language Mix**: C++ (core sim), Python (training/RL), mixed via pybind11

## Communication Preferences

### When Explaining Code
1. **Start with the "why"** before the "how"
2. Relate to RL/control theory concepts I know (policy gradients, value functions, PD controllers, etc.)
3. Use mathematical notation when clarifying algorithms (I'm comfortable with it)
4. Draw parallels to robotics concepts (forward/inverse kinematics, dynamics, etc.)

### When Suggesting Solutions
1. **Prioritize**: Correctness > Efficiency > Elegance
2. For RL components: Explain impact on training dynamics
3. For physics/simulation: Explain impact on stability and realism
4. Always mention potential edge cases in biomechanics (joint limits, muscle saturation, etc.)

### Code Style
- **Prefer explicit over implicit** for complex RL/physics code
- Use descriptive variable names: `target_joint_positions` not `tjp`
- Comment the "why" not the "what" (I can read code)
- For equations: Include reference to paper/algorithm name

## Project-Specific Context

### Key Files & Responsibilities
```
core/
├── Environment.cpp          # Main sim loop, reward computation, BVH integration
├── Character.cpp            # Musculoskeletal model, SPD controller
└── BVH.cpp                  # Motion data loading and sampling

python/
├── main.py                  # PPO training loop
├── Model.py                 # Policy NN (SimulationNN) and Muscle NN
└── PPO.py                   # PPO algorithm implementation

render/
└── Window.cpp               # Visualization and testing (no training)
```

### Current Architecture Constraints
1. **BVH Dependency**: System currently requires BVH reference at runtime (this is by design, not a bug)
2. **Phase Variable**: State includes phase φ ∈ [0,1] indicating position in motion cycle
3. **Residual Actions**: Policy outputs are scaled by 0.1 and added to BVH targets
4. **Muscle Redundancy**: Multiple muscle activation patterns can produce same motion

### Common Pitfalls to Avoid
- **Don't** assume this is standard locomotion RL (it's motion imitation)
- **Don't** remove BVH usage without understanding it's fundamental to architecture
- **Don't** ignore physics constraints (joint limits, muscle force limits)
- **Don't** forget that training and testing use the same pipeline (including BVH)

## Upcoming Work (Next 30 Days)

### Phase 0-1: MotionNN Foundation (Days 1-10)
**Goal**: Extract motion patterns from BVH, train network to predict future poses

**Key Tasks**:
1. BVH data extraction pipeline
   - Sample poses at regular intervals from BVH files
   - Create dataset: (phase, history) → future_pose
   - Handle multiple motion clips (walk, run, etc.)

2. MotionNN architecture
   - Input: phase φ, pose history (last N frames)
   - Output: predicted joint positions
   - Consider: LSTM, Transformer, or MLP with phase encoding

3. Training & validation
   - Loss: MSE on joint positions
   - Metric: Compare rollouts to ground truth BVH
   - Validate: Can it generate full motion cycles?

**Agent Guidance**:
- When implementing data pipeline: Ensure phase normalization [0,1] per cycle
- For network architecture: Start simple (MLP), iterate if needed
- Validation is critical: Show me rollout visualizations vs ground truth

### Phase 2: Integration (Days 11-20)
**Goal**: Replace BVH lookup with MotionNN predictions in main training loop

**Key Modifications**:
1. `Environment.cpp::SetAction()`
   - Currently: `mTargetPositions = GetTargetPosAndVel(t, ...)`
   - Change to: `mTargetPositions = MotionNN(phase, state)`
   - Keep residual addition: `p_des += mAction`

2. PPO training loop
   - Add MotionNN to inference pipeline
   - May need to retrain policy (behavior will differ from BVH baseline)

3. Comparison metrics
   - Track: reward, pose error, success rate vs BVH baseline
   - Expect: Initial performance drop, then recovery

**Agent Guidance**:
- This is a major architectural change - plan integration carefully
- Consider: Gradual transition (mix MotionNN and BVH during training)
- Watch for: Instabilities in early training (MotionNN errors compound)

### Phase 3: Terrain Generalization (Days 21-30)
**Goal**: Handle varied terrain without terrain-specific BVH data

**Implementation**:
1. Terrain observations
   - Add to state: heightmap, normal vectors, friction coefficients
   - Consider: Local grid around character vs global terrain representation

2. Terrain variations
   - Procedural generation: slopes, stairs, uneven ground
   - Curriculum: Start flat, gradually increase difficulty

3. Training strategy
   - Curriculum learning essential (flat → simple → complex terrain)
   - May need terrain-conditioned MotionNN or policy adaptation

**Agent Guidance**:
- Terrain adds significant complexity - iterate carefully
- For observations: More data ≠ better (information bottleneck matters)
- Consider: Does MotionNN need terrain info, or just policy?