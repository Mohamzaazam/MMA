#!/usr/bin/env python3
"""
Step 2: Test Single Environment Inference
==========================================
This script tests the single-motion environment (pymss) with model inference.
This is the standard MASS pipeline.

Run from project root:
    pixi shell
    python python/test_02_single_env.py
    
    # Or with a specific metadata file:
    python python/test_02_single_env.py data/metadata.txt
"""

import os
import sys
import time
import argparse
import numpy as np

# Ensure build directory is in path for pymss
sys.path.insert(0, 'build')
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_pymss_import():
    """Test 1: Can we import pymss?"""
    print("=" * 60)
    print("TEST 1: Import pymss")
    print("=" * 60)
    
    try:
        import pymss
        print("✅ Successfully imported pymss")
        return pymss
    except Exception as e:
        print(f"❌ Failed to import pymss: {e}")
        print("\n   Make sure you've built the project:")
        print("   pixi run build")
        import traceback
        traceback.print_exc()
        return None


def test_environment_creation(pymss, metadata_path, num_slaves=4):
    """Test 2: Can we create an environment?"""
    print("\n" + "=" * 60)
    print("TEST 2: Create Environment")
    print("=" * 60)
    
    if not os.path.exists(metadata_path):
        print(f"❌ Metadata file not found: {metadata_path}")
        return None
    
    print(f"   Metadata: {metadata_path}")
    print(f"   Slaves: {num_slaves}")
    
    try:
        env = pymss.pymss(metadata_path, num_slaves)
        print("✅ Environment created successfully")
        
        # Print environment info
        print(f"\n   Environment Info:")
        print(f"   - Num State: {env.GetNumState()}")
        print(f"   - Num Action: {env.GetNumAction()}")
        print(f"   - Simulation Hz: {env.GetSimulationHz()}")
        print(f"   - Control Hz: {env.GetControlHz()}")
        print(f"   - Num Steps per Control: {env.GetNumSteps()}")
        print(f"   - Use Muscle: {env.UseMuscle()}")
        
        if env.UseMuscle():
            print(f"   - Num Muscles: {env.GetNumMuscles()}")
            print(f"   - Num Total Muscle Related DOFs: {env.GetNumTotalMuscleRelatedDofs()}")
        
        return env
    except Exception as e:
        print(f"❌ Failed to create environment: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_environment_reset(env):
    """Test 3: Can we reset the environment?"""
    print("\n" + "=" * 60)
    print("TEST 3: Reset Environment")
    print("=" * 60)
    
    try:
        env.Resets(True)  # RSI = Random State Initialization
        print("✅ Environment reset successful (RSI=True)")
        
        env.Resets(False)
        print("✅ Environment reset successful (RSI=False)")
        
        return True
    except Exception as e:
        print(f"❌ Failed to reset environment: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_get_states(env, num_slaves):
    """Test 4: Can we get states from the environment?"""
    print("\n" + "=" * 60)
    print("TEST 4: Get States")
    print("=" * 60)
    
    try:
        states = env.GetStates()
        print(f"✅ Got states: shape={states.shape}")
        
        # Check for NaN
        if np.any(np.isnan(states)):
            print("⚠️  Warning: NaN values in states!")
        else:
            print("   - No NaN values in states")
        
        # Print stats
        print(f"   - State range: [{states.min():.4f}, {states.max():.4f}]")
        print(f"   - State mean: {states.mean():.4f}")
        
        assert states.shape[0] == num_slaves, f"Expected {num_slaves} states, got {states.shape[0]}"
        
        return states
    except Exception as e:
        print(f"❌ Failed to get states: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_step_simulation(env, num_slaves):
    """Test 5: Can we step the simulation?"""
    print("\n" + "=" * 60)
    print("TEST 5: Step Simulation")
    print("=" * 60)
    
    try:
        # Set random actions
        num_action = env.GetNumAction()
        actions = np.random.randn(num_slaves, num_action).astype(np.float64) * 0.1
        env.SetActions(actions)
        print(f"✅ Set actions: shape={actions.shape}")
        
        # Step the environment
        env.StepsAtOnce()
        print(f"✅ StepsAtOnce() successful")
        
        # Get new states
        new_states = env.GetStates()
        print(f"✅ Got new states after step: shape={new_states.shape}")
        
        # Check for NaN
        if np.any(np.isnan(new_states)):
            print("⚠️  Warning: NaN values in states after step!")
        
        # Get rewards
        rewards = env.GetRewards()
        print(f"✅ Got rewards: shape={rewards.shape}")
        print(f"   - Reward range: [{rewards.min():.4f}, {rewards.max():.4f}]")
        
        # Check episode ends
        eoe = env.IsEndOfEpisodes()
        print(f"✅ Got end-of-episode flags: {eoe}")
        
        return True
    except Exception as e:
        print(f"❌ Failed to step simulation: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_muscle_interface(env, num_slaves):
    """Test 6: Can we use the muscle interface?"""
    print("\n" + "=" * 60)
    print("TEST 6: Muscle Interface")
    print("=" * 60)
    
    if not env.UseMuscle():
        print("⚠️  Muscle not enabled, skipping test")
        return True
    
    try:
        # Get muscle torques
        muscle_torques = env.GetMuscleTorques()
        print(f"✅ Got muscle torques: shape={muscle_torques.shape}")
        
        # Get desired torques
        desired_torques = env.GetDesiredTorques()
        print(f"✅ Got desired torques: shape={desired_torques.shape}")
        
        # Set activation levels
        num_muscles = env.GetNumMuscles()
        activations = np.random.rand(num_slaves, num_muscles).astype(np.float64)
        env.SetActivationLevels(activations)
        print(f"✅ Set activation levels: shape={activations.shape}")
        
        # Compute muscle tuples (for training)
        env.ComputeMuscleTuples()
        print("✅ ComputeMuscleTuples() successful")
        
        JtA = env.GetMuscleTuplesJtA()
        TauDes = env.GetMuscleTuplesTauDes()
        L = env.GetMuscleTuplesL()
        b = env.GetMuscleTuplesb()
        
        print(f"   - JtA shape: {JtA.shape}")
        print(f"   - TauDes shape: {TauDes.shape}")
        print(f"   - L shape: {L.shape}")
        print(f"   - b shape: {b.shape}")
        
        return True
    except Exception as e:
        print(f"❌ Muscle interface failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_episode_rollout(env, num_slaves, max_steps=100):
    """Test 7: Can we run a complete episode?"""
    print("\n" + "=" * 60)
    print(f"TEST 7: Episode Rollout (max {max_steps} steps)")
    print("=" * 60)
    
    try:
        env.Resets(True)
        
        total_rewards = np.zeros(num_slaves)
        steps = 0
        all_done = False
        
        start_time = time.time()
        
        while steps < max_steps and not all_done:
            # Get states
            states = env.GetStates()
            
            # Check for NaN
            if np.any(np.isnan(states)):
                print(f"⚠️  NaN detected at step {steps}")
                break
            
            # Random actions
            actions = np.random.randn(num_slaves, env.GetNumAction()).astype(np.float64) * 0.1
            env.SetActions(actions)
            
            # Step
            env.StepsAtOnce()
            
            # Get rewards
            rewards = env.GetRewards()
            total_rewards += rewards
            
            # Check episode ends
            eoe = env.IsEndOfEpisodes()
            
            # Reset terminated episodes
            for j in range(num_slaves):
                if eoe[j] > 0.5:
                    env.Reset(True, j)
            
            steps += 1
            
            if steps % 20 == 0:
                print(f"   Step {steps}: avg_reward={np.mean(total_rewards)/steps:.4f}")
        
        elapsed = time.time() - start_time
        
        print(f"\n✅ Episode rollout completed")
        print(f"   - Steps: {steps}")
        print(f"   - Time: {elapsed:.2f}s ({steps/elapsed:.1f} steps/sec)")
        print(f"   - Total rewards: {total_rewards}")
        print(f"   - Avg reward per step: {np.mean(total_rewards)/steps:.4f}")
        
        return True
    except Exception as e:
        print(f"❌ Episode rollout failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_with_trained_model(env, num_slaves):
    """Test 8: Run inference with a trained model (if available)"""
    print("\n" + "=" * 60)
    print("TEST 8: Inference with Trained Model")
    print("=" * 60)
    
    # Check for trained models
    model_paths = [
        ('nn/current.pt', 'nn/current_muscle.pt'),
        ('nn/max.pt', 'nn/max_muscle.pt'),
    ]
    
    sim_path, muscle_path = None, None
    for sp, mp in model_paths:
        if os.path.exists(sp) and os.path.exists(mp):
            sim_path, muscle_path = sp, mp
            break
    
    if sim_path is None:
        print("⚠️  No trained model found, skipping inference test")
        print("   Train with: pixi run train")
        return True
    
    print(f"   Using model: {sim_path}")
    
    try:
        from Model import SimulationNN, MuscleNN, Tensor
        
        # Get dimensions from environment
        num_state = env.GetNumState()
        num_action = env.GetNumAction()
        num_muscles = env.GetNumMuscles()
        num_dofs = env.GetNumTotalMuscleRelatedDofs()
        
        # Create and load models
        sim_nn = SimulationNN(num_state, num_action)
        muscle_nn = MuscleNN(num_dofs, num_action, num_muscles)
        
        sim_nn.load(sim_path)
        muscle_nn.load(muscle_path)
        
        print("✅ Loaded pre-trained models")
        
        # Run a few steps with the trained model
        env.Resets(True)
        
        for step in range(20):
            states = env.GetStates()
            
            # Get actions from policy
            with torch.no_grad():
                state_tensor = Tensor(states.astype(np.float32))
                action_dist, _ = sim_nn(state_tensor)
                actions = action_dist.loc.cpu().numpy()  # Use mean action
            
            env.SetActions(actions)
            
            # Get muscle activations if using muscle
            if env.UseMuscle():
                mt = env.GetMuscleTorques()
                dt = env.GetDesiredTorques()
                with torch.no_grad():
                    activations = muscle_nn(
                        Tensor(mt.astype(np.float32)),
                        Tensor(dt.astype(np.float32))
                    ).cpu().numpy()
                env.SetActivationLevels(activations)
            
            env.StepsAtOnce()
            
            rewards = env.GetRewards()
            if step % 5 == 0:
                print(f"   Step {step}: avg_reward={np.mean(rewards):.4f}")
            
            # Reset terminated episodes
            for j in range(num_slaves):
                if env.IsEndOfEpisode(j):
                    env.Reset(True, j)
        
        print("✅ Trained model inference successful")
        return True
        
    except Exception as e:
        print(f"❌ Trained model inference failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description='Test Single Environment')
    parser.add_argument('metadata', nargs='?', default='data/metadata.txt',
                        help='Path to metadata file')
    parser.add_argument('--slaves', type=int, default=4,
                        help='Number of parallel environments')
    args = parser.parse_args()
    
    print("\n" + "=" * 60)
    print("MASS SINGLE ENVIRONMENT TEST (Step 2)")
    print("=" * 60)
    print(f"Metadata: {args.metadata}")
    print(f"Slaves: {args.slaves}")
    print()
    
    # Test 1: Import
    pymss = test_pymss_import()
    if pymss is None:
        return 1
    
    # Test 2: Create environment
    env = test_environment_creation(pymss, args.metadata, args.slaves)
    if env is None:
        return 1
    
    # Test 3: Reset
    if not test_environment_reset(env):
        return 1
    
    # Test 4: Get states
    states = test_get_states(env, args.slaves)
    if states is None:
        return 1
    
    # Test 5: Step
    if not test_step_simulation(env, args.slaves):
        return 1
    
    # Test 6: Muscle interface
    if not test_muscle_interface(env, args.slaves):
        return 1
    
    # Test 7: Episode rollout
    if not test_episode_rollout(env, args.slaves):
        return 1
    
    # Test 8: Trained model (optional)
    import torch  # Import here to avoid issues if not needed
    test_with_trained_model(env, args.slaves)
    
    print("\n" + "=" * 60)
    print("✅ ALL SINGLE ENVIRONMENT TESTS PASSED!")
    print("=" * 60)
    print("\nNext step: Run test_03_multimodal_env.py")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
