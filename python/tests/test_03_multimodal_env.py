#!/usr/bin/env python3
"""
Step 3: Test Multi-Modal Environment
=====================================
This script tests the MultimodalEnvManager with multiple motions.

Run from project root:
    pixi shell
    python python/test_03_multimodal_env.py
    
    # Or with specific files:
    python python/test_03_multimodal_env.py --motion_list data/motion_list.txt --template data/metadata.txt
"""

import os
import sys
import time
import argparse
import numpy as np

# Ensure build directory is in path for pymss
sys.path.insert(0, 'build')
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_multimodal_imports():
    """Test 1: Can we import multimodal components?"""
    print("=" * 60)
    print("TEST 1: Import Multimodal Components")
    print("=" * 60)
    
    try:
        from multimodal_env import (
            MultimodalEnvManager, 
            MotionConfig, 
            load_motion_configs_from_list
        )
        print("✅ Successfully imported MultimodalEnvManager")
        print("✅ Successfully imported MotionConfig")
        print("✅ Successfully imported load_motion_configs_from_list")
        return MultimodalEnvManager, MotionConfig, load_motion_configs_from_list
    except Exception as e:
        print(f"❌ Failed to import: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None


def test_load_motion_configs(load_motion_configs_from_list, motion_list_path, template_path):
    """Test 2: Can we load motion configurations?"""
    print("\n" + "=" * 60)
    print("TEST 2: Load Motion Configurations")
    print("=" * 60)
    
    if not os.path.exists(motion_list_path):
        print(f"❌ Motion list not found: {motion_list_path}")
        print("   Create it with: python python/tmps/scan_motions.py data/motion")
        return None
    
    if not os.path.exists(template_path):
        print(f"❌ Template not found: {template_path}")
        return None
    
    print(f"   Motion list: {motion_list_path}")
    print(f"   Template: {template_path}")
    
    try:
        configs = load_motion_configs_from_list(motion_list_path, template_path)
        
        print(f"\n✅ Loaded {len(configs)} motion configurations:")
        for i, config in enumerate(configs):
            print(f"   [{i}] {config.name}")
            print(f"       - metadata: {config.metadata_path}")
            print(f"       - cyclic: {config.cyclic}")
        
        return configs
    except Exception as e:
        print(f"❌ Failed to load configs: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_multimodal_env_creation(MultimodalEnvManager, configs, num_slaves_per_motion=2):
    """Test 3: Can we create a MultimodalEnvManager?"""
    print("\n" + "=" * 60)
    print("TEST 3: Create MultimodalEnvManager")
    print("=" * 60)
    
    try:
        import pymss
        
        print(f"   Slaves per motion: {num_slaves_per_motion}")
        print(f"   Total motions: {len(configs)}")
        print(f"   Expected total slaves: {num_slaves_per_motion * len(configs)}")
        
        env = MultimodalEnvManager(
            motion_configs=configs,
            num_slaves_per_motion=num_slaves_per_motion,
            pymss_module=pymss
        )
        
        print("\n   Initializing environments...")
        env.initialize()
        
        print(f"\n✅ MultimodalEnvManager created successfully")
        print(f"   - Total slaves: {env.total_slaves}")
        print(f"   - Num motions: {env.num_motions}")
        print(f"   - Num state: {env.GetNumState()}")
        print(f"   - Num action: {env.GetNumAction()}")
        print(f"   - Use muscle: {env.UseMuscle()}")
        
        if env.UseMuscle():
            print(f"   - Num muscles: {env.GetNumMuscles()}")
        
        return env
    except Exception as e:
        print(f"❌ Failed to create MultimodalEnvManager: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_multimodal_reset(env):
    """Test 4: Can we reset the multimodal environment?"""
    print("\n" + "=" * 60)
    print("TEST 4: Reset Multimodal Environment")
    print("=" * 60)
    
    try:
        env.Resets(True)
        print("✅ Reset successful (RSI=True)")
        return True
    except Exception as e:
        print(f"❌ Reset failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_multimodal_get_states(env):
    """Test 5: Can we get states from all slaves?"""
    print("\n" + "=" * 60)
    print("TEST 5: Get States (All Slaves)")
    print("=" * 60)
    
    try:
        states = env.GetStates()
        print(f"✅ Got states: shape={states.shape}")
        
        # Check expected shape
        expected_shape = (env.total_slaves, env.GetNumState())
        assert states.shape == expected_shape, f"Expected {expected_shape}, got {states.shape}"
        
        # Check for NaN
        if np.any(np.isnan(states)):
            print("⚠️  Warning: NaN values in states!")
        else:
            print("   - No NaN values")
        
        print(f"   - Range: [{states.min():.4f}, {states.max():.4f}]")
        
        return states
    except Exception as e:
        print(f"❌ Failed to get states: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_multimodal_per_slave_access(env, configs):
    """Test 6: Can we access individual slaves and their motions?"""
    print("\n" + "=" * 60)
    print("TEST 6: Per-Slave Access")
    print("=" * 60)
    
    try:
        print("   Slave -> Motion mapping:")
        for slave_id in range(env.total_slaves):
            motion_name = env.get_motion_for_slave(slave_id)
            reward = env.GetReward(slave_id)
            is_done = env.IsEndOfEpisode(slave_id)
            print(f"   - Slave {slave_id}: motion='{motion_name}', reward={reward:.4f}, done={is_done}")
        
        print("\n✅ Per-slave access successful")
        return True
    except Exception as e:
        print(f"❌ Per-slave access failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_multimodal_step(env):
    """Test 7: Can we step the multimodal environment?"""
    print("\n" + "=" * 60)
    print("TEST 7: Step Multimodal Environment")
    print("=" * 60)
    
    try:
        # Set random actions for all slaves
        actions = np.random.randn(env.total_slaves, env.GetNumAction()).astype(np.float64) * 0.1
        env.SetActions(actions)
        print(f"✅ Set actions: shape={actions.shape}")
        
        # Step
        env.StepsAtOnce()
        print("✅ StepsAtOnce() successful")
        
        # Get new states
        new_states = env.GetStates()
        print(f"✅ Got new states: shape={new_states.shape}")
        
        # Get rewards
        rewards = env.GetRewards()
        print(f"✅ Got rewards: shape={rewards.shape}")
        print(f"   - Reward range: [{rewards.min():.4f}, {rewards.max():.4f}]")
        
        return True
    except Exception as e:
        print(f"❌ Step failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_multimodal_muscle_interface(env):
    """Test 8: Can we use muscle interface with multimodal env?"""
    print("\n" + "=" * 60)
    print("TEST 8: Multimodal Muscle Interface")
    print("=" * 60)
    
    if not env.UseMuscle():
        print("⚠️  Muscle not enabled, skipping")
        return True
    
    try:
        # Get muscle torques
        mt = env.GetMuscleTorques()
        print(f"✅ Got muscle torques: shape={mt.shape}")
        
        # Get desired torques
        dt = env.GetDesiredTorques()
        print(f"✅ Got desired torques: shape={dt.shape}")
        
        # Set activation levels
        num_muscles = env.GetNumMuscles()
        activations = np.random.rand(env.total_slaves, num_muscles).astype(np.float64)
        env.SetActivationLevels(activations)
        print(f"✅ Set activations: shape={activations.shape}")
        
        # Compute tuples
        env.ComputeMuscleTuples()
        JtA = env.GetMuscleTuplesJtA()
        print(f"✅ Computed muscle tuples: JtA shape={JtA.shape}")
        
        return True
    except Exception as e:
        print(f"❌ Muscle interface failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_multimodal_episode_rollout(env, configs, max_steps=50):
    """Test 9: Can we run episodes across all motions?"""
    print("\n" + "=" * 60)
    print(f"TEST 9: Multimodal Episode Rollout ({max_steps} steps)")
    print("=" * 60)
    
    try:
        env.Resets(True)
        
        # Track per-motion rewards
        motion_rewards = {config.name: [] for config in configs}
        motion_steps = {config.name: 0 for config in configs}
        
        start_time = time.time()
        
        for step in range(max_steps):
            # Get states
            states = env.GetStates()
            
            if np.any(np.isnan(states)):
                print(f"⚠️  NaN detected at step {step}")
                break
            
            # Random actions
            actions = np.random.randn(env.total_slaves, env.GetNumAction()).astype(np.float64) * 0.1
            env.SetActions(actions)
            
            # Step
            env.StepsAtOnce()
            
            # Get rewards and track per-motion
            rewards = env.GetRewards()
            for slave_id in range(env.total_slaves):
                motion_name = env.get_motion_for_slave(slave_id)
                motion_rewards[motion_name].append(rewards[slave_id])
                motion_steps[motion_name] += 1
            
            # Reset terminated episodes
            for slave_id in range(env.total_slaves):
                if env.IsEndOfEpisode(slave_id):
                    env.Reset(True, slave_id)
            
            if step % 10 == 0:
                print(f"   Step {step}: avg_reward={np.mean(rewards):.4f}")
        
        elapsed = time.time() - start_time
        
        print(f"\n✅ Rollout completed ({elapsed:.2f}s, {max_steps/elapsed:.1f} steps/sec)")
        print("\n   Per-motion statistics:")
        for motion_name in motion_rewards:
            rewards = motion_rewards[motion_name]
            if rewards:
                print(f"   - {motion_name}: steps={motion_steps[motion_name]}, "
                      f"avg_reward={np.mean(rewards):.4f}")
        
        return True
    except Exception as e:
        print(f"❌ Rollout failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_multimodal_trained_model(env, configs):
    """Test 10: Run inference with trained multimodal model (if available)"""
    print("\n" + "=" * 60)
    print("TEST 10: Trained Multimodal Model Inference")
    print("=" * 60)
    
    # Check for multimodal models
    model_paths = [
        ('nn/multimodal_current.pt', 'nn/multimodal_current_muscle.pt'),
        ('nn/multimodal_max.pt', 'nn/multimodal_max_muscle.pt'),
    ]
    
    sim_path, muscle_path = None, None
    for sp, mp in model_paths:
        if os.path.exists(sp) and os.path.exists(mp):
            sim_path, muscle_path = sp, mp
            break
    
    if sim_path is None:
        print("⚠️  No trained multimodal model found")
        print("   Train with: pixi run train_multimodal")
        return True
    
    print(f"   Using model: {sim_path}")
    
    try:
        import torch
        from Model import SimulationNN, MuscleNN, Tensor
        
        # Create and load models
        num_state = env.GetNumState()
        num_action = env.GetNumAction()
        num_muscles = env.GetNumMuscles()
        num_dofs = env.GetNumTotalMuscleRelatedDofs()
        
        sim_nn = SimulationNN(num_state, num_action)
        muscle_nn = MuscleNN(num_dofs, num_action, num_muscles)
        
        sim_nn.load(sim_path)
        muscle_nn.load(muscle_path)
        print("✅ Loaded multimodal models")
        
        # Run inference
        env.Resets(True)
        
        motion_rewards = {config.name: [] for config in configs}
        
        for step in range(30):
            states = env.GetStates()
            
            # Policy inference
            with torch.no_grad():
                state_tensor = Tensor(states.astype(np.float32))
                action_dist, _ = sim_nn(state_tensor)
                actions = action_dist.loc.cpu().numpy()
            
            env.SetActions(actions)
            
            # Muscle inference
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
            
            # Track rewards
            rewards = env.GetRewards()
            for slave_id in range(env.total_slaves):
                motion_name = env.get_motion_for_slave(slave_id)
                motion_rewards[motion_name].append(rewards[slave_id])
            
            # Reset terminated
            for slave_id in range(env.total_slaves):
                if env.IsEndOfEpisode(slave_id):
                    env.Reset(True, slave_id)
            
            if step % 10 == 0:
                print(f"   Step {step}: avg_reward={np.mean(rewards):.4f}")
        
        print("\n✅ Trained model inference successful")
        print("\n   Per-motion performance:")
        for motion_name, rewards in motion_rewards.items():
            if rewards:
                print(f"   - {motion_name}: avg_reward={np.mean(rewards):.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Trained model inference failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description='Test Multimodal Environment')
    parser.add_argument('--motion_list', default='data/motion_list.txt',
                        help='Path to motion list file')
    parser.add_argument('--template', default='data/metadata.txt',
                        help='Path to template metadata file')
    parser.add_argument('--slaves', type=int, default=2,
                        help='Slaves per motion (default: 2)')
    args = parser.parse_args()
    
    print("\n" + "=" * 60)
    print("MASS MULTIMODAL ENVIRONMENT TEST (Step 3)")
    print("=" * 60)
    print(f"Motion list: {args.motion_list}")
    print(f"Template: {args.template}")
    print(f"Slaves per motion: {args.slaves}")
    print()
    
    # Test 1: Import
    components = test_multimodal_imports()
    MultimodalEnvManager, MotionConfig, load_motion_configs_from_list = components
    if MultimodalEnvManager is None:
        return 1
    
    # Test 2: Load configs
    configs = test_load_motion_configs(
        load_motion_configs_from_list,
        args.motion_list,
        args.template
    )
    if configs is None:
        return 1
    
    # Test 3: Create env
    env = test_multimodal_env_creation(MultimodalEnvManager, configs, args.slaves)
    if env is None:
        return 1
    
    # Test 4: Reset
    if not test_multimodal_reset(env):
        return 1
    
    # Test 5: Get states
    if test_multimodal_get_states(env) is None:
        return 1
    
    # Test 6: Per-slave access
    if not test_multimodal_per_slave_access(env, configs):
        return 1
    
    # Test 7: Step
    if not test_multimodal_step(env):
        return 1
    
    # Test 8: Muscle interface
    if not test_multimodal_muscle_interface(env):
        return 1
    
    # Test 9: Episode rollout
    if not test_multimodal_episode_rollout(env, configs):
        return 1
    
    # Test 10: Trained model (optional)
    import torch  # Import here for test 10
    test_multimodal_trained_model(env, configs)
    
    print("\n" + "=" * 60)
    print("✅ ALL MULTIMODAL ENVIRONMENT TESTS PASSED!")
    print("=" * 60)
    print("\nNext step: Run test_04_visualize.py to visualize trained models")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
