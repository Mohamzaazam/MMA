#!/usr/bin/env python3
"""
Step 1: Test Model Inference (No Environment)
==============================================
This script tests if pre-trained models can be loaded and run inference.
No environment needed - just testing the neural network part.

Run from project root:
    pixi shell
    python python/test_01_model_inference.py
"""

import os
import sys
import torch
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_model_loading():
    """Test 1: Can we import and create the model classes?"""
    print("=" * 60)
    print("TEST 1: Model Class Import")
    print("=" * 60)
    
    try:
        from Model import SimulationNN, MuscleNN, Tensor
        print("✅ Successfully imported SimulationNN, MuscleNN, Tensor")
        return SimulationNN, MuscleNN, Tensor
    except Exception as e:
        print(f"❌ Failed to import models: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None


def test_model_creation(SimulationNN, MuscleNN):
    """Test 2: Can we create model instances with typical dimensions?"""
    print("\n" + "=" * 60)
    print("TEST 2: Model Creation")
    print("=" * 60)
    
    # Typical dimensions from the MASS environment
    num_state = 100  # Approximate state dimension
    num_action = 57  # Approximate action dimension (DOFs - root)
    num_muscles = 326  # Number of muscles
    num_total_muscle_related_dofs = 500  # Approximate
    
    try:
        sim_nn = SimulationNN(num_state, num_action)
        print(f"✅ Created SimulationNN(num_state={num_state}, num_action={num_action})")
        
        muscle_nn = MuscleNN(num_total_muscle_related_dofs, num_action, num_muscles)
        print(f"✅ Created MuscleNN(dofs={num_total_muscle_related_dofs}, actions={num_action}, muscles={num_muscles})")
        
        return sim_nn, muscle_nn, num_state, num_action, num_muscles, num_total_muscle_related_dofs
    except Exception as e:
        print(f"❌ Failed to create models: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None, None, None, None


def test_forward_pass(sim_nn, muscle_nn, Tensor, num_state, num_action, num_muscles, num_dofs):
    """Test 3: Can we run forward passes?"""
    print("\n" + "=" * 60)
    print("TEST 3: Forward Pass")
    print("=" * 60)
    
    # Test SimulationNN forward pass
    try:
        # Create dummy state
        dummy_state = np.random.randn(1, num_state).astype(np.float32)
        state_tensor = Tensor(dummy_state)
        
        # Forward pass
        action_dist, value = sim_nn(state_tensor)
        
        print(f"✅ SimulationNN forward pass successful")
        print(f"   - Action mean shape: {action_dist.loc.shape}")
        print(f"   - Action std shape: {action_dist.scale.shape}")
        print(f"   - Value shape: {value.shape}")
        
        # Sample an action
        action = action_dist.sample()
        print(f"   - Sampled action shape: {action.shape}")
        
    except Exception as e:
        print(f"❌ SimulationNN forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test MuscleNN forward pass
    try:
        # Create dummy inputs
        dummy_muscle_tau = np.random.randn(1, num_dofs).astype(np.float32)
        dummy_tau_des = np.random.randn(1, num_action).astype(np.float32)
        
        mt_tensor = Tensor(dummy_muscle_tau)
        td_tensor = Tensor(dummy_tau_des)
        
        # Forward pass
        activations = muscle_nn(mt_tensor, td_tensor)
        
        print(f"✅ MuscleNN forward pass successful")
        print(f"   - Activations shape: {activations.shape}")
        print(f"   - Activations range: [{activations.min().item():.4f}, {activations.max().item():.4f}]")
        
    except Exception as e:
        print(f"❌ MuscleNN forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


def test_load_pretrained():
    """Test 4: Can we find and validate pre-trained weights?"""
    print("\n" + "=" * 60)
    print("TEST 4: Check Pre-trained Weights")
    print("=" * 60)
    
    # Check for model files
    model_paths = [
        ('nn/current.pt', 'nn/current_muscle.pt', 'Single-motion (current)'),
        ('nn/max.pt', 'nn/max_muscle.pt', 'Single-motion (best)'),
        ('nn/multimodal_current.pt', 'nn/multimodal_current_muscle.pt', 'Multi-modal (current)'),
        ('nn/multimodal_max.pt', 'nn/multimodal_max_muscle.pt', 'Multi-modal (best)'),
    ]
    
    found_models = []
    for sim_path, muscle_path, name in model_paths:
        if os.path.exists(sim_path) and os.path.exists(muscle_path):
            print(f"\n✅ Found: {name}")
            print(f"   - SimNN: {sim_path}")
            print(f"   - MuscleNN: {muscle_path}")
            found_models.append((sim_path, muscle_path, name))
            
            try:
                # Check if files are valid PyTorch models
                sim_state = torch.load(sim_path, map_location='cpu')
                muscle_state = torch.load(muscle_path, map_location='cpu')
                
                # Extract dimensions from saved state
                sim_keys = list(sim_state.keys())
                muscle_keys = list(muscle_state.keys())
                
                # Try to infer dimensions
                if 'p_fc1.weight' in sim_state:
                    num_state = sim_state['p_fc1.weight'].shape[1]
                    print(f"   - Inferred num_state: {num_state}")
                if 'p_fc3.weight' in sim_state:
                    num_action = sim_state['p_fc3.weight'].shape[0]
                    print(f"   - Inferred num_action: {num_action}")
                    
            except Exception as e:
                print(f"   ⚠️  Could not inspect model: {e}")
    
    if not found_models:
        print("\n⚠️  No pre-trained models found in nn/")
        print("   This is expected if you haven't trained yet.")
        print("   Train with: pixi run train")
        print("   Or: pixi run train_multimodal")
    
    return found_models


def test_batch_inference(sim_nn, Tensor, num_state, num_action):
    """Test 5: Can we run batch inference (like in training)?"""
    print("\n" + "=" * 60)
    print("TEST 5: Batch Inference")
    print("=" * 60)
    
    batch_sizes = [1, 4, 16, 64]
    
    for batch_size in batch_sizes:
        try:
            # Create batch of states
            dummy_states = np.random.randn(batch_size, num_state).astype(np.float32)
            states_tensor = Tensor(dummy_states)
            
            # Forward pass
            action_dist, values = sim_nn(states_tensor)
            
            # Sample actions
            actions = action_dist.sample()
            
            assert actions.shape == (batch_size, num_action), f"Wrong action shape"
            assert values.shape == (batch_size, 1), f"Wrong value shape"
            
            print(f"✅ Batch size {batch_size}: actions={actions.shape}, values={values.shape}")
            
        except Exception as e:
            print(f"❌ Batch size {batch_size} failed: {e}")
            return False
    
    return True


def main():
    print("\n" + "=" * 60)
    print("MASS MODEL INFERENCE TEST (Step 1)")
    print("=" * 60)
    print("Testing model loading and inference without environment")
    print()
    
    # Test 1: Import
    SimulationNN, MuscleNN, Tensor = test_model_loading()
    if SimulationNN is None:
        print("\n❌ FAILED: Cannot proceed without model imports")
        return 1
    
    # Test 2: Creation
    result = test_model_creation(SimulationNN, MuscleNN)
    sim_nn, muscle_nn, num_state, num_action, num_muscles, num_dofs = result
    if sim_nn is None:
        print("\n❌ FAILED: Cannot proceed without model instances")
        return 1
    
    # Test 3: Forward pass
    if not test_forward_pass(sim_nn, muscle_nn, Tensor, num_state, num_action, num_muscles, num_dofs):
        print("\n❌ FAILED: Forward pass failed")
        return 1
    
    # Test 4: Check pretrained (info only, doesn't fail)
    found_models = test_load_pretrained()
    
    # Test 5: Batch inference
    if not test_batch_inference(sim_nn, Tensor, num_state, num_action):
        print("\n❌ FAILED: Batch inference failed")
        return 1
    
    print("\n" + "=" * 60)
    print("✅ ALL BASIC MODEL TESTS PASSED!")
    print("=" * 60)
    
    if found_models:
        print(f"\nFound {len(found_models)} trained model(s).")
        print("Next step: Run test_02_single_env.py")
    else:
        print("\nNo trained models found. Train first with:")
        print("  pixi run train                # Single motion")
        print("  pixi run train_multimodal     # Multi-modal")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
