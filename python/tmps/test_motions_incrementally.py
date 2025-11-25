#!/usr/bin/env python3
"""
Diagnostic script to test multimodal training incrementally.
This helps identify which motions cause NaN issues.

Usage:
    pixi shell
    python python/test_motions_incremental.py
"""

import os
import sys
import time
import numpy as np

# Add build directory to path for pymss
sys.path.insert(0, 'build')

def generate_metadata(motion_path, template_path, output_path, cyclic=True):
    """Generate metadata file for a specific motion."""
    with open(template_path, 'r') as f:
        content = f.read()
    
    new_content = []
    for line in content.split('\n'):
        if line.strip().startswith('bvh_file'):
            cyclic_str = 'true' if cyclic else 'false'
            new_content.append(f"bvh_file {motion_path} {cyclic_str}")
        else:
            new_content.append(line)
            
    with open(output_path, 'w') as f:
        f.write('\n'.join(new_content))
    
    return output_path


def test_single_motion(motion_name, motion_path, template_path, num_steps=100, num_slaves=2):
    """Test a single motion to see if it causes NaN errors."""
    import pymss
    
    print(f"\n{'='*60}")
    print(f"Testing motion: {motion_name}")
    print(f"  Path: {motion_path}")
    print(f"  Steps: {num_steps}, Slaves: {num_slaves}")
    print(f"{'='*60}")
    
    # Generate metadata
    meta_file = f"data/metadata_test_{motion_name}.txt"
    generate_metadata(motion_path, template_path, meta_file)
    
    try:
        # Create environment
        print(f"  Creating environment...")
        env = pymss.pymss(meta_file, num_slaves)
        
        # Reset
        print(f"  Resetting...")
        env.Resets(True)
        
        # Get initial state
        states = env.GetStates()
        if np.any(np.isnan(states)):
            print(f"  ❌ FAILED: NaN in initial states!")
            return False, "NaN in initial states"
        print(f"  Initial states OK (shape: {states.shape})")
        
        # Run simulation steps
        print(f"  Running {num_steps} steps...")
        nan_step = -1
        for step in range(num_steps):
            # Set random actions
            actions = np.random.randn(num_slaves, env.GetNumAction()) * 0.1
            env.SetActions(actions)
            
            # Step the environment
            env.StepsAtOnce()
            
            # Check for NaN
            states = env.GetStates()
            rewards = env.GetRewards()
            
            if np.any(np.isnan(states)) or np.any(np.isnan(rewards)):
                nan_step = step
                print(f"  ❌ FAILED: NaN detected at step {step}!")
                break
            
            # Check for episode end and reset
            for j in range(num_slaves):
                if env.IsEndOfEpisode(j):
                    env.Reset(True, j)
            
            if step % 20 == 0:
                print(f"    Step {step}: OK (avg reward: {np.mean(rewards):.4f})")
        
        if nan_step == -1:
            print(f"  ✅ SUCCESS: Motion '{motion_name}' completed {num_steps} steps without NaN!")
            return True, None
        else:
            return False, f"NaN at step {nan_step}"
            
    except Exception as e:
        print(f"  ❌ EXCEPTION: {e}")
        import traceback
        traceback.print_exc()
        return False, str(e)
        
    finally:
        # Cleanup
        if os.path.exists(meta_file):
            os.remove(meta_file)


def load_motion_list(path):
    """Load motions from motion_list.txt."""
    motions = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) >= 2:
                motion_path = parts[0]
                cyclic = parts[1].lower() in ('true', '1')
                motion_name = os.path.splitext(os.path.basename(motion_path))[0]
                motions.append({
                    'name': motion_name,
                    'path': motion_path,
                    'cyclic': cyclic
                })
    return motions


def main():
    print("="*60)
    print("INCREMENTAL MOTION TESTING")
    print("="*60)
    print("This script tests each motion individually to identify")
    print("which ones cause NaN/instability issues.")
    print()
    
    motion_list_path = "data/motion_list.txt"
    template_path = "data/metadata.txt"
    
    if not os.path.exists(motion_list_path):
        print(f"Error: {motion_list_path} not found")
        print("Please run: python python/scan_motions.py data/motion")
        return
    
    if not os.path.exists(template_path):
        print(f"Error: {template_path} not found")
        return
    
    # Load all motions
    motions = load_motion_list(motion_list_path)
    print(f"Found {len(motions)} motions to test:")
    for m in motions:
        print(f"  - {m['name']} (cyclic={m['cyclic']})")
    
    # Test each motion
    results = {}
    working_motions = []
    failed_motions = []
    
    for motion in motions:
        success, error = test_single_motion(
            motion['name'],
            motion['path'],
            template_path,
            num_steps=100,
            num_slaves=4
        )
        results[motion['name']] = {'success': success, 'error': error}
        
        if success:
            working_motions.append(motion)
        else:
            failed_motions.append(motion)
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"\n✅ Working motions ({len(working_motions)}):")
    for m in working_motions:
        print(f"   - {m['name']}")
    
    print(f"\n❌ Failed motions ({len(failed_motions)}):")
    for m in failed_motions:
        print(f"   - {m['name']}: {results[m['name']]['error']}")
    
    # Generate a "safe" motion list
    if working_motions:
        safe_list_path = "data/motion_list_safe.txt"
        print(f"\n📝 Generating safe motion list: {safe_list_path}")
        with open(safe_list_path, 'w') as f:
            f.write("# Safe motion list - only motions that passed testing\n")
            f.write("# Generated by test_motions_incremental.py\n\n")
            for m in working_motions:
                cyclic_str = 'true' if m['cyclic'] else 'false'
                f.write(f"{m['path']} {cyclic_str}\n")
        
        print(f"\nTo run multimodal training with safe motions:")
        print(f"  pixi run train_multimodal --motion_list {safe_list_path}")


if __name__ == "__main__":
    main()