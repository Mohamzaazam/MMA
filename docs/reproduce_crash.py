
import pymss
import os
import sys
import time

def load_motion_list(path):
    motions = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) >= 1:
                motions.append(parts[0])
    return motions

def generate_metadata(motion_path, template_path, output_path):
    with open(template_path, 'r') as f:
        content = f.read()
    
    new_content = []
    for line in content.split('\n'):
        if line.strip().startswith('bvh_file'):
            new_content.append(f"bvh_file {motion_path} true")
        else:
            new_content.append(line)
            
    with open(output_path, 'w') as f:
        f.write('\n'.join(new_content))

def test_multimodal_minimal():
    print("Testing MINIMAL motions SIMULTANEOUSLY...")
    
    motion_list_path = "data/motion_list.txt"
    template_path = "data/metadata.txt"
    num_slaves_per_motion = 1  # Minimal load
    
    # Pick 2 motions
    target_motions = ['walk', 'backflip']
    
    if not os.path.exists(motion_list_path):
        print(f"Error: {motion_list_path} not found")
        return

    all_motions = load_motion_list(motion_list_path)
    motions_to_test = []
    for m in all_motions:
        name = os.path.splitext(os.path.basename(m))[0]
        if name in target_motions:
            motions_to_test.append(m)
            
    print(f"Testing {len(motions_to_test)} motions: {target_motions}")
    
    envs = []
    meta_files = []
    
    try:
        # Create environments
        for i, motion_path in enumerate(motions_to_test):
            motion_name = os.path.splitext(os.path.basename(motion_path))[0]
            print(f"  Creating env for {motion_name}...")
            
            meta_file = f"data/metadata_{motion_name}_test_minimal.txt"
            generate_metadata(motion_path, template_path, meta_file)
            meta_files.append(meta_file)
            
            env = pymss.pymss(meta_file, num_slaves_per_motion)
            envs.append(env)
            
        print(f"Created {len(envs)} environments.")
        
        # Reset all
        print("Resetting all environments...")
        for i, env in enumerate(envs):
            env.Resets(True)
            
        # Step all
        print("Stepping all environments (1000 steps)...")
        for step in range(1000):
            if step % 100 == 0:
                print(f"  Step {step}")
            for env in envs:
                env.Steps(1)
                
        print("Success! Minimal execution worked.")
        
    except Exception as e:
        print(f"FAILED during minimal execution: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        # Clean up
        for meta_file in meta_files:
            if os.path.exists(meta_file):
                os.remove(meta_file)

if __name__ == "__main__":
    test_multimodal_minimal()
