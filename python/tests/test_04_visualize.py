#!/usr/bin/env python3
"""
Step 4: Visualization Helper for Trained Models
================================================
This script helps launch visualization for both single and multimodal trained models.
It can generate the correct commands and metadata files.

Run from project root:
    pixi shell
    python python/test_04_visualize.py
    
    # Single motion visualization
    python python/test_04_visualize.py --mode single
    
    # Multi-modal visualization (select motion)
    python python/test_04_visualize.py --mode multi --motion walk
"""

import os
import sys
import argparse
import subprocess

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def find_trained_models():
    """Find all available trained models"""
    print("=" * 60)
    print("Finding Trained Models")
    print("=" * 60)
    
    models = {
        'single': [],
        'multimodal': []
    }
    
    # Check for single-motion models
    single_paths = [
        ('nn/max.pt', 'nn/max_muscle.pt', 'best'),
        ('nn/current.pt', 'nn/current_muscle.pt', 'current'),
    ]
    
    for sim_path, muscle_path, name in single_paths:
        if os.path.exists(sim_path) and os.path.exists(muscle_path):
            models['single'].append({
                'name': name,
                'sim_path': sim_path,
                'muscle_path': muscle_path
            })
            print(f"✅ Found single-motion model: {name}")
            print(f"   - {sim_path}")
            print(f"   - {muscle_path}")
    
    # Check for multimodal models
    multi_paths = [
        ('nn/multimodal_max.pt', 'nn/multimodal_max_muscle.pt', 'best'),
        ('nn/multimodal_current.pt', 'nn/multimodal_current_muscle.pt', 'current'),
    ]
    
    for sim_path, muscle_path, name in multi_paths:
        if os.path.exists(sim_path) and os.path.exists(muscle_path):
            models['multimodal'].append({
                'name': name,
                'sim_path': sim_path,
                'muscle_path': muscle_path
            })
            print(f"✅ Found multimodal model: {name}")
            print(f"   - {sim_path}")
            print(f"   - {muscle_path}")
    
    # Check for numbered checkpoints
    for i in range(100):
        # Single motion checkpoints
        sim_path = f'nn/{i}.pt'
        muscle_path = f'nn/{i}_muscle.pt'
        if os.path.exists(sim_path) and os.path.exists(muscle_path):
            models['single'].append({
                'name': f'checkpoint_{i*100}',
                'sim_path': sim_path,
                'muscle_path': muscle_path
            })
        
        # Multimodal checkpoints
        sim_path = f'nn/multimodal_{i}.pt'
        muscle_path = f'nn/multimodal_{i}_muscle.pt'
        if os.path.exists(sim_path) and os.path.exists(muscle_path):
            models['multimodal'].append({
                'name': f'checkpoint_{i*100}',
                'sim_path': sim_path,
                'muscle_path': muscle_path
            })
    
    if not models['single'] and not models['multimodal']:
        print("\n⚠️  No trained models found!")
        print("   Train with:")
        print("   - Single motion: pixi run train")
        print("   - Multi-modal: pixi run train_multimodal")
    
    return models


def find_available_motions():
    """Find available motions from motion_list.txt"""
    print("\n" + "=" * 60)
    print("Finding Available Motions")
    print("=" * 60)
    
    motions = []
    motion_list_path = 'data/motion_list.txt'
    
    if not os.path.exists(motion_list_path):
        print(f"⚠️  {motion_list_path} not found")
        return motions
    
    with open(motion_list_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) >= 2:
                bvh_path = parts[0]
                cyclic = parts[1].lower() in ('true', '1')
                motion_name = os.path.splitext(os.path.basename(bvh_path))[0]
                motions.append({
                    'name': motion_name,
                    'bvh_path': bvh_path,
                    'cyclic': cyclic
                })
                print(f"   - {motion_name}: {bvh_path} (cyclic={cyclic})")
    
    return motions


def generate_metadata_for_motion(motion, template_path='data/metadata.txt'):
    """Generate a metadata file for a specific motion"""
    output_path = f'data/metadata_{motion["name"]}.txt'
    
    with open(template_path, 'r') as f:
        content = f.read()
    
    new_content = []
    for line in content.split('\n'):
        if line.strip().startswith('bvh_file'):
            cyclic_str = 'true' if motion['cyclic'] else 'false'
            new_content.append(f"bvh_file {motion['bvh_path']} {cyclic_str}")
        else:
            new_content.append(line)
    
    with open(output_path, 'w') as f:
        f.write('\n'.join(new_content))
    
    return output_path


def get_render_command(metadata_path, model=None):
    """Generate the render command"""
    if model:
        return f"./build/render {metadata_path} {model['sim_path']} {model['muscle_path']}"
    else:
        return f"./build/render {metadata_path}"


def visualize_single(models, execute=False):
    """Visualize single-motion model"""
    print("\n" + "=" * 60)
    print("Single-Motion Visualization")
    print("=" * 60)
    
    metadata_path = 'data/metadata.txt'
    
    if not os.path.exists(metadata_path):
        print(f"❌ Metadata not found: {metadata_path}")
        return
    
    if models['single']:
        # Use best available model
        model = models['single'][0]
        cmd = get_render_command(metadata_path, model)
        print(f"\n📝 Command to visualize with '{model['name']}' model:")
        print(f"   {cmd}")
    else:
        cmd = get_render_command(metadata_path)
        print("\n📝 Command to visualize without trained model:")
        print(f"   {cmd}")
    
    print("\n📌 Controls:")
    print("   - SPACE: Start/stop simulation")
    print("   - 's': Single step")
    print("   - 'r': Reset")
    print("   - ESC: Exit")
    
    if execute:
        print(f"\n▶️  Launching render...")
        subprocess.run(cmd, shell=True)


def visualize_multimodal(models, motions, motion_name=None, execute=False):
    """Visualize multimodal model with specific motion"""
    print("\n" + "=" * 60)
    print("Multi-Modal Visualization")
    print("=" * 60)
    
    # Find the requested motion
    if motion_name:
        motion = next((m for m in motions if m['name'] == motion_name), None)
        if not motion:
            print(f"❌ Motion '{motion_name}' not found")
            print(f"   Available: {[m['name'] for m in motions]}")
            return
    else:
        if motions:
            motion = motions[0]
            print(f"⚠️  No motion specified, using: {motion['name']}")
        else:
            print("❌ No motions available")
            return
    
    # Generate metadata for this motion
    metadata_path = generate_metadata_for_motion(motion)
    print(f"\n📄 Generated metadata: {metadata_path}")
    
    if models['multimodal']:
        # Use best available model
        model = models['multimodal'][0]
        cmd = get_render_command(metadata_path, model)
        print(f"\n📝 Command to visualize '{motion['name']}' with '{model['name']}' multimodal model:")
        print(f"   {cmd}")
    else:
        cmd = get_render_command(metadata_path)
        print(f"\n📝 Command to visualize '{motion['name']}' without trained model:")
        print(f"   {cmd}")
    
    print("\n📌 Controls:")
    print("   - SPACE: Start/stop simulation")
    print("   - 's': Single step")
    print("   - 'r': Reset")
    print("   - ESC: Exit")
    
    if execute:
        print(f"\n▶️  Launching render...")
        subprocess.run(cmd, shell=True)


def print_all_commands(models, motions):
    """Print all possible visualization commands"""
    print("\n" + "=" * 60)
    print("All Visualization Commands")
    print("=" * 60)
    
    print("\n--- Single-Motion Commands ---")
    metadata_path = 'data/metadata.txt'
    if os.path.exists(metadata_path):
        # Without model
        print(f"\n# Without trained model:")
        print(f"./build/render {metadata_path}")
        
        # With models
        for model in models['single']:
            print(f"\n# With {model['name']} model:")
            print(f"./build/render {metadata_path} {model['sim_path']} {model['muscle_path']}")
    
    print("\n--- Multi-Modal Commands ---")
    for motion in motions:
        # Check if metadata exists
        meta_path = f'data/metadata_{motion["name"]}.txt'
        
        print(f"\n# Motion: {motion['name']}")
        if os.path.exists(meta_path):
            print(f"# Metadata exists: {meta_path}")
        else:
            print(f"# Generate metadata first:")
            print(f"#   python python/test_04_visualize.py --mode multi --motion {motion['name']}")
        
        for model in models['multimodal']:
            print(f"./build/render data/metadata_{motion['name']}.txt {model['sim_path']} {model['muscle_path']}")


def interactive_mode(models, motions):
    """Interactive mode for selecting visualization"""
    print("\n" + "=" * 60)
    print("Interactive Visualization Selector")
    print("=" * 60)
    
    print("\nSelect mode:")
    print("  [1] Single-motion visualization")
    print("  [2] Multi-modal visualization")
    print("  [3] Show all commands")
    print("  [q] Quit")
    
    choice = input("\nEnter choice: ").strip().lower()
    
    if choice == '1':
        visualize_single(models, execute=False)
    elif choice == '2':
        if motions:
            print("\nAvailable motions:")
            for i, m in enumerate(motions):
                print(f"  [{i}] {m['name']}")
            idx = input("\nSelect motion index: ").strip()
            try:
                motion = motions[int(idx)]
                visualize_multimodal(models, motions, motion['name'], execute=False)
            except (ValueError, IndexError):
                print("Invalid selection")
        else:
            print("No motions available")
    elif choice == '3':
        print_all_commands(models, motions)
    elif choice == 'q':
        return
    else:
        print("Invalid choice")


def main():
    parser = argparse.ArgumentParser(description='Visualization Helper')
    parser.add_argument('--mode', choices=['single', 'multi', 'all', 'interactive'],
                        default='interactive',
                        help='Visualization mode')
    parser.add_argument('--motion', type=str, default=None,
                        help='Motion name for multi-modal (e.g., walk, run)')
    parser.add_argument('--execute', action='store_true',
                        help='Actually launch the render (requires X server)')
    args = parser.parse_args()
    
    print("\n" + "=" * 60)
    print("MASS VISUALIZATION HELPER (Step 4)")
    print("=" * 60)
    
    # Find models
    models = find_trained_models()
    
    # Find motions
    motions = find_available_motions()
    
    if args.mode == 'single':
        visualize_single(models, execute=args.execute)
    elif args.mode == 'multi':
        visualize_multimodal(models, motions, args.motion, execute=args.execute)
    elif args.mode == 'all':
        print_all_commands(models, motions)
    else:
        interactive_mode(models, motions)
    
    print("\n" + "=" * 60)
    print("Notes:")
    print("=" * 60)
    print("1. The render executable requires an X server (or VcXsrv on WSL)")
    print("2. If no model is specified, the character uses random/zero actions")
    print("3. For WSL, run: export DISPLAY=:0 (or your display)")
    print("4. To train: pixi run train (or train_multimodal)")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
