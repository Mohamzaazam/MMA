#!/usr/bin/env python3
"""
Extract BVH state data to NPZ files for fast training.

This script extracts position+velocity states from BVH files and saves them
as NPZ files, preserving the directory structure. This enables fast training
without C++ BVH parsing overhead.

Usage:
    pixi run python scripts/extract_states.py --bvh_dir data/cmu --output_dir data/extracted
    pixi run python scripts/extract_states.py --bvh_dir data/cmu --output_dir data/extracted --max_subjects 10
"""

import argparse
import sys
from pathlib import Path
from collections import defaultdict

# Add project root to path (MMA root, not python/)
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import numpy as np


def extract_bvh_to_npz(
    bvh_dir: str,
    output_dir: str,
    max_subjects: int = None,
    build_dir: str = "build",
    verbose: bool = True,
):
    """
    Extract all BVH files to NPZ format.
    
    Args:
        bvh_dir: Directory containing BVH files
        output_dir: Output directory for NPZ files
        max_subjects: Maximum number of subjects to extract (for testing)
        build_dir: Build directory for pymss
        verbose: Print progress
    """
    from python.utils.state_extractor import create_state_extractor
    
    bvh_path = Path(bvh_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    # Suppress C++ output
    sys.path.insert(0, str(Path(build_dir).absolute()))
    try:
        import pymss
        pymss.set_verbose(False)
    except ImportError:
        pass
    
    # Find all BVH files
    all_bvh = sorted(bvh_path.glob("**/*.bvh"))
    
    if verbose:
        print(f"Found {len(all_bvh)} BVH files in {bvh_dir}")
    
    # Group by subject if limiting
    if max_subjects:
        subjects = defaultdict(list)
        for f in all_bvh:
            # Extract subject from path (e.g., data/cmu/01/01_01.bvh -> 01)
            parts = f.relative_to(bvh_path).parts
            if len(parts) >= 2:
                subject = parts[0]
            else:
                subject = "unknown"
            subjects[subject].append(f)
        
        # Take first N subjects
        selected_subjects = sorted(subjects.keys())[:max_subjects]
        all_bvh = []
        for s in selected_subjects:
            all_bvh.extend(subjects[s])
        
        if verbose:
            print(f"Limited to {max_subjects} subjects: {len(all_bvh)} files")
    
    # Create shared extractor with first file
    extractor = None
    success = 0
    failed = 0
    
    for i, bvh_file in enumerate(all_bvh):
        rel_path = bvh_file.relative_to(bvh_path)
        npz_path = out_path / rel_path.with_suffix('.npz')
        npz_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Skip if already extracted
        if npz_path.exists():
            if verbose and (i + 1) % 100 == 0:
                print(f"[{i+1}/{len(all_bvh)}] Skipped (exists): {rel_path}")
            success += 1
            continue
        
        try:
            if extractor is None:
                # Create first extractor
                extractor = create_state_extractor(
                    str(bvh_file),
                    build_dir=build_dir,
                    cyclic=False
                )
            else:
                # Reload BVH into existing extractor
                if not extractor.reload_bvh(str(bvh_file), cyclic=False):
                    raise ValueError("reload_bvh failed")
            
            # Extract states
            states = extractor.extract_all_states(include_phase=False)
            
            # Save to NPZ
            np.savez_compressed(npz_path, states=states)
            success += 1
            
            if verbose and (i + 1) % 50 == 0:
                print(f"[{i+1}/{len(all_bvh)}] ✓ {rel_path} ({len(states)} frames)")
                
        except Exception as e:
            failed += 1
            if verbose:
                print(f"[{i+1}/{len(all_bvh)}] ✗ {rel_path}: {e}")
    
    if verbose:
        print(f"\nExtraction complete:")
        print(f"  Success: {success}")
        print(f"  Failed:  {failed}")
        print(f"  Output:  {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Extract BVH states to NPZ files")
    parser.add_argument('--bvh_dir', type=str, default='data/cmu',
                        help='Directory containing BVH files')
    parser.add_argument('--output_dir', type=str, default='data/extracted',
                        help='Output directory for NPZ files')
    parser.add_argument('--max_subjects', type=int, default=None,
                        help='Maximum subjects to extract (for testing)')
    parser.add_argument('--build_dir', type=str, default='build',
                        help='Build directory for pymss')
    parser.add_argument('--quiet', action='store_true',
                        help='Suppress progress output')
    
    args = parser.parse_args()
    
    extract_bvh_to_npz(
        bvh_dir=args.bvh_dir,
        output_dir=args.output_dir,
        max_subjects=args.max_subjects,
        build_dir=args.build_dir,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
