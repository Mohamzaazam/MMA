#!/usr/bin/env python3
"""
Motion Scanner Utility
Scans data/motion directory for BVH files and validates their compatibility
with the MASS skeleton.

Run this first to understand what motions are available before enabling
multimodal training.
"""

import os
import sys
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class BVHInfo:
    """Information about a BVH file"""
    filepath: str
    filename: str
    root_joint: str
    num_joints: int
    num_channels: int
    num_frames: int
    frame_time: float
    duration: float
    joint_names: List[str]
    is_valid: bool
    error_message: str = ""


def parse_bvh_header(filepath: str) -> BVHInfo:
    """
    Parse BVH file header to extract metadata without loading full motion data.
    This is a lightweight check to validate BVH compatibility.
    """
    filename = os.path.basename(filepath)
    
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
    except Exception as e:
        return BVHInfo(
            filepath=filepath,
            filename=filename,
            root_joint="",
            num_joints=0,
            num_channels=0,
            num_frames=0,
            frame_time=0.0,
            duration=0.0,
            joint_names=[],
            is_valid=False,
            error_message=f"Cannot read file: {e}"
        )
    
    # Extract hierarchy section
    parts = re.split(r'\bMOTION\b', content, flags=re.IGNORECASE)
    if len(parts) < 2:
        return BVHInfo(
            filepath=filepath,
            filename=filename,
            root_joint="",
            num_joints=0,
            num_channels=0,
            num_frames=0,
            frame_time=0.0,
            duration=0.0,
            joint_names=[],
            is_valid=False,
            error_message="No MOTION section found"
        )
    
    hierarchy_section = parts[0]
    motion_section = parts[1]
    
    # Parse root joint
    root_match = re.search(r'ROOT\s+(\S+)', hierarchy_section, re.IGNORECASE)
    root_joint = root_match.group(1) if root_match else "Unknown"
    
    # Parse all joints (ROOT and JOINT)
    joint_names = re.findall(r'(?:ROOT|JOINT)\s+(\S+)', hierarchy_section, re.IGNORECASE)
    num_joints = len(joint_names)
    
    # Count total channels
    channels = re.findall(r'CHANNELS\s+(\d+)', hierarchy_section, re.IGNORECASE)
    num_channels = sum(int(c) for c in channels)
    
    # Parse motion info
    frames_match = re.search(r'Frames:\s*(\d+)', motion_section, re.IGNORECASE)
    num_frames = int(frames_match.group(1)) if frames_match else 0
    
    frame_time_match = re.search(r'Frame\s*Time:\s*([\d.]+)', motion_section, re.IGNORECASE)
    frame_time = float(frame_time_match.group(1)) if frame_time_match else 0.0
    
    duration = num_frames * frame_time
    
    # Basic validation
    is_valid = True
    error_message = ""
    
    if num_joints == 0:
        is_valid = False
        error_message = "No joints found"
    elif num_frames == 0:
        is_valid = False
        error_message = "No motion frames"
    elif frame_time <= 0:
        is_valid = False
        error_message = "Invalid frame time"
    
    return BVHInfo(
        filepath=filepath,
        filename=filename,
        root_joint=root_joint,
        num_joints=num_joints,
        num_channels=num_channels,
        num_frames=num_frames,
        frame_time=frame_time,
        duration=duration,
        joint_names=joint_names,
        is_valid=is_valid,
        error_message=error_message
    )


def scan_motion_directory(motion_dir: str) -> List[BVHInfo]:
    """Scan directory for all BVH files and parse their headers"""
    bvh_files = []
    
    motion_path = Path(motion_dir)
    if not motion_path.exists():
        print(f"ERROR: Motion directory not found: {motion_dir}")
        return []
    
    # Find all .bvh files (case insensitive)
    for filepath in motion_path.glob("**/*.[bB][vV][hH]"):
        info = parse_bvh_header(str(filepath))
        bvh_files.append(info)
    
    return bvh_files


def print_scan_results(bvh_files: List[BVHInfo]):
    """Print formatted scan results"""
    print("\n" + "=" * 80)
    print("MOTION FILE SCAN RESULTS")
    print("=" * 80)
    
    valid_files = [f for f in bvh_files if f.is_valid]
    invalid_files = [f for f in bvh_files if not f.is_valid]
    
    print(f"\nTotal files found: {len(bvh_files)}")
    print(f"Valid files: {len(valid_files)}")
    print(f"Invalid files: {len(invalid_files)}")
    
    if valid_files:
        print("\n" + "-" * 80)
        print("VALID BVH FILES:")
        print("-" * 80)
        print(f"{'Filename':<40} {'Joints':<8} {'Frames':<8} {'Duration':<10} {'Root'}")
        print("-" * 80)
        
        for info in sorted(valid_files, key=lambda x: x.filename):
            print(f"{info.filename:<40} {info.num_joints:<8} {info.num_frames:<8} {info.duration:>6.2f}s    {info.root_joint}")
    
    if invalid_files:
        print("\n" + "-" * 80)
        print("INVALID BVH FILES:")
        print("-" * 80)
        for info in invalid_files:
            print(f"  {info.filename}: {info.error_message}")
    
    # Check skeleton compatibility
    if valid_files:
        print("\n" + "-" * 80)
        print("SKELETON COMPATIBILITY CHECK:")
        print("-" * 80)
        
        # Group by root joint
        root_groups: Dict[str, List[BVHInfo]] = {}
        for info in valid_files:
            root = info.root_joint.lower()
            if root not in root_groups:
                root_groups[root] = []
            root_groups[root].append(info)
        
        for root, files in root_groups.items():
            print(f"\nRoot joint '{root}': {len(files)} file(s)")
            # Check joint count consistency
            joint_counts = set(f.num_joints for f in files)
            if len(joint_counts) > 1:
                print(f"  WARNING: Inconsistent joint counts: {joint_counts}")
            else:
                print(f"  Joint count: {joint_counts.pop()}")
    
    return valid_files


def generate_motion_list(valid_files: List[BVHInfo], output_path: str):
    """Generate a motion list file for the training system"""
    with open(output_path, 'w') as f:
        f.write("# Motion list for MASS multimodal training\n")
        f.write("# Format: <relative_path> <cyclic: true/false>\n")
        f.write("# Generated by scan_motions.py\n\n")
        
        for info in sorted(valid_files, key=lambda x: x.filename):
            # Determine if motion is likely cyclic based on name
            name_lower = info.filename.lower()
            cyclic = "true" if any(w in name_lower for w in ['walk', 'run', 'jog', 'cycle']) else "false"
            
            # Use relative path from data directory
            rel_path = info.filepath
            if '/data/' in rel_path:
                rel_path = '/data/' + rel_path.split('/data/')[-1]
            
            f.write(f"{rel_path} {cyclic}\n")
    
    print(f"\nMotion list written to: {output_path}")


def main():
    # Determine motion directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    # Default motion directory
    motion_dir = os.path.join(project_root, "data", "motion")
    
    # Allow override via command line
    if len(sys.argv) > 1:
        motion_dir = sys.argv[1]
    
    print(f"Scanning motion directory: {motion_dir}")
    
    # Scan for BVH files
    bvh_files = scan_motion_directory(motion_dir)
    
    if not bvh_files:
        print("No BVH files found!")
        return
    
    # Print results
    valid_files = print_scan_results(bvh_files)
    
    # Generate motion list if valid files found
    if valid_files:
        motion_list_path = os.path.join(project_root, "data", "motion_list.txt")
        generate_motion_list(valid_files, motion_list_path)
        
        print("\n" + "=" * 80)
        print("NEXT STEPS:")
        print("=" * 80)
        print("1. Review the generated motion_list.txt")
        print("2. Edit cyclic flags as needed (true for looping motions)")
        print("3. Ensure BVH joint names match your skeleton's bvh_map")
        print("4. Run the multimodal training with the motion list")


if __name__ == "__main__":
    main()