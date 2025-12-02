#!/usr/bin/env python3
"""
BVH File Compatibility Analyzer
Compares two BVH files to identify structural, coordinate system, and hierarchy differences
"""

import re
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import math

@dataclass
class Joint:
    name: str
    offset: Tuple[float, float, float]
    channels: List[str]
    children: List['Joint']
    depth: int

def parse_bvh_hierarchy(filepath: str) -> Tuple[Joint, Dict[str, Joint], Dict]:
    """Parse BVH file and extract hierarchy information"""
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Extract hierarchy section
    hierarchy_match = re.search(r'HIERARCHY\s+(.*?)MOTION', content, re.DOTALL)
    if not hierarchy_match:
        raise ValueError("Could not find HIERARCHY section")
    
    hierarchy_text = hierarchy_match.group(1)
    
    # Extract motion section info
    motion_section = content[content.find('MOTION'):]
    frames_match = re.search(r'Frames:\s*(\d+)', motion_section)
    frametime_match = re.search(r'Frame Time:\s*([\d.]+)', motion_section)
    
    motion_info = {
        'frames': int(frames_match.group(1)) if frames_match else None,
        'frame_time': float(frametime_match.group(1)) if frametime_match else None
    }
    
    joints = {}
    root = None
    
    # Parse joints
    lines = hierarchy_text.strip().split('\n')
    stack = []
    current_depth = 0
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        if line.startswith('ROOT') or line.startswith('JOINT'):
            joint_type = 'ROOT' if line.startswith('ROOT') else 'JOINT'
            name = line.split()[1]
            
            # Find OFFSET
            offset = (0.0, 0.0, 0.0)
            channels = []
            
            # Look ahead for OFFSET and CHANNELS
            j = i + 1
            while j < len(lines) and not lines[j].strip().startswith(('ROOT', 'JOINT', 'End Site', '}')):
                subline = lines[j].strip()
                if subline.startswith('OFFSET'):
                    parts = subline.split()
                    offset = (float(parts[1]), float(parts[2]), float(parts[3]))
                elif subline.startswith('CHANNELS'):
                    parts = subline.split()
                    channels = parts[2:]
                j += 1
            
            joint = Joint(
                name=name,
                offset=offset,
                channels=channels,
                children=[],
                depth=len(stack)
            )
            
            joints[name] = joint
            
            if joint_type == 'ROOT':
                root = joint
            
            if stack:
                stack[-1].children.append(joint)
            
            stack.append(joint)
        
        elif line == '{':
            pass
        elif line == '}':
            if stack:
                stack.pop()
        
        i += 1
    
    return root, joints, motion_info

def get_channel_order(channels: List[str]) -> str:
    """Extract rotation order from channels"""
    rot_channels = [c for c in channels if 'rotation' in c.lower()]
    return ''.join([c[0] for c in rot_channels])  # e.g., 'ZXY' or 'ZYX'

def analyze_hierarchy(root: Joint, prefix: str = "") -> List[str]:
    """Generate hierarchy tree representation"""
    lines = []
    lines.append(f"{prefix}{root.name} (channels: {root.channels}, offset: {root.offset})")
    for i, child in enumerate(root.children):
        is_last = i == len(root.children) - 1
        child_prefix = prefix + ("    " if is_last else "│   ")
        connector = "└── " if is_last else "├── "
        child_lines = analyze_hierarchy(child, child_prefix)
        child_lines[0] = prefix + connector + child_lines[0][len(child_prefix):]
        lines.extend(child_lines)
    return lines

def compare_joints(joints1: Dict[str, Joint], joints2: Dict[str, Joint]) -> Dict:
    """Compare joint structures between two files"""
    comparison = {
        'common_joints': [],
        'only_in_file1': [],
        'only_in_file2': [],
        'channel_differences': [],
        'offset_differences': []
    }
    
    # Normalize joint names for comparison
    def normalize_name(name: str) -> str:
        # Remove common prefixes
        name = re.sub(r'^Character1_', '', name)
        name = re.sub(r'_RIGMESH$', '', name)
        return name.lower()
    
    norm_to_orig1 = {normalize_name(j): j for j in joints1.keys()}
    norm_to_orig2 = {normalize_name(j): j for j in joints2.keys()}
    
    all_normalized = set(norm_to_orig1.keys()) | set(norm_to_orig2.keys())
    
    for norm_name in all_normalized:
        in_file1 = norm_name in norm_to_orig1
        in_file2 = norm_name in norm_to_orig2
        
        if in_file1 and in_file2:
            orig1 = norm_to_orig1[norm_name]
            orig2 = norm_to_orig2[norm_name]
            joint1 = joints1[orig1]
            joint2 = joints2[orig2]
            
            comparison['common_joints'].append((orig1, orig2))
            
            # Check channel order
            if joint1.channels and joint2.channels:
                order1 = get_channel_order(joint1.channels)
                order2 = get_channel_order(joint2.channels)
                if order1 != order2:
                    comparison['channel_differences'].append({
                        'joint1': orig1,
                        'joint2': orig2,
                        'order1': order1,
                        'order2': order2
                    })
            
            # Check offset magnitude differences
            mag1 = math.sqrt(sum(x**2 for x in joint1.offset))
            mag2 = math.sqrt(sum(x**2 for x in joint2.offset))
            if mag1 > 0 and mag2 > 0:
                ratio = max(mag1, mag2) / min(mag1, mag2) if min(mag1, mag2) > 0.001 else float('inf')
                if ratio > 1.5:  # Significant difference
                    comparison['offset_differences'].append({
                        'joint1': orig1,
                        'joint2': orig2,
                        'offset1': joint1.offset,
                        'offset2': joint2.offset,
                        'ratio': ratio
                    })
        elif in_file1:
            comparison['only_in_file1'].append(norm_to_orig1[norm_name])
        else:
            comparison['only_in_file2'].append(norm_to_orig2[norm_name])
    
    return comparison

def extract_first_frame_rotations(filepath: str, joints: Dict[str, Joint]) -> Dict[str, List[float]]:
    """Extract rotation values from first frame for key joints"""
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Find motion data
    motion_start = content.find('MOTION')
    lines = content[motion_start:].strip().split('\n')
    
    # Skip header lines to get first frame data
    for i, line in enumerate(lines):
        if line.strip() and not line.strip().startswith(('MOTION', 'Frames:', 'Frame Time:')):
            first_frame = line.strip().split()
            break
    
    # Map values to joints based on channel count
    values = [float(v) for v in first_frame]
    rotations = {}
    
    idx = 0
    def traverse(joint: Joint):
        nonlocal idx
        if not joint.channels:
            return
        
        num_channels = len(joint.channels)
        joint_values = values[idx:idx+num_channels]
        idx += num_channels
        
        # Extract rotations
        rot_values = []
        for i, ch in enumerate(joint.channels):
            if 'rotation' in ch.lower():
                rot_values.append((ch, joint_values[i]))
        
        if rot_values:
            rotations[joint.name] = rot_values
        
        for child in joint.children:
            traverse(child)
    
    # Find root and traverse
    root = next((j for j in joints.values() if j.depth == 0), None)
    if root:
        traverse(root)
    
    return rotations

def main():
    print("=" * 80)
    print("BVH FILE COMPATIBILITY ANALYSIS")
    print("=" * 80)
    
    base_file = "data/motion/walk.bvh"
    new_file = "data/motion/29_01.bvh"
    
    # Parse both files
    print("\n[1] PARSING FILES...")
    root1, joints1, motion1 = parse_bvh_hierarchy(base_file)
    root2, joints2, motion2 = parse_bvh_hierarchy(new_file)
    
    print(f"\nBase file (walk.bvh):")
    print(f"  - Root joint: {root1.name}")
    print(f"  - Total joints: {len(joints1)}")
    print(f"  - Frames: {motion1['frames']}")
    print(f"  - Frame time: {motion1['frame_time']} ({1/motion1['frame_time']:.1f} fps)")
    
    print(f"\nNew file (29_01_.bvh):")
    print(f"  - Root joint: {root2.name}")
    print(f"  - Total joints: {len(joints2)}")
    print(f"  - Frames: {motion2['frames']}")
    print(f"  - Frame time: {motion2['frame_time']} ({1/motion2['frame_time']:.1f} fps)")
    
    # Channel order analysis
    print("\n" + "=" * 80)
    print("[2] CHANNEL ORDER ANALYSIS (CRITICAL FOR ROTATION)")
    print("=" * 80)
    
    print("\nBase file (walk.bvh) - Root channels:")
    print(f"  {root1.name}: {root1.channels}")
    
    print("\nNew file (29_01_.bvh) - Root channels:")
    print(f"  {root2.name}: {root2.channels}")
    
    # Compare rotation orders for matching joints
    comparison = compare_joints(joints1, joints2)
    
    if comparison['channel_differences']:
        print("\n⚠️  ROTATION ORDER DIFFERENCES DETECTED:")
        print("-" * 60)
        for diff in comparison['channel_differences']:
            print(f"  {diff['joint1']:30} -> {diff['order1']}")
            print(f"  {diff['joint2']:30} -> {diff['order2']}")
            print()
    
    # Hierarchy comparison
    print("\n" + "=" * 80)
    print("[3] HIERARCHY COMPARISON")
    print("=" * 80)
    
    print(f"\nJoints only in BASE file (walk.bvh): {len(comparison['only_in_file1'])}")
    for j in sorted(comparison['only_in_file1']):
        print(f"  - {j}")
    
    print(f"\nJoints only in NEW file (29_01_.bvh): {len(comparison['only_in_file2'])}")
    for j in sorted(comparison['only_in_file2']):
        print(f"  - {j}")
    
    # Coordinate system analysis
    print("\n" + "=" * 80)
    print("[4] COORDINATE SYSTEM & SCALE ANALYSIS")
    print("=" * 80)
    
    print("\nRoot position offsets:")
    print(f"  Base: {root1.offset}")
    print(f"  New:  {root2.offset}")
    
    print("\nKey joint offsets comparison:")
    key_joints = ['leftarm', 'rightarm', 'leftupleg', 'rightupleg', 'leftleg', 'rightleg']
    
    for norm_name in key_joints:
        for orig1, orig2 in comparison['common_joints']:
            n1 = re.sub(r'^Character1_', '', orig1).lower()
            n2 = orig2.lower()
            if norm_name in n1 or norm_name in n2:
                j1 = joints1[orig1]
                j2 = joints2[orig2]
                print(f"\n  {orig1} vs {orig2}:")
                print(f"    Base offset: {j1.offset}")
                print(f"    New offset:  {j2.offset}")
                mag1 = math.sqrt(sum(x**2 for x in j1.offset))
                mag2 = math.sqrt(sum(x**2 for x in j2.offset))
                if mag1 > 0 and mag2 > 0:
                    print(f"    Scale ratio: {mag1/mag2:.2f}x")
                break
    
    # First frame rotation analysis
    print("\n" + "=" * 80)
    print("[5] FIRST FRAME ROTATION ANALYSIS")
    print("=" * 80)
    
    rot1 = extract_first_frame_rotations(base_file, joints1)
    rot2 = extract_first_frame_rotations(new_file, joints2)
    
    print("\nKey joint rotations (first frame):")
    
    critical_joints = ['Hips', 'LeftArm', 'RightArm', 'LeftForeArm', 'RightForeArm', 
                       'LeftUpLeg', 'RightUpLeg', 'LeftLeg', 'RightLeg', 
                       'LeftFoot', 'RightFoot']
    
    for cj in critical_joints:
        print(f"\n  {cj}:")
        # Find in base file
        for name, rots in rot1.items():
            if cj.lower() in name.lower():
                print(f"    Base ({name}): {rots}")
                break
        # Find in new file
        for name, rots in rot2.items():
            if cj.lower() in name.lower():
                print(f"    New ({name}):  {rots}")
                break
    
    # Summary and recommendations
    print("\n" + "=" * 80)
    print("[6] DIAGNOSIS & RECOMMENDATIONS")
    print("=" * 80)
    
    issues = []
    
    # Check rotation order
    if comparison['channel_differences']:
        issues.append({
            'severity': 'CRITICAL',
            'issue': 'Rotation Order Mismatch',
            'details': f"Base file uses ZXY rotation order, new file uses ZYX",
            'effect': "This causes inverted joints (elbows, ankles) because rotations are applied in wrong order",
            'fix': "In MotionBuilder, ensure export uses ZXY rotation order, or apply rotation order conversion"
        })
    
    # Check frame rate
    if motion1['frame_time'] != motion2['frame_time']:
        issues.append({
            'severity': 'MEDIUM',
            'issue': 'Frame Rate Mismatch',
            'details': f"Base: {1/motion1['frame_time']:.1f} fps, New: {1/motion2['frame_time']:.1f} fps",
            'effect': "Animation speed will differ",
            'fix': "Resample animation to match target frame rate"
        })
    
    # Check hierarchy differences
    if comparison['only_in_file2']:
        extra_joints = [j for j in comparison['only_in_file2'] 
                       if not j.endswith('Joint') and 'Thumb' not in j and 'Finger' not in j]
        if extra_joints:
            issues.append({
                'severity': 'MEDIUM',
                'issue': 'Extra Intermediate Joints',
                'details': f"New file has additional joints: {extra_joints}",
                'effect': "May cause skeleton mismatch in pipeline",
                'fix': "Flatten hierarchy or map intermediate joints appropriately in human.xml"
            })
    
    # Check for missing required joints
    if comparison['only_in_file1']:
        missing = [j for j in comparison['only_in_file1'] if '_RIGMESH' not in j]
        if missing:
            issues.append({
                'severity': 'LOW',
                'issue': 'Missing Mesh Helper Joints',
                'details': f"Joints in base not in new: {missing}",
                'effect': "May not affect core animation",
                'fix': "Only add if required by rendering pipeline"
            })
    
    for i, issue in enumerate(issues, 1):
        print(f"\n{'⛔' if issue['severity'] == 'CRITICAL' else '⚠️' if issue['severity'] == 'MEDIUM' else 'ℹ️'} Issue {i}: [{issue['severity']}] {issue['issue']}")
        print(f"   Details: {issue['details']}")
        print(f"   Effect:  {issue['effect']}")
        print(f"   Fix:     {issue['fix']}")
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"""
The primary cause of inverted elbows and ankles is the ROTATION ORDER MISMATCH:

  Base file (walk.bvh):  Zrotation Xrotation Yrotation (ZXY Euler order)
  New file (29_01_.bvh): Zrotation Yrotation Xrotation (ZYX Euler order)

This difference means when the same rotation values are applied, they produce
completely different poses because the order of matrix multiplication changes.

RECOMMENDED SOLUTIONS:

1. RE-EXPORT FROM MOTIONBUILDER:
   - Set rotation order to ZXY to match base file
   - Ensure "Use Scene Rotation Order" is disabled
   - Check that the character definition uses same conventions

2. POST-PROCESS CONVERSION:
   - Write a conversion script to transform ZYX rotations to ZXY
   - This requires quaternion conversion to avoid gimbal lock

3. PIPELINE ADAPTATION:
   - If modifying human.xml, also update rotation interpretation
   - Add rotation order parameter to your skeleton definition
""")

if __name__ == "__main__":
    main()