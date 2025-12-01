#!/usr/bin/env python3
"""
MASS BVH Pipeline Compatibility Tool

This is the main script for analyzing and converting BVH files for the
MASS (Muscle-Actuated Skeletal System) pipeline.

Based on analysis of:
- human.xml skeleton definition
- BVH.cpp parsing code
- JOINT_COMPARISON_ANALYSIS.md

Usage:
    # Analyze a single file
    python mass_bvh_tool.py analyze 29_01.bvh
    
    # Compare two files
    python mass_bvh_tool.py compare walk.bvh 29_01.bvh
    
    # Convert CMU to MASS format
    python mass_bvh_tool.py convert 29_01.bvh --output 29_01_mass.bvh --reference walk.bvh
    
    # Generate human.xml mappings for a BVH file
    python mass_bvh_tool.py generate-xml 29_01.bvh
"""

import sys
import re
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field
from collections import OrderedDict
import xml.etree.ElementTree as ET


# ============================================================================
# MASS Pipeline Configuration (from human.xml)
# ============================================================================

# These are the joints defined in human.xml that need BVH mapping
MASS_SKELETON_JOINTS = [
    'Pelvis', 'FemurR', 'TibiaR', 'TalusR', 'FootThumbR', 'FootPinkyR',
    'FemurL', 'TibiaL', 'TalusL', 'FootThumbL', 'FootPinkyL',
    'Spine', 'Torso', 'Neck', 'Head',
    'ShoulderR', 'ArmR', 'ForeArmR', 'HandR',
    'ShoulderL', 'ArmL', 'ForeArmL', 'HandL'
]

# BVH mappings currently in human.xml (MASS joint -> BVH joint name)
CURRENT_MASS_BVH_MAP = {
    'Pelvis': 'Character1_Hips',
    'FemurR': 'Character1_RightUpLeg',
    'TibiaR': 'Character1_RightLeg',
    'TalusR': 'Character1_RightFoot',
    'FemurL': 'Character1_LeftUpLeg',
    'TibiaL': 'Character1_LeftLeg',
    'TalusL': 'Character1_LeftFoot',
    'Spine': 'Character1_Spine',
    'Torso': 'Character1_Spine1',
    'Neck': 'Character1_Neck',
    'ShoulderR': 'Character1_RightShoulder',
    'ArmR': 'Character1_RightArm',
    'ForeArmR': 'Character1_RightForeArm',
    'HandR': 'Character1_RightHand',
    'ShoulderL': 'Character1_LeftShoulder',
    'ArmL': 'Character1_LeftArm',
    'ForeArmL': 'Character1_LeftForeArm',
    'HandL': 'Character1_LeftHand',
    # Note: FootThumb*, FootPinky*, Head don't have BVH mappings in human.xml
}

# Standard CMU skeleton joint mappings (multiple possible names)
CMU_JOINT_ALIASES = {
    'root': ['root', 'Hips', 'hip'],
    'lowerback': ['lowerback', 'LowerBack', 'Spine'],
    'upperback': ['upperback', 'UpperBack', 'Spine1'],
    'thorax': ['thorax', 'Thorax', 'Chest', 'Spine2'],
    'lowerneck': ['lowerneck', 'LowerNeck'],
    'upperneck': ['upperneck', 'UpperNeck'],
    'neck': ['Neck', 'Neck1'],
    'head': ['head', 'Head'],
    'rclavicle': ['rclavicle', 'RightCollar', 'RightShoulder'],
    'rhumerus': ['rhumerus', 'RightUpArm', 'RightArm'],
    'rradius': ['rradius', 'RightLowArm', 'RightForeArm'],
    'rwrist': ['rwrist', 'RightHand'],
    'rhand': ['rhand', 'RightHandEnd'],
    'rfingers': ['rfingers', 'RightFingers'],
    'rthumb': ['rthumb', 'RightThumb'],
    'lclavicle': ['lclavicle', 'LeftCollar', 'LeftShoulder'],
    'lhumerus': ['lhumerus', 'LeftUpArm', 'LeftArm'],
    'lradius': ['lradius', 'LeftLowArm', 'LeftForeArm'],
    'lwrist': ['lwrist', 'LeftHand'],
    'lhand': ['lhand', 'LeftHandEnd'],
    'lfingers': ['lfingers', 'LeftFingers'],
    'lthumb': ['lthumb', 'LeftThumb'],
    'rfemur': ['rfemur', 'RightUpLeg', 'RightHip'],
    'rtibia': ['rtibia', 'RightLowLeg', 'RightLeg'],
    'rfoot': ['rfoot', 'RightFoot'],
    'rtoes': ['rtoes', 'RightToe', 'RightToeBase'],
    'lfemur': ['lfemur', 'LeftUpLeg', 'LeftHip'],
    'ltibia': ['ltibia', 'LeftLowLeg', 'LeftLeg'],
    'lfoot': ['lfoot', 'LeftFoot'],
    'ltoes': ['ltoes', 'LeftToe', 'LeftToeBase'],
}

# Mapping from CMU standard name to MASS skeleton joint
CMU_TO_MASS_JOINT = {
    'root': 'Pelvis',
    'lowerback': 'Spine',
    'upperback': 'Torso',
    'thorax': 'Torso',  # MASS has 2-segment spine, CMU has 3
    'lowerneck': 'Neck',
    'upperneck': 'Neck',
    'neck': 'Neck',
    'head': 'Head',
    'rclavicle': 'ShoulderR',
    'rhumerus': 'ArmR',
    'rradius': 'ForeArmR',
    'rwrist': 'HandR',
    'rhand': 'HandR',
    'lclavicle': 'ShoulderL',
    'lhumerus': 'ArmL',
    'lradius': 'ForeArmL',
    'lwrist': 'HandL',
    'lhand': 'HandL',
    'rfemur': 'FemurR',
    'rtibia': 'TibiaR',
    'rfoot': 'TalusR',
    'rtoes': 'FootThumbR',  # Approximate
    'lfemur': 'FemurL',
    'ltibia': 'TibiaL',
    'lfoot': 'TalusL',
    'ltoes': 'FootThumbL',
}


@dataclass
class BVHJoint:
    """Represents a joint in BVH hierarchy"""
    name: str
    parent: Optional[str] = None
    offset: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    channels: List[str] = field(default_factory=list)
    channel_start: int = 0
    children: List[str] = field(default_factory=list)


@dataclass
class BVHData:
    """Complete BVH file data"""
    filepath: str
    root_name: str
    joints: Dict[str, BVHJoint]
    num_frames: int
    frame_time: float
    num_channels: int
    hierarchy_text: str
    motion_data: Optional[List[List[float]]] = None


class BVHParser:
    """Parse BVH files"""
    
    def parse(self, filepath: str, load_motion: bool = True) -> BVHData:
        """Parse a BVH file"""
        with open(filepath, 'r') as f:
            content = f.read()
        
        # Split hierarchy and motion
        upper_content = content.upper()
        motion_start = upper_content.find('MOTION')
        hierarchy_text = content[:motion_start].strip()
        motion_text = content[motion_start:].strip()
        
        # Parse hierarchy
        joints, root_name, num_channels = self._parse_hierarchy(hierarchy_text)
        
        # Parse motion
        num_frames, frame_time, motion_data = self._parse_motion(motion_text, load_motion)
        
        return BVHData(
            filepath=filepath,
            root_name=root_name,
            joints=joints,
            num_frames=num_frames,
            frame_time=frame_time,
            num_channels=num_channels,
            hierarchy_text=hierarchy_text,
            motion_data=motion_data
        )
    
    def _parse_hierarchy(self, text: str) -> Tuple[Dict[str, BVHJoint], str, int]:
        """Parse hierarchy section"""
        joints = {}
        root_name = ""
        channel_offset = 0
        
        lines = text.split('\n')
        stack = []  # Stack of parent joint names
        current_joint = None
        
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            if line.upper().startswith('ROOT'):
                name = line.split()[1]
                root_name = name
                current_joint = BVHJoint(name=name)
                joints[name] = current_joint
                stack.append(name)
            
            elif line.upper().startswith('JOINT'):
                name = line.split()[1]
                parent = stack[-1] if stack else None
                current_joint = BVHJoint(name=name, parent=parent)
                joints[name] = current_joint
                if parent:
                    joints[parent].children.append(name)
                stack.append(name)
            
            elif line.startswith('{'):
                pass
            
            elif line.startswith('}'):
                if stack:
                    stack.pop()
            
            elif line.upper().startswith('OFFSET'):
                parts = line.split()
                if current_joint and len(parts) >= 4:
                    current_joint.offset = (float(parts[1]), float(parts[2]), float(parts[3]))
            
            elif line.upper().startswith('CHANNELS'):
                parts = line.split()
                num_ch = int(parts[1])
                channels = parts[2:2+num_ch]
                if current_joint:
                    current_joint.channels = channels
                    current_joint.channel_start = channel_offset
                channel_offset += num_ch
            
            elif line.upper().startswith('END SITE'):
                # Skip end site block
                while i < len(lines) and '}' not in lines[i]:
                    i += 1
            
            i += 1
        
        return joints, root_name, channel_offset
    
    def _parse_motion(self, text: str, load_data: bool) -> Tuple[int, float, Optional[List]]:
        """Parse motion section"""
        lines = text.split('\n')
        num_frames = 0
        frame_time = 0.0
        motion_data = None
        
        data_start = 0
        for i, line in enumerate(lines):
            line = line.strip()
            if line.upper().startswith('FRAMES:'):
                num_frames = int(line.split(':')[1].strip())
            elif line.upper().startswith('FRAME TIME:'):
                frame_time = float(line.split(':')[1].strip())
                data_start = i + 1
                break
        
        if load_data:
            motion_data = []
            for line in lines[data_start:]:
                line = line.strip()
                if line:
                    values = [float(v) for v in line.split()]
                    motion_data.append(values)
        
        return num_frames, frame_time, motion_data


def identify_joint_type(joint_name: str) -> Tuple[Optional[str], str]:
    """
    Identify the CMU standard type of a joint.
    
    Returns:
        (cmu_standard_name, match_type) or (None, 'unknown')
    """
    name_lower = joint_name.lower()
    
    # Check exact matches and aliases
    for std_name, aliases in CMU_JOINT_ALIASES.items():
        for alias in aliases:
            if joint_name == alias or name_lower == alias.lower():
                return std_name, 'exact'
    
    # Check partial matches
    for std_name, aliases in CMU_JOINT_ALIASES.items():
        for alias in aliases:
            if alias.lower() in name_lower:
                return std_name, 'partial'
    
    return None, 'unknown'


def analyze_bvh(bvh: BVHData) -> Dict:
    """Analyze BVH file for MASS compatibility"""
    
    # Categorize joints
    functional_joints = []
    helper_joints = []
    unmapped_joints = []
    
    # Build mapping to MASS
    joint_to_mass = {}  # BVH joint -> MASS joint
    joint_type_info = {}  # BVH joint -> (cmu_type, match_type)
    
    for name, joint in bvh.joints.items():
        # Skip RIGMESH helpers
        if 'RIGMESH' in name.upper():
            helper_joints.append(name)
            continue
        
        functional_joints.append(name)
        
        # Try to identify joint type
        cmu_type, match_type = identify_joint_type(name)
        joint_type_info[name] = (cmu_type, match_type)
        
        if cmu_type and cmu_type in CMU_TO_MASS_JOINT:
            joint_to_mass[name] = CMU_TO_MASS_JOINT[cmu_type]
        else:
            unmapped_joints.append(name)
    
    # Check MASS coverage
    mass_coverage = {}
    for mass_joint in MASS_SKELETON_JOINTS:
        if mass_joint in CURRENT_MASS_BVH_MAP:
            required_bvh = CURRENT_MASS_BVH_MAP[mass_joint]
            # Check if this BVH joint exists or can be mapped
            found = False
            for bvh_name, mapped_mass in joint_to_mass.items():
                if mapped_mass == mass_joint:
                    mass_coverage[mass_joint] = ('mapped', bvh_name)
                    found = True
                    break
            if not found:
                # Check direct match
                if required_bvh in bvh.joints:
                    mass_coverage[mass_joint] = ('direct', required_bvh)
                else:
                    mass_coverage[mass_joint] = ('missing', required_bvh)
        else:
            mass_coverage[mass_joint] = ('no_bvh', None)
    
    return {
        'filepath': bvh.filepath,
        'root': bvh.root_name,
        'num_frames': bvh.num_frames,
        'frame_time': bvh.frame_time,
        'duration': bvh.num_frames * bvh.frame_time,
        'fps': 1.0 / bvh.frame_time if bvh.frame_time > 0 else 0,
        'num_channels': bvh.num_channels,
        'total_joints': len(bvh.joints),
        'functional_joints': functional_joints,
        'helper_joints': helper_joints,
        'unmapped_joints': unmapped_joints,
        'joint_to_mass': joint_to_mass,
        'joint_type_info': joint_type_info,
        'mass_coverage': mass_coverage,
        'joints': bvh.joints,
    }


def print_analysis(analysis: Dict, verbose: bool = True):
    """Print analysis results"""
    print("\n" + "=" * 70)
    print(f"BVH FILE ANALYSIS: {Path(analysis['filepath']).name}")
    print("=" * 70)
    
    print(f"\n📊 Basic Info:")
    print(f"   Root Joint: {analysis['root']}")
    print(f"   Total Joints: {analysis['total_joints']}")
    print(f"   Functional: {len(analysis['functional_joints'])}")
    print(f"   Helper (RIGMESH): {len(analysis['helper_joints'])}")
    print(f"   Frames: {analysis['num_frames']}")
    print(f"   Duration: {analysis['duration']:.2f}s @ {analysis['fps']:.1f} FPS")
    print(f"   Channels: {analysis['num_channels']}")
    
    print(f"\n🦴 Joint Hierarchy:")
    _print_hierarchy(analysis)
    
    print(f"\n🎯 MASS Pipeline Mapping:")
    mapped = 0
    missing = 0
    for mass_joint, (status, bvh_joint) in analysis['mass_coverage'].items():
        if status == 'direct':
            print(f"   ✓ {mass_joint:15s} -> {bvh_joint} (direct match)")
            mapped += 1
        elif status == 'mapped':
            print(f"   ✓ {mass_joint:15s} -> {bvh_joint} (via CMU mapping)")
            mapped += 1
        elif status == 'missing':
            print(f"   ✗ {mass_joint:15s} -> {bvh_joint} (NOT FOUND)")
            missing += 1
        else:
            print(f"   - {mass_joint:15s} (no BVH mapping in human.xml)")
    
    print(f"\n   Coverage: {mapped}/{mapped+missing} required joints")
    
    if analysis['unmapped_joints']:
        print(f"\n⚠️  Unmapped Joints ({len(analysis['unmapped_joints'])}):")
        for j in analysis['unmapped_joints']:
            print(f"   - {j}")
    
    if analysis['helper_joints'] and verbose:
        print(f"\n📦 Helper Joints (can be ignored):")
        for j in analysis['helper_joints']:
            print(f"   - {j}")


def _print_hierarchy(analysis: Dict, max_depth: int = 10):
    """Print joint hierarchy tree"""
    joints = analysis['joints']
    root = analysis['root']
    
    def print_joint(name: str, depth: int, prefix: str = ""):
        if depth > max_depth or name not in joints:
            return
        
        joint = joints[name]
        cmu_type, _ = analysis['joint_type_info'].get(name, (None, 'unknown'))
        mass_joint = analysis['joint_to_mass'].get(name, '?')
        
        ch_str = f"[{len(joint.channels)}ch]" if joint.channels else ""
        type_str = f"-> {mass_joint}" if mass_joint != '?' else ""
        
        print(f"   {prefix}{name} {ch_str} {type_str}")
        
        for i, child in enumerate(joint.children):
            is_last = (i == len(joint.children) - 1)
            child_prefix = prefix + ("    " if is_last else "│   ")
            child_marker = "└── " if is_last else "├── "
            
            if child in joints:
                cj = joints[child]
                cmu_type, _ = analysis['joint_type_info'].get(child, (None, 'unknown'))
                mass_joint = analysis['joint_to_mass'].get(child, '?')
                ch_str = f"[{len(cj.channels)}ch]" if cj.channels else ""
                type_str = f"-> {mass_joint}" if mass_joint != '?' else ""
                print(f"   {prefix}{child_marker}{child} {ch_str} {type_str}")
                
                # Recurse for children
                for j, grandchild in enumerate(cj.children):
                    _print_subtree(joints, grandchild, depth + 2, child_prefix, 
                                   j == len(cj.children) - 1, analysis)
    
    print_joint(root, 0)


def _print_subtree(joints: Dict, name: str, depth: int, prefix: str, is_last: bool, analysis: Dict):
    """Helper to print subtree"""
    if depth > 10 or name not in joints:
        return
    
    joint = joints[name]
    marker = "└── " if is_last else "├── "
    next_prefix = prefix + ("    " if is_last else "│   ")
    
    mass_joint = analysis['joint_to_mass'].get(name, '?')
    ch_str = f"[{len(joint.channels)}ch]" if joint.channels else ""
    type_str = f"-> {mass_joint}" if mass_joint != '?' else ""
    
    print(f"   {prefix}{marker}{name} {ch_str} {type_str}")
    
    for i, child in enumerate(joint.children):
        _print_subtree(joints, child, depth + 1, next_prefix, 
                      i == len(joint.children) - 1, analysis)


def compare_bvh_files(base_analysis: Dict, target_analysis: Dict) -> Dict:
    """Compare two BVH files"""
    base_functional = set(base_analysis['functional_joints'])
    target_functional = set(target_analysis['functional_joints'])
    
    # Joints unique to each
    base_only = base_functional - target_functional
    target_only = target_functional - base_functional
    common = base_functional & target_functional
    
    # Compare MASS coverage
    base_coverage = set()
    target_coverage = set()
    
    for mass_joint, (status, _) in base_analysis['mass_coverage'].items():
        if status in ('direct', 'mapped'):
            base_coverage.add(mass_joint)
    
    for mass_joint, (status, _) in target_analysis['mass_coverage'].items():
        if status in ('direct', 'mapped'):
            target_coverage.add(mass_joint)
    
    return {
        'base': base_analysis,
        'target': target_analysis,
        'base_only': list(base_only),
        'target_only': list(target_only),
        'common': list(common),
        'base_mass_coverage': list(base_coverage),
        'target_mass_coverage': list(target_coverage),
        'coverage_diff': list(base_coverage - target_coverage),
    }


def print_comparison(comparison: Dict):
    """Print comparison results"""
    print("\n" + "=" * 70)
    print("BVH FILE COMPARISON")
    print("=" * 70)
    
    base_name = Path(comparison['base']['filepath']).name
    target_name = Path(comparison['target']['filepath']).name
    
    print(f"\n📁 Base:   {base_name}")
    print(f"📁 Target: {target_name}")
    
    print(f"\n📊 Joint Count Comparison:")
    print(f"   {'':20s} {'Base':>10s} {'Target':>10s}")
    print(f"   {'-'*42}")
    print(f"   {'Total':20s} {comparison['base']['total_joints']:>10d} {comparison['target']['total_joints']:>10d}")
    print(f"   {'Functional':20s} {len(comparison['base']['functional_joints']):>10d} {len(comparison['target']['functional_joints']):>10d}")
    print(f"   {'Helper (RIGMESH)':20s} {len(comparison['base']['helper_joints']):>10d} {len(comparison['target']['helper_joints']):>10d}")
    
    print(f"\n🔍 Joint Differences:")
    
    if comparison['common']:
        print(f"\n   Common joints ({len(comparison['common'])}):")
        for j in sorted(comparison['common'])[:10]:
            print(f"     - {j}")
        if len(comparison['common']) > 10:
            print(f"     ... and {len(comparison['common']) - 10} more")
    
    if comparison['base_only']:
        print(f"\n   Only in BASE ({len(comparison['base_only'])}):")
        for j in sorted(comparison['base_only']):
            print(f"     - {j}")
    
    if comparison['target_only']:
        print(f"\n   Only in TARGET ({len(comparison['target_only'])}):")
        for j in sorted(comparison['target_only']):
            print(f"     - {j}")
    
    print(f"\n🎯 MASS Coverage Comparison:")
    print(f"   Base coverage:   {len(comparison['base_mass_coverage'])}/{len(MASS_SKELETON_JOINTS)}")
    print(f"   Target coverage: {len(comparison['target_mass_coverage'])}/{len(MASS_SKELETON_JOINTS)}")
    
    if comparison['coverage_diff']:
        print(f"\n   Joints missing in TARGET but present in BASE:")
        for j in sorted(comparison['coverage_diff']):
            print(f"     - {j}")
    
    # Compatibility assessment
    print(f"\n" + "=" * 70)
    target_missing = len(comparison['coverage_diff'])
    if target_missing == 0:
        print("✅ TARGET FILE IS FULLY COMPATIBLE")
    elif target_missing <= 2:
        print(f"⚠️  TARGET FILE IS MOSTLY COMPATIBLE ({target_missing} joints need handling)")
    else:
        print(f"❌ TARGET FILE HAS COMPATIBILITY ISSUES ({target_missing} missing joints)")
    print("=" * 70)


def generate_xml_mapping(analysis: Dict) -> str:
    """Generate XML snippet for human.xml BVH mappings"""
    lines = [
        "<!-- BVH Mappings for: {} -->".format(Path(analysis['filepath']).name),
        "<!-- Add these 'bvh' attributes to Joint elements in human.xml -->\n"
    ]
    
    for bvh_joint, mass_joint in analysis['joint_to_mass'].items():
        lines.append(f'<!-- MASS: {mass_joint} -->')
        lines.append(f'<Joint ... bvh="{bvh_joint}" ...>')
        lines.append("")
    
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description='MASS BVH Pipeline Compatibility Tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Analyze a single BVH file
    python mass_bvh_tool.py analyze 29_01.bvh
    
    # Compare walk.bvh (base) with 29_01.bvh (target)
    python mass_bvh_tool.py compare walk.bvh 29_01.bvh
    
    # Generate human.xml mappings
    python mass_bvh_tool.py generate-xml 29_01.bvh
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze a BVH file')
    analyze_parser.add_argument('file', help='BVH file to analyze')
    analyze_parser.add_argument('-v', '--verbose', action='store_true', help='Show more details')
    
    # Compare command
    compare_parser = subparsers.add_parser('compare', help='Compare two BVH files')
    compare_parser.add_argument('base', help='Base/reference BVH file (e.g., walk.bvh)')
    compare_parser.add_argument('target', help='Target BVH file to check (e.g., 29_01.bvh)')
    
    # Generate XML command
    xml_parser = subparsers.add_parser('generate-xml', help='Generate human.xml mapping snippet')
    xml_parser.add_argument('file', help='BVH file to generate mappings for')
    xml_parser.add_argument('-o', '--output', help='Output file (default: stdout)')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    bvh_parser = BVHParser()
    
    if args.command == 'analyze':
        bvh = bvh_parser.parse(args.file)
        analysis = analyze_bvh(bvh)
        print_analysis(analysis, verbose=args.verbose)
    
    elif args.command == 'compare':
        base_bvh = bvh_parser.parse(args.base)
        target_bvh = bvh_parser.parse(args.target)
        
        base_analysis = analyze_bvh(base_bvh)
        target_analysis = analyze_bvh(target_bvh)
        
        comparison = compare_bvh_files(base_analysis, target_analysis)
        print_comparison(comparison)
    
    elif args.command == 'generate-xml':
        bvh = bvh_parser.parse(args.file)
        analysis = analyze_bvh(bvh)
        xml_snippet = generate_xml_mapping(analysis)
        
        if args.output:
            with open(args.output, 'w') as f:
                f.write(xml_snippet)
            print(f"XML mapping saved to: {args.output}")
        else:
            print(xml_snippet)


if __name__ == "__main__":
    main()