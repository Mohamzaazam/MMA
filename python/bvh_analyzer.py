#!/usr/bin/env python3
"""
BVH Compatibility Analyzer for MASS Pipeline

This tool analyzes BVH files and checks compatibility with the MASS
musculoskeletal simulation pipeline.

Key features:
1. Parse BVH hierarchy and extract joint structure
2. Compare two BVH files for compatibility
3. Generate retargeting configuration
4. Identify missing/extra joints
5. Check channel configurations

Based on the MASS pipeline requirements from human.xml
"""

import re
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set
from pathlib import Path


@dataclass
class BVHJoint:
    """Represents a joint in the BVH hierarchy"""
    name: str
    parent: Optional[str]
    offset: Tuple[float, float, float]
    channels: List[str]
    channel_offset: int
    children: List[str] = field(default_factory=list)
    is_end_site: bool = False


@dataclass 
class BVHFile:
    """Parsed BVH file data"""
    filepath: str
    joints: Dict[str, BVHJoint]
    root_name: str
    num_frames: int
    frame_time: float
    num_channels: int
    motion_data: Optional[List[List[float]]] = None


class BVHParser:
    """Parser for BVH files"""
    
    def __init__(self):
        self.joints: Dict[str, BVHJoint] = {}
        self.root_name: str = ""
        self.channel_offset: int = 0
        
    def parse(self, filepath: str, load_motion: bool = False) -> BVHFile:
        """Parse a BVH file and return structured data"""
        self.joints = {}
        self.channel_offset = 0
        
        with open(filepath, 'r') as f:
            content = f.read()
        
        lines = content.split('\n')
        line_idx = 0
        
        # Parse HIERARCHY
        while line_idx < len(lines):
            line = lines[line_idx].strip()
            if line == "HIERARCHY":
                line_idx += 1
                break
            line_idx += 1
        
        # Parse root
        while line_idx < len(lines):
            line = lines[line_idx].strip()
            if line.startswith("ROOT"):
                self.root_name = line.split()[1]
                line_idx = self._parse_joint(lines, line_idx + 1, self.root_name, None)
                break
            line_idx += 1
        
        # Parse MOTION section
        num_frames = 0
        frame_time = 0.0
        motion_data = None
        
        while line_idx < len(lines):
            line = lines[line_idx].strip()
            if line == "MOTION":
                line_idx += 1
                break
            line_idx += 1
        
        while line_idx < len(lines):
            line = lines[line_idx].strip()
            if line.startswith("Frames:"):
                num_frames = int(line.split(":")[1].strip())
            elif line.startswith("Frame Time:"):
                frame_time = float(line.split(":")[1].strip())
                line_idx += 1
                break
            line_idx += 1
        
        # Optionally load motion data
        if load_motion:
            motion_data = []
            while line_idx < len(lines):
                line = lines[line_idx].strip()
                if line:
                    values = [float(v) for v in line.split()]
                    motion_data.append(values)
                line_idx += 1
        
        return BVHFile(
            filepath=filepath,
            joints=self.joints,
            root_name=self.root_name,
            num_frames=num_frames,
            frame_time=frame_time,
            num_channels=self.channel_offset,
            motion_data=motion_data
        )
    
    def _parse_joint(self, lines: List[str], line_idx: int, 
                     name: str, parent: Optional[str]) -> int:
        """Parse a joint and its children recursively"""
        offset = (0.0, 0.0, 0.0)
        channels = []
        channel_start = self.channel_offset
        children = []
        is_end_site = name == "End Site" or name.startswith("End")
        
        # Skip opening brace
        while line_idx < len(lines):
            line = lines[line_idx].strip()
            if line == "{":
                line_idx += 1
                break
            line_idx += 1
        
        while line_idx < len(lines):
            line = lines[line_idx].strip()
            
            if line == "}":
                break
            elif line.startswith("OFFSET"):
                parts = line.split()
                offset = (float(parts[1]), float(parts[2]), float(parts[3]))
            elif line.startswith("CHANNELS"):
                parts = line.split()
                num_channels = int(parts[1])
                channels = parts[2:2+num_channels]
                self.channel_offset += num_channels
            elif line.startswith("JOINT"):
                child_name = line.split()[1]
                children.append(child_name)
                line_idx = self._parse_joint(lines, line_idx + 1, child_name, name)
            elif line.startswith("End Site"):
                end_name = f"{name}_End"
                line_idx = self._parse_joint(lines, line_idx + 1, end_name, name)
            
            line_idx += 1
        
        # Store joint
        if not is_end_site:
            self.joints[name] = BVHJoint(
                name=name,
                parent=parent,
                offset=offset,
                channels=channels,
                channel_offset=channel_start,
                children=children,
                is_end_site=is_end_site
            )
        
        return line_idx


class MASSCompatibilityChecker:
    """Check BVH compatibility with MASS pipeline"""
    
    # Required joints in MASS pipeline (from human.xml bvh mappings)
    MASS_BVH_MAPPING = {
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
    }
    
    # CMU joint name patterns (common naming)
    CMU_JOINT_PATTERNS = {
        'root': ['root', 'Hips', 'hip'],
        'lowerback': ['lowerback', 'LowerBack', 'Spine'],
        'upperback': ['upperback', 'UpperBack', 'Spine1'],
        'thorax': ['thorax', 'Thorax', 'Chest', 'Spine2'],
        'lowerneck': ['lowerneck', 'LowerNeck', 'Neck'],
        'upperneck': ['upperneck', 'UpperNeck', 'Neck1'],
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
    
    # CMU to MASS Character1_* mapping
    CMU_TO_MASS_MAPPING = {
        'root': 'Character1_Hips',
        'Hips': 'Character1_Hips',
        'hip': 'Character1_Hips',
        'lowerback': 'Character1_Spine',
        'LowerBack': 'Character1_Spine',
        'Spine': 'Character1_Spine',
        'upperback': 'Character1_Spine1',
        'UpperBack': 'Character1_Spine1',
        'Spine1': 'Character1_Spine1',
        'thorax': 'Character1_Spine1',  # Map to Spine1 since MASS doesn't have chest
        'Thorax': 'Character1_Spine1',
        'Chest': 'Character1_Spine1',
        'lowerneck': 'Character1_Neck',
        'LowerNeck': 'Character1_Neck',
        'Neck': 'Character1_Neck',
        'upperneck': 'Character1_Neck',  # Merge upper/lower neck
        'UpperNeck': 'Character1_Neck',
        'Neck1': 'Character1_Neck',
        'head': 'Character1_Head',
        'Head': 'Character1_Head',
        'rclavicle': 'Character1_RightShoulder',
        'RightCollar': 'Character1_RightShoulder',
        'RightShoulder': 'Character1_RightShoulder',
        'rhumerus': 'Character1_RightArm',
        'RightUpArm': 'Character1_RightArm',
        'RightArm': 'Character1_RightArm',
        'rradius': 'Character1_RightForeArm',
        'RightLowArm': 'Character1_RightForeArm',
        'RightForeArm': 'Character1_RightForeArm',
        'rwrist': 'Character1_RightHand',
        'RightHand': 'Character1_RightHand',
        'rhand': 'Character1_RightHand',
        'lclavicle': 'Character1_LeftShoulder',
        'LeftCollar': 'Character1_LeftShoulder',
        'LeftShoulder': 'Character1_LeftShoulder',
        'lhumerus': 'Character1_LeftArm',
        'LeftUpArm': 'Character1_LeftArm',
        'LeftArm': 'Character1_LeftArm',
        'lradius': 'Character1_LeftForeArm',
        'LeftLowArm': 'Character1_LeftForeArm',
        'LeftForeArm': 'Character1_LeftForeArm',
        'lwrist': 'Character1_LeftHand',
        'LeftHand': 'Character1_LeftHand',
        'lhand': 'Character1_LeftHand',
        'rfemur': 'Character1_RightUpLeg',
        'RightUpLeg': 'Character1_RightUpLeg',
        'RightHip': 'Character1_RightUpLeg',
        'rtibia': 'Character1_RightLeg',
        'RightLowLeg': 'Character1_RightLeg',
        'RightLeg': 'Character1_RightLeg',
        'rfoot': 'Character1_RightFoot',
        'RightFoot': 'Character1_RightFoot',
        'rtoes': 'Character1_RightToeBase',
        'RightToe': 'Character1_RightToeBase',
        'RightToeBase': 'Character1_RightToeBase',
        'lfemur': 'Character1_LeftUpLeg',
        'LeftUpLeg': 'Character1_LeftUpLeg',
        'LeftHip': 'Character1_LeftUpLeg',
        'ltibia': 'Character1_LeftLeg',
        'LeftLowLeg': 'Character1_LeftLeg',
        'LeftLeg': 'Character1_LeftLeg',
        'lfoot': 'Character1_LeftFoot',
        'LeftFoot': 'Character1_LeftFoot',
        'ltoes': 'Character1_LeftToeBase',
        'LeftToe': 'Character1_LeftToeBase',
        'LeftToeBase': 'Character1_LeftToeBase',
    }
    
    def __init__(self):
        self.parser = BVHParser()
    
    def analyze_bvh(self, filepath: str) -> Dict:
        """Analyze a single BVH file"""
        bvh = self.parser.parse(filepath)
        
        # Identify joint types
        functional_joints = []
        helper_joints = []
        end_sites = []
        
        for name, joint in bvh.joints.items():
            if 'RIGMESH' in name or 'rigmesh' in name.lower():
                helper_joints.append(name)
            elif joint.is_end_site or name.endswith('_End'):
                end_sites.append(name)
            else:
                functional_joints.append(name)
        
        # Build hierarchy tree
        hierarchy = self._build_hierarchy_string(bvh)
        
        # Check for MASS compatibility
        mass_compatible_joints = []
        unmapped_joints = []
        
        for joint_name in functional_joints:
            if joint_name in self.CMU_TO_MASS_MAPPING:
                mass_compatible_joints.append((joint_name, self.CMU_TO_MASS_MAPPING[joint_name]))
            elif joint_name.startswith('Character1_'):
                # Already in MASS format
                mass_compatible_joints.append((joint_name, joint_name))
            else:
                unmapped_joints.append(joint_name)
        
        return {
            'filepath': filepath,
            'root': bvh.root_name,
            'num_frames': bvh.num_frames,
            'frame_time': bvh.frame_time,
            'duration': bvh.num_frames * bvh.frame_time,
            'fps': 1.0 / bvh.frame_time if bvh.frame_time > 0 else 0,
            'total_joints': len(bvh.joints),
            'functional_joints': functional_joints,
            'helper_joints': helper_joints,
            'end_sites': end_sites,
            'num_channels': bvh.num_channels,
            'hierarchy': hierarchy,
            'mass_compatible': mass_compatible_joints,
            'unmapped': unmapped_joints,
            'joints_detail': bvh.joints
        }
    
    def _build_hierarchy_string(self, bvh: BVHFile, indent: int = 0) -> str:
        """Build a visual hierarchy string"""
        lines = []
        
        def add_joint(name: str, level: int):
            if name not in bvh.joints:
                return
            joint = bvh.joints[name]
            prefix = "  " * level + ("└── " if level > 0 else "")
            channels = ", ".join(joint.channels) if joint.channels else "none"
            lines.append(f"{prefix}{name} [{channels}]")
            for child in joint.children:
                add_joint(child, level + 1)
        
        add_joint(bvh.root_name, 0)
        return "\n".join(lines)
    
    def compare_bvh_files(self, base_path: str, target_path: str) -> Dict:
        """Compare two BVH files for compatibility"""
        base = self.analyze_bvh(base_path)
        target = self.analyze_bvh(target_path)
        
        base_functional = set(base['functional_joints'])
        target_functional = set(target['functional_joints'])
        
        # Find mappings between the two
        mapping_suggestions = {}
        
        # If base uses Character1_* and target uses CMU names
        if any(j.startswith('Character1_') for j in base_functional):
            # Base is MASS format, target needs mapping
            for target_joint in target_functional:
                if target_joint in self.CMU_TO_MASS_MAPPING:
                    mass_name = self.CMU_TO_MASS_MAPPING[target_joint]
                    if mass_name in base_functional:
                        mapping_suggestions[target_joint] = mass_name
        
        # Check which MASS joints are covered
        required_bvh_names = set(self.MASS_BVH_MAPPING.values())
        covered = set()
        missing = set()
        
        for name in required_bvh_names:
            if name in base_functional:
                covered.add(name)
            else:
                missing.add(name)
        
        # Check target coverage
        target_covers = set()
        target_missing = set()
        
        for mass_name in required_bvh_names:
            found = False
            # Direct match
            if mass_name in target_functional:
                target_covers.add(mass_name)
                found = True
            else:
                # Check via mapping
                for target_joint in target_functional:
                    if target_joint in self.CMU_TO_MASS_MAPPING:
                        if self.CMU_TO_MASS_MAPPING[target_joint] == mass_name:
                            target_covers.add(mass_name)
                            found = True
                            break
            if not found:
                target_missing.add(mass_name)
        
        return {
            'base_analysis': base,
            'target_analysis': target,
            'base_joints_only': list(base_functional - target_functional),
            'target_joints_only': list(target_functional - base_functional),
            'common_joints': list(base_functional & target_functional),
            'mapping_suggestions': mapping_suggestions,
            'mass_required_joints': list(required_bvh_names),
            'base_coverage': list(covered),
            'base_missing': list(missing),
            'target_coverage': list(target_covers),
            'target_missing': list(target_missing),
            'compatible': len(target_missing) == 0 or len(target_missing) <= 2  # Allow some missing
        }
    
    def generate_retargeting_config(self, source_bvh: str, 
                                    output_path: Optional[str] = None) -> str:
        """Generate a retargeting configuration for CMU to MASS"""
        analysis = self.analyze_bvh(source_bvh)
        
        config_lines = [
            "# BVH Retargeting Configuration",
            f"# Source: {source_bvh}",
            f"# Generated for MASS pipeline compatibility",
            "",
            "# Joint Mapping (Source -> MASS)",
            "# Format: source_joint_name = mass_joint_name",
            ""
        ]
        
        # Generate mapping
        for joint_name in analysis['functional_joints']:
            if joint_name in self.CMU_TO_MASS_MAPPING:
                mass_name = self.CMU_TO_MASS_MAPPING[joint_name]
                config_lines.append(f"{joint_name} = {mass_name}")
            else:
                config_lines.append(f"# {joint_name} = ???  # UNMAPPED")
        
        config_content = "\n".join(config_lines)
        
        if output_path:
            with open(output_path, 'w') as f:
                f.write(config_content)
        
        return config_content


def print_analysis(analysis: Dict):
    """Pretty print BVH analysis"""
    print("=" * 70)
    print(f"BVH Analysis: {analysis['filepath']}")
    print("=" * 70)
    print(f"Root Joint: {analysis['root']}")
    print(f"Frames: {analysis['num_frames']}")
    print(f"Frame Time: {analysis['frame_time']:.6f}s")
    print(f"Duration: {analysis['duration']:.2f}s")
    print(f"FPS: {analysis['fps']:.1f}")
    print(f"Total Channels: {analysis['num_channels']}")
    print()
    print(f"Joint Counts:")
    print(f"  Total: {analysis['total_joints']}")
    print(f"  Functional: {len(analysis['functional_joints'])}")
    print(f"  Helper/RIGMESH: {len(analysis['helper_joints'])}")
    print()
    print("Hierarchy:")
    print(analysis['hierarchy'])
    print()
    print("Functional Joints:")
    for joint in analysis['functional_joints']:
        print(f"  - {joint}")
    print()
    if analysis['helper_joints']:
        print("Helper Joints (can be ignored):")
        for joint in analysis['helper_joints']:
            print(f"  - {joint}")
        print()
    if analysis['unmapped']:
        print("Unmapped Joints (need manual mapping):")
        for joint in analysis['unmapped']:
            print(f"  - {joint}")
    print()


def print_comparison(comparison: Dict):
    """Pretty print comparison results"""
    print("=" * 70)
    print("BVH COMPARISON RESULTS")
    print("=" * 70)
    print()
    print(f"Base File: {comparison['base_analysis']['filepath']}")
    print(f"Target File: {comparison['target_analysis']['filepath']}")
    print()
    
    print("JOINT DIFFERENCES:")
    print("-" * 40)
    
    if comparison['base_joints_only']:
        print(f"\nJoints ONLY in BASE ({len(comparison['base_joints_only'])}):")
        for j in sorted(comparison['base_joints_only']):
            print(f"  - {j}")
    
    if comparison['target_joints_only']:
        print(f"\nJoints ONLY in TARGET ({len(comparison['target_joints_only'])}):")
        for j in sorted(comparison['target_joints_only']):
            print(f"  - {j}")
    
    print(f"\nCommon Joints ({len(comparison['common_joints'])}):")
    for j in sorted(comparison['common_joints']):
        print(f"  - {j}")
    
    print()
    print("MASS PIPELINE COMPATIBILITY:")
    print("-" * 40)
    print(f"\nRequired MASS joints: {len(comparison['mass_required_joints'])}")
    
    print(f"\nBase file coverage: {len(comparison['base_coverage'])}/{len(comparison['mass_required_joints'])}")
    if comparison['base_missing']:
        print(f"  Missing: {', '.join(comparison['base_missing'])}")
    
    print(f"\nTarget file coverage: {len(comparison['target_coverage'])}/{len(comparison['mass_required_joints'])}")
    if comparison['target_missing']:
        print(f"  Missing: {', '.join(comparison['target_missing'])}")
    
    print()
    if comparison['mapping_suggestions']:
        print("SUGGESTED MAPPINGS (Target -> Base):")
        print("-" * 40)
        for src, dst in sorted(comparison['mapping_suggestions'].items()):
            print(f"  {src} -> {dst}")
    
    print()
    print("=" * 70)
    if comparison['compatible']:
        print("✓ TARGET FILE IS COMPATIBLE (with retargeting)")
    else:
        print("✗ TARGET FILE MAY NOT BE FULLY COMPATIBLE")
        print("  Missing critical joints for MASS simulation")
    print("=" * 70)


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='BVH Compatibility Analyzer for MASS')
    parser.add_argument('files', nargs='+', help='BVH file(s) to analyze')
    parser.add_argument('--compare', action='store_true', 
                       help='Compare two files (first is base, second is target)')
    parser.add_argument('--retarget-config', metavar='OUTPUT',
                       help='Generate retargeting config file')
    
    args = parser.parse_args()
    
    checker = MASSCompatibilityChecker()
    
    if args.compare and len(args.files) >= 2:
        comparison = checker.compare_bvh_files(args.files[0], args.files[1])
        print_comparison(comparison)
    else:
        for filepath in args.files:
            analysis = checker.analyze_bvh(filepath)
            print_analysis(analysis)
            
            if args.retarget_config:
                config = checker.generate_retargeting_config(filepath, args.retarget_config)
                print(f"\nRetargeting config saved to: {args.retarget_config}")


if __name__ == "__main__":
    main()