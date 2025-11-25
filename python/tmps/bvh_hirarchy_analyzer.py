#!/usr/bin/env python3
"""
BVH Hierarchy Analyzer
Analyzes different BVH file structures and creates a common skeleton mapping
"""

import re
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import json


class BVHJoint:
    """Represents a joint in BVH hierarchy"""

    def __init__(self, name: str, parent: Optional['BVHJoint'] = None):
        self.name = name
        self.parent = parent
        self.children: List['BVHJoint'] = []
        self.offset = [0.0, 0.0, 0.0]
        self.channels = []
        self.depth = 0 if parent is None else parent.depth + 1

    def add_child(self, child: 'BVHJoint'):
        self.children.append(child)

    def get_path(self) -> str:
        """Get the path from root to this joint"""
        path = []
        current = self
        while current:
            path.insert(0, current.name)
            current = current.parent
        return " -> ".join(path)


class BVHParser:
    """Parse BVH file and extract hierarchy"""

    def __init__(self, filepath: str):
        self.filepath = filepath
        self.root = None
        self.joints_dict = {}
        self.total_channels = 0
        self.frame_count = 0
        self.frame_time = 0.0

    def parse(self):
        """Parse the BVH file"""
        with open(self.filepath, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()

        # Split into hierarchy and motion sections
        parts = re.split(r'\bMOTION\b', content, flags=re.IGNORECASE)
        hierarchy_section = parts[0]

        if len(parts) > 1:
            motion_section = parts[1]
            self._parse_motion_header(motion_section)

        self._parse_hierarchy(hierarchy_section)

    def _parse_motion_header(self, motion_section: str):
        """Parse motion section header"""
        frames_match = re.search(r'Frames:\s*(\d+)', motion_section, re.IGNORECASE)
        if frames_match:
            self.frame_count = int(frames_match.group(1))

        frame_time_match = re.search(r'Frame Time:\s*([\d.]+)', motion_section, re.IGNORECASE)
        if frame_time_match:
            self.frame_time = float(frame_time_match.group(1))

    def _parse_hierarchy(self, hierarchy_section: str):
        """Parse the hierarchy section"""
        lines = hierarchy_section.split('\n')

        stack = []
        current_joint = None
        in_end_site = False

        for line in lines:
            line = line.strip().rstrip('\r')

            # ROOT joint
            root_match = re.match(r'ROOT\s+(\S+)', line, re.IGNORECASE)
            if root_match:
                joint_name = root_match.group(1)
                current_joint = BVHJoint(joint_name, parent=None)
                self.root = current_joint
                self.joints_dict[joint_name] = current_joint
                stack.append(current_joint)
                continue

            # Regular JOINT
            joint_match = re.match(r'JOINT\s+(\S+)', line, re.IGNORECASE)
            if joint_match:
                joint_name = joint_match.group(1)
                parent = stack[-1] if stack else None
                current_joint = BVHJoint(joint_name, parent=parent)
                if parent:
                    parent.add_child(current_joint)
                self.joints_dict[joint_name] = current_joint
                stack.append(current_joint)
                continue

            # End Site
            if re.match(r'End\s+Site', line, re.IGNORECASE):
                in_end_site = True
                # Create a pseudo joint for end site
                parent = stack[-1] if stack else None
                end_site_name = f"{parent.name}_End" if parent else "End"
                current_joint = BVHJoint(end_site_name, parent=parent)
                if parent:
                    parent.add_child(current_joint)
                stack.append(current_joint)
                continue

            # OFFSET
            offset_match = re.match(r'OFFSET\s+([-\d.e]+)\s+([-\d.e]+)\s+([-\d.e]+)', line, re.IGNORECASE)
            if offset_match and current_joint:
                current_joint.offset = [
                    float(offset_match.group(1)),
                    float(offset_match.group(2)),
                    float(offset_match.group(3))
                ]
                continue

            # CHANNELS
            channels_match = re.match(r'CHANNELS\s+(\d+)\s+(.+)', line, re.IGNORECASE)
            if channels_match and current_joint:
                num_channels = int(channels_match.group(1))
                channel_names = channels_match.group(2).strip().split()
                current_joint.channels = channel_names[:num_channels]
                self.total_channels += num_channels
                continue

            # Closing brace
            if line == '}':
                if stack:
                    stack.pop()
                if in_end_site:
                    in_end_site = False
                continue

    def get_joint_list(self) -> List[BVHJoint]:
        """Get flat list of all joints"""
        joints = []

        def traverse(joint):
            joints.append(joint)
            for child in joint.children:
                traverse(child)

        if self.root:
            traverse(self.root)

        return joints

    def print_hierarchy(self):
        """Print the hierarchy tree"""

        def print_joint(joint, indent=0):
            channel_info = f" [{len(joint.channels)} channels]" if joint.channels else ""
            print("  " * indent + f"- {joint.name}{channel_info}")
            for child in joint.children:
                print_joint(child, indent + 1)

        if self.root:
            print(f"\nHierarchy for: {self.filepath}")
            print(f"Total joints: {len(self.joints_dict)}")
            print(f"Total channels: {self.total_channels}")
            print(f"Frames: {self.frame_count}, Frame time: {self.frame_time}s")
            print("\nJoint tree:")
            print_joint(self.root)


class SkeletonMapper:
    """Map different BVH skeletons to a common hierarchy"""

    # Standard bone name mappings (various naming conventions -> standard name)
    BONE_MAPPINGS = {
        # Root/Hips
        'hips': 'hips',
        'character1_hips': 'hips',
        'root': 'hips',
        'reference': 'reference_frame',

        # Spine
        'spine': 'spine',
        'spine1': 'spine1',
        'character1_spine': 'spine',
        'character1_spine1': 'spine1',
        'lowerback': 'lower_back',
        'upperback': 'upper_back',
        'thorax': 'chest',

        # Neck/Head
        'neck': 'neck',
        'neck1': 'neck1',
        'lowerneck': 'lower_neck',
        'upperneck': 'upper_neck',
        'head': 'head',

        # Left Leg
        'lhipjoint': 'left_hip',
        'leftupleg': 'left_upper_leg',
        'lfemur': 'left_upper_leg',
        'leftleg': 'left_lower_leg',
        'ltibia': 'left_lower_leg',
        'leftfoot': 'left_foot',
        'lfoot': 'left_foot',
        'lefttoebase': 'left_toes',
        'ltoes': 'left_toes',

        # Right Leg
        'rhipjoint': 'right_hip',
        'rightupleg': 'right_upper_leg',
        'rfemur': 'right_upper_leg',
        'rightleg': 'right_lower_leg',
        'rtibia': 'right_lower_leg',
        'rightfoot': 'right_foot',
        'rfoot': 'right_foot',
        'righttoebase': 'right_toes',
        'rtoes': 'right_toes',

        # Left Arm
        'lclavicle': 'left_clavicle',
        'leftshoulder': 'left_shoulder',
        'character1_leftshoulder': 'left_shoulder',
        'leftarm': 'left_upper_arm',
        'character1_leftarm': 'left_upper_arm',
        'lhumerus': 'left_upper_arm',
        'leftforearm': 'left_lower_arm',
        'character1_leftforearm': 'left_lower_arm',
        'lradius': 'left_lower_arm',
        'lefthand': 'left_hand',
        'character1_lefthand': 'left_hand',
        'lhand': 'left_hand',

        # Right Arm
        'rclavicle': 'right_clavicle',
        'rightshoulder': 'right_shoulder',
        'character1_rightshoulder': 'right_shoulder',
        'rightarm': 'right_upper_arm',
        'character1_rightarm': 'right_upper_arm',
        'rhumerus': 'right_upper_arm',
        'rightforearm': 'right_lower_arm',
        'character1_rightforearm': 'right_lower_arm',
        'rradius': 'right_lower_arm',
        'righthand': 'right_hand',
        'character1_righthand': 'right_hand',
        'rhand': 'right_hand',

        # Fingers (left)
        'lthumb': 'left_thumb',
        'lfingers': 'left_fingers',

        # Fingers (right)
        'rthumb': 'right_thumb',
        'rfingers': 'right_fingers',
    }

    def __init__(self):
        self.parsers: Dict[str, BVHParser] = {}
        self.mappings: Dict[str, Dict[str, str]] = {}

    def add_bvh(self, name: str, parser: BVHParser):
        """Add a BVH parser to the mapper"""
        self.parsers[name] = parser
        self._create_mapping(name, parser)

    def _create_mapping(self, name: str, parser: BVHParser):
        """Create mapping from BVH joints to standard names"""
        mapping = {}

        for joint_name, joint in parser.joints_dict.items():
            # Normalize joint name
            normalized = joint_name.lower().replace('_', '').replace('-', '')

            # Look for standard name
            if normalized in self.BONE_MAPPINGS:
                standard_name = self.BONE_MAPPINGS[normalized]
                mapping[joint_name] = standard_name
            else:
                # Keep original if no mapping found
                mapping[joint_name] = joint_name.lower()

        self.mappings[name] = mapping

    def get_common_joints(self) -> List[str]:
        """Get joints that appear in all BVH files"""
        if not self.mappings:
            return []

        # Get standard names from each file
        standard_sets = []
        for name, mapping in self.mappings.items():
            standard_sets.append(set(mapping.values()))

        # Find intersection
        common = set.intersection(*standard_sets)
        return sorted(list(common))

    def print_comparison(self):
        """Print comparison of all BVH hierarchies"""
        print("\n" + "=" * 80)
        print("SKELETON COMPARISON")
        print("=" * 80)

        # Print individual mappings
        for name, parser in self.parsers.items():
            print(f"\n{name.upper()}:")
            print(f"  Total joints: {len(parser.joints_dict)}")
            print(f"  Total channels: {parser.total_channels}")
            print(f"  Root: {parser.root.name if parser.root else 'None'}")

            print(f"\n  Joint mapping:")
            mapping = self.mappings[name]
            for orig, standard in sorted(mapping.items()):
                print(f"    {orig:30} -> {standard}")

        # Print common joints
        common = self.get_common_joints()
        print(f"\n{'=' * 80}")
        print(f"COMMON JOINTS ({len(common)} total):")
        print(f"{'=' * 80}")
        for joint in common:
            print(f"  - {joint}")

    def export_mapping_config(self, output_file: str):
        """Export mapping configuration as JSON"""
        config = {
            'files': {},
            'common_joints': self.get_common_joints(),
            'standard_hierarchy': self._build_standard_hierarchy()
        }

        for name, parser in self.parsers.items():
            config['files'][name] = {
                'original_file': parser.filepath,
                'root_joint': parser.root.name if parser.root else None,
                'total_joints': len(parser.joints_dict),
                'total_channels': parser.total_channels,
                'frame_count': parser.frame_count,
                'frame_time': parser.frame_time,
                'joint_mapping': self.mappings[name]
            }

        with open(output_file, 'w') as f:
            json.dump(config, f, indent=2)

        print(f"\nMapping configuration saved to: {output_file}")

    def _build_standard_hierarchy(self) -> Dict:
        """Build a standard skeleton hierarchy structure"""
        return {
            'root': 'hips',
            'structure': {
                'hips': {
                    'children': ['spine', 'left_upper_leg', 'right_upper_leg'],
                    'type': 'pelvis'
                },
                'spine': {
                    'children': ['spine1', 'left_shoulder', 'right_shoulder'],
                    'type': 'spine'
                },
                'spine1': {
                    'children': ['chest', 'neck'],
                    'type': 'spine'
                },
                'neck': {
                    'children': ['head'],
                    'type': 'neck'
                },
                'left_upper_leg': {
                    'children': ['left_lower_leg'],
                    'type': 'leg'
                },
                'left_lower_leg': {
                    'children': ['left_foot'],
                    'type': 'leg'
                },
                'left_foot': {
                    'children': ['left_toes'],
                    'type': 'leg'
                },
                'left_shoulder': {
                    'children': ['left_upper_arm'],
                    'type': 'arm'
                },
                'left_upper_arm': {
                    'children': ['left_lower_arm'],
                    'type': 'arm'
                },
                'left_lower_arm': {
                    'children': ['left_hand'],
                    'type': 'arm'
                },
                # Mirror for right side
                'right_upper_leg': {
                    'children': ['right_lower_leg'],
                    'type': 'leg'
                },
                'right_lower_leg': {
                    'children': ['right_foot'],
                    'type': 'leg'
                },
                'right_foot': {
                    'children': ['right_toes'],
                    'type': 'leg'
                },
                'right_shoulder': {
                    'children': ['right_upper_arm'],
                    'type': 'arm'
                },
                'right_upper_arm': {
                    'children': ['right_lower_arm'],
                    'type': 'arm'
                },
                'right_lower_arm': {
                    'children': ['right_hand'],
                    'type': 'arm'
                }
            }
        }


def main():
    """Main function to analyze BVH files"""

    # File paths
    files = {
        'accad': '../data/motion/Female1_B01_StandToWalk.bvh',
        'Generic_Walk': '../data/motion/walk.bvh',
        'CMU': '../data/motion/02_01.bvh'
    }

    # Parse each file
    parsers = {}
    for name, filepath in files.items():
        print(f"Parsing {name}...")
        parser = BVHParser(filepath)
        parser.parse()
        parser.print_hierarchy()
        parsers[name] = parser

    # Create mapper
    print("\n" + "=" * 80)
    print("CREATING SKELETON MAPPER")
    print("=" * 80)

    mapper = SkeletonMapper()
    for name, parser in parsers.items():
        mapper.add_bvh(name, parser)

    # Print comparison
    mapper.print_comparison()

    # Export configuration
    mapper.export_mapping_config('../bvh_mapping_config.json')


if __name__ == '__main__':
    main()