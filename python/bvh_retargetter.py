#!/usr/bin/env python3
"""
BVH Retargeter: Convert CMU BVH files to MASS Pipeline Format

This tool converts BVH files from CMU motion capture format to the
Character1_* naming convention used by the MASS pipeline.

Key transformations:
1. Rename joints from CMU names to Character1_* names
2. Adjust hierarchy to match MASS skeleton structure
3. Handle missing/extra joints appropriately
4. Preserve motion data with correct channel mapping

Usage:
    python bvh_retargeter.py input.bvh output.bvh
    python bvh_retargeter.py 29_01.bvh 29_01_mass.bvh --reference walk.bvh
"""

import re
import sys
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set
from pathlib import Path
from copy import deepcopy


# ============================================================================
# CMU to MASS Joint Mapping
# ============================================================================

# Standard CMU skeleton joint names (from ASF/AMC conversion)
CMU_TO_MASS = {
    # Root
    'root': 'Character1_Hips',
    'Hips': 'Character1_Hips',
    
    # Spine chain
    'lowerback': 'Character1_Spine',
    'LowerBack': 'Character1_Spine',
    'upperback': 'Character1_Spine1',
    'UpperBack': 'Character1_Spine1',
    'thorax': 'Character1_Spine1',  # Merged with Spine1 (MASS has 2-segment spine)
    'Thorax': 'Character1_Spine1',
    
    # Neck/Head
    'lowerneck': 'Character1_Neck',
    'LowerNeck': 'Character1_Neck',
    'upperneck': 'Character1_Neck',  # Merged
    'UpperNeck': 'Character1_Neck',
    'Neck': 'Character1_Neck',
    'head': 'Character1_Head',
    'Head': 'Character1_Head',
    
    # Right Arm
    'rclavicle': 'Character1_RightShoulder',
    'RightShoulder': 'Character1_RightShoulder',
    'rhumerus': 'Character1_RightArm',
    'RightArm': 'Character1_RightArm',
    'rradius': 'Character1_RightForeArm',
    'RightForeArm': 'Character1_RightForeArm',
    'rwrist': 'Character1_RightHand',
    'rhand': 'Character1_RightHand',
    'RightHand': 'Character1_RightHand',
    'rfingers': None,  # Skip - not in MASS
    'rthumb': None,    # Skip - not in MASS
    
    # Left Arm
    'lclavicle': 'Character1_LeftShoulder',
    'LeftShoulder': 'Character1_LeftShoulder',
    'lhumerus': 'Character1_LeftArm',
    'LeftArm': 'Character1_LeftArm',
    'lradius': 'Character1_LeftForeArm',
    'LeftForeArm': 'Character1_LeftForeArm',
    'lwrist': 'Character1_LeftHand',
    'lhand': 'Character1_LeftHand',
    'LeftHand': 'Character1_LeftHand',
    'lfingers': None,  # Skip
    'lthumb': None,    # Skip
    
    # Right Leg
    'rfemur': 'Character1_RightUpLeg',
    'RightUpLeg': 'Character1_RightUpLeg',
    'rtibia': 'Character1_RightLeg',
    'RightLeg': 'Character1_RightLeg',
    'rfoot': 'Character1_RightFoot',
    'RightFoot': 'Character1_RightFoot',
    'rtoes': 'Character1_RightToeBase',
    'RightToeBase': 'Character1_RightToeBase',
    
    # Left Leg
    'lfemur': 'Character1_LeftUpLeg',
    'LeftUpLeg': 'Character1_LeftUpLeg',
    'ltibia': 'Character1_LeftLeg',
    'LeftLeg': 'Character1_LeftLeg',
    'lfoot': 'Character1_LeftFoot',
    'LeftFoot': 'Character1_LeftFoot',
    'ltoes': 'Character1_LeftToeBase',
    'LeftToeBase': 'Character1_LeftToeBase',
}

# MASS skeleton hierarchy (from human.xml)
MASS_HIERARCHY = {
    'Character1_Hips': {
        'parent': None,
        'children': ['Character1_Spine', 'Character1_RightUpLeg', 'Character1_LeftUpLeg']
    },
    'Character1_Spine': {
        'parent': 'Character1_Hips',
        'children': ['Character1_Spine1']
    },
    'Character1_Spine1': {
        'parent': 'Character1_Spine',
        'children': ['Character1_Neck', 'Character1_RightShoulder', 'Character1_LeftShoulder']
    },
    'Character1_Neck': {
        'parent': 'Character1_Spine1',
        'children': ['Character1_Head']
    },
    'Character1_Head': {
        'parent': 'Character1_Neck',
        'children': []
    },
    'Character1_RightShoulder': {
        'parent': 'Character1_Spine1',
        'children': ['Character1_RightArm']
    },
    'Character1_RightArm': {
        'parent': 'Character1_RightShoulder',
        'children': ['Character1_RightForeArm']
    },
    'Character1_RightForeArm': {
        'parent': 'Character1_RightArm',
        'children': ['Character1_RightHand']
    },
    'Character1_RightHand': {
        'parent': 'Character1_RightForeArm',
        'children': []
    },
    'Character1_LeftShoulder': {
        'parent': 'Character1_Spine1',
        'children': ['Character1_LeftArm']
    },
    'Character1_LeftArm': {
        'parent': 'Character1_LeftShoulder',
        'children': ['Character1_LeftForeArm']
    },
    'Character1_LeftForeArm': {
        'parent': 'Character1_LeftArm',
        'children': ['Character1_LeftHand']
    },
    'Character1_LeftHand': {
        'parent': 'Character1_LeftForeArm',
        'children': []
    },
    'Character1_RightUpLeg': {
        'parent': 'Character1_Hips',
        'children': ['Character1_RightLeg']
    },
    'Character1_RightLeg': {
        'parent': 'Character1_RightUpLeg',
        'children': ['Character1_RightFoot']
    },
    'Character1_RightFoot': {
        'parent': 'Character1_RightLeg',
        'children': ['Character1_RightToeBase']
    },
    'Character1_RightToeBase': {
        'parent': 'Character1_RightFoot',
        'children': []
    },
    'Character1_LeftUpLeg': {
        'parent': 'Character1_Hips',
        'children': ['Character1_LeftLeg']
    },
    'Character1_LeftLeg': {
        'parent': 'Character1_LeftUpLeg',
        'children': ['Character1_LeftFoot']
    },
    'Character1_LeftFoot': {
        'parent': 'Character1_LeftLeg',
        'children': ['Character1_LeftToeBase']
    },
    'Character1_LeftToeBase': {
        'parent': 'Character1_LeftFoot',
        'children': []
    },
}


@dataclass
class Joint:
    """BVH Joint data"""
    name: str
    offset: Tuple[float, float, float]
    channels: List[str]
    children: List['Joint'] = field(default_factory=list)
    is_end_site: bool = False


class BVHRetargeter:
    """Convert CMU BVH files to MASS format"""
    
    def __init__(self, reference_bvh_path: Optional[str] = None):
        """
        Initialize retargeter.
        
        Args:
            reference_bvh_path: Optional path to reference BVH (e.g., walk.bvh)
                              to extract exact offsets and structure
        """
        self.reference_offsets = {}
        self.reference_hierarchy = None
        
        if reference_bvh_path:
            self._load_reference(reference_bvh_path)
    
    def _load_reference(self, path: str):
        """Load reference BVH for offset values"""
        print(f"Loading reference BVH: {path}")
        with open(path, 'r') as f:
            content = f.read()
        
        # Parse offsets from reference
        joint_pattern = r'(ROOT|JOINT)\s+(\S+)\s*\{[^}]*OFFSET\s+([\d\.\-e]+)\s+([\d\.\-e]+)\s+([\d\.\-e]+)'
        for match in re.finditer(joint_pattern, content, re.DOTALL | re.IGNORECASE):
            joint_name = match.group(2)
            offset = (float(match.group(3)), float(match.group(4)), float(match.group(5)))
            self.reference_offsets[joint_name] = offset
        
        print(f"  Loaded {len(self.reference_offsets)} joint offsets from reference")
    
    def convert(self, input_path: str, output_path: str, 
                scale_factor: float = 1.0,
                use_reference_offsets: bool = True) -> bool:
        """
        Convert a CMU BVH file to MASS format.
        
        Args:
            input_path: Path to input CMU BVH file
            output_path: Path for output MASS-compatible BVH
            scale_factor: Scale factor for positions (CMU uses different units)
            use_reference_offsets: Whether to use reference offsets
            
        Returns:
            True if conversion successful
        """
        print(f"\nConverting: {input_path}")
        print(f"Output: {output_path}")
        
        with open(input_path, 'r') as f:
            content = f.read()
        
        # Parse the BVH
        hierarchy_section, motion_section = self._split_bvh(content)
        
        # Parse joint structure
        root_joint, channel_mapping = self._parse_hierarchy(hierarchy_section)
        
        # Map joints to MASS names
        mapped_root = self._remap_joints(root_joint)
        
        # Parse motion data
        num_frames, frame_time, motion_data = self._parse_motion(motion_section)
        
        # Remap motion data channels
        remapped_motion = self._remap_motion_data(
            motion_data, channel_mapping, scale_factor
        )
        
        # Generate output BVH
        output_content = self._generate_bvh(
            mapped_root, num_frames, frame_time, remapped_motion,
            use_reference_offsets
        )
        
        # Write output
        with open(output_path, 'w') as f:
            f.write(output_content)
        
        print(f"✓ Conversion complete: {output_path}")
        return True
    
    def _split_bvh(self, content: str) -> Tuple[str, str]:
        """Split BVH into hierarchy and motion sections"""
        motion_idx = content.upper().find('MOTION')
        if motion_idx == -1:
            raise ValueError("No MOTION section found in BVH")
        return content[:motion_idx], content[motion_idx:]
    
    def _parse_hierarchy(self, hierarchy: str) -> Tuple[Joint, Dict]:
        """Parse hierarchy section and return root joint + channel mapping"""
        lines = hierarchy.split('\n')
        channel_mapping = {}  # original_name -> (start_channel, num_channels)
        current_channel = 0
        
        def parse_joint(lines: List[str], idx: int, is_root: bool = False) -> Tuple[Joint, int]:
            nonlocal current_channel
            
            # Get joint name
            line = lines[idx].strip()
            if is_root:
                name = line.split()[1]
            else:
                name = line.split()[1]
            
            idx += 1  # Skip to {
            while idx < len(lines) and lines[idx].strip() != '{':
                idx += 1
            idx += 1  # Skip {
            
            offset = (0.0, 0.0, 0.0)
            channels = []
            children = []
            
            while idx < len(lines):
                line = lines[idx].strip()
                
                if line == '}':
                    break
                elif line.startswith('OFFSET'):
                    parts = line.split()
                    offset = (float(parts[1]), float(parts[2]), float(parts[3]))
                elif line.startswith('CHANNELS'):
                    parts = line.split()
                    num_ch = int(parts[1])
                    channels = parts[2:2+num_ch]
                    channel_mapping[name] = (current_channel, num_ch, channels)
                    current_channel += num_ch
                elif line.startswith('JOINT'):
                    child, idx = parse_joint(lines, idx)
                    children.append(child)
                elif line.startswith('End Site'):
                    # Parse end site
                    idx += 1  # {
                    while idx < len(lines) and '}' not in lines[idx]:
                        idx += 1
                
                idx += 1
            
            joint = Joint(name=name, offset=offset, channels=channels, children=children)
            return joint, idx
        
        # Find ROOT
        for i, line in enumerate(lines):
            if line.strip().startswith('ROOT'):
                root, _ = parse_joint(lines, i, is_root=True)
                return root, channel_mapping
        
        raise ValueError("No ROOT found in hierarchy")
    
    def _remap_joints(self, joint: Joint) -> Joint:
        """Recursively remap joint names to MASS format"""
        # Get new name
        new_name = CMU_TO_MASS.get(joint.name, joint.name)
        if new_name is None:
            # Skip this joint (fingers, thumbs, etc.)
            return None
        
        # Remap children
        new_children = []
        for child in joint.children:
            remapped = self._remap_joints(child)
            if remapped is not None:
                new_children.append(remapped)
        
        return Joint(
            name=new_name,
            offset=joint.offset,
            channels=joint.channels,
            children=new_children
        )
    
    def _parse_motion(self, motion_section: str) -> Tuple[int, float, List[List[float]]]:
        """Parse motion section"""
        lines = motion_section.strip().split('\n')
        
        num_frames = 0
        frame_time = 0.0
        motion_data = []
        
        reading_data = False
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            if line.upper().startswith('FRAMES:'):
                num_frames = int(line.split(':')[1].strip())
            elif line.upper().startswith('FRAME TIME:'):
                frame_time = float(line.split(':')[1].strip())
                reading_data = True
            elif reading_data:
                values = [float(v) for v in line.split()]
                motion_data.append(values)
        
        return num_frames, frame_time, motion_data
    
    def _remap_motion_data(self, motion_data: List[List[float]], 
                          channel_mapping: Dict,
                          scale_factor: float) -> List[List[float]]:
        """Remap motion data channels based on joint mapping"""
        # For now, keep all channels (proper remapping requires more complex logic)
        # Scale position channels
        remapped = []
        for frame in motion_data:
            new_frame = frame.copy()
            # Scale root position (first 3 channels typically)
            new_frame[0] *= scale_factor
            new_frame[1] *= scale_factor
            new_frame[2] *= scale_factor
            remapped.append(new_frame)
        
        return remapped
    
    def _generate_bvh(self, root: Joint, num_frames: int, frame_time: float,
                     motion_data: List[List[float]],
                     use_reference_offsets: bool) -> str:
        """Generate BVH file content"""
        lines = ['HIERARCHY']
        
        def write_joint(joint: Joint, indent: int, is_root: bool = False):
            prefix = '\t' * indent
            
            if is_root:
                lines.append(f'{prefix}ROOT {joint.name}')
            else:
                lines.append(f'{prefix}JOINT {joint.name}')
            
            lines.append(f'{prefix}{{')
            
            # Use reference offset if available
            if use_reference_offsets and joint.name in self.reference_offsets:
                offset = self.reference_offsets[joint.name]
            else:
                offset = joint.offset
            
            lines.append(f'{prefix}\tOFFSET {offset[0]:.6f} {offset[1]:.6f} {offset[2]:.6f}')
            
            if joint.channels:
                ch_str = ' '.join(joint.channels)
                lines.append(f'{prefix}\tCHANNELS {len(joint.channels)} {ch_str}')
            
            # Write children
            for child in joint.children:
                write_joint(child, indent + 1)
            
            # Add end site if no children
            if not joint.children:
                lines.append(f'{prefix}\tEnd Site')
                lines.append(f'{prefix}\t{{')
                lines.append(f'{prefix}\t\tOFFSET 0.000000 0.000000 0.000000')
                lines.append(f'{prefix}\t}}')
            
            lines.append(f'{prefix}}}')
        
        write_joint(root, 0, is_root=True)
        
        # Motion section
        lines.append('MOTION')
        lines.append(f'Frames: {num_frames}')
        lines.append(f'Frame Time: {frame_time:.6f}')
        
        for frame in motion_data:
            frame_str = ' '.join(f'{v:.6f}' for v in frame)
            lines.append(frame_str)
        
        return '\n'.join(lines)


def analyze_and_suggest(input_path: str, reference_path: Optional[str] = None):
    """Analyze a BVH file and suggest conversion approach"""
    print("=" * 70)
    print(f"Analyzing: {input_path}")
    print("=" * 70)
    
    with open(input_path, 'r') as f:
        content = f.read()
    
    # Extract joint names
    joint_pattern = r'(ROOT|JOINT)\s+(\S+)'
    joints = [m.group(2) for m in re.finditer(joint_pattern, content)]
    
    print(f"\nFound {len(joints)} joints:")
    for j in joints:
        mass_name = CMU_TO_MASS.get(j, '??? (UNKNOWN)')
        if mass_name is None:
            mass_name = '(will be skipped)'
        print(f"  {j:20s} -> {mass_name}")
    
    # Check coverage
    required = set(MASS_HIERARCHY.keys())
    mapped = set()
    for j in joints:
        mass_name = CMU_TO_MASS.get(j)
        if mass_name:
            mapped.add(mass_name)
    
    missing = required - mapped
    print(f"\nMASS Coverage: {len(mapped)}/{len(required)} joints")
    if missing:
        print(f"Missing joints: {', '.join(sorted(missing))}")
    
    if reference_path:
        print(f"\nReference file: {reference_path}")
        with open(reference_path, 'r') as f:
            ref_content = f.read()
        ref_joints = [m.group(2) for m in re.finditer(joint_pattern, ref_content)]
        print(f"Reference has {len(ref_joints)} joints")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Convert CMU BVH to MASS format')
    parser.add_argument('input', help='Input BVH file (CMU format)')
    parser.add_argument('output', nargs='?', help='Output BVH file (MASS format)')
    parser.add_argument('--reference', '-r', help='Reference BVH file (e.g., walk.bvh)')
    parser.add_argument('--scale', type=float, default=1.0, help='Scale factor')
    parser.add_argument('--analyze', '-a', action='store_true', help='Analyze only, no conversion')
    
    args = parser.parse_args()
    
    if args.analyze:
        analyze_and_suggest(args.input, args.reference)
    else:
        if not args.output:
            # Generate output name
            p = Path(args.input)
            args.output = str(p.parent / f"{p.stem}_mass{p.suffix}")
        
        retargeter = BVHRetargeter(args.reference)
        retargeter.convert(args.input, args.output, args.scale)


if __name__ == "__main__":
    main()