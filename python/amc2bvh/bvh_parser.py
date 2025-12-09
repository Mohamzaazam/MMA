#!/usr/bin/env python3
"""
BVH Parser - Pure Python BVH file parser.

This module provides pure Python parsing for BVH motion capture files,
useful for metadata exploration without C++ dependencies.

For proper DART-compatible state extraction, use state_extractor.py instead.
"""

import re
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple


@dataclass
class Joint:
    """Represents a single joint in the skeleton hierarchy."""
    name: str
    offset: np.ndarray
    channels: List[str]
    channel_indices: List[int] = field(default_factory=list)
    children: List['Joint'] = field(default_factory=list)
    parent: Optional['Joint'] = None
    is_end_site: bool = False
    
    @property
    def num_channels(self) -> int:
        return len(self.channels)
    
    def get_rotation_order(self) -> str:
        """Extract rotation order from channels (e.g., 'ZXY')."""
        rot_channels = [c for c in self.channels if 'rotation' in c.lower()]
        return ''.join([c[0].upper() for c in rot_channels])


@dataclass
class BVHData:
    """
    Complete BVH file data structure.
    
    Attributes:
        root: Root joint of the skeleton hierarchy
        joints: Dictionary mapping joint names to Joint objects
        frame_count: Number of motion frames
        frame_time: Time per frame in seconds
        motion_data: Raw motion data array (frames x channels)
        filename: Original filename
        subject_id: Subject identifier parsed from filename
    """
    root: Joint
    joints: Dict[str, Joint]
    frame_count: int
    frame_time: float
    motion_data: np.ndarray
    filename: str = ""
    subject_id: Optional[int] = None
    
    @property
    def duration(self) -> float:
        return self.frame_count * self.frame_time
    
    @property
    def fps(self) -> float:
        return 1.0 / self.frame_time if self.frame_time > 0 else 0
    
    @property
    def num_joints(self) -> int:
        return sum(1 for j in self.joints.values() if not j.is_end_site)
    
    @property
    def total_channels(self) -> int:
        return self.motion_data.shape[1] if len(self.motion_data.shape) > 1 else 0
    
    def get_frame(self, frame_idx: int) -> np.ndarray:
        return self.motion_data[frame_idx].copy()


class BVHParser:
    """Pure Python parser for BVH motion capture files."""
    
    def __init__(self, filepath: str):
        self.filepath = Path(filepath)
        self.joints: Dict[str, Joint] = {}
        self.channel_count = 0
        
    def parse(self) -> BVHData:
        """Parse the BVH file and return structured data."""
        with open(self.filepath, 'r') as f:
            content = f.read()
        
        parts = content.split('MOTION')
        if len(parts) != 2:
            raise ValueError(f"Invalid BVH file format: {self.filepath}")
        
        root = self._parse_hierarchy(parts[0])
        frame_count, frame_time, motion_data = self._parse_motion('MOTION' + parts[1])
        subject_id = self._extract_subject_id()
        
        return BVHData(
            root=root, joints=self.joints, frame_count=frame_count,
            frame_time=frame_time, motion_data=motion_data,
            filename=self.filepath.name, subject_id=subject_id,
        )
    
    def _extract_subject_id(self) -> Optional[int]:
        match = re.match(r'^(\d+)', self.filepath.stem)
        return int(match.group(1)) if match else None
    
    def _parse_hierarchy(self, content: str) -> Joint:
        lines = [l.strip() for l in content.split('\n') if l.strip()]
        root, joint_stack, current_joint = None, [], None
        
        for line in lines:
            tokens = line.split()
            if not tokens:
                continue
            
            if tokens[0] in ('ROOT', 'JOINT'):
                new_joint = Joint(name=tokens[1], offset=np.zeros(3), channels=[])
                if tokens[0] == 'ROOT':
                    root = new_joint
                elif current_joint:
                    current_joint.children.append(new_joint)
                    new_joint.parent = current_joint
                self.joints[tokens[1]] = new_joint
                joint_stack.append(new_joint)
                current_joint = new_joint
            
            elif tokens[0] == 'End' and len(tokens) > 1 and tokens[1] == 'Site':
                end_joint = Joint(
                    name=f"{current_joint.name}_End", offset=np.zeros(3),
                    channels=[], is_end_site=True
                )
                current_joint.children.append(end_joint)
                end_joint.parent = current_joint
                joint_stack.append(end_joint)
                current_joint = end_joint
            
            elif tokens[0] == 'OFFSET' and current_joint:
                current_joint.offset = np.array([float(t) for t in tokens[1:4]])
            
            elif tokens[0] == 'CHANNELS' and current_joint:
                n = int(tokens[1])
                current_joint.channels = tokens[2:2+n]
                current_joint.channel_indices = list(range(self.channel_count, self.channel_count + n))
                self.channel_count += n
            
            elif tokens[0] == '}' and joint_stack:
                joint_stack.pop()
                current_joint = joint_stack[-1] if joint_stack else None
        
        return root
    
    def _parse_motion(self, content: str) -> Tuple[int, float, np.ndarray]:
        lines = [l.strip() for l in content.split('\n') if l.strip()]
        frame_count, frame_time, motion_lines = 0, 0.0, []
        parsing_frames = False
        
        for line in lines:
            if line.startswith('Frames:'):
                frame_count = int(line.split(':')[1].strip())
            elif line.startswith('Frame Time:'):
                frame_time = float(line.split(':')[1].strip())
                parsing_frames = True
            elif parsing_frames:
                motion_lines.append([float(v) for v in line.split()])
        
        return frame_count, frame_time, np.array(motion_lines, dtype=np.float64)


def load_bvh(filepath: str) -> BVHData:
    """Load and parse a BVH file."""
    return BVHParser(filepath).parse()


def load_bvh_files(directory: str, pattern: str = "*.bvh") -> List[BVHData]:
    """Load all BVH files from a directory."""
    dir_path = Path(directory)
    bvh_data_list = []
    for filepath in sorted(dir_path.glob(pattern)):
        try:
            bvh_data_list.append(load_bvh(str(filepath)))
        except Exception as e:
            print(f"Warning: Failed to load {filepath}: {e}")
    return bvh_data_list
