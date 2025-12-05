"""
BVH (BioVision Hierarchy) file writer.

Writes skeleton and motion data to BVH format.
Supports optional joint collapsing, renaming, and scaling.
"""

import numpy as np
from typing import List

from .data_structs import Joint, Skeleton, Motion, MotionFrame
from .config import ConversionConfig
from .rot_converter import RotationConverter


class BVHWriter:
    """Writer for BVH files with configurable output options"""
    
    def __init__(self, skeleton: Skeleton, motion: Motion, config: ConversionConfig):
        self.skeleton = skeleton
        self.motion = motion
        self.config = config
        self.rot_converter = RotationConverter()
    
    def _get_name(self, joint_name: str) -> str:
        """Get mapped joint name (or original if no mapping)"""
        return self.config.joint_name_map.get(joint_name, joint_name)
    
    def _is_collapse(self, name: str) -> bool:
        """Check if joint should be collapsed into parent"""
        return name in self.config.collapse_joints
    
    def _is_skip(self, name: str) -> bool:
        """Check if joint should be skipped entirely"""
        return name in self.config.skip_joints
    
    def _is_end(self, name: str) -> bool:
        """Check if joint is a terminal joint"""
        return name in self.config.end_joints
    
    def _get_ordered_children(self, joint: Joint) -> List[Joint]:
        """
        Get children in correct order for walk_.bvh compatibility.
        
        Root order: Spine, RightUpLeg, LeftUpLeg
        Thorax order (becomes Spine1's children): LeftShoulder, RightShoulder, Neck
        """
        valid_children = [c for c in joint.children if not self._is_skip(c.name)]
        
        # Only reorder when walk-bvh mode (collapse_joints set)
        if not self.config.collapse_joints:
            return valid_children
        
        # Root joint: Spine first, then right leg, then left leg
        if joint.name == 'root':
            order_map = {
                'lowerback': 0,  # Spine first
                'rfemur': 1,     # Right leg
                'lfemur': 2,     # Left leg
                'rhipjoint': 1,
                'lhipjoint': 2,
            }
            valid_children.sort(key=lambda c: order_map.get(c.name, 999))
        
        # Thorax (collapsed into Spine1): order its children (shoulders before neck)
        # walk_.bvh order: LeftShoulder, RightShoulder, Neck
        elif joint.name == 'thorax':
            order_map = {
                'lclavicle': 0,  # LeftShoulder first
                'rclavicle': 1,  # RightShoulder second
                'lowerneck': 2,  # Neck third
            }
            valid_children.sort(key=lambda c: order_map.get(c.name, 999))
        
        return valid_children
    
    def write(self, filepath: str):
        """Write BVH file with configured options."""
        with open(filepath, 'w') as f:
            f.write("HIERARCHY\n")
            self._write_skeleton(f)
            f.write("MOTION\n")
            f.write(f"Frames:\t{self.motion.frame_count}\n")
            f.write(f"Frame Time:\t{1.0 / self.config.fps:.6f}\n")
            self._write_motion(f)
    
    def _has_translation(self, joint: Joint) -> bool:
        """Check if joint has translation channels"""
        for dof in joint.dof:
            if dof in ['TX', 'TY', 'TZ']:
                return True
        return False
    
    def _write_skeleton(self, f):
        """Write skeleton hierarchy"""
        root = self.skeleton.root
        root_pos = self.skeleton.root_position * self.config.scale
        self._write_joint(f, root, root_pos, 0, np.zeros(3))
    
    def _write_joint(self, f, joint: Joint, offset: np.ndarray, depth: int, accumulated: np.ndarray):
        """Write a joint and its children with optional collapsing"""
        # Skip joints entirely
        if self._is_skip(joint.name):
            return
        
        # Collapse joint - skip but process children with accumulated offset
        if self._is_collapse(joint.name):
            new_accumulated = accumulated + self._get_child_offset(joint)
            for child in self._get_ordered_children(joint):
                self._write_joint(f, child, offset, depth, new_accumulated)
            return
        
        indent = "\t" * depth
        name = self._get_name(joint.name)
        
        # ROOT or JOINT
        joint_type = "ROOT" if depth == 0 else "JOINT"
        f.write(f"{indent}{joint_type} {name}\n")
        f.write(f"{indent}{{\n")
        
        # OFFSET (includes accumulated from collapsed parents)
        total_offset = offset + accumulated
        f.write(f"{indent}\tOFFSET\t{total_offset[0]:.6f}\t{total_offset[1]:.6f}\t{total_offset[2]:.6f}\n")
        
        # CHANNELS
        has_trans = self._has_translation(joint)
        rot_order = self.config.rotation_order.value
        if has_trans:
            f.write(f"{indent}\tCHANNELS 6 Xposition Yposition Zposition {rot_order}\n")
        else:
            f.write(f"{indent}\tCHANNELS 3 {rot_order}\n")
        
        # Get child offset
        child_offset = self._get_child_offset(joint)
        
        # Terminal joint - write end site
        if self._is_end(joint.name):
            f.write(f"{indent}\tEnd Site\n")
            f.write(f"{indent}\t{{\n")
            f.write(f"{indent}\t\tOFFSET\t{child_offset[0]:.6f}\t{child_offset[1]:.6f}\t{child_offset[2]:.6f}\n")
            f.write(f"{indent}\t}}\n")
        else:
            # Process children in correct order
            ordered_children = self._get_ordered_children(joint)
            
            if ordered_children:
                for child in ordered_children:
                    self._write_joint(f, child, child_offset, depth + 1, np.zeros(3))
            else:
                # No valid children - add end site
                f.write(f"{indent}\tEnd Site\n")
                f.write(f"{indent}\t{{\n")
                f.write(f"{indent}\t\tOFFSET\t{child_offset[0]:.6f}\t{child_offset[1]:.6f}\t{child_offset[2]:.6f}\n")
                f.write(f"{indent}\t}}\n")
        
        f.write(f"{indent}}}\n")
    
    def _get_child_offset(self, joint: Joint) -> np.ndarray:
        """Calculate child offset from joint direction and length"""
        if joint.length > 0 and np.linalg.norm(joint.direction) > 0:
            dir_norm = joint.direction / np.linalg.norm(joint.direction)
            return dir_norm * joint.length * self.config.scale
        return np.zeros(3)
    
    def _write_motion(self, f):
        """Write motion data in hierarchy order"""
        for frame in self.motion.frames:
            values = []
            self._write_joint_sample(self.skeleton.root, frame, values)
            f.write("\t".join(f"{v:.6f}" for v in values) + "\n")
    
    def _write_joint_sample(self, joint: Joint, frame: MotionFrame, values: List[float]):
        """Write motion sample for a joint with optional collapsing"""
        # Skip joints
        if self._is_skip(joint.name):
            return
        
        # Collapse joints - skip motion but process children in order
        if self._is_collapse(joint.name):
            for child in self._get_ordered_children(joint):
                self._write_joint_sample(child, frame, values)
            return
        
        data = frame.joint_data.get(joint.name, np.zeros(len(joint.dof)))
        
        # Translation (scaled)
        if self._has_translation(joint):
            tx, ty, tz = 0.0, 0.0, 0.0
            for i, dof in enumerate(joint.dof):
                if i < len(data):
                    if dof == 'TX':
                        tx = data[i] * self.config.scale
                    elif dof == 'TY':
                        ty = data[i] * self.config.scale
                    elif dof == 'TZ':
                        tz = data[i] * self.config.scale
            values.extend([tx, ty, tz])
        
        # Rotation
        rotations = self.rot_converter.convert_joint_rotation(joint, data)
        values.extend(rotations)
        
        # Children in order (skip terminal joints' children)
        if not self._is_end(joint.name):
            for child in self._get_ordered_children(joint):
                self._write_joint_sample(child, frame, values)