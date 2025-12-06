#!/usr/bin/env python3
"""
ASF/AMC to BVH Converter - Fixed for walk_.bvh (Maya) compatibility

This converter properly handles the ZXY rotation order used in walk_.bvh.

Key differences from the original:
1. Uses ZXY rotation order (Maya format) instead of ZYX (CMU format)
2. Properly extracts Euler angles using intrinsic rotation convention
3. The BVH CHANNELS line matches the actual rotation decomposition

Usage:
    python convert_asfamc_to_bvh.py skeleton.asf motion.amc -o output.bvh --walk-bvh
"""

import argparse
import numpy as np
import math
from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional
from enum import Enum


# ============================================================================
# Transformations (minimal subset needed)
# ============================================================================

_EPS = np.finfo(float).eps * 4.0
_NEXT_AXIS = [1, 2, 0, 1]

_AXES2TUPLE = {
    'sxyz': (0, 0, 0, 0), 'sxyx': (0, 0, 1, 0), 'sxzy': (0, 1, 0, 0),
    'sxzx': (0, 1, 1, 0), 'syzx': (1, 0, 0, 0), 'syzy': (1, 0, 1, 0),
    'syxz': (1, 1, 0, 0), 'syxy': (1, 1, 1, 0), 'szxy': (2, 0, 0, 0),
    'szxz': (2, 0, 1, 0), 'szyx': (2, 1, 0, 0), 'szyz': (2, 1, 1, 0),
    'rzyx': (0, 0, 0, 1), 'rxyx': (0, 0, 1, 1), 'ryzx': (0, 1, 0, 1),
    'rxzx': (0, 1, 1, 1), 'rxzy': (1, 0, 0, 1), 'ryzy': (1, 0, 1, 1),
    'rzxy': (1, 1, 0, 1), 'ryxy': (1, 1, 1, 1), 'ryxz': (2, 0, 0, 1),
    'rzxz': (2, 0, 1, 1), 'rxyz': (2, 1, 0, 1), 'rzyz': (2, 1, 1, 1)}


def quaternion_from_euler(ai, aj, ak, axes='sxyz'):
    """Return quaternion from Euler angles and axis sequence."""
    try:
        firstaxis, parity, repetition, frame = _AXES2TUPLE[axes.lower()]
    except (AttributeError, KeyError):
        firstaxis, parity, repetition, frame = axes

    i = firstaxis + 1
    j = _NEXT_AXIS[i+parity-1] + 1
    k = _NEXT_AXIS[i-parity] + 1

    if frame:
        ai, ak = ak, ai
    if parity:
        aj = -aj

    ai /= 2.0
    aj /= 2.0
    ak /= 2.0
    ci = math.cos(ai)
    si = math.sin(ai)
    cj = math.cos(aj)
    sj = math.sin(aj)
    ck = math.cos(ak)
    sk = math.sin(ak)
    cc = ci*ck
    cs = ci*sk
    sc = si*ck
    ss = si*sk

    q = np.empty((4, ))
    if repetition:
        q[0] = cj*(cc - ss)
        q[i] = cj*(cs + sc)
        q[j] = sj*(cc + ss)
        q[k] = sj*(cs - sc)
    else:
        q[0] = cj*cc + sj*ss
        q[i] = cj*sc - sj*cs
        q[j] = cj*ss + sj*cc
        q[k] = cj*cs - sj*sc
    if parity:
        q[j] *= -1.0

    return q


def quaternion_matrix(quaternion):
    """Return homogeneous rotation matrix from quaternion."""
    q = np.array(quaternion, dtype=np.float64, copy=True)
    n = np.dot(q, q)
    if n < _EPS:
        return np.identity(4)
    q *= math.sqrt(2.0 / n)
    q = np.outer(q, q)
    return np.array([
        [1.0-q[2, 2]-q[3, 3],     q[1, 2]-q[3, 0],     q[1, 3]+q[2, 0], 0.0],
        [    q[1, 2]+q[3, 0], 1.0-q[1, 1]-q[3, 3],     q[2, 3]-q[1, 0], 0.0],
        [    q[1, 3]-q[2, 0],     q[2, 3]+q[1, 0], 1.0-q[1, 1]-q[2, 2], 0.0],
        [                0.0,                 0.0,                 0.0, 1.0]], dtype=np.float64)


def euler_from_matrix(matrix, axes='sxyz'):
    """Return Euler angles from rotation matrix for specified axis sequence."""
    try:
        firstaxis, parity, repetition, frame = _AXES2TUPLE[axes.lower()]
    except (AttributeError, KeyError):
        firstaxis, parity, repetition, frame = axes

    i = firstaxis
    j = _NEXT_AXIS[i+parity]
    k = _NEXT_AXIS[i-parity+1]

    M = np.array(matrix, dtype=np.float64, copy=False)[:3, :3]
    if repetition:
        sy = math.sqrt(M[i, j]*M[i, j] + M[i, k]*M[i, k])
        if sy > _EPS:
            ax = math.atan2( M[i, j],  M[i, k])
            ay = math.atan2( sy,       M[i, i])
            az = math.atan2( M[j, i], -M[k, i])
        else:
            ax = math.atan2(-M[j, k],  M[j, j])
            ay = math.atan2( sy,       M[i, i])
            az = 0.0
    else:
        cy = math.sqrt(M[i, i]*M[i, i] + M[j, i]*M[j, i])
        if cy > _EPS:
            ax = math.atan2( M[k, j],  M[k, k])
            ay = math.atan2(-M[k, i],  cy)
            az = math.atan2( M[j, i],  M[i, i])
        else:
            ax = math.atan2(-M[j, k],  M[j, j])
            ay = math.atan2(-M[k, i],  cy)
            az = 0.0

    if parity:
        ax, ay, az = -ax, -ay, -az
    if frame:
        ax, az = az, ax
    return ax, ay, az


def euler_from_quaternion(quaternion, axes='sxyz'):
    """Return Euler angles from quaternion for specified axis sequence."""
    return euler_from_matrix(quaternion_matrix(quaternion), axes)


def quaternion_multiply(quaternion1, quaternion0):
    """Return multiplication of two quaternions."""
    w0, x0, y0, z0 = quaternion0
    w1, x1, y1, z1 = quaternion1
    return np.array([
        -x1*x0 - y1*y0 - z1*z0 + w1*w0,
        x1*w0 + y1*z0 - z1*y0 + w1*x0,
        -x1*z0 + y1*w0 + z1*x0 + w1*y0,
        x1*y0 - y1*x0 + z1*w0 + w1*z0], dtype=np.float64)


def quaternion_inverse(quaternion):
    """Return inverse of quaternion."""
    q = np.array(quaternion, dtype=np.float64, copy=True)
    q[1:] *= -1.0
    return q / np.dot(q, q)


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class Joint:
    """Represents a joint in the skeleton hierarchy"""
    name: str
    direction: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0]))
    length: float = 0.0
    axis: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0]))
    axis_order: str = "XYZ"
    rotation: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0]))
    dof: List[str] = field(default_factory=list)
    children: List['Joint'] = field(default_factory=list)
    parent: Optional['Joint'] = None
    offset: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0]))


@dataclass
class Skeleton:
    """Represents the complete skeleton structure"""
    name: str = "VICON"
    joints: Dict[str, Joint] = field(default_factory=dict)
    root: Optional[Joint] = None
    root_position: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0]))
    length_unit: float = 1.0
    angle_unit: str = "deg"


@dataclass
class MotionFrame:
    """Represents a single frame of motion data"""
    frame_number: int
    joint_data: Dict[str, np.ndarray] = field(default_factory=dict)


@dataclass
class Motion:
    """Contains all motion data"""
    frames: List[MotionFrame] = field(default_factory=list)
    
    @property
    def frame_count(self) -> int:
        return len(self.frames)


# ============================================================================
# Configuration
# ============================================================================

class RotationOrder(Enum):
    """
    Supported rotation orders for BVH output.
    The value is the BVH CHANNELS string for rotations.
    """
    ZXY = "Zrotation Xrotation Yrotation"  # Maya/walk_.bvh format
    ZYX = "Zrotation Yrotation Xrotation"  # CMU default
    XYZ = "Xrotation Yrotation Zrotation"


@dataclass
class ConversionConfig:
    """Configuration for the conversion process"""
    rotation_order: RotationOrder = RotationOrder.ZXY
    scale: float = 1.0
    fps: float = 120.0
    joint_name_map: Dict[str, str] = field(default_factory=dict)
    collapse_joints: Set[str] = field(default_factory=set)
    skip_joints: Set[str] = field(default_factory=set)
    end_joints: Set[str] = field(default_factory=set)
    
    @staticmethod
    def walk_bvh_joint_map() -> Dict[str, str]:
        return {
            'root': 'Character1_Hips',
            'lowerback': 'Character1_Spine',
            'upperback': 'Character1_Spine1',
            'lowerneck': 'Character1_Neck',
            'head': 'Character1_Head',
            'lclavicle': 'Character1_LeftShoulder',
            'lhumerus': 'Character1_LeftArm',
            'lradius': 'Character1_LeftForeArm',
            'lwrist': 'Character1_LeftHand',
            'rclavicle': 'Character1_RightShoulder',
            'rhumerus': 'Character1_RightArm',
            'rradius': 'Character1_RightForeArm',
            'rwrist': 'Character1_RightHand',
            'lfemur': 'Character1_LeftUpLeg',
            'ltibia': 'Character1_LeftLeg',
            'lfoot': 'Character1_LeftFoot',
            'ltoes': 'Character1_LeftToeBase',
            'rfemur': 'Character1_RightUpLeg',
            'rtibia': 'Character1_RightLeg',
            'rfoot': 'Character1_RightFoot',
            'rtoes': 'Character1_RightToeBase',
        }
    
    @staticmethod
    def walk_bvh_collapse_joints() -> Set[str]:
        return {'lhipjoint', 'rhipjoint', 'thorax', 'upperneck'}
    
    @staticmethod
    def walk_bvh_skip_joints() -> Set[str]:
        return {'lhand', 'lfingers', 'lthumb', 'rhand', 'rfingers', 'rthumb'}
    
    @staticmethod
    def walk_bvh_end_joints() -> Set[str]:
        return {'lwrist', 'rwrist', 'head', 'ltoes', 'rtoes'}


# ============================================================================
# ASF Parser
# ============================================================================

def asf_order_to_axes(order: str) -> str:
    """Convert ASF axis order to transformations axes format."""
    return 's' + order.lower()


class ASFParser:
    """Parser for ASF (Acclaim Skeleton Format) files"""
    
    def __init__(self):
        self.skeleton = Skeleton()
    
    def parse(self, filepath: str) -> Skeleton:
        with open(filepath, 'r') as f:
            lines = f.readlines()
        
        mode = None
        current_joint = None
        
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            if not line or line.startswith('#'):
                i += 1
                continue
            
            if line.startswith(':'):
                keyword = line.split()[0]
                if keyword == ':name':
                    parts = line.split(None, 1)
                    self.skeleton.name = parts[1] if len(parts) > 1 else "VICON"
                elif keyword == ':units':
                    mode = 'units'
                elif keyword == ':root':
                    mode = 'root'
                elif keyword == ':bonedata':
                    mode = 'bonedata'
                elif keyword == ':hierarchy':
                    mode = 'hierarchy'
                else:
                    mode = keyword[1:]
                i += 1
                continue
            
            if mode == 'units':
                self._parse_units(line)
            elif mode == 'root':
                self._parse_root(line)
            elif mode == 'bonedata':
                current_joint = self._parse_bonedata(line, current_joint)
            elif mode == 'hierarchy':
                result = self._parse_hierarchy(line)
                if result == 'end':
                    mode = None
            
            i += 1
        
        return self.skeleton
    
    def _parse_units(self, line: str):
        parts = line.split()
        if len(parts) >= 2:
            if parts[0] == 'length':
                self.skeleton.length_unit = float(parts[1])
            elif parts[0] == 'angle':
                self.skeleton.angle_unit = parts[1]
    
    def _parse_root(self, line: str):
        parts = line.split()
        if parts[0] == 'position' and len(parts) >= 4:
            self.skeleton.root_position = np.array([float(x) for x in parts[1:4]])
    
    def _parse_bonedata(self, line: str, current_joint: Joint) -> Joint:
        if line == 'begin':
            return Joint(name="")
        elif line == 'end':
            if current_joint and current_joint.name:
                self.skeleton.joints[current_joint.name] = current_joint
            return None
        elif current_joint is not None:
            parts = line.split()
            if parts[0] == 'name':
                current_joint.name = parts[1]
            elif parts[0] == 'direction':
                current_joint.direction = np.array([float(x) for x in parts[1:4]])
            elif parts[0] == 'length':
                current_joint.length = float(parts[1])
            elif parts[0] == 'axis':
                axis_angles = np.array([float(x) for x in parts[1:4]])
                current_joint.axis = axis_angles
                if len(parts) > 4:
                    current_joint.axis_order = parts[4]
                
                # Convert to quaternion
                axis_rad = np.radians(axis_angles)
                axes = asf_order_to_axes(current_joint.axis_order)
                current_joint.rotation = quaternion_from_euler(*axis_rad, axes=axes)
            elif parts[0] == 'dof':
                current_joint.dof = [d.upper() for d in parts[1:]]
        
        return current_joint
    
    def _parse_hierarchy(self, line: str) -> str:
        if line == 'begin':
            return 'begin'
        elif line == 'end':
            return 'end'
        else:
            parts = line.split()
            if len(parts) >= 2:
                parent_name = parts[0]
                child_names = parts[1:]
                
                if parent_name == 'root':
                    if self.skeleton.root is None:
                        self.skeleton.root = Joint(name='root')
                        self.skeleton.root.dof = ['TX', 'TY', 'TZ', 'RX', 'RY', 'RZ']
                        self.skeleton.joints['root'] = self.skeleton.root
                    parent_joint = self.skeleton.root
                else:
                    parent_joint = self.skeleton.joints.get(parent_name)
                
                if parent_joint:
                    for child_name in child_names:
                        child = self.skeleton.joints.get(child_name)
                        if child:
                            child.parent = parent_joint
                            parent_joint.children.append(child)
                            child.offset = parent_joint.direction * parent_joint.length
        
        return 'continue'


# ============================================================================
# AMC Parser
# ============================================================================

class AMCParser:
    """Parser for AMC (Acclaim Motion Capture) files"""
    
    def __init__(self, skeleton: Skeleton):
        self.skeleton = skeleton
        self.motion = Motion()
    
    def parse(self, filepath: str) -> Motion:
        with open(filepath, 'r') as f:
            lines = f.readlines()
        
        current_frame = None
        is_degrees = True
        
        for line in lines:
            line = line.strip()
            
            if not line or line.startswith('#'):
                continue
            
            if line.startswith(':'):
                if line == ':DEGREES':
                    is_degrees = True
                elif line == ':RADIANS':
                    is_degrees = False
                continue
            
            if line.isdigit():
                if current_frame is not None:
                    self.motion.frames.append(current_frame)
                current_frame = MotionFrame(frame_number=int(line))
                continue
            
            if current_frame is not None:
                self._parse_joint_data(line, current_frame, is_degrees)
        
        if current_frame is not None:
            self.motion.frames.append(current_frame)
        
        return self.motion
    
    def _parse_joint_data(self, line: str, frame: MotionFrame, is_degrees: bool):
        parts = line.split()
        joint_name = parts[0]
        values = [float(x) for x in parts[1:]]
        
        joint = self.skeleton.joints.get(joint_name)
        if joint:
            data = np.zeros(len(joint.dof))
            for i, (dof, val) in enumerate(zip(joint.dof, values)):
                if is_degrees and dof in ['RX', 'RY', 'RZ']:
                    data[i] = np.radians(val)
                else:
                    data[i] = val
            frame.joint_data[joint_name] = data


# ============================================================================
# Rotation Converter
# ============================================================================

class RotationConverter:
    """Handles rotation conversions for motion data"""
    
    def __init__(self, rotation_order: RotationOrder = RotationOrder.ZXY):
        self.rotation_order = rotation_order
        
        # Map rotation order to transformations axes string
        # BVH uses intrinsic rotations (rotating frame), hence 'r' prefix
        self._axes_map = {
            RotationOrder.ZXY: 'rzxy',  # walk_.bvh format
            RotationOrder.ZYX: 'rzyx',  # CMU format
            RotationOrder.XYZ: 'rxyz',
        }
    
    def convert_joint_rotation(self, joint: Joint, data: np.ndarray) -> List[float]:
        """
        Convert rotation from ASF format to BVH format.
        
        Returns angles in degrees, ordered to match BVH channels.
        """
        # Build input rotation from ASF DOF channels
        xyz_angles = [0.0, 0.0, 0.0]
        has_rotation = False
        
        for i, dof in enumerate(joint.dof):
            if dof == 'RX':
                xyz_angles[0] = data[i] if i < len(data) else 0.0
                has_rotation = True
            elif dof == 'RY':
                xyz_angles[1] = data[i] if i < len(data) else 0.0
                has_rotation = True
            elif dof == 'RZ':
                xyz_angles[2] = data[i] if i < len(data) else 0.0
                has_rotation = True
        
        if not has_rotation:
            return [0.0, 0.0, 0.0]
        
        # Convert ASF motion angles to quaternion (extrinsic XYZ)
        q_motion = quaternion_from_euler(
            xyz_angles[0], xyz_angles[1], xyz_angles[2], 
            axes='sxyz'
        )
        
        # Apply joint's local axis transform
        q_axis = joint.rotation
        q_axis_inv = quaternion_inverse(q_axis)
        
        q_combined = quaternion_multiply(q_axis, q_motion)
        q_combined = quaternion_multiply(q_combined, q_axis_inv)
        
        # Extract Euler angles in target rotation order
        target_axes = self._axes_map.get(self.rotation_order, 'rzxy')
        result = euler_from_quaternion(q_combined, axes=target_axes)
        
        # Convert to degrees and return in channel order
        result_deg = np.degrees(result)
        return [result_deg[0], result_deg[1], result_deg[2]]


# ============================================================================
# BVH Writer
# ============================================================================

class BVHWriter:
    """Writer for BVH files with configurable output options"""
    
    def __init__(self, skeleton: Skeleton, motion: Motion, config: ConversionConfig):
        self.skeleton = skeleton
        self.motion = motion
        self.config = config
        self.rot_converter = RotationConverter(config.rotation_order)
    
    def _get_name(self, joint_name: str) -> str:
        return self.config.joint_name_map.get(joint_name, joint_name)
    
    def _is_collapse(self, name: str) -> bool:
        return name in self.config.collapse_joints
    
    def _is_skip(self, name: str) -> bool:
        return name in self.config.skip_joints
    
    def _is_end(self, name: str) -> bool:
        return name in self.config.end_joints
    
    def _get_ordered_children(self, joint: Joint) -> List[Joint]:
        valid_children = [c for c in joint.children if not self._is_skip(c.name)]
        
        if not self.config.collapse_joints:
            return valid_children
        
        if joint.name == 'root':
            order_map = {'lowerback': 0, 'rfemur': 1, 'lfemur': 2, 
                        'rhipjoint': 1, 'lhipjoint': 2}
            valid_children.sort(key=lambda c: order_map.get(c.name, 999))
        elif joint.name == 'thorax':
            order_map = {'lclavicle': 0, 'rclavicle': 1, 'lowerneck': 2}
            valid_children.sort(key=lambda c: order_map.get(c.name, 999))
        
        return valid_children
    
    def write(self, filepath: str):
        with open(filepath, 'w') as f:
            f.write("HIERARCHY\n")
            self._write_skeleton(f)
            f.write("MOTION\n")
            f.write(f"Frames:\t{self.motion.frame_count}\n")
            f.write(f"Frame Time:\t{1.0 / self.config.fps:.8f}\n")
            self._write_motion(f)
    
    def _has_translation(self, joint: Joint) -> bool:
        return any(dof in ['TX', 'TY', 'TZ'] for dof in joint.dof)
    
    def _get_channel_string(self, has_translation: bool) -> str:
        rot_order = self.config.rotation_order.value
        if has_translation:
            return f"CHANNELS 6 Xposition Yposition Zposition {rot_order}"
        else:
            return f"CHANNELS 3 {rot_order}"
    
    def _write_skeleton(self, f):
        root = self.skeleton.root
        root_pos = self.skeleton.root_position * self.config.scale
        self._write_joint(f, root, root_pos, 0, np.zeros(3))
    
    def _write_joint(self, f, joint: Joint, offset: np.ndarray, depth: int, accumulated: np.ndarray):
        if self._is_skip(joint.name):
            return
        
        if self._is_collapse(joint.name):
            new_accumulated = accumulated + self._get_child_offset(joint)
            for child in self._get_ordered_children(joint):
                self._write_joint(f, child, offset, depth, new_accumulated)
            return
        
        indent = "  " * depth
        name = self._get_name(joint.name)
        
        joint_type = "ROOT" if depth == 0 else "JOINT"
        f.write(f"{indent}{joint_type} {name}\n")
        f.write(f"{indent}{{\n")
        
        total_offset = offset + accumulated
        f.write(f"{indent}  OFFSET {total_offset[0]:.6f} {total_offset[1]:.6f} {total_offset[2]:.6f}\n")
        
        has_trans = self._has_translation(joint)
        f.write(f"{indent}  {self._get_channel_string(has_trans)}\n")
        
        child_offset = self._get_child_offset(joint)
        
        if self._is_end(joint.name):
            f.write(f"{indent}  End Site\n")
            f.write(f"{indent}  {{\n")
            f.write(f"{indent}    OFFSET {child_offset[0]:.6f} {child_offset[1]:.6f} {child_offset[2]:.6f}\n")
            f.write(f"{indent}  }}\n")
        else:
            ordered_children = self._get_ordered_children(joint)
            
            if ordered_children:
                for child in ordered_children:
                    self._write_joint(f, child, child_offset, depth + 1, np.zeros(3))
            else:
                f.write(f"{indent}  End Site\n")
                f.write(f"{indent}  {{\n")
                f.write(f"{indent}    OFFSET {child_offset[0]:.6f} {child_offset[1]:.6f} {child_offset[2]:.6f}\n")
                f.write(f"{indent}  }}\n")
        
        f.write(f"{indent}}}\n")
    
    def _get_child_offset(self, joint: Joint) -> np.ndarray:
        if joint.length > 0 and np.linalg.norm(joint.direction) > 0:
            dir_norm = joint.direction / np.linalg.norm(joint.direction)
            return dir_norm * joint.length * self.config.scale
        return np.zeros(3)
    
    def _write_motion(self, f):
        for frame in self.motion.frames:
            values = []
            self._write_joint_sample(self.skeleton.root, frame, values)
            f.write(" ".join(f"{v:.6f}" for v in values) + "\n")
    
    def _write_joint_sample(self, joint: Joint, frame: MotionFrame, values: List[float]):
        if self._is_skip(joint.name):
            return
        
        if self._is_collapse(joint.name):
            for child in self._get_ordered_children(joint):
                self._write_joint_sample(child, frame, values)
            return
        
        data = frame.joint_data.get(joint.name, np.zeros(len(joint.dof)))
        
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
        
        rotations = self.rot_converter.convert_joint_rotation(joint, data)
        values.extend(rotations)
        
        if not self._is_end(joint.name):
            for child in self._get_ordered_children(joint):
                self._write_joint_sample(child, frame, values)


# ============================================================================
# Main Converter
# ============================================================================

class ASFAMCtoBVH:
    """Main converter class for ASF/AMC to BVH conversion"""
    
    def __init__(self, config: ConversionConfig = None):
        self.config = config or ConversionConfig()
    
    def convert(self, asf_path: str, amc_path: str, output_path: str) -> str:
        print(f"Parsing skeleton: {asf_path}")
        asf_parser = ASFParser()
        skeleton = asf_parser.parse(asf_path)
        
        print(f"Parsing motion: {amc_path}")
        amc_parser = AMCParser(skeleton)
        motion = amc_parser.parse(amc_path)
        
        print(f"Writing BVH: {output_path}")
        writer = BVHWriter(skeleton, motion, self.config)
        writer.write(output_path)
        
        print(f"\nConversion complete!")
        print(f"  Frames: {motion.frame_count}")
        print(f"  Joints: {len(skeleton.joints)}")
        print(f"  FPS: {self.config.fps}")
        print(f"  Scale: {self.config.scale}")
        print(f"  Rotation Order: {self.config.rotation_order.name} ({self.config.rotation_order.value})")
        
        return output_path


def main():
    parser = argparse.ArgumentParser(
        description='Convert ASF/AMC to BVH with proper rotation order',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  # Convert with ZXY rotation order (matches walk_.bvh/Maya format)
  %(prog)s skeleton.asf motion.amc -o output.bvh
  
  # Use walk_.bvh format (collapse joints, rename, scale, ZXY rotation)
  %(prog)s skeleton.asf motion.amc -o output.bvh --walk-bvh
  
  # Use ZYX rotation order (CMU format)
  %(prog)s skeleton.asf motion.amc -o output.bvh -r ZYX
        """
    )
    
    parser.add_argument('asf', help='Input ASF skeleton file')
    parser.add_argument('amc', help='Input AMC motion file')
    parser.add_argument('-o', '--output', default='output.bvh', help='Output BVH file')
    parser.add_argument('-f', '--fps', type=float, default=120.0, help='Frames per second (default: 120)')
    parser.add_argument('-r', '--rotation', choices=['ZXY', 'ZYX', 'XYZ'], default='ZXY',
                       help='Rotation order (default: ZXY for Maya compatibility)')
    parser.add_argument('-s', '--scale', type=float, default=1.0,
                       help='Scale factor for positions (default: 1.0)')
    parser.add_argument('--collapse', action='store_true',
                       help='Collapse intermediate joints (lhipjoint, rhipjoint, etc.)')
    parser.add_argument('--rename', action='store_true',
                       help='Rename joints to Character1_* format')
    parser.add_argument('--walk-bvh', action='store_true',
                       help='Use walk_.bvh format (enables --collapse, --rename, ZXY rotation, and proper scaling)')
    
    args = parser.parse_args()
    
    config = ConversionConfig()
    config.fps = args.fps
    config.rotation_order = RotationOrder[args.rotation]
    config.scale = args.scale
    
    if args.walk_bvh:
        config.rotation_order = RotationOrder.ZXY  # Critical: Maya uses ZXY
        config.scale = (1 / 0.45) * 2.54  # Convert CMU units
        config.collapse_joints = ConversionConfig.walk_bvh_collapse_joints()
        config.skip_joints = ConversionConfig.walk_bvh_skip_joints()
        config.end_joints = ConversionConfig.walk_bvh_end_joints()
        config.joint_name_map = ConversionConfig.walk_bvh_joint_map()
    else:
        if args.collapse:
            config.collapse_joints = ConversionConfig.walk_bvh_collapse_joints()
            config.skip_joints = ConversionConfig.walk_bvh_skip_joints()
            config.end_joints = ConversionConfig.walk_bvh_end_joints()
        
        if args.rename:
            config.joint_name_map = ConversionConfig.walk_bvh_joint_map()
    
    converter = ASFAMCtoBVH(config)
    converter.convert(args.asf, args.amc, args.output)


if __name__ == '__main__':
    main()