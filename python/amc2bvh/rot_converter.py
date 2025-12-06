"""
Rotation conversion utilities for ASF/AMC to BVH conversion.

Handles conversion of rotation data between different formats and
coordinate systems used in ASF and BVH files.
"""

import numpy as np
from typing import List

from .data_structs import Joint
from .config import RotationOrder
from .transformations import (
    quaternion_from_euler,
    quaternion_multiply,
    quaternion_inverse,
    euler_from_quaternion,
)


def asf_order_to_axes(order: str) -> str:
    """
    Convert ASF axis order (e.g., "XYZ") to transformations.py axes format (e.g., "sxyz").
    
    The 's' prefix indicates static/extrinsic rotations (fixed frame of reference).
    
    Args:
        order: ASF-style axis order like "XYZ", "ZXY", etc.
        
    Returns:
        transformations.py-style axes string like "sxyz", "szxy", etc.
    """
    return 's' + order.lower()


class RotationConverter:
    """Handles rotation conversions for motion data"""
    
    def __init__(self, rotation_order: RotationOrder = RotationOrder.ZXY):
        self.rotation_order = rotation_order
    
    def convert_joint_rotation(self, joint: Joint, data: np.ndarray) -> List[float]:
        """
        Convert rotation from ASF format to BVH format.
        
        Matches C's write_bvh_joint_sample() (lines 438-458):
        1. Build sample_rotation from channel data
        2. Apply: q_combined = q_axis * q_motion * q_axis_inv  
        3. Convert to XYZ Euler angles
        4. Output as degrees in reverse order (Z, Y, X for BVH)
        
        Args:
            joint: Joint containing rotation quaternion and DOF
            data: Motion data array for the joint (already in radians)
            
        Returns:
            List of 3 rotation values in degrees [Zrot, Yrot, Xrot]
        """
        # Build input rotation from DOF channels
        # Match C's sample_rotation construction (lines 439-447)
        # Build a full XYZ rotation with zeros for missing axes
        xyz_angles = [0.0, 0.0, 0.0]  # X, Y, Z
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
            # No rotation DOF
            return [0.0, 0.0, 0.0]
        
        # Convert motion angles to quaternion using transformations.py
        # Always use XYZ order with all 3 angles
        q_motion = quaternion_from_euler(xyz_angles[0], xyz_angles[1], xyz_angles[2], axes='sxyz')
        
        # Apply joint local transform (C lines 450-453):
        # q_combined = q_axis * q_motion * q_axis_inv
        q_axis = joint.rotation  # Pre-computed quaternion from ASF axis
        q_axis_inv = quaternion_inverse(q_axis)
        
        q_combined = quaternion_multiply(q_axis, q_motion)
        q_combined = quaternion_multiply(q_combined, q_axis_inv)
        
        # Convert to XYZ Euler angles
        # euler_from_quaternion returns angles in the specified axis order
        result = euler_from_quaternion(q_combined, axes='sxyz')
        
        # Convert to degrees
        result_deg = np.degrees(result)
        
        # Return in ZYX order for BVH output (C lines 456-458)
        # C outputs: angles[2], angles[1], angles[0] which is Z, Y, X
        return [result_deg[2], result_deg[1], result_deg[0]]
    
    @staticmethod
    def degrees_to_radians(angles: np.ndarray) -> np.ndarray:
        """Convert angles from degrees to radians"""
        return np.radians(angles)
    
    @staticmethod
    def radians_to_degrees(angles: np.ndarray) -> np.ndarray:
        """Convert angles from radians to degrees"""
        return np.degrees(angles)