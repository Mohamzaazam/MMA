"""
Rotation conversion utilities for ASF/AMC to BVH conversion.

Handles conversion of rotation data between different formats and
coordinate systems used in ASF and BVH files.
"""

import numpy as np
from typing import List

from .data_structs import Joint
from .config import RotationOrder
from .quat_math import QuaternionMath


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
        input_order = ""
        input_angles = []
        
        for i, dof in enumerate(joint.dof):
            if dof in ['RX', 'RY', 'RZ']:
                input_order += dof[1]  # Extract 'X', 'Y', or 'Z'
                input_angles.append(data[i] if i < len(data) else 0.0)
        
        if not input_order:
            # No rotation DOF
            return [0.0, 0.0, 0.0]
        
        # Convert motion angles to quaternion
        # Note: data should already be in radians from AMC parser
        q_motion = QuaternionMath.euler_to_quat(np.array(input_angles), input_order)
        
        # Apply joint local transform (C lines 450-453):
        # q_combined = q_axis * q_motion * q_axis_inv
        q_axis = joint.rotation  # Pre-computed quaternion from ASF axis
        q_axis_inv = QuaternionMath.quat_inverse(q_axis)
        
        q_combined = QuaternionMath.quat_multiply(q_axis, q_motion)
        q_combined = QuaternionMath.quat_multiply(q_combined, q_axis_inv)
        
        # Convert to XYZ Euler angles (C's quat_to_euler_xyz)
        # Returns [roll, pitch, yaw] which are [X, Y, Z] rotations
        result = QuaternionMath.quat_to_euler_xyz(q_combined)
        
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