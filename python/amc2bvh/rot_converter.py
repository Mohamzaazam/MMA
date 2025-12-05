"""
Rotation conversion utilities for ASF/AMC to BVH conversion.

FIXED VERSION: Corrected axis rotation handling for CMU ASF files.

CHANGE LOG:
-----------
- Fixed: Changed from similarity transform (q_axis * q_motion * q_axis_inv)
  to simple composition (R_axis * R_motion)
  
- The ASF "axis" defines the local DOF coordinate frame orientation.
- Rotations in AMC are in this local frame.
- BVH expects rotations in the parent frame.
- Solution: R_global = R_axis * R_local_motion

This fixes the "crossed legs" issue when using converted BVH files
in simulation pipelines that expect standard BVH conventions.
"""

import numpy as np
from typing import List

from .data_structs import Joint
from .config import RotationOrder


class RotationConverter:
    """Handles rotation conversions for motion data"""
    
    def __init__(self, rotation_order: RotationOrder = RotationOrder.ZXY):
        self.rotation_order = rotation_order
    
    def convert_joint_rotation(self, joint: Joint, data: np.ndarray) -> List[float]:
        """
        Convert rotation from ASF format to BVH format.
        
        CORRECTED ALGORITHM:
        1. Build rotation matrix from AMC channel data in DOF order
        2. Build axis rotation matrix from ASF axis field  
        3. Compose: R_global = R_axis * R_motion
        4. Convert to ZYX Euler angles for BVH output
        
        Args:
            joint: Joint containing axis rotation and DOF info
            data: Motion data array for the joint (already in radians)
            
        Returns:
            List of 3 rotation values in degrees [Zrot, Yrot, Xrot]
        """
        # Build input rotation from DOF channels
        input_order = ""
        input_angles = []
        
        for i, dof in enumerate(joint.dof):
            if dof in ['RX', 'RY', 'RZ']:
                input_order += dof[1]  # Extract 'X', 'Y', or 'Z'
                input_angles.append(data[i] if i < len(data) else 0.0)
        
        if not input_order:
            # No rotation DOF
            return [0.0, 0.0, 0.0]
        
        # Build rotation matrix from motion data
        # Note: input_angles are already in radians from AMC parser
        R_motion = self._euler_to_matrix(np.array(input_angles), input_order)
        
        # Build axis rotation matrix from ASF axis field
        # joint.axis is stored in degrees, axis_order is typically "XYZ"
        axis_angles_rad = np.radians(joint.axis)
        R_axis = self._euler_to_matrix(axis_angles_rad, joint.axis_order)
        
        # FIXED: Use composition instead of similarity transform
        # The motion is defined in the local axis frame, so we compose
        # with the axis rotation to get the parent-relative rotation
        #
        # WRONG: R_axis @ R_motion @ R_axis.T (similarity transform)
        # RIGHT: R_axis @ R_motion (composition)
        R_combined = R_axis @ R_motion
        
        # Convert to ZYX Euler angles for BVH output
        result_deg = self._matrix_to_euler_zyx(R_combined)
        
        # Return in ZYX order for BVH (matches "Zrotation Yrotation Xrotation")
        return [result_deg[0], result_deg[1], result_deg[2]]
    
    def _euler_to_matrix(self, angles_rad: np.ndarray, order: str) -> np.ndarray:
        """
        Convert Euler angles (radians) to rotation matrix.
        
        Applies rotations in the order specified using intrinsic (local axis)
        convention, which matches ASF and standard animation conventions.
        
        Args:
            angles_rad: Array of angles in radians
            order: String specifying rotation order, e.g., "XYZ"
            
        Returns:
            3x3 rotation matrix
        """
        R = np.eye(3)
        
        for i, axis in enumerate(order):
            if i >= len(angles_rad):
                break
            
            c, s = np.cos(angles_rad[i]), np.sin(angles_rad[i])
            
            if axis == 'X':
                Ri = np.array([
                    [1, 0,  0],
                    [0, c, -s],
                    [0, s,  c]
                ])
            elif axis == 'Y':
                Ri = np.array([
                    [ c, 0, s],
                    [ 0, 1, 0],
                    [-s, 0, c]
                ])
            else:  # Z
                Ri = np.array([
                    [c, -s, 0],
                    [s,  c, 0],
                    [0,  0, 1]
                ])
            
            # Intrinsic rotation: post-multiply
            R = R @ Ri
        
        return R
    
    def _matrix_to_euler_zyx(self, R: np.ndarray) -> np.ndarray:
        """
        Extract ZYX Euler angles (degrees) from rotation matrix.
        
        For BVH output with "Zrotation Yrotation Xrotation" channel order.
        
        The ZYX decomposition is: R = Rz * Ry * Rx
        
        Args:
            R: 3x3 rotation matrix
            
        Returns:
            Array [Z, Y, X] angles in degrees
        """
        # ZYX decomposition
        # sin(Y) = -R[2,0]
        sy = np.clip(-R[2, 0], -1.0, 1.0)
        cy = np.sqrt(max(0, 1 - sy*sy))
        
        if cy > 1e-6:
            # Normal case
            z = np.arctan2(R[1, 0], R[0, 0])
            y = np.arcsin(sy)
            x = np.arctan2(R[2, 1], R[2, 2])
        else:
            # Gimbal lock: Y = ±90°
            z = np.arctan2(-R[0, 1], R[1, 1])
            y = np.pi / 2 * np.sign(sy)
            x = 0.0
        
        return np.degrees(np.array([z, y, x]))
    
    @staticmethod
    def degrees_to_radians(angles: np.ndarray) -> np.ndarray:
        """Convert angles from degrees to radians"""
        return np.radians(angles)
    
    @staticmethod
    def radians_to_degrees(angles: np.ndarray) -> np.ndarray:
        """Convert angles from radians to degrees"""
        return np.degrees(angles)