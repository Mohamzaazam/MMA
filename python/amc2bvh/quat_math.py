"""
Quaternion mathematics utilities for rotation operations.

Provides quaternion operations including multiplication, inverse,
and conversions between Euler angles and quaternions.
"""

import numpy as np


class QuaternionMath:
    """Quaternion and rotation utilities"""
    
    @staticmethod
    def euler_to_quat(angles: np.ndarray, order: str = "XYZ") -> np.ndarray:
        """Convert Euler angles (radians) to quaternion [w, x, y, z]"""
        q = np.array([1.0, 0.0, 0.0, 0.0])
        
        for i, axis in enumerate(order):
            angle = angles[i]
            c, s = np.cos(angle / 2), np.sin(angle / 2)
            
            if axis == 'X':
                q_axis = np.array([c, s, 0, 0])
            elif axis == 'Y':
                q_axis = np.array([c, 0, s, 0])
            elif axis == 'Z':
                q_axis = np.array([c, 0, 0, s])
            else:
                continue
            
            q = QuaternionMath.quat_multiply(q_axis, q)
        
        return q
    
    @staticmethod
    def quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
        """Multiply two quaternions"""
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        
        return np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2
        ])
    
    @staticmethod
    def quat_inverse(q: np.ndarray) -> np.ndarray:
        """Compute quaternion inverse"""
        norm_sq = np.sum(q**2)
        return np.array([q[0], -q[1], -q[2], -q[3]]) / norm_sq
    
    @staticmethod
    def quat_to_euler_zxy(q: np.ndarray) -> np.ndarray:
        """Convert quaternion to ZXY Euler angles"""
        w, x, y, z = q
        
        # ZXY decomposition
        sinx = np.clip(2 * (w * x + y * z), -1.0, 1.0)
        cosx = np.sqrt(1 - sinx * sinx)
        
        if cosx > 1e-6:
            rz = np.arctan2(2 * (w * z - x * y), 1 - 2 * (x * x + z * z))
            rx = np.arcsin(sinx)
            ry = np.arctan2(2 * (w * y - x * z), 1 - 2 * (x * x + y * y))
        else:
            # Gimbal lock
            rz = np.arctan2(-2 * (x * y - w * z), 1 - 2 * (y * y + z * z))
            rx = np.pi / 2 * np.sign(sinx)
            ry = 0
        
        return np.array([rz, rx, ry])
    
    @staticmethod
    def quat_to_euler_zyx(q: np.ndarray) -> np.ndarray:
        """Convert quaternion to ZYX Euler angles"""
        w, x, y, z = q
        
        roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
        pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1.0, 1.0))
        yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        
        return np.array([yaw, pitch, roll])
    
    @staticmethod
    def quat_to_euler_xyz(q: np.ndarray) -> np.ndarray:
        """
        Convert quaternion to XYZ Euler angles (roll, pitch, yaw).
        
        Matches C implementation in amc2bvh.c lines 784-793:
        - roll = atan2(2*(w*x + y*z), 1-2*(x*x + y*y))
        - pitch = asin(clamp(2*(w*y - z*x), -0.9999, 0.9999))
        - yaw = atan2(2*(w*z + x*y), 1-2*(y*y + z*z))
        
        Returns:
            Array [roll, pitch, yaw] in radians (XYZ order)
        """
        w, x, y, z = q
        
        roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
        pitch = np.arcsin(np.clip(2 * (w * y - z * x), -0.9999, 0.9999))
        yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        
        return np.array([roll, pitch, yaw])