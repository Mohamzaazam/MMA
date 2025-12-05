"""
AMC (Acclaim Motion Capture) parser.

Parses AMC motion files and creates Motion objects containing
frame-by-frame joint rotation and position data.
"""

import numpy as np

from .data_structs import Skeleton, Motion, MotionFrame


class AMCParser:
    """Parser for AMC (Acclaim Motion Capture) files"""
    
    def __init__(self, skeleton: Skeleton):
        """
        Initialize parser with skeleton reference.
        
        Args:
            skeleton: Skeleton object containing joint definitions
        """
        self.skeleton = skeleton
        self.motion = Motion()
    
    def parse(self, filepath: str) -> Motion:
        """
        Parse an AMC file.
        
        Args:
            filepath: Path to the AMC file
            
        Returns:
            Motion object containing all frames
        """
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
        """
        Parse joint data from a line.
        
        Args:
            line: Line containing joint name and values
            frame: Current frame to add data to
            is_degrees: Whether angles are in degrees (True) or radians (False)
        """
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