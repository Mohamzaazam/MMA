"""
ASF (Acclaim Skeleton Format) parser.

Parses ASF skeleton files and creates Skeleton objects with
complete joint hierarchy, axes, and degrees of freedom.
"""

import numpy as np

from .data_structs import Joint, Skeleton
from .quat_math import QuaternionMath


class ASFParser:
    """Parser for ASF (Acclaim Skeleton Format) files"""
    
    def __init__(self):
        self.skeleton = Skeleton()
    
    def parse(self, filepath: str) -> Skeleton:
        """
        Parse an ASF file.
        
        Args:
            filepath: Path to the ASF file
            
        Returns:
            Skeleton object with complete hierarchy
        """
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
        """Parse units section"""
        parts = line.split()
        if len(parts) >= 2:
            if parts[0] == 'length':
                self.skeleton.length_unit = float(parts[1])
            elif parts[0] == 'angle':
                self.skeleton.angle_unit = parts[1]
    
    def _parse_root(self, line: str):
        """Parse root section"""
        parts = line.split()
        if parts[0] == 'position' and len(parts) >= 4:
            self.skeleton.root_position = np.array([float(x) for x in parts[1:4]])
    
    def _parse_bonedata(self, line: str, current_joint: Joint) -> Joint:
        """Parse bonedata section"""
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
                # Parse axis angles (in degrees)
                axis_angles = np.array([float(x) for x in parts[1:4]])
                current_joint.axis = axis_angles
                if len(parts) > 4:
                    current_joint.axis_order = parts[4]
                
                # Convert to quaternion matching C's parse_joint_rotation()
                # C converts degrees to radians, then uses euler_to_quat
                axis_rad = np.radians(axis_angles)
                current_joint.rotation = QuaternionMath.euler_to_quat(axis_rad, current_joint.axis_order)
            elif parts[0] == 'dof':
                current_joint.dof = [d.upper() for d in parts[1:]]
        
        return current_joint
    
    def _parse_hierarchy(self, line: str) -> str:
        """Parse hierarchy section"""
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