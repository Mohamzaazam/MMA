"""
Configuration classes for ASF/AMC to BVH conversion.

Contains conversion configuration options and rotation order definitions.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Set, Optional


class RotationOrder(Enum):
    """Supported rotation orders for BVH output"""
    ZYX = "Zrotation Yrotation Xrotation"  # C default
    ZXY = "Zrotation Xrotation Yrotation"
    XYZ = "Xrotation Yrotation Zrotation"


@dataclass
class ConversionConfig:
    """Configuration for the conversion process"""
    # Rotation order for BVH output (default: ZYX to match C)
    rotation_order: RotationOrder = RotationOrder.ZYX
    
    # Scale factor for positions (default: 1.0 to match C, no scaling)
    scale: float = 1.0
    
    # Frame rate (frames per second)
    fps: float = 120.0
    
    # Joint renaming map (empty = keep original names like C)
    joint_name_map: Dict[str, str] = field(default_factory=dict)
    
    # Joints to collapse into parent (empty = no collapsing like C)
    collapse_joints: Set[str] = field(default_factory=set)
    
    # Joints to skip entirely
    skip_joints: Set[str] = field(default_factory=set)
    
    # Joints that terminate a chain (become end sites)
    end_joints: Set[str] = field(default_factory=set)
    
    @staticmethod
    def walk_bvh_joint_map() -> Dict[str, str]:
        """Joint mapping to match walk_.bvh naming"""
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
        """Joints to collapse for walk_.bvh structure"""
        return {'lhipjoint', 'rhipjoint', 'thorax', 'upperneck'}
    
    @staticmethod
    def walk_bvh_skip_joints() -> Set[str]:
        """Joints to skip for walk_.bvh structure"""
        return {'lhand', 'lfingers', 'lthumb', 'rhand', 'rfingers', 'rthumb'}
    
    @staticmethod
    def walk_bvh_end_joints() -> Set[str]:
        """Terminal joints for walk_.bvh structure"""
        return {'lwrist', 'rwrist', 'head', 'ltoes', 'rtoes'}