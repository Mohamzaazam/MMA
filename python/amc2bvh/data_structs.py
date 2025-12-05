"""
Data structures for ASF/AMC to BVH conversion.

Contains dataclasses for representing skeleton joints, skeleton hierarchy,
motion frames, and complete motion data.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class Joint:
    """Represents a joint in the skeleton hierarchy"""
    name: str
    direction: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0]))
    length: float = 0.0
    axis: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0]))
    axis_order: str = "XYZ"
    rotation: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0]))  # Quaternion [w,x,y,z]
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