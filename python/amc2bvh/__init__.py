"""
ASF/AMC to BVH Converter Package

This converter produces BVH files that match the walk_.bvh skeleton structure:
- Collapses intermediate hip joints (lhipjoint/rhipjoint)
- Collapses thorax into Spine1 
- Collapses upperneck into Neck
- Terminates hands at LeftHand/RightHand (no finger joints)
- Correct root child order: Spine, RightUpLeg, LeftUpLeg
"""

from .data_structs import Joint, Skeleton, MotionFrame, Motion
from .config import RotationOrder, ConversionConfig
from . import quat_math
from .rot_converter import RotationConverter
from .skel_converter import SkeletonConverter
from .asf_parser import ASFParser
from .amc_parser import AMCParser
from .bvh_writer import BVHWriter
# Note: main module not imported here to avoid RuntimeWarning when running as -m
from .truncate_mocap import truncate_bvh, truncate_amc

__version__ = "1.0.0"

__all__ = [
    # Data structures
    'Joint',
    'Skeleton',
    'MotionFrame',
    'Motion',
    
    # Configuration
    'RotationOrder',
    'ConversionConfig',
    
    # Math utilities module
    'quat_math',
    
    # Converters
    'RotationConverter',
    'SkeletonConverter',
    
    # Parsers
    'ASFParser',
    'AMCParser',
    
    # Writers
    'BVHWriter',
    
    # Main converter - import directly from src.main if needed
    # 'ASFAMCtoBVH',
    
    # Utilities
    'truncate_bvh',
    'truncate_amc',
]