"""
Skeleton conversion utilities for ASF/AMC to BVH conversion.

Handles skeleton hierarchy manipulation, joint collapsing, and
offset calculations for converting ASF skeletons to BVH format.
"""

import numpy as np
from typing import Dict, List, Set

from .data_structs import Joint, Skeleton


class SkeletonConverter:
    """Handles skeleton structure conversions and manipulations"""
    
    # Joints to completely skip (collapse into parent)
    DEFAULT_COLLAPSE_JOINTS: Set[str] = {'lhipjoint', 'rhipjoint', 'thorax', 'upperneck'}
    
    # Joints that terminate a chain (become end sites, children are ignored)
    DEFAULT_END_JOINTS: Set[str] = {'lwrist', 'rwrist', 'head', 'ltoes', 'rtoes'}
    
    # Joints to skip entirely (not included in output)
    DEFAULT_SKIP_JOINTS: Set[str] = {'lhand', 'lfingers', 'lthumb', 'rhand', 'rfingers', 'rthumb'}
    
    def __init__(self,
                 collapse_joints: Set[str] = None,
                 end_joints: Set[str] = None,
                 skip_joints: Set[str] = None):
        """
        Initialize skeleton converter with joint classification.
        
        Args:
            collapse_joints: Joints to collapse into their parent
            end_joints: Joints that become end sites
            skip_joints: Joints to skip entirely
        """
        self.collapse_joints = collapse_joints or self.DEFAULT_COLLAPSE_JOINTS
        self.end_joints = end_joints or self.DEFAULT_END_JOINTS
        self.skip_joints = skip_joints or self.DEFAULT_SKIP_JOINTS
        
        # Build collapse mapping
        self.collapse_map = self._build_collapse_mapping()
    
    def _build_collapse_mapping(self) -> Dict[str, str]:
        """
        Build mapping for collapsed joint motion accumulation.
        
        Returns:
            Dictionary mapping collapsed joints to their target joints
        """
        return {
            'lhipjoint': 'lfemur',
            'rhipjoint': 'rfemur',
            'thorax': 'upperback',
            'upperneck': 'lowerneck',
        }
    
    def get_accumulated_offset(self, joint: Joint, scale: float = 1.0) -> np.ndarray:
        """
        Get offset, accumulating through any collapsed parents.
        
        Args:
            joint: The joint to get offset for
            scale: Scale factor to apply
            
        Returns:
            Accumulated offset array
        """
        offset = joint.offset.copy()
        
        # Check if parent was collapsed
        parent = joint.parent
        while parent and parent.name in self.collapse_joints:
            offset += parent.offset
            parent = parent.parent
        
        return offset * scale
    
    def get_end_site_offset(self, joint: Joint, skeleton: Skeleton, scale: float = 1.0) -> np.ndarray:
        """
        Get end site offset for terminal joints.
        
        Args:
            joint: The terminal joint
            skeleton: The skeleton structure
            scale: Scale factor to apply
            
        Returns:
            End site offset array
        """
        # For joints like lwrist, we want to include the length of the hand chain
        if joint.name in ['lwrist', 'rwrist']:
            # Find lhand/rhand and use their direction * length
            hand_name = 'lhand' if joint.name == 'lwrist' else 'rhand'
            hand = skeleton.joints.get(hand_name)
            if hand:
                return hand.direction * hand.length * scale
        
        # Default: use the joint's own direction * length
        return joint.direction * joint.length * scale
    
    def get_children_to_process(self, joint: Joint) -> List[Joint]:
        """
        Get children that should be processed (skip collapsed/skip joints).
        
        Args:
            joint: The parent joint
            
        Returns:
            List of children joints to process
        """
        result = []
        for child in joint.children:
            if child.name in self.skip_joints:
                continue
            elif child.name in self.collapse_joints:
                # Add the collapsed joint's children instead
                result.extend(self.get_children_to_process(child))
            else:
                result.append(child)
        
        # Reorder children to match walk_.bvh structure
        # Under Spine1 (upperback), order should be: LeftShoulder, RightShoulder, Neck
        if joint.name == 'upperback':
            order_map = {'lclavicle': 0, 'rclavicle': 1, 'lowerneck': 2}
            result.sort(key=lambda x: order_map.get(x.name, 999))
        
        return result
    
    def get_ordered_root_children(self, skeleton: Skeleton, scale: float = 1.0) -> List[dict]:
        """
        Get root children in walk_.bvh order: Spine, RightUpLeg, LeftUpLeg.
        
        Args:
            skeleton: The skeleton structure
            scale: Scale factor to apply
            
        Returns:
            List of dicts with 'joint' and 'accumulated_offset' keys
        """
        result = []
        
        # Find lowerback (Spine)
        lowerback = skeleton.joints.get('lowerback')
        if lowerback:
            result.append({'joint': lowerback, 'accumulated_offset': np.zeros(3)})
        
        # Find rfemur through rhipjoint (RightUpLeg)
        rhipjoint = skeleton.joints.get('rhipjoint')
        if rhipjoint:
            rfemur = skeleton.joints.get('rfemur')
            if rfemur:
                # Accumulate hipjoint offset
                accumulated = rhipjoint.direction * rhipjoint.length * scale
                result.append({'joint': rfemur, 'accumulated_offset': accumulated})
        
        # Find lfemur through lhipjoint (LeftUpLeg)
        lhipjoint = skeleton.joints.get('lhipjoint')
        if lhipjoint:
            lfemur = skeleton.joints.get('lfemur')
            if lfemur:
                accumulated = lhipjoint.direction * lhipjoint.length * scale
                result.append({'joint': lfemur, 'accumulated_offset': accumulated})
        
        return result
    
    def is_collapse_joint(self, joint_name: str) -> bool:
        """Check if joint should be collapsed"""
        return joint_name in self.collapse_joints
    
    def is_end_joint(self, joint_name: str) -> bool:
        """Check if joint is an end joint"""
        return joint_name in self.end_joints
    
    def is_skip_joint(self, joint_name: str) -> bool:
        """Check if joint should be skipped"""
        return joint_name in self.skip_joints