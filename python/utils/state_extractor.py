#!/usr/bin/env python3
"""
State Extractor - C++ pymss integration for DART-compatible state extraction.

This module uses the pymss C++ backend to extract states that are properly
mapped to the DART skeleton, matching GetTargetPositions() output.
"""

import sys
import numpy as np
from pathlib import Path
from typing import Tuple, Optional


class BVHStateExtractor:
    """
    Extract states from BVH files using the C++ pymss backend.
    
    Ensures proper mapping from BVH channels to DART skeleton DOFs,
    matching exactly what GetTargetPositions() and GetTargetPosAndVel() return.
    
    Attributes:
        num_dofs: Number of skeleton DOFs (typically 56)
        frame_count: Number of frames in the loaded BVH
        frame_time: Time per frame in seconds
    """
    
    def __init__(self, metadata_file: str, build_dir: str = "build"):
        """
        Initialize the state extractor.
        
        Args:
            metadata_file: Path to metadata.txt file (defines skeleton + BVH)
            build_dir: Directory containing the built pymss module
        """
        self.metadata_file = Path(metadata_file)
        self.build_dir = Path(build_dir)
        
        # Add build directory to path for pymss import
        build_path = str(self.build_dir.absolute())
        if build_path not in sys.path:
            sys.path.insert(0, build_path)
        
        try:
            import pymss
            self._pymss = pymss
        except ImportError as e:
            raise ImportError(
                f"Could not import pymss from {build_path}. "
                f"Make sure the C++ code is built. Error: {e}"
            )
        
        # Create environment with 1 slave
        self.env = self._pymss.pymss(str(self.metadata_file.absolute()), 1)
        
        # Get dimensions
        self.num_dofs = self.env.GetNumAction() + 6  # Actions + root DOFs
        self.num_states = self.env.GetNumState()
        self._control_hz = self.env.GetControlHz()
        self._dt = 1.0 / self._control_hz
    
    @property
    def frame_count(self) -> int:
        return self.env.GetBVHFrameCount()
    
    @property
    def frame_time(self) -> float:
        return self.env.GetBVHFrameTime()
    
    @property
    def max_time(self) -> float:
        return self.env.GetBVHMaxTime()
    
    def get_target_positions(self, t: float) -> np.ndarray:
        """Get target joint positions at time t."""
        return self.env.GetTargetPositions(t)
    
    def get_target_pos_and_vel(self, t: float) -> Tuple[np.ndarray, np.ndarray]:
        """Get target positions and velocities at time t."""
        return self.env.GetTargetPosAndVel(t)
    
    def get_state(self, t: float, include_phase: bool = False) -> np.ndarray:
        """Get state at time t (positions + velocities, optionally phase)."""
        pos, vel = self.get_target_pos_and_vel(t)
        
        if include_phase:
            phase = (t % self.max_time) / self.max_time
            return np.concatenate([pos, vel, [phase]])
        return np.concatenate([pos, vel])
    
    def get_state_pair(self, t: float, include_phase: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """Get consecutive state pair (s_t, s_{t+dt})."""
        return self.get_state(t, include_phase), self.get_state(t + self._dt, include_phase)
    
    def extract_all_states(self, include_phase: bool = False, step_frames: int = 1) -> np.ndarray:
        """Extract states for all frames."""
        states = []
        for frame_idx in range(0, self.frame_count, step_frames):
            t = frame_idx * self.frame_time
            states.append(self.get_state(t, include_phase))
        return np.array(states)
    
    def extract_all_pairs(self, include_phase: bool = False, step_frames: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """Extract all consecutive state pairs."""
        states_t, states_t1 = [], []
        for frame_idx in range(0, self.frame_count - 1, step_frames):
            t = frame_idx * self.frame_time
            s_t, s_t1 = self.get_state_pair(t, include_phase)
            states_t.append(s_t)
            states_t1.append(s_t1)
        return np.array(states_t), np.array(states_t1)


def create_state_extractor(
    bvh_file: str,
    skeleton_file: str = "data/human.xml",
    muscle_file: str = "data/muscle284.xml",
    build_dir: str = "build",
    cyclic: bool = False
) -> BVHStateExtractor:
    """
    Create a BVHStateExtractor with a temporary metadata file.
    
    Args:
        bvh_file: Path to BVH file
        skeleton_file: Path to skeleton XML
        muscle_file: Path to muscle XML
        build_dir: Directory containing pymss module
        cyclic: Whether motion is cyclic
        
    Returns:
        Configured BVHStateExtractor
    """
    import tempfile
    
    metadata_content = f"""use_muscle true
con_hz 30
sim_hz 600
skel_file /{skeleton_file}
muscle_file /{muscle_file}
bvh_file /{bvh_file} {'true' if cyclic else 'false'}
reward_param 0.75 0.1 0.0 0.15
"""
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(metadata_content)
        temp_path = f.name
    
    return BVHStateExtractor(temp_path, build_dir)
