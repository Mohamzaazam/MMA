"""
MultimodalEnvManager: Python wrapper for multimodal motion training

This module provides a higher-level interface for training with multiple motions
without requiring C++ modifications to the core MASS library.

Strategy:
    Instead of modifying the C++ Environment/Character to switch BVH files at runtime,
    we create multiple environment instances (one per motion) and randomly sample
    which environment to use during training.

This is a clean, non-invasive approach that:
1. Preserves the original MASS code unchanged
2. Allows flexible motion weighting and selection
3. Enables per-motion performance tracking
4. Supports curriculum learning strategies

Usage:
    # Instead of:
    #   env = pymss.pymss(metadata_file, num_slaves)
    # Use:
    #   env = MultimodalEnvManager(motion_list_file, num_slaves_per_motion)
"""

import os
import random
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class MotionConfig:
    """Configuration for a single motion"""
    name: str
    metadata_path: str
    cyclic: bool
    weight: float = 1.0
    

@dataclass 
class MotionStats:
    """Statistics for tracking per-motion performance"""
    name: str
    episodes: int = 0
    total_reward: float = 0.0
    total_steps: int = 0
    
    @property
    def avg_reward(self) -> float:
        return self.total_reward / max(1, self.episodes)
    
    @property
    def avg_steps(self) -> float:
        return self.total_steps / max(1, self.episodes)


class MultimodalEnvManager:
    """
    Manager for multimodal motion training.
    
    This wraps multiple pymss instances, each configured with a different motion,
    and provides a unified interface for training.
    """
    
    def __init__(self, 
                 motion_configs: List[MotionConfig],
                 num_slaves_per_motion: int = 4,
                 pymss_module = None):
        """
        Initialize multimodal environment manager.
        
        Args:
            motion_configs: List of MotionConfig objects defining available motions
            num_slaves_per_motion: Number of parallel environments per motion
            pymss_module: The pymss module (import pymss), passed in to avoid import issues
        """
        self.motion_configs = motion_configs
        self.num_slaves_per_motion = num_slaves_per_motion
        self.num_motions = len(motion_configs)
        self.total_slaves = num_slaves_per_motion * self.num_motions
        
        self.pymss = pymss_module
        self.envs: List = []  # List of pymss instances
        self.motion_stats: Dict[str, MotionStats] = {}
        
        # Track which motion each "virtual" slave belongs to
        self.slave_to_motion: Dict[int, int] = {}
        
        # Current active motion indices for each slave
        self.active_motion_indices: List[int] = []
        
        # Weights for motion selection (can be adjusted during training)
        self.motion_weights = [c.weight for c in motion_configs]
        
        self._initialized = False
        
    def initialize(self):
        """Initialize all environment instances"""
        if self.pymss is None:
            raise RuntimeError("pymss module not provided. Pass it in constructor.")
            
        print(f"Initializing MultimodalEnvManager with {self.num_motions} motions...")
        
        for i, config in enumerate(self.motion_configs):
            print(f"  [{i+1}/{self.num_motions}] Loading '{config.name}'...")
            
            if not os.path.exists(config.metadata_path):
                raise FileNotFoundError(f"Metadata not found: {config.metadata_path}")
            
            # Create pymss instance for this motion
            env = self.pymss.pymss(config.metadata_path, self.num_slaves_per_motion)
            self.envs.append(env)
            
            # Initialize stats tracking
            self.motion_stats[config.name] = MotionStats(name=config.name)
            
            # Map slaves to motion
            for j in range(self.num_slaves_per_motion):
                slave_id = i * self.num_slaves_per_motion + j
                self.slave_to_motion[slave_id] = i
        
        # Initialize active motion indices
        self.active_motion_indices = list(range(self.num_motions)) * self.num_slaves_per_motion
        
        self._initialized = True
        print(f"  Initialized {len(self.envs)} environments, {self.total_slaves} total slaves")
        
    def _check_initialized(self):
        if not self._initialized:
            raise RuntimeError("Call initialize() before using the environment")
    
    # =========================================================================
    # Environment Interface (mirrors pymss interface)
    # =========================================================================
    
    def GetNumState(self) -> int:
        """Get state dimension (same for all motions)"""
        self._check_initialized()
        return self.envs[0].GetNumState()
    
    def GetNumAction(self) -> int:
        """Get action dimension (same for all motions)"""
        self._check_initialized()
        return self.envs[0].GetNumAction()
    
    def GetSimulationHz(self) -> int:
        self._check_initialized()
        return self.envs[0].GetSimulationHz()
    
    def GetControlHz(self) -> int:
        self._check_initialized()
        return self.envs[0].GetControlHz()
    
    def GetNumSteps(self) -> int:
        self._check_initialized()
        return self.envs[0].GetNumSteps()
    
    def UseMuscle(self) -> bool:
        self._check_initialized()
        return self.envs[0].UseMuscle()
    
    def GetNumMuscles(self) -> int:
        self._check_initialized()
        return self.envs[0].GetNumMuscles()
    
    def GetNumTotalMuscleRelatedDofs(self) -> int:
        self._check_initialized()
        return self.envs[0].GetNumTotalMuscleRelatedDofs()
    
    def Resets(self, RSI: bool = True, randomize_motion: bool = True):
        """
        Reset all environments.
        
        Args:
            RSI: Random State Initialization
            randomize_motion: If True, randomly assign motions to slaves
        """
        self._check_initialized()
        
        if randomize_motion:
            # Randomly assign motions to environment groups
            # (This determines which motion each slave will train on this episode)
            self._randomize_motion_assignments()
        
        # Reset each environment
        for env in self.envs:
            env.Resets(RSI)
    
    def _randomize_motion_assignments(self):
        """
        Randomly assign motions based on weights.
        
        This is called at the start of each episode batch to ensure
        diverse motion coverage during training.
        """
        # For now, we keep the mapping fixed (each env group handles one motion)
        # A more sophisticated approach would dynamically reassign
        pass
    
    def GetStates(self) -> np.ndarray:
        """Get states from all slaves across all motions"""
        self._check_initialized()
        
        states_list = []
        for env in self.envs:
            states_list.append(env.GetStates())
        
        return np.vstack(states_list)
    
    def SetActions(self, actions: np.ndarray):
        """Set actions for all slaves"""
        self._check_initialized()
        
        for i, env in enumerate(self.envs):
            start = i * self.num_slaves_per_motion
            end = start + self.num_slaves_per_motion
            env.SetActions(actions[start:end])
    
    def StepsAtOnce(self):
        """Step all environments"""
        self._check_initialized()
        for env in self.envs:
            env.StepsAtOnce()
    
    def Steps(self, num: int):
        """Step all environments by num steps"""
        self._check_initialized()
        for env in self.envs:
            env.Steps(num)
    
    def IsEndOfEpisodes(self) -> np.ndarray:
        """Check episode termination for all slaves"""
        self._check_initialized()
        
        eoe_list = []
        for env in self.envs:
            eoe_list.append(env.IsEndOfEpisodes())
        
        return np.concatenate(eoe_list)
    
    def GetRewards(self) -> np.ndarray:
        """Get rewards from all slaves"""
        self._check_initialized()
        
        rewards_list = []
        for env in self.envs:
            rewards_list.append(env.GetRewards())
        
        return np.concatenate(rewards_list)
    
    def Reset(self, RSI: bool, slave_id: int):
        """Reset a specific slave"""
        self._check_initialized()
        
        motion_idx = slave_id // self.num_slaves_per_motion
        local_id = slave_id % self.num_slaves_per_motion
        self.envs[motion_idx].Reset(RSI, local_id)
    
    def IsEndOfEpisode(self, slave_id: int) -> bool:
        """Check if specific slave's episode ended"""
        self._check_initialized()
        
        motion_idx = slave_id // self.num_slaves_per_motion
        local_id = slave_id % self.num_slaves_per_motion
        return self.envs[motion_idx].IsEndOfEpisode(local_id)
    
    def GetReward(self, slave_id: int) -> float:
        """Get reward for specific slave"""
        self._check_initialized()
        
        motion_idx = slave_id // self.num_slaves_per_motion
        local_id = slave_id % self.num_slaves_per_motion
        return self.envs[motion_idx].GetReward(local_id)
    
    # =========================================================================
    # Muscle-related methods
    # =========================================================================
    
    def GetMuscleTorques(self) -> np.ndarray:
        self._check_initialized()
        torques_list = []
        for env in self.envs:
            torques_list.append(env.GetMuscleTorques())
        return np.vstack(torques_list)
    
    def GetDesiredTorques(self) -> np.ndarray:
        self._check_initialized()
        torques_list = []
        for env in self.envs:
            torques_list.append(env.GetDesiredTorques())
        return np.vstack(torques_list)
    
    def SetActivationLevels(self, activations: np.ndarray):
        self._check_initialized()
        for i, env in enumerate(self.envs):
            start = i * self.num_slaves_per_motion
            end = start + self.num_slaves_per_motion
            env.SetActivationLevels(activations[start:end])
    
    def ComputeMuscleTuples(self):
        self._check_initialized()
        for env in self.envs:
            env.ComputeMuscleTuples()
    
    def GetMuscleTuplesJtA(self) -> np.ndarray:
        self._check_initialized()
        tuples_list = []
        for env in self.envs:
            tuples_list.append(env.GetMuscleTuplesJtA())
        return np.vstack(tuples_list) if tuples_list else np.array([])
    
    def GetMuscleTuplesTauDes(self) -> np.ndarray:
        self._check_initialized()
        tuples_list = []
        for env in self.envs:
            tuples_list.append(env.GetMuscleTuplesTauDes())
        return np.vstack(tuples_list) if tuples_list else np.array([])
    
    def GetMuscleTuplesL(self) -> np.ndarray:
        self._check_initialized()
        tuples_list = []
        for env in self.envs:
            tuples_list.append(env.GetMuscleTuplesL())
        return np.vstack(tuples_list) if tuples_list else np.array([])
    
    def GetMuscleTuplesb(self) -> np.ndarray:
        self._check_initialized()
        tuples_list = []
        for env in self.envs:
            tuples_list.append(env.GetMuscleTuplesb())
        return np.vstack(tuples_list) if tuples_list else np.array([])
    
    # =========================================================================
    # Multimodal-specific methods
    # =========================================================================
    
    def get_motion_for_slave(self, slave_id: int) -> str:
        """Get the motion name for a given slave"""
        motion_idx = slave_id // self.num_slaves_per_motion
        return self.motion_configs[motion_idx].name
    
    def update_stats(self, slave_id: int, reward: float, steps: int):
        """Update statistics for a completed episode"""
        motion_name = self.get_motion_for_slave(slave_id)
        stats = self.motion_stats[motion_name]
        stats.episodes += 1
        stats.total_reward += reward
        stats.total_steps += steps
    
    def get_stats_summary(self) -> Dict:
        """Get summary of per-motion statistics"""
        return {
            name: {
                'episodes': stats.episodes,
                'avg_reward': stats.avg_reward,
                'avg_steps': stats.avg_steps
            }
            for name, stats in self.motion_stats.items()
        }
    
    def print_stats(self):
        """Print per-motion statistics"""
        print("\n--- Per-Motion Statistics ---")
        for name, stats in self.motion_stats.items():
            print(f"  {name}: episodes={stats.episodes}, "
                  f"avg_reward={stats.avg_reward:.3f}, "
                  f"avg_steps={stats.avg_steps:.1f}")
    
    def set_motion_weights(self, weights: Dict[str, float]):
        """Set weights for motion sampling"""
        for i, config in enumerate(self.motion_configs):
            if config.name in weights:
                self.motion_weights[i] = weights[config.name]


def load_motion_configs_from_list(motion_list_path: str, 
                                   metadata_template: str) -> List[MotionConfig]:
    """
    Load motion configurations from a motion list file.
    
    This creates metadata files for each motion if they don't exist.
    
    Args:
        motion_list_path: Path to motion_list.txt
        metadata_template: Path to template metadata file
        
    Returns:
        List of MotionConfig objects
    """
    configs = []
    
    # Read template
    with open(metadata_template, 'r') as f:
        template_content = f.read()
    
    # Get directory for generated metadata files
    metadata_dir = os.path.dirname(motion_list_path)
    
    # Parse motion list
    with open(motion_list_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            parts = line.split()
            if len(parts) < 2:
                continue
            
            bvh_path = parts[0]
            cyclic = parts[1].lower() in ('true', '1')
            
            # Extract motion name from BVH filename
            motion_name = os.path.splitext(os.path.basename(bvh_path))[0]
            
            # Generate metadata file for this motion
            metadata_path = os.path.join(metadata_dir, f"metadata_{motion_name}.txt")
            
            # Create metadata by modifying template
            # Replace bvh_file line
            new_content = []
            for tline in template_content.split('\n'):
                if tline.strip().startswith('bvh_file'):
                    cyclic_str = 'true' if cyclic else 'false'
                    new_content.append(f"bvh_file {bvh_path} {cyclic_str}")
                else:
                    new_content.append(tline)
            
            with open(metadata_path, 'w') as f:
                f.write('\n'.join(new_content))
            
            configs.append(MotionConfig(
                name=motion_name,
                metadata_path=metadata_path,
                cyclic=cyclic
            ))
            
            print(f"  Generated metadata for '{motion_name}': {metadata_path}")
    
    return configs


# Example usage
if __name__ == "__main__":
    print("MultimodalEnvManager module loaded.")
    print("Usage:")
    print("  from multimodal_env import MultimodalEnvManager, MotionConfig")
    print("  configs = [MotionConfig('walk', 'data/metadata_walk.txt', True)]")
    print("  env = MultimodalEnvManager(configs, num_slaves_per_motion=4)")
    print("  env.initialize()")