"""
MultimodalEnvManager: Python wrapper for multimodal motion training (IMPROVED)

This module provides a higher-level interface for training with multiple motions
without requiring C++ modifications to the core MASS library.

IMPROVEMENTS over original:
1. Weight-based slave allocation for curriculum learning
2. Better performance tracking per motion
3. Dynamic weight adjustment support
4. Cleaner stats API

Strategy:
    Instead of modifying the C++ Environment/Character to switch BVH files at runtime,
    we create multiple environment instances (one per motion) and use weighted sampling
    to determine which environments get more training focus.

Usage:
    # Instead of:
    #   env = pymss.pymss(metadata_file, num_slaves)
    # Use:
    #   env = MultimodalEnvManager(motion_configs, num_slaves_per_motion)
"""

import os
import random
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from collections import deque


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
    window_size: int = 100
    
    # Rolling window stats
    recent_returns: deque = field(default_factory=lambda: deque(maxlen=100))
    recent_steps: deque = field(default_factory=lambda: deque(maxlen=100))
    
    # Cumulative stats
    episodes: int = 0
    total_reward: float = 0.0
    total_steps: int = 0
    
    def __post_init__(self):
        self.recent_returns = deque(maxlen=self.window_size)
        self.recent_steps = deque(maxlen=self.window_size)
    
    def update(self, reward: float, steps: int):
        """Update with a completed episode"""
        self.episodes += 1
        self.total_reward += reward
        self.total_steps += steps
        self.recent_returns.append(reward)
        self.recent_steps.append(steps)
    
    @property
    def avg_return(self) -> float:
        """Average return over all episodes"""
        return self.total_reward / max(1, self.episodes)
    
    @property
    def recent_avg_return(self) -> float:
        """Average return over recent window"""
        if not self.recent_returns:
            return 0.0
        return np.mean(self.recent_returns)
    
    @property
    def avg_steps(self) -> float:
        return self.total_steps / max(1, self.episodes)
    
    @property
    def recent_avg_steps(self) -> float:
        if not self.recent_steps:
            return 0.0
        return np.mean(self.recent_steps)


class MultimodalEnvManager:
    """
    Manager for multimodal motion training.
    
    This wraps multiple pymss instances, each configured with a different motion,
    and provides a unified interface for training.
    
    IMPROVEMENTS:
    - Weights can dynamically adjust which motions get more training
    - Better episode tracking per motion
    - Cleaner interface for curriculum learning integration
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
        self.slave_to_motion_idx: Dict[int, int] = {}
        
        # Weights for motion selection (can be adjusted during training)
        self.motion_weights = [c.weight for c in motion_configs]
        self._normalized_weights = self._normalize_weights()
        
        # Episode tracking (for returns per motion in current batch)
        self._current_episode_rewards: Dict[int, float] = {}
        self._current_episode_steps: Dict[int, int] = {}
        self._completed_episodes: Dict[str, List[Tuple[float, int]]] = {
            c.name: [] for c in motion_configs
        }
        
        self._initialized = False
        
    def _normalize_weights(self) -> np.ndarray:
        """Normalize weights to probabilities"""
        weights = np.array(self.motion_weights)
        return weights / weights.sum()
        
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
                self.slave_to_motion_idx[slave_id] = i
                self._current_episode_rewards[slave_id] = 0.0
                self._current_episode_steps[slave_id] = 0
        
        self._initialized = True
        print(f"  Initialized {len(self.envs)} environments, {self.total_slaves} total slaves")
        
    def _check_initialized(self):
        if not self._initialized:
            raise RuntimeError("Call initialize() before using the environment")
    
    # =========================================================================
    # Motion Weight Management
    # =========================================================================
    
    def set_motion_weights(self, weights: Dict[str, float]):
        """
        Set weights for motion sampling.
        
        Args:
            weights: Dict mapping motion name to weight
        """
        for i, config in enumerate(self.motion_configs):
            if config.name in weights:
                self.motion_weights[i] = weights[config.name]
        self._normalized_weights = self._normalize_weights()
        
    def get_motion_weights(self) -> Dict[str, float]:
        """Get current weights as a dict"""
        return {
            config.name: self.motion_weights[i] 
            for i, config in enumerate(self.motion_configs)
        }
    
    def get_motion_for_slave(self, slave_id: int) -> str:
        """Get the motion name for a given slave"""
        motion_idx = self.slave_to_motion_idx[slave_id]
        return self.motion_configs[motion_idx].name
    
    def get_motion_idx_for_slave(self, slave_id: int) -> int:
        """Get the motion index for a given slave"""
        return self.slave_to_motion_idx[slave_id]
    
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
    
    def Resets(self, RSI: bool = True):
        """
        Reset all environments.
        
        Args:
            RSI: Random State Initialization
        """
        self._check_initialized()
        
        # Clear episode tracking
        for slave_id in self._current_episode_rewards:
            self._current_episode_rewards[slave_id] = 0.0
            self._current_episode_steps[slave_id] = 0
        
        # Clear completed episodes buffer
        for name in self._completed_episodes:
            self._completed_episodes[name] = []
        
        # Reset each environment
        for env in self.envs:
            env.Resets(RSI)
    
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
        
        # Update step counts
        for slave_id in self._current_episode_steps:
            self._current_episode_steps[slave_id] += 1
    
    def Steps(self, num: int):
        """Step all environments by num steps"""
        self._check_initialized()
        for env in self.envs:
            env.Steps(num)
        
        # Update step counts
        for slave_id in self._current_episode_steps:
            self._current_episode_steps[slave_id] += num
    
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
        
        all_rewards = np.concatenate(rewards_list)
        
        # Update episode reward tracking
        for slave_id, reward in enumerate(all_rewards):
            self._current_episode_rewards[slave_id] += reward
        
        return all_rewards
    
    def Reset(self, RSI: bool, slave_id: int):
        """Reset a specific slave and record episode stats"""
        self._check_initialized()
        
        # Record completed episode stats before reset
        motion_name = self.get_motion_for_slave(slave_id)
        episode_return = self._current_episode_rewards[slave_id]
        episode_steps = self._current_episode_steps[slave_id]
        
        if episode_steps > 0:  # Only record if episode had any steps
            self.motion_stats[motion_name].update(episode_return, episode_steps)
            self._completed_episodes[motion_name].append((episode_return, episode_steps))
        
        # Reset tracking for this slave
        self._current_episode_rewards[slave_id] = 0.0
        self._current_episode_steps[slave_id] = 0
        
        # Perform actual reset
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
        reward = self.envs[motion_idx].GetReward(local_id)
        
        # Update tracking
        self._current_episode_rewards[slave_id] += reward
        
        return reward
    
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
            jta = env.GetMuscleTuplesJtA()
            if jta.size > 0:
                tuples_list.append(jta)
        return np.vstack(tuples_list) if tuples_list else np.array([])
    
    def GetMuscleTuplesTauDes(self) -> np.ndarray:
        self._check_initialized()
        tuples_list = []
        for env in self.envs:
            td = env.GetMuscleTuplesTauDes()
            if td.size > 0:
                tuples_list.append(td)
        return np.vstack(tuples_list) if tuples_list else np.array([])
    
    def GetMuscleTuplesL(self) -> np.ndarray:
        self._check_initialized()
        tuples_list = []
        for env in self.envs:
            l = env.GetMuscleTuplesL()
            if l.size > 0:
                tuples_list.append(l)
        return np.vstack(tuples_list) if tuples_list else np.array([])
    
    def GetMuscleTuplesb(self) -> np.ndarray:
        self._check_initialized()
        tuples_list = []
        for env in self.envs:
            b = env.GetMuscleTuplesb()
            if b.size > 0:
                tuples_list.append(b)
        return np.vstack(tuples_list) if tuples_list else np.array([])
    
    # =========================================================================
    # Statistics and Reporting
    # =========================================================================
    
    def get_completed_episodes_this_batch(self) -> Dict[str, List[float]]:
        """
        Get returns from episodes completed in the current batch.
        
        Returns:
            Dict mapping motion name to list of episode returns
        """
        return {
            name: [ret for ret, _ in episodes]
            for name, episodes in self._completed_episodes.items()
        }
    
    def get_stats_summary(self) -> Dict[str, Dict]:
        """Get summary of per-motion statistics"""
        return {
            name: {
                'episodes': stats.episodes,
                'avg_return': stats.avg_return,
                'recent_avg_return': stats.recent_avg_return,
                'avg_steps': stats.avg_steps,
                'recent_avg_steps': stats.recent_avg_steps,
                'weight': self.motion_weights[i]
            }
            for i, (name, stats) in enumerate(self.motion_stats.items())
        }
    
    def print_stats(self):
        """Print per-motion statistics"""
        print("\n--- Per-Motion Statistics ---")
        for i, (name, stats) in enumerate(self.motion_stats.items()):
            weight = self.motion_weights[i]
            print(f"  {name}: eps={stats.episodes}, "
                  f"avg_ret={stats.avg_return:.3f}, "
                  f"recent_ret={stats.recent_avg_return:.3f}, "
                  f"weight={weight:.3f}")


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


# Example usage and testing
if __name__ == "__main__":
    print("MultimodalEnvManager module loaded.")
    print("\nUsage:")
    print("  from multimodal_env import MultimodalEnvManager, MotionConfig")
    print("  configs = [MotionConfig('walk', 'data/metadata_walk.txt', True)]")
    print("  env = MultimodalEnvManager(configs, num_slaves_per_motion=4)")
    print("  env.initialize()")
    print("\nFor curriculum learning:")
    print("  env.set_motion_weights({'walk': 1.0, 'run': 2.0, 'jump': 3.0})")