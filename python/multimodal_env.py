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

PROGRESSIVE CURRICULUM - FULL RESOURCE UTILIZATION:
    - ALWAYS uses all available slaves (total_budget)
    - Distributes slaves evenly among active motions
    - Example with 36 total slaves:
      - 1 motion active: 36 slaves for that motion
      - 2 motions active: 18 slaves each
      - 3 motions active: 12 slaves each
      - 6 motions active: 6 slaves each

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
    
    PROGRESSIVE CURRICULUM - FULL RESOURCE UTILIZATION:
        Always uses ALL available slaves, distributed among active motions.
        When 1 motion active: all slaves go to that motion.
        When N motions active: slaves split evenly (total_budget / N each).
    """
    
    def __init__(self, 
                 motion_configs: List[MotionConfig],
                 num_slaves_per_motion: int = 4,
                 pymss_module = None,
                 progressive_order: Optional[List[str]] = None):
        """
        Initialize multimodal environment manager.
        
        Args:
            motion_configs: List of MotionConfig objects defining available motions
            num_slaves_per_motion: Number of parallel environments per motion (used for total budget)
            pymss_module: The pymss module (import pymss), passed in to avoid import issues
            progressive_order: Order for progressive curriculum (determines slave allocation)
        """
        self.motion_configs = motion_configs
        self.num_slaves_per_motion = num_slaves_per_motion
        self.num_motions = len(motion_configs)
        
        # Total slave budget = what we'd have if all motions were active with num_slaves_per_motion each
        self.total_slave_budget = num_slaves_per_motion * self.num_motions
        
        self.pymss = pymss_module
        self.envs: Dict[int, object] = {}  # motion_index -> pymss instance
        self.motion_stats: Dict[str, MotionStats] = {}
        
        # Progressive order (determines slave allocation per motion)
        if progressive_order:
            self._progressive_order = progressive_order
        else:
            self._progressive_order = [c.name for c in motion_configs]
        
        # Map motion name to index in motion_configs
        self._name_to_idx = {c.name: i for i, c in enumerate(motion_configs)}
        
        # Map motion name to its position in progressive order
        self._name_to_progressive_pos = {name: i for i, name in enumerate(self._progressive_order)}
        
        # Weights for motion selection (can be adjusted during training)
        self.motion_weights = [c.weight for c in motion_configs]
        
        # =====================================================================
        # PROGRESSIVE CURRICULUM: Active motion tracking
        # =====================================================================
        # Names of motions that are currently active (unlocked)
        self._active_motion_names: List[str] = []
        
        # Pre-calculated slave counts for each motion (based on progressive position)
        # Motion at position i gets total_budget / (i+1) slaves when it unlocks
        self._max_slaves_per_motion: Dict[str, int] = {}
        
        # Current slaves being used per motion (changes as more motions unlock)
        self._current_slaves_per_motion: Dict[str, int] = {}
        # =====================================================================
        
        self._initialized = False
        
    def _calculate_slave_allocation(self):
        """
        Pre-calculate how many slaves each motion needs.
        
        Motion at progressive position i unlocks when i+1 motions are active.
        At that point, total_budget is split (i+1) ways.
        So motion[i] gets total_budget // (i+1) slaves at unlock time.
        
        We create each motion with this "peak" allocation, then use subsets
        as more motions unlock.
        """
        for i, name in enumerate(self._progressive_order):
            # When motion at position i unlocks, there are (i+1) active motions
            # Each gets total_budget // (i+1) slaves
            slaves_at_unlock = self.total_slave_budget // (i + 1)
            self._max_slaves_per_motion[name] = slaves_at_unlock
            
        print(f"  Slave allocation (progressive order):")
        for name, count in self._max_slaves_per_motion.items():
            print(f"    {name}: {count} slaves (max)")
        
    def initialize(self):
        """Initialize environment instances with appropriate slave counts."""
        if self.pymss is None:
            raise RuntimeError("pymss module not provided. Pass it in constructor.")
            
        print(f"Initializing MultimodalEnvManager with {self.num_motions} motions...")
        print(f"  Total slave budget: {self.total_slave_budget}")
        
        # Calculate slave allocation based on progressive order
        self._calculate_slave_allocation()
        
        # Create pymss instances with appropriate slave counts
        for i, config in enumerate(self.motion_configs):
            motion_name = config.name
            slave_count = self._max_slaves_per_motion.get(motion_name, self.num_slaves_per_motion)
            
            print(f"  [{i+1}/{self.num_motions}] Loading '{motion_name}' with {slave_count} slaves...")
            
            if not os.path.exists(config.metadata_path):
                raise FileNotFoundError(f"Metadata not found: {config.metadata_path}")
            
            # Create pymss instance for this motion with calculated slave count
            env = self.pymss.pymss(config.metadata_path, slave_count)
            self.envs[i] = env
            
            # Initialize stats tracking
            self.motion_stats[config.name] = MotionStats(name=config.name)
        
        # Initialize with no active motions (will be set by set_active_motions)
        self._active_motion_names = []
        self._current_slaves_per_motion = {}
        
        self._initialized = True
        
        total_created = sum(self._max_slaves_per_motion.values())
        print(f"  Initialized {len(self.envs)} environments, {total_created} total slave instances")
        
    def _check_initialized(self):
        if not self._initialized:
            raise RuntimeError("Call initialize() before using the environment")
    
    # =========================================================================
    # PROGRESSIVE CURRICULUM: Active motion management with FULL RESOURCE USE
    # =========================================================================
    
    def set_active_motions(self, motion_names: List[str]):
        """
        Set which motions are currently active (unlocked).
        
        FULL RESOURCE UTILIZATION:
        - Always uses total_slave_budget slaves
        - Distributes evenly among active motions
        - Each active motion gets total_slave_budget // num_active slaves
        
        Args:
            motion_names: List of motion names that are currently unlocked
        """
        self._check_initialized()
        
        # Check for newly added motions
        old_set = set(self._active_motion_names)
        new_set = set(motion_names)
        newly_added = new_set - old_set
        
        self._active_motion_names = list(motion_names)
        num_active = len(motion_names)
        
        if num_active == 0:
            self._current_slaves_per_motion = {}
            return
        
        # Calculate even distribution of slaves
        slaves_per_active_motion = self.total_slave_budget // num_active
        
        # Update current slave counts
        self._current_slaves_per_motion = {}
        for name in motion_names:
            # Can't use more slaves than the motion was created with
            max_for_this = self._max_slaves_per_motion.get(name, slaves_per_active_motion)
            self._current_slaves_per_motion[name] = min(slaves_per_active_motion, max_for_this)
        
        # Report allocation
        total_used = sum(self._current_slaves_per_motion.values())
        print(f"  Active motions: {num_active}, slaves/motion: {slaves_per_active_motion}, total: {total_used}")
        for name, count in self._current_slaves_per_motion.items():
            max_avail = self._max_slaves_per_motion.get(name, 0)
            status = "(NEW)" if name in newly_added else ""
            print(f"    {name}: using {count}/{max_avail} slaves {status}")
        
        # Reset newly added motion environments
        if newly_added:
            for name in newly_added:
                motion_idx = self._name_to_idx[name]
                print(f"  Resetting environment for newly unlocked motion: {name}")
                self.envs[motion_idx].Resets(True)
    
    def get_active_motion_names(self) -> List[str]:
        """Get names of currently active motions."""
        return list(self._active_motion_names)
    
    @property
    def active_slave_count(self) -> int:
        """
        Total number of active slaves (should always equal total_slave_budget 
        when at least one motion is active).
        """
        return sum(self._current_slaves_per_motion.values())
    
    @property
    def total_slaves(self) -> int:
        """Alias for active_slave_count (for compatibility)."""
        return self.active_slave_count
    
    def _get_slaves_for_motion(self, motion_name: str) -> int:
        """Get current slave count for a specific motion."""
        return self._current_slaves_per_motion.get(motion_name, 0)
    
    def _external_to_internal(self, external_slave_id: int) -> Tuple[str, int]:
        """
        Map external slave ID to internal (motion_name, local_slave_id).
        
        External IDs are contiguous: 0 to active_slave_count-1
        Distributed across active motions based on current allocation.
        
        Args:
            external_slave_id: Slave ID as seen by training code (0 to active_slave_count-1)
            
        Returns:
            Tuple of (motion_name, local_slave_id_within_motion)
        """
        if not self._active_motion_names:
            raise RuntimeError("No active motions!")
        
        # Find which motion this slave belongs to
        cumulative = 0
        for name in self._active_motion_names:
            slaves_for_this = self._current_slaves_per_motion[name]
            if external_slave_id < cumulative + slaves_for_this:
                local_id = external_slave_id - cumulative
                return name, local_id
            cumulative += slaves_for_this
        
        raise ValueError(f"Invalid external slave ID {external_slave_id}, "
                        f"active_slave_count={self.active_slave_count}")
    
    def _get_motion_name_for_external_slave(self, external_slave_id: int) -> str:
        """Get the motion name for a given external slave ID."""
        motion_name, _ = self._external_to_internal(external_slave_id)
        return motion_name
    
    def _get_motion_idx_for_external_slave(self, external_slave_id: int) -> int:
        """Get the motion config index for a given external slave ID."""
        motion_name, _ = self._external_to_internal(external_slave_id)
        return self._name_to_idx[motion_name]
    
    # =========================================================================
    # Environment Interface (mirrors pymss interface) - FULL RESOURCE USE
    # =========================================================================
    
    def GetNumState(self) -> int:
        """Get state dimension (same for all motions)"""
        self._check_initialized()
        first_motion_idx = self._name_to_idx[self.motion_configs[0].name]
        return self.envs[first_motion_idx].GetNumState()
    
    def GetNumAction(self) -> int:
        """Get action dimension (same for all motions)"""
        self._check_initialized()
        first_motion_idx = self._name_to_idx[self.motion_configs[0].name]
        return self.envs[first_motion_idx].GetNumAction()
    
    def GetSimulationHz(self) -> int:
        self._check_initialized()
        first_motion_idx = self._name_to_idx[self.motion_configs[0].name]
        return self.envs[first_motion_idx].GetSimulationHz()
    
    def GetControlHz(self) -> int:
        self._check_initialized()
        first_motion_idx = self._name_to_idx[self.motion_configs[0].name]
        return self.envs[first_motion_idx].GetControlHz()
    
    def GetNumSteps(self) -> int:
        self._check_initialized()
        first_motion_idx = self._name_to_idx[self.motion_configs[0].name]
        return self.envs[first_motion_idx].GetNumSteps()
    
    def UseMuscle(self) -> bool:
        self._check_initialized()
        first_motion_idx = self._name_to_idx[self.motion_configs[0].name]
        return self.envs[first_motion_idx].UseMuscle()
    
    def GetNumMuscles(self) -> int:
        self._check_initialized()
        first_motion_idx = self._name_to_idx[self.motion_configs[0].name]
        return self.envs[first_motion_idx].GetNumMuscles()
    
    def GetNumTotalMuscleRelatedDofs(self) -> int:
        self._check_initialized()
        first_motion_idx = self._name_to_idx[self.motion_configs[0].name]
        return self.envs[first_motion_idx].GetNumTotalMuscleRelatedDofs()
    
    def Resets(self, RSI: bool = True, randomize_motion: bool = True):
        """
        Reset active motion environments.
        
        Args:
            RSI: Random State Initialization
            randomize_motion: Not used in progressive mode
        """
        self._check_initialized()
        
        # Reset each active motion's environment
        for name in self._active_motion_names:
            motion_idx = self._name_to_idx[name]
            self.envs[motion_idx].Resets(RSI)
    
    def GetStates(self) -> np.ndarray:
        """Get states from all active slaves."""
        self._check_initialized()
        
        states_list = []
        for name in self._active_motion_names:
            motion_idx = self._name_to_idx[name]
            slaves_to_use = self._current_slaves_per_motion[name]
            
            # Get all states from this motion's environment
            all_states = self.envs[motion_idx].GetStates()
            # Only use the first 'slaves_to_use' states
            states_list.append(all_states[:slaves_to_use])
        
        return np.vstack(states_list)
    
    def SetActions(self, actions: np.ndarray):
        """Set actions for all active slaves."""
        self._check_initialized()
        
        offset = 0
        for name in self._active_motion_names:
            motion_idx = self._name_to_idx[name]
            slaves_to_use = self._current_slaves_per_motion[name]
            max_slaves = self._max_slaves_per_motion[name]
            
            # Extract actions for this motion
            motion_actions = actions[offset:offset + slaves_to_use]
            
            # Pad with zeros if we're using fewer slaves than the env has
            if slaves_to_use < max_slaves:
                padding = np.zeros((max_slaves - slaves_to_use, actions.shape[1]))
                motion_actions = np.vstack([motion_actions, padding])
            
            self.envs[motion_idx].SetActions(motion_actions)
            offset += slaves_to_use
    
    def StepsAtOnce(self):
        """Step all active motion environments."""
        self._check_initialized()
        for name in self._active_motion_names:
            motion_idx = self._name_to_idx[name]
            self.envs[motion_idx].StepsAtOnce()
    
    def Steps(self, num: int):
        """Step all active motion environments by num steps."""
        self._check_initialized()
        for name in self._active_motion_names:
            motion_idx = self._name_to_idx[name]
            self.envs[motion_idx].Steps(num)
    
    def IsEndOfEpisodes(self) -> np.ndarray:
        """Check episode termination for all active slaves."""
        self._check_initialized()
        
        eoe_list = []
        for name in self._active_motion_names:
            motion_idx = self._name_to_idx[name]
            slaves_to_use = self._current_slaves_per_motion[name]
            
            all_eoe = self.envs[motion_idx].IsEndOfEpisodes()
            eoe_list.append(all_eoe[:slaves_to_use])
        
        return np.concatenate(eoe_list)
    
    def GetRewards(self) -> np.ndarray:
        """Get rewards from all active slaves."""
        self._check_initialized()
        
        rewards_list = []
        for name in self._active_motion_names:
            motion_idx = self._name_to_idx[name]
            slaves_to_use = self._current_slaves_per_motion[name]
            
            all_rewards = self.envs[motion_idx].GetRewards()
            rewards_list.append(all_rewards[:slaves_to_use])
        
        return np.concatenate(rewards_list)
    
    def Reset(self, RSI: bool, slave_id: int):
        """Reset a specific slave (using external slave ID)."""
        self._check_initialized()
        
        motion_name, local_id = self._external_to_internal(slave_id)
        motion_idx = self._name_to_idx[motion_name]
        self.envs[motion_idx].Reset(RSI, local_id)
    
    def IsEndOfEpisode(self, slave_id: int) -> bool:
        """Check if specific slave's episode ended (using external slave ID)."""
        self._check_initialized()
        
        motion_name, local_id = self._external_to_internal(slave_id)
        motion_idx = self._name_to_idx[motion_name]
        return self.envs[motion_idx].IsEndOfEpisode(local_id)
    
    def GetReward(self, slave_id: int) -> float:
        """Get reward for specific slave (using external slave ID)."""
        self._check_initialized()
        
        motion_name, local_id = self._external_to_internal(slave_id)
        motion_idx = self._name_to_idx[motion_name]
        return self.envs[motion_idx].GetReward(local_id)
    
    # =========================================================================
    # Muscle-related methods - FULL RESOURCE USE
    # =========================================================================
    
    def GetMuscleTorques(self) -> np.ndarray:
        self._check_initialized()
        torques_list = []
        for name in self._active_motion_names:
            motion_idx = self._name_to_idx[name]
            slaves_to_use = self._current_slaves_per_motion[name]
            
            all_torques = self.envs[motion_idx].GetMuscleTorques()
            torques_list.append(all_torques[:slaves_to_use])
        return np.vstack(torques_list)
    
    def GetDesiredTorques(self) -> np.ndarray:
        self._check_initialized()
        torques_list = []
        for name in self._active_motion_names:
            motion_idx = self._name_to_idx[name]
            slaves_to_use = self._current_slaves_per_motion[name]
            
            all_torques = self.envs[motion_idx].GetDesiredTorques()
            torques_list.append(all_torques[:slaves_to_use])
        return np.vstack(torques_list)
    
    def SetActivationLevels(self, activations: np.ndarray):
        self._check_initialized()
        offset = 0
        for name in self._active_motion_names:
            motion_idx = self._name_to_idx[name]
            slaves_to_use = self._current_slaves_per_motion[name]
            max_slaves = self._max_slaves_per_motion[name]
            
            # Extract activations for this motion
            motion_activations = activations[offset:offset + slaves_to_use]
            
            # Pad with zeros if we're using fewer slaves than the env has
            if slaves_to_use < max_slaves:
                padding = np.zeros((max_slaves - slaves_to_use, activations.shape[1]))
                motion_activations = np.vstack([motion_activations, padding])
            
            self.envs[motion_idx].SetActivationLevels(motion_activations)
            offset += slaves_to_use
    
    def ComputeMuscleTuples(self):
        self._check_initialized()
        for name in self._active_motion_names:
            motion_idx = self._name_to_idx[name]
            self.envs[motion_idx].ComputeMuscleTuples()
    
    def GetMuscleTuplesJtA(self) -> np.ndarray:
        self._check_initialized()
        tuples_list = []
        for name in self._active_motion_names:
            motion_idx = self._name_to_idx[name]
            data = self.envs[motion_idx].GetMuscleTuplesJtA()
            if data.size > 0:
                tuples_list.append(data)
        return np.vstack(tuples_list) if tuples_list else np.array([])
    
    def GetMuscleTuplesTauDes(self) -> np.ndarray:
        self._check_initialized()
        tuples_list = []
        for name in self._active_motion_names:
            motion_idx = self._name_to_idx[name]
            data = self.envs[motion_idx].GetMuscleTuplesTauDes()
            if data.size > 0:
                tuples_list.append(data)
        return np.vstack(tuples_list) if tuples_list else np.array([])
    
    def GetMuscleTuplesL(self) -> np.ndarray:
        self._check_initialized()
        tuples_list = []
        for name in self._active_motion_names:
            motion_idx = self._name_to_idx[name]
            data = self.envs[motion_idx].GetMuscleTuplesL()
            if data.size > 0:
                tuples_list.append(data)
        return np.vstack(tuples_list) if tuples_list else np.array([])
    
    def GetMuscleTuplesb(self) -> np.ndarray:
        self._check_initialized()
        tuples_list = []
        for name in self._active_motion_names:
            motion_idx = self._name_to_idx[name]
            data = self.envs[motion_idx].GetMuscleTuplesb()
            if data.size > 0:
                tuples_list.append(data)
        return np.vstack(tuples_list) if tuples_list else np.array([])
    
    # =========================================================================
    # Multimodal-specific methods
    # =========================================================================
    
    def get_motion_for_slave(self, slave_id: int) -> str:
        """Get the motion name for a given external slave ID."""
        return self._get_motion_name_for_external_slave(slave_id)
    
    def get_motion_index_for_slave(self, slave_id: int) -> int:
        """Get the motion index for a given external slave ID."""
        return self._get_motion_idx_for_external_slave(slave_id)
    
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
            active = "ACTIVE" if name in self._active_motion_names else "locked"
            slaves = self._current_slaves_per_motion.get(name, 0)
            print(f"  {name} [{active}, {slaves} slaves]: episodes={stats.episodes}, "
                  f"avg_reward={stats.avg_reward:.3f}, "
                  f"avg_steps={stats.avg_steps:.1f}")
    
    def set_motion_weights(self, weights: Dict[str, float]):
        """Set weights for motion sampling (not used in progressive mode)."""
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
    print("")
    print("FULL RESOURCE UTILIZATION in Progressive Mode:")
    print("  - Always uses ALL slaves (total_slave_budget)")
    print("  - Distributes evenly among active motions")
    print("  - Example with budget=36:")
    print("    * 1 motion:  36 slaves for motion 1")
    print("    * 2 motions: 18 slaves each")
    print("    * 3 motions: 12 slaves each")
    print("    * 6 motions: 6 slaves each")
    print("")
    print("Usage:")
    print("  from multimodal_env import MultimodalEnvManager, MotionConfig")
    print("  configs = [MotionConfig('walk', 'data/metadata_walk.txt', True), ...]")
    print("  env = MultimodalEnvManager(configs, num_slaves_per_motion=6,")
    print("                              progressive_order=['walk', 'run', 'jump'])")
    print("  env.initialize()")
    print("  env.set_active_motions(['walk'])  # All 36 slaves on walk")
    print("  # ... train ...")
    print("  env.set_active_motions(['walk', 'run'])  # 18 slaves each")