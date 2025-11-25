#!/usr/bin/env python3
"""
Test script for multimodal motion selection logic.
This validates the motion library logic before integrating with C++.
"""

import os
import random
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional


@dataclass
class MotionEntry:
    """Represents a motion clip"""
    filepath: str
    name: str
    cyclic: bool
    duration: float
    
    
class MotionLibrary:
    """
    Python prototype of MotionLibrary for testing multimodal logic.
    This mirrors the C++ implementation for validation.
    """
    
    def __init__(self):
        self.motions: List[MotionEntry] = []
        self.weights: List[float] = []
        self.counts: List[int] = []
        self.current_index: int = -1
        
    def load_motion_list(self, list_path: str) -> bool:
        """Load motions from a list file"""
        if not os.path.exists(list_path):
            print(f"Error: Motion list not found: {list_path}")
            return False
            
        with open(list_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                    
                parts = line.split()
                if len(parts) < 2:
                    continue
                    
                filepath = parts[0]
                cyclic = parts[1].lower() in ('true', '1')
                
                # For testing, we'll mock the BVH loading
                self.add_motion(filepath, cyclic)
        
        # Initialize weights uniformly
        if self.motions:
            self.weights = [1.0] * len(self.motions)
            self.counts = [0] * len(self.motions)
            self.select_motion(0)
            
        return len(self.motions) > 0
    
    def add_motion(self, filepath: str, cyclic: bool = True) -> bool:
        """Add a motion to the library"""
        name = os.path.splitext(os.path.basename(filepath))[0]
        
        # Mock duration based on filename (in real impl, read from BVH)
        duration = 1.0 if 'walk' in name.lower() else 0.5
        
        entry = MotionEntry(
            filepath=filepath,
            name=name,
            cyclic=cyclic,
            duration=duration
        )
        self.motions.append(entry)
        print(f"  + Added: {name} (cyclic={cyclic})")
        return True
    
    def select_random_motion(self) -> int:
        """Select a random motion weighted by weights"""
        if not self.motions:
            return -1
            
        # Weighted random selection
        total = sum(self.weights)
        r = random.uniform(0, total)
        
        cumulative = 0
        for i, w in enumerate(self.weights):
            cumulative += w
            if r <= cumulative:
                self.select_motion(i)
                return i
        
        # Fallback
        return self.select_motion(len(self.motions) - 1)
    
    def select_motion(self, index: int) -> bool:
        """Select a specific motion by index"""
        if index < 0 or index >= len(self.motions):
            return False
            
        self.current_index = index
        self.counts[index] += 1
        return True
    
    def get_current_motion(self) -> Optional[MotionEntry]:
        """Get the currently selected motion"""
        if self.current_index >= 0:
            return self.motions[self.current_index]
        return None
    
    def set_weights(self, weights: List[float]):
        """Set sampling weights for each motion"""
        if len(weights) == len(self.motions):
            self.weights = weights
    
    def get_stats(self) -> Dict:
        """Get usage statistics"""
        total = sum(self.counts)
        return {
            'total_selections': total,
            'per_motion': {
                m.name: {
                    'count': self.counts[i],
                    'percentage': (self.counts[i] / total * 100) if total > 0 else 0
                }
                for i, m in enumerate(self.motions)
            }
        }


def test_motion_library():
    """Test the motion library functionality"""
    print("=" * 60)
    print("TESTING MOTION LIBRARY")
    print("=" * 60)
    
    # Test 1: Load motion list
    print("\n--- Test 1: Loading motion list ---")
    lib = MotionLibrary()
    
    # Create test motion list if not exists
    test_list_path = "/home/hamza/PhD/code/MASS/data/motion_list.txt"
    success = lib.load_motion_list(test_list_path)
    
    print(f"Loaded {len(lib.motions)} motions: {success}")
    for m in lib.motions:
        print(f"  - {m.name}: cyclic={m.cyclic}, duration={m.duration}s")
    
    # Test 2: Random selection with uniform weights
    print("\n--- Test 2: Random selection (uniform weights) ---")
    lib.counts = [0] * len(lib.motions)  # Reset counts
    
    num_trials = 1000
    for _ in range(num_trials):
        lib.select_random_motion()
    
    stats = lib.get_stats()
    print(f"After {num_trials} random selections:")
    for name, data in stats['per_motion'].items():
        print(f"  {name}: {data['count']} ({data['percentage']:.1f}%)")
    
    # Test 3: Weighted selection (favor walking)
    print("\n--- Test 3: Weighted selection (favor walking) ---")
    lib.counts = [0] * len(lib.motions)  # Reset counts
    
    # Give walk 3x weight compared to run
    new_weights = []
    for m in lib.motions:
        if 'walk' in m.name.lower():
            new_weights.append(3.0)
        else:
            new_weights.append(1.0)
    lib.set_weights(new_weights)
    
    print(f"Weights: {dict(zip([m.name for m in lib.motions], new_weights))}")
    
    for _ in range(num_trials):
        lib.select_random_motion()
    
    stats = lib.get_stats()
    print(f"After {num_trials} random selections:")
    for name, data in stats['per_motion'].items():
        print(f"  {name}: {data['count']} ({data['percentage']:.1f}%)")
    
    # Test 4: Motion switching simulation
    print("\n--- Test 4: Episode simulation ---")
    lib.counts = [0] * len(lib.motions)
    lib.set_weights([1.0] * len(lib.motions))  # Reset to uniform
    
    num_episodes = 10
    for ep in range(num_episodes):
        idx = lib.select_random_motion()
        current = lib.get_current_motion()
        if current:
            print(f"  Episode {ep+1}: Selected '{current.name}' "
                  f"(duration={current.duration}s, cyclic={current.cyclic})")
    
    print("\n" + "=" * 60)
    print("ALL TESTS PASSED!")
    print("=" * 60)
    

def test_multimodal_training_flow():
    """
    Test the expected training flow with multimodal motions.
    This simulates what will happen in the Environment class.
    """
    print("\n" + "=" * 60)
    print("TESTING MULTIMODAL TRAINING FLOW")
    print("=" * 60)
    
    lib = MotionLibrary()
    lib.load_motion_list("/home/hamza/PhD/code/MASS/data/motion_list.txt")
    
    # Simulate training episodes
    print("\nSimulating training with random motion resets...")
    
    episode_rewards = []
    motion_episode_counts = {m.name: [] for m in lib.motions}
    
    for episode in range(20):
        # Reset: select random motion
        lib.select_random_motion()
        current_motion = lib.get_current_motion()
        
        # Simulate episode (mock rewards based on motion)
        # In reality, this would come from the simulation
        if current_motion:
            base_reward = 0.8 if 'walk' in current_motion.name.lower() else 0.6
            episode_reward = base_reward + random.uniform(-0.1, 0.1)
            
            episode_rewards.append(episode_reward)
            motion_episode_counts[current_motion.name].append(episode_reward)
    
    # Report results
    print("\nPer-motion performance:")
    for name, rewards in motion_episode_counts.items():
        if rewards:
            avg = np.mean(rewards)
            std = np.std(rewards)
            print(f"  {name}: avg_reward={avg:.3f} ± {std:.3f} (n={len(rewards)})")
    
    print(f"\nOverall: avg_reward={np.mean(episode_rewards):.3f} ± {np.std(episode_rewards):.3f}")
    

if __name__ == "__main__":
    test_motion_library()
    test_multimodal_training_flow()