#!/usr/bin/env python3
"""
Test MultimodalEnvManager with a mock pymss module.
This validates the interface before integrating with real training.
"""
import os 
print(os.getcwd())

import numpy as np
from multimodal_env import MultimodalEnvManager, MotionConfig


class MockPymssEnv:
    """Mock pymss environment for testing"""
    
    def __init__(self, metadata_path: str, num_slaves: int):
        self.metadata_path = metadata_path
        self.num_slaves = num_slaves
        self.num_state = 100
        self.num_action = 30
        self.num_muscles = 326
        self._step_count = np.zeros(num_slaves)
        print(f"    MockPymss created: {metadata_path}, {num_slaves} slaves")
        
    def GetNumState(self): return self.num_state
    def GetNumAction(self): return self.num_action
    def GetSimulationHz(self): return 900
    def GetControlHz(self): return 30
    def GetNumSteps(self): return 30
    def UseMuscle(self): return True
    def GetNumMuscles(self): return self.num_muscles
    def GetNumTotalMuscleRelatedDofs(self): return 500
    
    def Resets(self, RSI):
        self._step_count = np.zeros(self.num_slaves)
        
    def Reset(self, RSI, slave_id):
        self._step_count[slave_id] = 0
        
    def GetStates(self):
        return np.random.randn(self.num_slaves, self.num_state).astype(np.float32)
    
    def SetActions(self, actions):
        assert actions.shape == (self.num_slaves, self.num_action)
        
    def StepsAtOnce(self):
        self._step_count += self.GetNumSteps()
        
    def Steps(self, num):
        self._step_count += num
        
    def IsEndOfEpisodes(self):
        # Random termination after ~300 steps on average
        return (self._step_count > 300) | (np.random.rand(self.num_slaves) < 0.01)
    
    def IsEndOfEpisode(self, slave_id):
        return self._step_count[slave_id] > 300 or np.random.rand() < 0.01
    
    def GetRewards(self):
        return np.random.uniform(0.5, 1.0, self.num_slaves).astype(np.float32)
    
    def GetReward(self, slave_id):
        return np.random.uniform(0.5, 1.0)
    
    def GetMuscleTorques(self):
        return np.random.randn(self.num_slaves, self.GetNumTotalMuscleRelatedDofs()).astype(np.float32)
    
    def GetDesiredTorques(self):
        return np.random.randn(self.num_slaves, self.num_action).astype(np.float32)
    
    def SetActivationLevels(self, activations):
        assert activations.shape == (self.num_slaves, self.num_muscles)
        
    def ComputeMuscleTuples(self):
        pass
    
    def GetMuscleTuplesJtA(self):
        return np.random.randn(10, self.GetNumTotalMuscleRelatedDofs()).astype(np.float32)
    
    def GetMuscleTuplesTauDes(self):
        return np.random.randn(10, self.num_action).astype(np.float32)
    
    def GetMuscleTuplesL(self):
        return np.random.randn(10, self.num_action * self.num_muscles).astype(np.float32)
    
    def GetMuscleTuplesb(self):
        return np.random.randn(10, self.num_action).astype(np.float32)


class MockPymss:
    """Mock pymss module"""
    @staticmethod
    def pymss(metadata_path: str, num_slaves: int):
        return MockPymssEnv(metadata_path, num_slaves)


def test_multimodal_env_manager():
    """Test the MultimodalEnvManager interface"""
    print("=" * 60)
    print("TESTING MultimodalEnvManager")
    print("=" * 60)
    
    # Create motion configs with absolute paths
    import os
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    configs = [
        MotionConfig(name="walk", metadata_path=os.path.join(base_dir, "data/metadata_walk.txt"), cyclic=True),
        MotionConfig(name="run", metadata_path=os.path.join(base_dir, "data/metadata_run.txt"), cyclic=True),
    ]
    
    num_slaves_per_motion = 4
    
    # Create manager with mock pymss
    print("\n--- Creating MultimodalEnvManager ---")
    env = MultimodalEnvManager(
        motion_configs=configs,
        num_slaves_per_motion=num_slaves_per_motion,
        pymss_module=MockPymss
    )
    env.initialize()
    
    # Verify properties
    print("\n--- Verifying properties ---")
    print(f"Total slaves: {env.total_slaves}")
    print(f"Num motions: {env.num_motions}")
    print(f"Num state: {env.GetNumState()}")
    print(f"Num action: {env.GetNumAction()}")
    print(f"Use muscle: {env.UseMuscle()}")
    
    assert env.total_slaves == len(configs) * num_slaves_per_motion
    assert env.num_motions == len(configs)
    
    # Test reset
    print("\n--- Testing Resets ---")
    env.Resets(RSI=True)
    print("  Resets completed")
    
    # Test get states
    print("\n--- Testing GetStates ---")
    states = env.GetStates()
    print(f"  States shape: {states.shape}")
    assert states.shape == (env.total_slaves, env.GetNumState())
    
    # Test set actions
    print("\n--- Testing SetActions ---")
    actions = np.random.randn(env.total_slaves, env.GetNumAction()).astype(np.float32)
    env.SetActions(actions)
    print(f"  Actions shape: {actions.shape}")
    
    # Test step
    print("\n--- Testing StepsAtOnce ---")
    env.StepsAtOnce()
    print("  Steps completed")
    
    # Test rewards
    print("\n--- Testing GetRewards ---")
    rewards = env.GetRewards()
    print(f"  Rewards shape: {rewards.shape}")
    assert rewards.shape == (env.total_slaves,)
    
    # Test end of episodes
    print("\n--- Testing IsEndOfEpisodes ---")
    eoe = env.IsEndOfEpisodes()
    print(f"  EOE shape: {eoe.shape}")
    assert eoe.shape == (env.total_slaves,)
    
    # Test muscle methods
    print("\n--- Testing Muscle Methods ---")
    mt = env.GetMuscleTorques()
    print(f"  MuscleTorques shape: {mt.shape}")
    
    dt = env.GetDesiredTorques()
    print(f"  DesiredTorques shape: {dt.shape}")
    
    # Test per-slave access
    print("\n--- Testing Per-Slave Access ---")
    for slave_id in [0, 3, 5]:
        motion = env.get_motion_for_slave(slave_id)
        reward = env.GetReward(slave_id)
        print(f"  Slave {slave_id}: motion='{motion}', reward={reward:.3f}")
    
    # Simulate training loop
    print("\n--- Simulating Training Episodes ---")
    num_episodes = 20
    episode_rewards = {name: [] for name in [c.name for c in configs]}
    
    for ep in range(num_episodes):
        env.Resets(RSI=True)
        
        total_rewards = np.zeros(env.total_slaves)
        steps = 0
        
        while steps < 10:  # Limit steps for testing
            states = env.GetStates()
            actions = np.random.randn(env.total_slaves, env.GetNumAction()).astype(np.float32)
            env.SetActions(actions)
            env.StepsAtOnce()
            
            rewards = env.GetRewards()
            total_rewards += rewards
            steps += 1
        
        # Record per-motion rewards
        for slave_id in range(env.total_slaves):
            motion = env.get_motion_for_slave(slave_id)
            episode_rewards[motion].append(total_rewards[slave_id])
            env.update_stats(slave_id, total_rewards[slave_id], steps)
    
    # Print statistics
    print("\n--- Training Statistics ---")
    env.print_stats()
    
    print("\nPer-motion average rewards:")
    for motion, rewards in episode_rewards.items():
        print(f"  {motion}: {np.mean(rewards):.3f} ± {np.std(rewards):.3f}")
    
    print("\n" + "=" * 60)
    print("ALL TESTS PASSED!")
    print("=" * 60)


if __name__ == "__main__":
    test_multimodal_env_manager()