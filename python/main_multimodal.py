"""
main_multimodal.py - Multimodal Motion Training for MASS with Curriculum Learning

This script extends the original PPO training to support multiple motion clips
with adaptive curriculum learning that adjusts motion sampling weights based
on training performance.

Key features:
1. Uses MultimodalEnvManager instead of single pymss instance
2. Tracks per-motion statistics during training
3. Supports curriculum learning for motion weighting
4. Logs per-motion performance to TensorBoard
5. Progressive curriculum: gradually unlock motions over training

PROGRESSIVE CURRICULUM FIX:
    In progressive mode, only active (unlocked) motions are simulated.
    This saves computation and ensures proper training focus.

Usage:
    # Basic usage with balanced curriculum
    python main_multimodal.py --motion_list data/motion_list.txt --template data/metadata.txt
    
    # Progressive curriculum: train on one motion, then add more
    python main_multimodal.py --motion_list data/motion_list.txt --template data/metadata.txt \\
        --curriculum progressive --epochs_per_motion 1000
    
    # Progressive with custom order
    python main_multimodal.py --motion_list data/motion_list.txt --template data/metadata.txt \\
        --curriculum progressive --epochs_per_motion 500 --progressive_order walk run jump
"""

import argparse
import os
import time
import numpy as np
from datetime import datetime
from typing import Optional, List, Union

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

# Import MASS modules
from PPO import BasePPO
from Model import SimulationNN, MuscleNN
from utils import (
    device, Tensor,
    EpisodeBuffer, EpisodeBufferWithMotion, ReplayBuffer,
    Episode, EpisodeWithMotion, Transition,
    generate_shuffle_indices, format_time
)
from multimodal_env import MultimodalEnvManager, MotionConfig, load_motion_configs_from_list
from curriculum_manager import CurriculumManager

print(f"Using device: {device}")


class MultimodalPPO(BasePPO):
    """
    PPO trainer for multimodal motion imitation with curriculum learning.
    
    Extends BasePPO to support training on multiple motions with per-motion tracking
    and adaptive curriculum learning.
    
    PROGRESSIVE MODE:
        - Only processes active (unlocked) motions
        - Dynamically adjusts num_slaves as motions are unlocked
        - Episode buffers are sized for current active slaves only
    """
    
    def __init__(self, motion_list_path: str, metadata_template_path: str,
                 num_slaves_per_motion: int = 4,
                 curriculum_strategy: str = 'progressive',
                 curriculum_warmup: int = 1000,
                 curriculum_update_freq: int = 20,
                 epochs_per_motion: Union[int, List[int]] = 1000,
                 progressive_order: Optional[List[str]] = None):
        """
        Initialize multimodal PPO trainer with curriculum learning.
        
        Args:
            motion_list_path: Path to motion_list.txt
            metadata_template_path: Path to template metadata file
            num_slaves_per_motion: Number of parallel environments per motion
            curriculum_strategy: Strategy for curriculum learning 
                ('uniform', 'performance', 'progress', 'balanced', 'ucb', 'progressive')
            curriculum_warmup: Number of epochs before starting curriculum
            curriculum_update_freq: How often to update curriculum weights
            epochs_per_motion: (Progressive only) Epochs before unlocking next motion
            progressive_order: (Progressive only) Order to unlock motions
        """
        super().__init__()
        
        np.random.seed(seed=int(time.time()))
        
        # Store curriculum parameters
        self.curriculum_strategy = curriculum_strategy
        self.curriculum_warmup = curriculum_warmup
        self.curriculum_update_freq = curriculum_update_freq
        self.epochs_per_motion = epochs_per_motion if epochs_per_motion is not None else 1000
        self.progressive_order = progressive_order
        
        # Load motion configurations
        print("Loading motion configurations...")
        self.motion_configs = load_motion_configs_from_list(
            motion_list_path,
            metadata_template_path
        )
        
        if not self.motion_configs:
            raise ValueError("No motions loaded! Check motion_list.txt")
        
        print(f"Loaded {len(self.motion_configs)} motions:")
        for config in self.motion_configs:
            print(f"  - {config.name} (cyclic={config.cyclic})")
        
        # Use provided progressive_order or default to motion_names order
        motion_names = [config.name for config in self.motion_configs]
        prog_order = progressive_order if progressive_order else motion_names
        
        # Import pymss module
        import pymss
        
        # Create multimodal environment manager with progressive order
        self.num_slaves_per_motion = num_slaves_per_motion
        self.env = MultimodalEnvManager(
            motion_configs=self.motion_configs,
            num_slaves_per_motion=num_slaves_per_motion,
            pymss_module=pymss,
            progressive_order=prog_order if curriculum_strategy == 'progressive' else None
        )
        self.env.initialize()
        
        # Store total slave budget (same for all phases)
        self.num_motions = self.env.num_motions
        self.total_slave_budget = self.env.total_slave_budget
        
        # =====================================================================
        # CURRICULUM LEARNING INTEGRATION
        # =====================================================================
        self.curriculum = CurriculumManager(
            motion_names=motion_names,
            strategy=curriculum_strategy,
            warmup_epochs=curriculum_warmup,
            update_frequency=curriculum_update_freq,
            min_weight=0.5,
            max_weight=1.0,
            temperature=1.0,
            epochs_per_motion=epochs_per_motion,
            progressive_order=prog_order
        )
        print(f"\nCurriculum Learning:")
        print(f"  Strategy: {curriculum_strategy}")
        print(f"  Warmup epochs: {curriculum_warmup}")
        print(f"  Update frequency: {curriculum_update_freq}")
        if curriculum_strategy == 'progressive':
            print(f"  Epochs per motion: {epochs_per_motion}")
            print(f"  Progressive order: {prog_order}")
        
        # =====================================================================
        # SET INITIAL ACTIVE MOTIONS (FULL RESOURCE UTILIZATION)
        # =====================================================================
        if curriculum_strategy == 'progressive':
            # Start with only first motion active - it gets ALL slaves
            initial_unlocked = self.curriculum.get_unlocked_motions()
            self.env.set_active_motions(initial_unlocked)
            self.num_slaves = self.env.active_slave_count  # Should equal total_slave_budget
            print(f"  Initial active motions: {initial_unlocked}")
            print(f"  Initial active slaves: {self.num_slaves} (= total budget)")
        else:
            # All motions active for non-progressive strategies
            all_motion_names = [c.name for c in self.motion_configs]
            self.env.set_active_motions(all_motion_names)
            self.num_slaves = self.env.active_slave_count
        # =====================================================================
        
        # Set up from environment
        self.use_muscle = self.env.UseMuscle()
        self.num_state = self.env.GetNumState()
        self.num_action = self.env.GetNumAction()
        self.num_muscles = self.env.GetNumMuscles()
        self.num_simulation_Hz = self.env.GetSimulationHz()
        self.num_control_Hz = self.env.GetControlHz()
        self.num_simulation_per_control = self.num_simulation_Hz // self.num_control_Hz
        
        print(f"\nEnvironment info:")
        print(f"  Total slave budget: {self.total_slave_budget}")
        print(f"  Current active slaves: {self.num_slaves}")
        print(f"  Num state: {self.num_state}")
        print(f"  Num action: {self.num_action}")
        print(f"  Use muscle: {self.use_muscle}")
        
        # Initialize buffers
        self.replay_buffer = ReplayBuffer(30000)
        
        # Episode buffers - sized for current active slaves
        # Will be resized when motions are unlocked
        self.episodes = [EpisodeBufferWithMotion() for _ in range(self.num_slaves)]
        
        # Initialize neural networks
        self.model = SimulationNN(self.num_state, self.num_action).to(device)
        self.muscle_model = MuscleNN(
            self.env.GetNumTotalMuscleRelatedDofs(),
            self.num_action,
            self.num_muscles
        ).to(device)
        
        # Initialize optimizers
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.optimizer_muscle = optim.Adam(self.muscle_model.parameters(), lr=self.learning_rate)
        
        # Per-motion tracking (only for active motions)
        self.motion_returns = {config.name: [] for config in self.motion_configs}
        self.motion_episode_counts = {config.name: 0 for config in self.motion_configs}
        
        # Reset active environments
        self.env.Resets(True)
        
        # Initialize TensorBoard with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._init_tensorboard(f'./runs/multimodal_{curriculum_strategy}_{timestamp}')

    def SaveModel(self):
        """Save models with multimodal prefix."""
        nn_dir = './nn'
        os.makedirs(nn_dir, exist_ok=True)
        
        self.model.save(os.path.join(nn_dir, 'multimodal_current.pt'))
        self.muscle_model.save(os.path.join(nn_dir, 'multimodal_current_muscle.pt'))

        if self.max_return_epoch == self.num_evaluation:
            self.model.save(os.path.join(nn_dir, 'multimodal_max.pt'))
            self.muscle_model.save(os.path.join(nn_dir, 'multimodal_max_muscle.pt'))
            
        if self.num_evaluation % 100 == 0:
            epoch_num = self.num_evaluation // 100
            self.model.save(os.path.join(nn_dir, f'multimodal_{epoch_num}.pt'))
            self.muscle_model.save(os.path.join(nn_dir, f'multimodal_{epoch_num}_muscle.pt'))

    def LoadModel(self, path):
        """Load models from checkpoint."""
        nn_dir = './nn'
        self.model.load(os.path.join(nn_dir, f'{path}.pt'))
        self.muscle_model.load(os.path.join(nn_dir, f'{path}_muscle.pt'))

    def ComputeTDandGAE(self):
        """
        Compute TD targets and GAE advantages with per-motion tracking.
        
        Only processes episodes from active motions.
        """
        self.replay_buffer.Clear()
        self.muscle_buffer = {}
        self.sum_return = 0.0
        
        # Reset per-motion tracking for this evaluation (only active motions)
        active_motion_names = self.env.get_active_motion_names()
        motion_returns_this_epoch = {name: [] for name in active_motion_names}
        
        for epi in self.total_episodes:
            data = epi.GetData()
            size = len(data)
            if size == 0:
                continue
                
            # Unpack with motion index
            states, actions, rewards, values, logprobs, motion_indices = zip(*data)

            values = np.concatenate((values, np.zeros(1)), axis=0)
            advantages = np.zeros(size)
            ad_t = 0

            epi_return = 0.0
            for i in reversed(range(size)):
                epi_return += rewards[i]
                delta = rewards[i] + values[i + 1] * self.gamma - values[i]
                ad_t = delta + self.gamma * self.lb * ad_t
                advantages[i] = ad_t

            self.sum_return += epi_return
            TD = values[:size] + advantages
            
            # Track per-motion returns using external slave ID
            external_slave_id = motion_indices[0]
            motion_name = self.env.get_motion_for_slave(external_slave_id)
            
            if motion_name in motion_returns_this_epoch:
                motion_returns_this_epoch[motion_name].append(epi_return)
            self.motion_episode_counts[motion_name] += 1

            for i in range(size):
                self.replay_buffer.Push(states[i], actions[i], logprobs[i], TD[i], advantages[i])

        # Store motion returns for logging
        for name, returns in motion_returns_this_epoch.items():
            if returns:
                self.motion_returns[name] = returns

        self.num_episode = len(self.total_episodes)
        self.num_tuple = len(self.replay_buffer.buffer)
        print(f'SIM : {self.num_tuple}')
        self.num_tuple_so_far += self.num_tuple

        # Compute muscle tuples
        self._compute_muscle_tuples()

    def GenerateTransitions(self):
        """
        Generate transitions with motion index tracking.
        
        Only processes active motion slaves.
        """
        self.total_episodes = []
        states = self.env.GetStates()  # Only returns states for active slaves
        local_step = 0
        counter = 0

        while True:
            counter += 1
            if counter % 10 == 0:
                print(f'SIM : {local_step}', end='\r')

            # Get action from policy
            a_dist, v = self.model(Tensor(states))
            actions = a_dist.sample().cpu().detach().numpy()
            logprobs = a_dist.log_prob(Tensor(actions)).cpu().detach().numpy().reshape(-1)
            values = v.cpu().detach().numpy().reshape(-1)

            # Set actions and step (only for active slaves)
            self.env.SetActions(actions)

            if self.use_muscle:
                mt = Tensor(self.env.GetMuscleTorques())
                for i in range(self.num_simulation_per_control // 2):
                    dt = Tensor(self.env.GetDesiredTorques())
                    activations = self.muscle_model(mt, dt).cpu().detach().numpy()
                    self.env.SetActivationLevels(activations)
                    self.env.Steps(2)
            else:
                self.env.StepsAtOnce()

            # Process each active slave
            for j in range(self.num_slaves):
                nan_occur = False
                terminated_state = True

                if (np.any(np.isnan(states[j])) or np.any(np.isnan(actions[j])) or
                    np.any(np.isnan(values[j])) or np.any(np.isnan(logprobs[j]))):
                    nan_occur = True
                elif self.env.IsEndOfEpisode(j) is False:
                    terminated_state = False
                    reward = self.env.GetReward(j)
                    # External slave ID for tracking which motion this is from
                    external_slave_id = j
                    self.episodes[j].Push(states[j], actions[j], reward, values[j], logprobs[j], external_slave_id)
                    local_step += 1

                if terminated_state or nan_occur:
                    if nan_occur:
                        self.episodes[j].Pop()
                    self.total_episodes.append(self.episodes[j])
                    self.episodes[j] = EpisodeBufferWithMotion()
                    self.env.Reset(True, j)

            if local_step >= self.buffer_size:
                break

            states = self.env.GetStates()

    def _update_curriculum(self):
        """
        Update curriculum manager with current epoch's performance data.
        
        For progressive mode:
        - Updates active motions when new ones are unlocked
        - Slave redistribution is handled automatically by env manager
        - num_slaves stays constant (always total_slave_budget)
        """
        # Update curriculum with this epoch's motion returns
        self.curriculum.update(self.motion_returns, self.epoch)
        
        # Get updated weights from curriculum
        new_weights = self.curriculum.get_weights()
        
        # =====================================================================
        # PROGRESSIVE MODE: Update active motions
        # =====================================================================
        if self.curriculum_strategy == 'progressive':
            unlocked = self.curriculum.get_unlocked_motions()
            
            # Check if new motions were unlocked
            old_active = set(self.env.get_active_motion_names())
            new_active = set(unlocked)
            
            if new_active != old_active:
                print(f"\n*** CURRICULUM UPDATE: {len(old_active)} -> {len(new_active)} motions ***")
                
                # Update active motions (this redistributes slaves automatically)
                self.env.set_active_motions(unlocked)
                
                # num_slaves stays the same (always total_slave_budget)
                # But episode buffers may need adjustment if allocation changed
                new_slave_count = self.env.active_slave_count
                if new_slave_count != len(self.episodes):
                    # Resize episode buffers if needed
                    while len(self.episodes) < new_slave_count:
                        self.episodes.append(EpisodeBufferWithMotion())
                    while len(self.episodes) > new_slave_count:
                        self.episodes.pop()
                    print(f"    Episode buffers resized: {len(self.episodes)}")
        else:
            # Non-progressive: apply weights to environment (for sampling strategies)
            self.env.set_motion_weights(new_weights)
        # =====================================================================
        
        return new_weights

    def Evaluate(self):
        """
        Evaluate and log training progress with per-motion statistics and curriculum.
        
        Only logs stats for active motions in progressive mode.
        """
        self.num_evaluation += 1

        elapsed = time.time() - self.tic
        time_str = format_time(elapsed)

        if self.num_episode == 0:
            self.num_episode = 1
        if self.num_tuple == 0:
            self.num_tuple = 1

        avg_return = self.sum_return / self.num_episode
        if self.max_return < avg_return:
            self.max_return = avg_return
            self.max_return_epoch = self.num_evaluation

        # Get active motions for display
        active_motion_names = self.env.get_active_motion_names()

        # Print overall stats
        print(f'# {self.num_evaluation} === {time_str} ===')
        print(f'||Loss Actor               : {self.loss_actor:.4f}')
        print(f'||Loss Critic              : {self.loss_critic:.4f}')
        print(f'||Loss Muscle              : {self.loss_muscle:.4f}')
        print(f'||Noise                    : {self.model.log_std.exp().mean():.3f}')
        print(f'||Num Transition So far    : {self.num_tuple_so_far}')
        print(f'||Num Transition           : {self.num_tuple}')
        print(f'||Num Episode              : {self.num_episode}')
        print(f'||Avg Return per episode   : {avg_return:.3f}')
        print(f'||Avg Reward per transition: {self.sum_return / self.num_tuple:.3f}')
        print(f'||Avg Step per episode     : {self.num_tuple / self.num_episode:.1f}')
        print(f'||Max Avg Return So far    : {self.max_return:.3f} at #{self.max_return_epoch}')
        
        # Print active slave info for progressive mode
        if self.curriculum_strategy == 'progressive':
            print(f'||Active Slaves            : {self.num_slaves} (budget: {self.total_slave_budget})')
            per_motion = self.total_slave_budget // len(active_motion_names) if active_motion_names else 0
            print(f'||Slaves per motion        : {per_motion}')

        # Print per-motion stats (only active motions)
        print('||--- Per-Motion Statistics (Active Only) ---')
        for name in active_motion_names:
            returns = self.motion_returns.get(name, [])
            if returns:
                avg = np.mean(returns)
                count = self.motion_episode_counts[name]
                print(f'||  {name}: avg_return={avg:.3f}, episodes={count}')

        # =====================================================================
        # CURRICULUM LEARNING UPDATE
        # =====================================================================
        curriculum_weights = self._update_curriculum()
        
        # Print curriculum info
        print('||--- Curriculum Status ---')
        if self.curriculum_strategy == 'progressive':
            unlocked = self.curriculum.get_unlocked_motions()
            locked = [c.name for c in self.motion_configs if c.name not in unlocked]
            print(f'||  Unlocked ({len(unlocked)}): {", ".join(unlocked)}')
            if locked:
                print(f'||  Locked ({len(locked)}): {", ".join(locked)}')
        else:
            for name, weight in curriculum_weights.items():
                print(f'||  {name}: weight={weight:.3f}')
        # =====================================================================

        self.rewards.append(avg_return)

        # Log to TensorBoard
        if self.writer:
            self.writer.add_scalar('Loss/Actor', self.loss_actor, self.num_evaluation)
            self.writer.add_scalar('Loss/Critic', self.loss_critic, self.num_evaluation)
            self.writer.add_scalar('Loss/Muscle', self.loss_muscle, self.num_evaluation)
            self.writer.add_scalar('Train/Noise', self.model.log_std.exp().mean(), self.num_evaluation)
            self.writer.add_scalar('Train/AvgReturn', avg_return, self.num_evaluation)
            self.writer.add_scalar('Train/AvgReward', self.sum_return / self.num_tuple, self.num_evaluation)
            self.writer.add_scalar('Train/AvgStep', self.num_tuple / self.num_episode, self.num_evaluation)
            self.writer.add_scalar('Train/MaxAvgReturn', self.max_return, self.num_evaluation)
            
            # Log active slave count for progressive mode
            if self.curriculum_strategy == 'progressive':
                num_active_motions = len(self.curriculum.get_unlocked_motions())
                slaves_per_motion = self.total_slave_budget // num_active_motions if num_active_motions > 0 else 0
                self.writer.add_scalar('Progressive/NumUnlockedMotions', num_active_motions, self.num_evaluation)
                self.writer.add_scalar('Progressive/SlavesPerMotion', slaves_per_motion, self.num_evaluation)

            # Log per-motion returns (only active motions)
            for name in active_motion_names:
                returns = self.motion_returns.get(name, [])
                if returns:
                    self.writer.add_scalar(f'PerMotion/{name}_AvgReturn', np.mean(returns), self.num_evaluation)
                    self.writer.add_scalar(f'PerMotion/{name}_Episodes', 
                                          self.motion_episode_counts[name], self.num_evaluation)

            # Log curriculum weights (for non-progressive)
            if self.curriculum_strategy != 'progressive':
                for name, weight in curriculum_weights.items():
                    self.writer.add_scalar(f'Curriculum/{name}_Weight', weight, self.num_evaluation)

        self.SaveModel()

        print('=' * 45)
        return np.array(self.rewards)


def main():
    parser = argparse.ArgumentParser(description='Multimodal Motion Training for MASS with Curriculum Learning')
    parser.add_argument('-m', '--model', help='Model checkpoint to resume from')
    parser.add_argument('--motion_list', required=True, help='Path to motion_list.txt')
    parser.add_argument('--template', required=True, help='Path to template metadata file')
    parser.add_argument('--slaves', type=int, default=6, help='Slaves per motion (default: 6)')
    
    # Curriculum learning arguments
    parser.add_argument('--curriculum', type=str, default='progressive',
                        choices=['uniform', 'performance', 'progress', 'balanced', 'ucb', 'progressive'],
                        help='Curriculum strategy (default: progressive)')
    parser.add_argument('--warmup', type=int, default=1000,
                        help='Warmup epochs before curriculum starts (default: 1000)')
    parser.add_argument('--update_freq', type=int, default=20,
                        help='Curriculum weight update frequency (default: 20)')
    
    # Progressive curriculum arguments
    parser.add_argument('--epochs_per_motion', type=int, nargs='+', default=[1000, 4000, 5000],
                        help='(Progressive only) Epochs before unlocking next motion (default: 1000)')
    parser.add_argument('--progressive_order', type=str, nargs='+', default=['walk', 'walk_fullbody', 'balance', 'run', 'dance', 'kick'],
                        help='(Progressive only) Order to unlock motions, e.g., --progressive_order walk run jump')

    args = parser.parse_args()

    # Create multimodal PPO trainer with curriculum
    ppo = MultimodalPPO(
        motion_list_path=args.motion_list,
        metadata_template_path=args.template,
        num_slaves_per_motion=args.slaves,
        curriculum_strategy=args.curriculum,
        curriculum_warmup=args.warmup,
        curriculum_update_freq=args.update_freq,
        epochs_per_motion=args.epochs_per_motion,
        progressive_order=args.progressive_order
    )

    # Create nn directory
    nn_dir = './nn'
    if not os.path.exists(nn_dir):
        os.makedirs(nn_dir)

    # Load or save initial model
    if args.model is not None:
        ppo.LoadModel(args.model)
    else:
        ppo.SaveModel()

    print(f'\nNum states: {ppo.env.GetNumState()}, num actions: {ppo.env.GetNumAction()}')
    print(f'Starting multimodal training with {ppo.num_motions} total motions...')
    print(f'Curriculum strategy: {args.curriculum}')
    if args.curriculum == 'progressive':
        print(f'Progressive mode: FULL RESOURCE UTILIZATION')
        print(f'  Total slave budget: {ppo.total_slave_budget}')
        print(f'  Initial active: {ppo.curriculum.get_unlocked_motions()}')
        print(f'  Initial slaves per motion: {ppo.total_slave_budget}')
    print('')

    # Training loop
    for i in range(ppo.max_iteration - 5):
        ppo.Train()
        rewards = ppo.Evaluate()


if __name__ == "__main__":
    main()