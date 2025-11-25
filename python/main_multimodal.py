"""
main_multimodal.py - Multimodal Motion Training for MASS (IMPROVED)

This script extends the original PPO training to support multiple motion clips
with ADAPTIVE CURRICULUM LEARNING.

Key improvements over original:
1. Integrated CurriculumManager for adaptive motion weighting
2. Better per-motion statistics tracking
3. Configurable curriculum strategies
4. Improved logging and monitoring

Usage:
    # Basic multimodal training with curriculum
    python main_multimodal.py --motion_list data/motion_list.txt --template data/metadata.txt
    
    # With specific curriculum strategy
    python main_multimodal.py --motion_list data/motion_list.txt --template data/metadata.txt --curriculum balanced
    
    # Resume from checkpoint
    python main_multimodal.py --motion_list data/motion_list.txt --template data/metadata.txt -m checkpoint
"""

import math
import random
import time
import os
import sys
import argparse
from datetime import datetime

import collections
from collections import namedtuple
from collections import deque
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

torch.backends.cudnn.benchmark = True  # Auto-tune cuDNN kernels

import numpy as np

# Import MASS modules
from Model import SimulationNN, MuscleNN, Tensor
from multimodal_env import MultimodalEnvManager, MotionConfig, load_motion_configs_from_list
from curriculum_manager import CurriculumManager

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


Episode = namedtuple('Episode', ('s', 'a', 'r', 'value', 'logprob', 'motion_idx'))


class EpisodeBuffer:
    def __init__(self):
        self.data = []

    def Push(self, *args):
        self.data.append(Episode(*args))
        
    def Pop(self):
        if self.data:
            self.data.pop()
        
    def GetData(self):
        return self.data
    
    def Clear(self):
        self.data = []


Transition = namedtuple('Transition', ('s', 'a', 'logprob', 'TD', 'GAE'))


class ReplayBuffer:
    def __init__(self, buff_size=10000):
        super(ReplayBuffer, self).__init__()
        self.buffer = deque(maxlen=buff_size)

    def Push(self, *args):
        self.buffer.append(Transition(*args))

    def Clear(self):
        self.buffer.clear()


# Dataset classes for DataLoader (for faster training)
class TransitionDataset(Dataset):
    def __init__(self, transitions):
        self.transitions = transitions
        
    def __len__(self):
        return len(self.transitions)
    
    def __getitem__(self, idx):
        transition = self.transitions[idx]
        return {
            's': transition.s.astype(np.float32),
            'a': transition.a.astype(np.float32),
            'logprob': np.float32(transition.logprob),
            'TD': np.float32(transition.TD),
            'GAE': np.float32(transition.GAE)
        }


class MuscleDataset(Dataset):
    def __init__(self, JtA, tau_des, L, b, num_action, num_muscles):
        self.JtA = JtA
        self.tau_des = tau_des
        self.L = L
        self.b = b
        self.num_action = num_action
        self.num_muscles = num_muscles
        
    def __len__(self):
        return len(self.JtA)
    
    def __getitem__(self, idx):
        return {
            'JtA': self.JtA[idx].astype(np.float32),
            'tau_des': self.tau_des[idx].astype(np.float32),
            'L': self.L[idx].astype(np.float32).reshape(self.num_action, self.num_muscles),
            'b': self.b[idx].astype(np.float32)
        }


class MultimodalPPO:
    """
    PPO trainer for multimodal motion imitation with curriculum learning.
    """
    
    def __init__(self, 
                 motion_list_path: str, 
                 metadata_template_path: str, 
                 num_slaves_per_motion: int = 4,
                 curriculum_strategy: str = 'balanced',
                 curriculum_warmup: int = 20,
                 curriculum_update_freq: int = 5):
        """
        Initialize multimodal PPO trainer.
        
        Args:
            motion_list_path: Path to motion_list.txt
            metadata_template_path: Path to template metadata file
            num_slaves_per_motion: Number of parallel environments per motion
            curriculum_strategy: Strategy for curriculum learning 
                                 ('uniform', 'performance', 'progress', 'balanced', 'ucb')
            curriculum_warmup: Number of epochs before starting curriculum
            curriculum_update_freq: How often to update curriculum weights
        """
        np.random.seed(seed=int(time.time()))
        
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
        
        # Import pymss module
        import pymss
        
        # Create multimodal environment manager
        self.num_slaves_per_motion = num_slaves_per_motion
        self.env = MultimodalEnvManager(
            motion_configs=self.motion_configs,
            num_slaves_per_motion=num_slaves_per_motion,
            pymss_module=pymss
        )
        self.env.initialize()
        
        self.use_muscle = self.env.UseMuscle()
        self.num_state = self.env.GetNumState()
        self.num_action = self.env.GetNumAction()
        self.num_muscles = self.env.GetNumMuscles()
        self.num_slaves = self.env.total_slaves
        self.num_motions = self.env.num_motions
        
        print(f"\nEnvironment info:")
        print(f"  Total slaves: {self.num_slaves}")
        print(f"  Num motions: {self.num_motions}")
        print(f"  Num state: {self.num_state}")
        print(f"  Num action: {self.num_action}")
        print(f"  Use muscle: {self.use_muscle}")

        # =====================================================================
        # CURRICULUM LEARNING SETUP
        # =====================================================================
        self.curriculum_strategy = curriculum_strategy
        motion_names = [c.name for c in self.motion_configs]
        
        self.curriculum = CurriculumManager(
            motion_names=motion_names,
            strategy=curriculum_strategy,
            min_weight=0.1,
            max_weight=3.0,
            temperature=1.0,
            update_frequency=curriculum_update_freq,
            warmup_epochs=curriculum_warmup
        )
        
        print(f"\nCurriculum Learning:")
        print(f"  Strategy: {curriculum_strategy}")
        print(f"  Warmup epochs: {curriculum_warmup}")
        print(f"  Update frequency: {curriculum_update_freq}")

        # Training hyperparameters
        self.num_epochs = 10
        self.num_epochs_muscle = 3
        self.num_evaluation = 0
        self.num_tuple_so_far = 0
        self.num_episode = 0
        self.num_tuple = 0
        self.num_simulation_Hz = self.env.GetSimulationHz()
        self.num_control_Hz = self.env.GetControlHz()
        self.num_simulation_per_control = self.num_simulation_Hz // self.num_control_Hz

        self.gamma = 0.99
        self.lb = 0.99

        self.buffer_size = 2048
        self.batch_size = 128
        self.muscle_batch_size = 128
        self.replay_buffer = ReplayBuffer(30000)
        self.muscle_buffer = {}

        # Neural networks
        self.model = SimulationNN(self.num_state, self.num_action).to(device)
        self.muscle_model = MuscleNN(
            self.env.GetNumTotalMuscleRelatedDofs(),
            self.num_action,
            self.num_muscles
        ).to(device)

        # Optimizer settings
        self.default_learning_rate = 1E-4
        self.default_clip_ratio = 0.2
        self.learning_rate = self.default_learning_rate
        self.clip_ratio = self.default_clip_ratio
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.optimizer_muscle = optim.Adam(self.muscle_model.parameters(), lr=self.learning_rate)
        self.max_iteration = 50000

        self.w_entropy = -0.001

        # Tracking
        self.loss_actor = 0.0
        self.loss_critic = 0.0
        self.loss_muscle = 0.0
        self.rewards = []
        self.sum_return = 0.0
        self.max_return = -1.0
        self.max_return_epoch = 1
        self.tic = time.time()
        
        # Per-motion tracking (for this batch)
        self.motion_returns_this_batch: Dict[str, List[float]] = {}

        # Episode buffers
        self.episodes = [EpisodeBuffer() for _ in range(self.num_slaves)]
        self.env.Resets(True)

        # TensorBoard logging
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.writer = SummaryWriter(f'runs/multimodal_{curriculum_strategy}_{timestamp}')

    def SaveModel(self):
        """Save current and best models"""
        os.makedirs('nn', exist_ok=True)
        
        self.model.save('nn/multimodal_current.pt')
        self.muscle_model.save('nn/multimodal_current_muscle.pt')

        if self.max_return_epoch == self.num_evaluation:
            self.model.save('nn/multimodal_max.pt')
            self.muscle_model.save('nn/multimodal_max_muscle.pt')
            
        if self.num_evaluation % 100 == 0:
            self.model.save(f'nn/multimodal_{self.num_evaluation//100}.pt')
            self.muscle_model.save(f'nn/multimodal_{self.num_evaluation//100}_muscle.pt')

    def LoadModel(self, path):
        """Load model from checkpoint"""
        self.model.load('nn/' + path + '.pt')
        self.muscle_model.load('nn/' + path + '_muscle.pt')

    def ComputeTDandGAE(self):
        """Compute TD targets and GAE advantages"""
        self.replay_buffer.Clear()
        self.muscle_buffer = {}
        self.sum_return = 0.0
        
        # Reset per-motion tracking for this batch
        self.motion_returns_this_batch = {c.name: [] for c in self.motion_configs}
        
        for epi in self.total_episodes:
            data = epi.GetData()
            size = len(data)
            if size == 0:
                continue
            states, actions, rewards, values, logprobs, motion_indices = zip(*data)

            values = np.concatenate((values, np.zeros(1)), axis=0)
            advantages = np.zeros(size)
            ad_t = 0

            epi_return = 0.0
            for i in reversed(range(len(data))):
                epi_return += rewards[i]
                delta = rewards[i] + values[i + 1] * self.gamma - values[i]
                ad_t = delta + self.gamma * self.lb * ad_t
                advantages[i] = ad_t
                
            self.sum_return += epi_return
            TD = values[:size] + advantages
            
            # Track per-motion returns
            # motion_indices[0] is the slave_id, use it to get motion name
            slave_id = motion_indices[0]
            motion_name = self.env.get_motion_for_slave(slave_id)
            self.motion_returns_this_batch[motion_name].append(epi_return)

            for i in range(size):
                self.replay_buffer.Push(states[i], actions[i], logprobs[i], TD[i], advantages[i])
                
        self.num_episode = len(self.total_episodes)
        self.num_tuple = len(self.replay_buffer.buffer)
        print(f'SIM : {self.num_tuple}')
        self.num_tuple_so_far += self.num_tuple

        # Compute muscle tuples
        self.env.ComputeMuscleTuples()
        self.muscle_buffer['JtA'] = self.env.GetMuscleTuplesJtA()
        self.muscle_buffer['TauDes'] = self.env.GetMuscleTuplesTauDes()
        self.muscle_buffer['L'] = self.env.GetMuscleTuplesL()
        self.muscle_buffer['b'] = self.env.GetMuscleTuplesb()
        
        if self.muscle_buffer['JtA'].size > 0:
            print(f"Muscle tuples shape: {self.muscle_buffer['JtA'].shape}")
        
        # =====================================================================
        # UPDATE CURRICULUM
        # =====================================================================
        self.curriculum.update(self.motion_returns_this_batch)
        
        # Apply new weights to environment
        new_weights = self.curriculum.get_weights()
        self.env.set_motion_weights(new_weights)

    def GenerateTransitions(self):
        """Generate transitions by running the policy"""
        self.total_episodes = []
        states = self.env.GetStates()
        local_step = 0
        counter = 0
        
        while True:
            counter += 1
            if counter % 10 == 0:
                print(f'SIM : {local_step}', end='\r')
                
            a_dist, v = self.model(Tensor(states))
            actions = a_dist.sample().cpu().detach().numpy()
            logprobs = a_dist.log_prob(Tensor(actions)).cpu().detach().numpy().reshape(-1)
            values = v.cpu().detach().numpy().reshape(-1)
            
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

            for j in range(self.num_slaves):
                nan_occur = False
                terminated_state = True

                if (np.any(np.isnan(states[j])) or np.any(np.isnan(actions[j])) or 
                    np.any(np.isnan(values[j])) or np.any(np.isnan(logprobs[j]))):
                    nan_occur = True
                elif self.env.IsEndOfEpisode(j) is False:
                    terminated_state = False
                    rewards_j = self.env.GetReward(j)
                    # Store slave_id as motion_idx for later identification
                    self.episodes[j].Push(states[j], actions[j], rewards_j, values[j], logprobs[j], j)
                    local_step += 1

                if terminated_state or nan_occur:
                    if nan_occur and self.episodes[j].data:
                        self.episodes[j].Pop()
                    
                    if self.episodes[j].data:  # Only add non-empty episodes
                        self.total_episodes.append(self.episodes[j])
                    
                    self.episodes[j] = EpisodeBuffer()
                    self.env.Reset(True, j)

            if local_step >= self.buffer_size:
                break

            states = self.env.GetStates()

    def OptimizeSimulationNN(self):
        """Optimize the policy and value networks using DataLoader"""
        all_transitions = list(self.replay_buffer.buffer)
        dataset = TransitionDataset(all_transitions)
        
        for j in range(self.num_epochs):
            dataloader = DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=0,
                pin_memory=True,
                drop_last=False
            )
            
            for batch in dataloader:
                stack_s = batch['s']
                stack_a = batch['a']
                stack_lp = batch['logprob']
                stack_td = batch['TD']
                stack_gae = batch['GAE']

                a_dist, v = self.model(Tensor(stack_s))
                
                # Critic loss
                loss_critic = ((v - Tensor(stack_td)).pow(2)).mean()

                # Actor loss
                ratio = torch.exp(a_dist.log_prob(Tensor(stack_a)) - Tensor(stack_lp))
                stack_gae_np = stack_gae.numpy()
                stack_gae_np = (stack_gae_np - stack_gae_np.mean()) / (stack_gae_np.std() + 1E-5)
                stack_gae_t = Tensor(stack_gae_np)
                surrogate1 = ratio * stack_gae_t
                surrogate2 = torch.clamp(ratio, min=1.0 - self.clip_ratio, max=1.0 + self.clip_ratio) * stack_gae_t
                loss_actor = -torch.min(surrogate1, surrogate2).mean()
                
                # Entropy loss
                loss_entropy = -self.w_entropy * a_dist.entropy().mean()

                self.loss_actor = loss_actor.cpu().detach().numpy().tolist()
                self.loss_critic = loss_critic.cpu().detach().numpy().tolist()

                loss = loss_actor + loss_entropy + loss_critic

                self.optimizer.zero_grad()
                loss.backward()
                for param in self.model.parameters():
                    if param.grad is not None:
                        param.grad.data.clamp_(-0.5, 0.5)
                self.optimizer.step()
                
            print(f'Optimizing sim nn : {j + 1}/{self.num_epochs}', end='\r')
        print('')

    def OptimizeMuscleNN(self):
        """Optimize the muscle network using DataLoader"""
        if self.muscle_buffer['JtA'].size == 0:
            return
        
        dataset = MuscleDataset(
            self.muscle_buffer['JtA'],
            self.muscle_buffer['TauDes'],
            self.muscle_buffer['L'],
            self.muscle_buffer['b'],
            self.num_action,
            self.num_muscles
        )
            
        for j in range(self.num_epochs_muscle):
            dataloader = DataLoader(
                dataset,
                batch_size=self.muscle_batch_size,
                shuffle=True,
                num_workers=0,
                pin_memory=True,
                drop_last=False
            )
            
            for batch in dataloader:
                stack_JtA = Tensor(batch['JtA'])
                stack_tau_des = Tensor(batch['tau_des'])
                stack_L = Tensor(batch['L'])
                stack_b = Tensor(batch['b'])

                activation = self.muscle_model(stack_JtA, stack_tau_des)
                tau = torch.einsum('ijk,ik->ij', (stack_L, activation)) + stack_b

                loss_reg = activation.pow(2).mean()
                loss_target = (((tau - stack_tau_des) / 100.0).pow(2)).mean()

                loss = 0.01 * loss_reg + loss_target

                self.optimizer_muscle.zero_grad()
                loss.backward()
                for param in self.muscle_model.parameters():
                    if param.grad is not None:
                        param.grad.data.clamp_(-0.5, 0.5)
                self.optimizer_muscle.step()

            print(f'Optimizing muscle nn : {j + 1}/{self.num_epochs_muscle}', end='\r')
        self.loss_muscle = loss.cpu().detach().numpy().tolist()
        print('')

    def OptimizeModel(self):
        """Run one optimization iteration"""
        self.ComputeTDandGAE()
        self.OptimizeSimulationNN()
        if self.use_muscle:
            self.OptimizeMuscleNN()

    def Train(self):
        """Run one training iteration"""
        self.GenerateTransitions()
        self.OptimizeModel()

    def Evaluate(self):
        """Evaluate and log training progress"""
        self.num_evaluation += 1
        
        h = int((time.time() - self.tic) // 3600.0)
        m = int((time.time() - self.tic) // 60.0)
        s = int((time.time() - self.tic))
        m = m - h * 60
        s = s - h * 3600 - m * 60
        
        if self.num_episode == 0:
            self.num_episode = 1
        if self.num_tuple == 0:
            self.num_tuple = 1
            
        avg_return = self.sum_return / self.num_episode
        if self.max_return < avg_return:
            self.max_return = avg_return
            self.max_return_epoch = self.num_evaluation

        # Print overall stats
        print(f'# {self.num_evaluation} === {h}h:{m}m:{s}s ===')
        print(f'||Curriculum Strategy       : {self.curriculum_strategy}')
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
        
        # Print per-motion stats with curriculum weights
        print('||--- Per-Motion Statistics (with curriculum weights) ---')
        curriculum_stats = self.curriculum.get_stats()
        for name, stats in curriculum_stats.items():
            weight = stats['current_weight']
            avg_ret = stats['avg_return']
            total_eps = stats['total_episodes']
            print(f'||  {name}: avg_ret={avg_ret:.3f}, total_eps={total_eps}, weight={weight:.3f}')

        self.rewards.append(avg_return)

        # Log to TensorBoard
        self.writer.add_scalar('Loss/Actor', self.loss_actor, self.num_evaluation)
        self.writer.add_scalar('Loss/Critic', self.loss_critic, self.num_evaluation)
        self.writer.add_scalar('Loss/Muscle', self.loss_muscle, self.num_evaluation)
        self.writer.add_scalar('Train/Noise', self.model.log_std.exp().mean(), self.num_evaluation)
        self.writer.add_scalar('Train/AvgReturn', avg_return, self.num_evaluation)
        self.writer.add_scalar('Train/AvgReward', self.sum_return / self.num_tuple, self.num_evaluation)
        self.writer.add_scalar('Train/AvgStep', self.num_tuple / self.num_episode, self.num_evaluation)
        self.writer.add_scalar('Train/MaxAvgReturn', self.max_return, self.num_evaluation)
        
        # Log per-motion returns and weights
        for name, stats in curriculum_stats.items():
            self.writer.add_scalar(f'PerMotion/{name}_AvgReturn', stats['avg_return'], self.num_evaluation)
            self.writer.add_scalar(f'PerMotion/{name}_Episodes', stats['total_episodes'], self.num_evaluation)
            self.writer.add_scalar(f'Curriculum/{name}_Weight', stats['current_weight'], self.num_evaluation)

        self.SaveModel()

        print('=' * 60)
        return np.array(self.rewards)


def main():
    parser = argparse.ArgumentParser(description='Multimodal Motion Training for MASS with Curriculum Learning')
    parser.add_argument('-m', '--model', help='Model checkpoint to resume from')
    parser.add_argument('--motion_list', required=True, help='Path to motion_list.txt')
    parser.add_argument('--template', required=True, help='Path to template metadata file')
    parser.add_argument('--slaves', type=int, default=4, help='Slaves per motion (default: 4)')
    parser.add_argument('--curriculum', type=str, default='balanced', 
                        choices=['uniform', 'performance', 'progress', 'balanced', 'ucb'],
                        help='Curriculum learning strategy (default: balanced)')
    parser.add_argument('--warmup', type=int, default=20, 
                        help='Curriculum warmup epochs (default: 20)')
    parser.add_argument('--update_freq', type=int, default=5,
                        help='Curriculum weight update frequency (default: 5)')
    
    args = parser.parse_args()
    
    # Create multimodal PPO trainer
    ppo = MultimodalPPO(
        motion_list_path=args.motion_list,
        metadata_template_path=args.template,
        num_slaves_per_motion=args.slaves,
        curriculum_strategy=args.curriculum,
        curriculum_warmup=args.warmup,
        curriculum_update_freq=args.update_freq
    )
    
    # Create nn directory
    nn_dir = 'nn'
    if not os.path.exists(nn_dir):
        os.makedirs(nn_dir)
        
    # Load or save initial model
    if args.model is not None:
        ppo.LoadModel(args.model)
    else:
        ppo.SaveModel()
        
    print(f'\nNum states: {ppo.env.GetNumState()}, num actions: {ppo.env.GetNumAction()}')
    print(f'Starting multimodal training with {ppo.num_motions} motions...')
    print(f'Curriculum strategy: {args.curriculum}\n')
    
    # Training loop
    for i in range(ppo.max_iteration - 5):
        ppo.Train()
        rewards = ppo.Evaluate()


if __name__ == "__main__":
    main()