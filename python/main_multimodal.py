"""
main_multimodal.py - Multimodal Motion Training for MASS

This script extends the original PPO training to support multiple motion clips.
It uses the MultimodalEnvManager to train a single policy that can imitate
multiple different motions (walking, running, etc.).

Key differences from main.py:
1. Uses MultimodalEnvManager instead of single pymss instance
2. Tracks per-motion statistics during training
3. Supports motion weighting for curriculum learning
4. Logs per-motion performance to TensorBoard

Usage:
    # Basic multimodal training
    python main_multimodal.py --motion_list data/motion_list.txt --template data/metadata.txt
    
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
from torch.utils.tensorboard import SummaryWriter

import numpy as np

# Import MASS modules
from Model import SimulationNN, MuscleNN, Tensor
from multimodal_env import MultimodalEnvManager, MotionConfig, load_motion_configs_from_list

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


Episode = namedtuple('Episode', ('s', 'a', 'r', 'value', 'logprob', 'motion_idx'))


class EpisodeBuffer:
    def __init__(self):
        self.data = []

    def Push(self, *args):
        self.data.append(Episode(*args))
        
    def Pop(self):
        self.data.pop()
        
    def GetData(self):
        return self.data


MuscleTransition = namedtuple('MuscleTransition', ('JtA', 'tau_des', 'L', 'b'))


class MuscleBuffer:
    def __init__(self, buff_size=10000):
        super(MuscleBuffer, self).__init__()
        self.buffer = deque(maxlen=buff_size)

    def Push(self, *args):
        self.buffer.append(MuscleTransition(*args))

    def Clear(self):
        self.buffer.clear()


Transition = namedtuple('Transition', ('s', 'a', 'logprob', 'TD', 'GAE'))


class ReplayBuffer:
    def __init__(self, buff_size=10000):
        super(ReplayBuffer, self).__init__()
        self.buffer = deque(maxlen=buff_size)

    def Push(self, *args):
        self.buffer.append(Transition(*args))

    def Clear(self):
        self.buffer.clear()


class MultimodalPPO:
    """
    PPO trainer for multimodal motion imitation.
    
    Extends the original PPO class to support training on multiple motions.
    """
    
    def __init__(self, motion_list_path: str, metadata_template_path: str, 
                 num_slaves_per_motion: int = 4):
        """
        Initialize multimodal PPO trainer.
        
        Args:
            motion_list_path: Path to motion_list.txt
            metadata_template_path: Path to template metadata file
            num_slaves_per_motion: Number of parallel environments per motion
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
        
        # Per-motion tracking
        self.motion_returns = {config.name: [] for config in self.motion_configs}
        self.motion_episode_counts = {config.name: 0 for config in self.motion_configs}

        # Episode buffers
        self.episodes = [None] * self.num_slaves
        for j in range(self.num_slaves):
            self.episodes[j] = EpisodeBuffer()
        self.env.Resets(True)

        # TensorBoard logging
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.writer = SummaryWriter(f'runs/multimodal_{timestamp}')

    def SaveModel(self):
        """Save current and best models"""
        os.makedirs('./nn', exist_ok=True)
        
        self.model.save('./nn/multimodal_current.pt')
        self.muscle_model.save('./nn/multimodal_current_muscle.pt')

        if self.max_return_epoch == self.num_evaluation:
            self.model.save('./nn/multimodal_max.pt')
            self.muscle_model.save('./nn/multimodal_max_muscle.pt')
            
        if self.num_evaluation % 100 == 0:
            self.model.save(f'./nn/multimodal_{self.num_evaluation//100}.pt')
            self.muscle_model.save(f'../nn/multimodal_{self.num_evaluation//100}_muscle.pt')

    def LoadModel(self, path):
        """Load model from checkpoint"""
        self.model.load('../nn/' + path + '.pt')
        self.muscle_model.load('../nn/' + path + '_muscle.pt')

    def ComputeTDandGAE(self):
        """Compute TD targets and GAE advantages"""
        self.replay_buffer.Clear()
        self.muscle_buffer = {}
        self.sum_return = 0.0
        
        # Reset per-motion tracking for this evaluation
        motion_returns_this_epoch = {name: [] for name in self.motion_returns}
        
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
            motion_name = self.motion_configs[motion_indices[0] // self.num_slaves_per_motion].name
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
        self.env.ComputeMuscleTuples()
        self.muscle_buffer['JtA'] = self.env.GetMuscleTuplesJtA()
        self.muscle_buffer['TauDes'] = self.env.GetMuscleTuplesTauDes()
        self.muscle_buffer['L'] = self.env.GetMuscleTuplesL()
        self.muscle_buffer['b'] = self.env.GetMuscleTuplesb()
        
        if self.muscle_buffer['JtA'].size > 0:
            print(f"Muscle tuples shape: {self.muscle_buffer['JtA'].shape}")

    def GenerateTransitions(self):
        """Generate transitions by running the policy"""
        self.total_episodes = []
        states = [None] * self.num_slaves
        actions = [None] * self.num_slaves
        rewards = [None] * self.num_slaves
        
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
                    rewards[j] = self.env.GetReward(j)
                    # Include motion index in episode data
                    motion_idx = j  # Will be used to determine which motion this episode is from
                    self.episodes[j].Push(states[j], actions[j], rewards[j], values[j], logprobs[j], motion_idx)
                    local_step += 1

                if terminated_state or nan_occur:
                    if nan_occur:
                        self.episodes[j].Pop()
                    self.total_episodes.append(self.episodes[j])
                    self.episodes[j] = EpisodeBuffer()
                    self.env.Reset(True, j)

            if local_step >= self.buffer_size:
                break

            states = self.env.GetStates()

    def OptimizeSimulationNN(self):
        """Optimize the policy and value networks"""
        all_transitions = np.array(self.replay_buffer.buffer, dtype=object)
        
        for j in range(self.num_epochs):
            np.random.shuffle(all_transitions)
            
            for i in range(len(all_transitions) // self.batch_size):
                transitions = all_transitions[i * self.batch_size:(i + 1) * self.batch_size]
                batch = Transition(*zip(*transitions))

                stack_s = np.vstack(batch.s).astype(np.float32)
                stack_a = np.vstack(batch.a).astype(np.float32)
                stack_lp = np.vstack(batch.logprob).astype(np.float32)
                stack_td = np.vstack(batch.TD).astype(np.float32)
                stack_gae = np.vstack(batch.GAE).astype(np.float32)

                a_dist, v = self.model(Tensor(stack_s))
                
                # Critic loss
                loss_critic = ((v - Tensor(stack_td)).pow(2)).mean()

                # Actor loss
                ratio = torch.exp(a_dist.log_prob(Tensor(stack_a)) - Tensor(stack_lp))
                stack_gae = (stack_gae - stack_gae.mean()) / (stack_gae.std() + 1E-5)
                stack_gae = Tensor(stack_gae)
                surrogate1 = ratio * stack_gae
                surrogate2 = torch.clamp(ratio, min=1.0 - self.clip_ratio, max=1.0 + self.clip_ratio) * stack_gae
                loss_actor = -torch.min(surrogate1, surrogate2).mean()
                
                # Entropy loss
                loss_entropy = -self.w_entropy * a_dist.entropy().mean()

                self.loss_actor = loss_actor.cpu().detach().numpy().tolist()
                self.loss_critic = loss_critic.cpu().detach().numpy().tolist()

                loss = loss_actor + loss_entropy + loss_critic

                self.optimizer.zero_grad()
                loss.backward(retain_graph=True)
                for param in self.model.parameters():
                    if param.grad is not None:
                        param.grad.data.clamp_(-0.5, 0.5)
                self.optimizer.step()
                
            print(f'Optimizing sim nn : {j + 1}/{self.num_epochs}', end='\r')
        print('')

    def generate_shuffle_indices(self, batch_size, minibatch_size):
        """Generate shuffled indices for minibatch training"""
        n = batch_size
        m = minibatch_size
        p = np.random.permutation(n)

        r = m - n % m
        if r > 0:
            p = np.hstack([p, np.random.randint(0, n, r)])

        p = p.reshape(-1, m)
        return p

    def OptimizeMuscleNN(self):
        """Optimize the muscle network"""
        if self.muscle_buffer['JtA'].size == 0:
            return
            
        for j in range(self.num_epochs_muscle):
            minibatches = self.generate_shuffle_indices(
                self.muscle_buffer['JtA'].shape[0], 
                self.muscle_batch_size
            )

            for minibatch in minibatches:
                stack_JtA = self.muscle_buffer['JtA'][minibatch].astype(np.float32)
                stack_tau_des = self.muscle_buffer['TauDes'][minibatch].astype(np.float32)
                stack_L = self.muscle_buffer['L'][minibatch].astype(np.float32)
                stack_L = stack_L.reshape(self.muscle_batch_size, self.num_action, self.num_muscles)
                stack_b = self.muscle_buffer['b'][minibatch].astype(np.float32)

                stack_JtA = Tensor(stack_JtA)
                stack_tau_des = Tensor(stack_tau_des)
                stack_L = Tensor(stack_L)
                stack_b = Tensor(stack_b)

                activation = self.muscle_model(stack_JtA, stack_tau_des)
                tau = torch.einsum('ijk,ik->ij', (stack_L, activation)) + stack_b

                loss_reg = activation.pow(2).mean()
                loss_target = (((tau - stack_tau_des) / 100.0).pow(2)).mean()

                loss = 0.01 * loss_reg + loss_target

                self.optimizer_muscle.zero_grad()
                loss.backward(retain_graph=True)
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
        
        # Print per-motion stats
        print('||--- Per-Motion Statistics ---')
        for name, returns in self.motion_returns.items():
            if returns:
                avg = np.mean(returns)
                count = self.motion_episode_counts[name]
                print(f'||  {name}: avg_return={avg:.3f}, episodes={count}')

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
        
        # Log per-motion returns
        for name, returns in self.motion_returns.items():
            if returns:
                self.writer.add_scalar(f'PerMotion/{name}_AvgReturn', np.mean(returns), self.num_evaluation)
                self.writer.add_scalar(f'PerMotion/{name}_Episodes', self.motion_episode_counts[name], self.num_evaluation)

        self.SaveModel()

        print('=' * 45)
        return np.array(self.rewards)


def main():
    parser = argparse.ArgumentParser(description='Multimodal Motion Training for MASS')
    parser.add_argument('-m', '--model', help='Model checkpoint to resume from')
    parser.add_argument('--motion_list', required=True, help='Path to motion_list.txt')
    parser.add_argument('--template', required=True, help='Path to template metadata file')
    parser.add_argument('--slaves', type=int, default=2, help='Slaves per motion (default: 4)')
    
    args = parser.parse_args()
    
    # Create multimodal PPO trainer
    ppo = MultimodalPPO(
        motion_list_path=args.motion_list,
        metadata_template_path=args.template,
        num_slaves_per_motion=args.slaves
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
    print(f'Starting multimodal training with {ppo.num_motions} motions...\n')
    
    # Training loop
    for i in range(ppo.max_iteration - 5):
        ppo.Train()
        rewards = ppo.Evaluate()


if __name__ == "__main__":
    main()