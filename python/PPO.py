"""
PPO.py - Base PPO (Proximal Policy Optimization) class for MASS training

This module provides the core PPO training functionality shared by:
- main.py (single motion training)
- main_multimodal.py (multimodal training)

The BasePPO class contains:
- Hyperparameter management
- Neural network optimization (simulation and muscle networks)
- Transition generation and TD/GAE computation
- Model saving/loading
- TensorBoard logging
"""

import math
import random
import time
import os
import numpy as np
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from utils import (
    device, Tensor, 
    Episode, Transition, MuscleTransition,
    EpisodeBuffer, ReplayBuffer, MuscleBuffer,
    generate_shuffle_indices, format_time
)
from Model import SimulationNN, MuscleNN


class BasePPO:
    """
    Base class for PPO training in MASS.
    
    This class provides the core training loop and optimization methods.
    Subclasses can override specific methods for customization (e.g., multimodal training).
    """
    
    def __init__(self):
        """Initialize base PPO with default hyperparameters."""
        # These will be set by subclasses after environment initialization
        self.env = None
        self.use_muscle = False
        self.num_state = 0
        self.num_action = 0
        self.num_muscles = 0
        self.num_slaves = 0
        
        # Training hyperparameters
        self.num_epochs = 10
        self.num_epochs_muscle = 3
        self.num_evaluation = 0
        self.num_tuple_so_far = 0
        self.num_episode = 0
        self.num_tuple = 0
        
        self.gamma = 0.99
        self.lb = 0.99  # Lambda for GAE

        self.buffer_size = 2048
        self.batch_size = 128
        self.muscle_batch_size = 128
        
        # Learning rate and clipping
        self.default_learning_rate = 1E-4
        self.default_clip_ratio = 0.2
        self.learning_rate = self.default_learning_rate
        self.clip_ratio = self.default_clip_ratio
        
        self.max_iteration = 50000
        self.w_entropy = -0.001

        # Tracking variables
        self.loss_actor = 0.0
        self.loss_critic = 0.0
        self.loss_muscle = 0.0
        self.rewards = []
        self.sum_return = 0.0
        self.max_return = -1.0
        self.max_return_epoch = 1
        self.tic = time.time()
        
        # Buffers (initialized in subclass)
        self.replay_buffer = ReplayBuffer(30000)
        self.muscle_buffer = {}
        self.episodes = []
        self.total_episodes = []
        
        # Neural networks (initialized in subclass)
        self.model = None
        self.muscle_model = None
        self.optimizer = None
        self.optimizer_muscle = None
        
        # TensorBoard writer (initialized in subclass)
        self.writer = None
        
        # Simulation parameters (set after env initialization)
        self.num_simulation_Hz = 0
        self.num_control_Hz = 0
        self.num_simulation_per_control = 0

    def _init_from_env(self, env, num_slaves):
        """
        Initialize parameters from environment.
        
        Called by subclasses after environment is created.
        
        Args:
            env: The pymss environment instance
            num_slaves: Number of parallel environments
        """
        self.env = env
        self.num_slaves = num_slaves
        self.use_muscle = env.UseMuscle()
        self.num_state = env.GetNumState()
        self.num_action = env.GetNumAction()
        self.num_muscles = env.GetNumMuscles()
        self.num_simulation_Hz = env.GetSimulationHz()
        self.num_control_Hz = env.GetControlHz()
        self.num_simulation_per_control = self.num_simulation_Hz // self.num_control_Hz
        
        # Initialize buffers
        self.replay_buffer = ReplayBuffer(30000)
        
        # Initialize episode buffers for each slave
        self.episodes = [EpisodeBuffer() for _ in range(num_slaves)]
        
        # Initialize neural networks
        self.model = SimulationNN(self.num_state, self.num_action).to(device)
        self.muscle_model = MuscleNN(
            env.GetNumTotalMuscleRelatedDofs(),
            self.num_action,
            self.num_muscles
        ).to(device)
        
        # Initialize optimizers
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.optimizer_muscle = optim.Adam(self.muscle_model.parameters(), lr=self.learning_rate)
        
        # Reset environments
        env.Resets(True)

    def _init_tensorboard(self, log_dir=None):
        """
        Initialize TensorBoard writer.
        
        Args:
            log_dir: Optional log directory. If None, uses default 'runs/' directory.
        """
        if log_dir:
            self.writer = SummaryWriter(log_dir)
        else:
            self.writer = SummaryWriter()

    # =========================================================================
    # Model Save/Load
    # =========================================================================

    def SaveModel(self, prefix='', nn_dir='./nn'):
        """
        Save current and best models.
        
        Args:
            prefix: Prefix for model filenames
            nn_dir: Directory to save models
        """
        os.makedirs(nn_dir, exist_ok=True)
        
        model_path = os.path.join(nn_dir, f'{prefix}current.pt')
        muscle_path = os.path.join(nn_dir, f'{prefix}current_muscle.pt')
        
        self.model.save(model_path)
        self.muscle_model.save(muscle_path)

        if self.max_return_epoch == self.num_evaluation:
            self.model.save(os.path.join(nn_dir, f'{prefix}max.pt'))
            self.muscle_model.save(os.path.join(nn_dir, f'{prefix}max_muscle.pt'))
            
        if self.num_evaluation % 100 == 0:
            epoch_num = self.num_evaluation // 100
            self.model.save(os.path.join(nn_dir, f'{prefix}{epoch_num}.pt'))
            self.muscle_model.save(os.path.join(nn_dir, f'{prefix}{epoch_num}_muscle.pt'))

    def LoadModel(self, path, nn_dir='./nn'):
        """
        Load model from checkpoint.
        
        Args:
            path: Model name (without extension)
            nn_dir: Directory containing models
        """
        model_path = os.path.join(nn_dir, f'{path}.pt')
        muscle_path = os.path.join(nn_dir, f'{path}_muscle.pt')
        
        print(f'Loading simulation nn: {model_path}')
        self.model.load(model_path)
        
        print(f'Loading muscle nn: {muscle_path}')
        self.muscle_model.load(muscle_path)

    # =========================================================================
    # Core Training Methods
    # =========================================================================

    def ComputeTDandGAE(self):
        """
        Compute TD targets and GAE advantages from collected episodes.
        
        This method processes self.total_episodes and fills self.replay_buffer
        with transitions containing TD targets and GAE advantages.
        """
        self.replay_buffer.Clear()
        self.muscle_buffer = {}
        self.sum_return = 0.0
        
        for epi in self.total_episodes:
            data = epi.GetData()
            size = len(data)
            if size == 0:
                continue
                
            states, actions, rewards, values, logprobs = zip(*data)

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

            for i in range(size):
                self.replay_buffer.Push(states[i], actions[i], logprobs[i], TD[i], advantages[i])
                
        self.num_episode = len(self.total_episodes)
        self.num_tuple = len(self.replay_buffer.buffer)
        print(f'SIM : {self.num_tuple}')
        self.num_tuple_so_far += self.num_tuple

        # Compute muscle tuples
        self._compute_muscle_tuples()

    def _compute_muscle_tuples(self):
        """Compute and store muscle tuples for muscle network training."""
        self.env.ComputeMuscleTuples()
        self.muscle_buffer['JtA'] = self.env.GetMuscleTuplesJtA()
        self.muscle_buffer['TauDes'] = self.env.GetMuscleTuplesTauDes()
        self.muscle_buffer['L'] = self.env.GetMuscleTuplesL()
        self.muscle_buffer['b'] = self.env.GetMuscleTuplesb()
        
        if self.muscle_buffer['JtA'].size > 0:
            print(f"Muscle tuples shape: {self.muscle_buffer['JtA'].shape}")

    def GenerateTransitions(self):
        """
        Generate transitions by running the policy in the environment.
        
        Fills self.total_episodes with episode data.
        """
        self.total_episodes = []
        states = self.env.GetStates()
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
            
            # Set actions and step
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

            # Process each slave
            for j in range(self.num_slaves):
                nan_occur = False
                terminated_state = True

                if (np.any(np.isnan(states[j])) or np.any(np.isnan(actions[j])) or
                    np.any(np.isnan(values[j])) or np.any(np.isnan(logprobs[j]))):
                    nan_occur = True
                elif self.env.IsEndOfEpisode(j) is False:
                    terminated_state = False
                    reward = self.env.GetReward(j)
                    self.episodes[j].Push(states[j], actions[j], reward, values[j], logprobs[j])
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
        """Optimize the policy and value networks using PPO."""
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

                # Actor loss (PPO clipped objective)
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

    def OptimizeMuscleNN(self):
        """Optimize the muscle network."""
        if self.muscle_buffer['JtA'].size == 0:
            return

        for j in range(self.num_epochs_muscle):
            minibatches = generate_shuffle_indices(
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
        """Run one optimization iteration (TD/GAE computation + network optimization)."""
        self.ComputeTDandGAE()
        self.OptimizeSimulationNN()
        if self.use_muscle:
            self.OptimizeMuscleNN()

    def Train(self):
        """Run one training iteration (generate transitions + optimize)."""
        self.GenerateTransitions()
        self.OptimizeModel()

    # =========================================================================
    # Evaluation and Logging
    # =========================================================================

    def Evaluate(self):
        """
        Evaluate and log training progress.
        
        Returns:
            numpy array of historical rewards
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

        # Print stats
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

        print('=' * 45)
        return np.array(self.rewards)