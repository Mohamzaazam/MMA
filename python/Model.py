"""
Model.py - Neural Network Models for MASS

This module contains the neural network architectures:
- SimulationNN: Policy and value network for PPO
- MuscleNN: Network for computing muscle activations from desired torques
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

from rl_utils import device, Tensor

# =============================================================================
# Distribution Setup
# =============================================================================

MultiVariateNormal = torch.distributions.Normal
temp = MultiVariateNormal.log_prob
MultiVariateNormal.log_prob = lambda self, val: temp(self, val).sum(-1, keepdim=True)

temp2 = MultiVariateNormal.entropy
MultiVariateNormal.entropy = lambda self: temp2(self).sum(-1)
MultiVariateNormal.mode = lambda self: self.mean


# =============================================================================
# Weight Initialization
# =============================================================================

def weights_init(m):
    """Xavier uniform initialization for Linear layers."""
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.zero_()


# =============================================================================
# Muscle Neural Network
# =============================================================================

class MuscleNN(nn.Module):
    """
    Neural network for computing muscle activations from desired torques.
    
    Takes as input:
    - JtA (muscle Jacobian transpose times activation): related DOF torques
    - tau (desired torques): target joint torques
    
    Outputs:
    - Muscle activation levels (0 to 1 range via Tanh + ReLU)
    """
    
    def __init__(self, num_total_muscle_related_dofs, num_dofs, num_muscles):
        super(MuscleNN, self).__init__()
        self.num_total_muscle_related_dofs = num_total_muscle_related_dofs
        self.num_dofs = num_dofs
        self.num_muscles = num_muscles

        num_h1 = 1024
        num_h2 = 512
        num_h3 = 512
        
        self.fc = nn.Sequential(
            nn.Linear(num_total_muscle_related_dofs + num_dofs, num_h1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(num_h1, num_h2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(num_h2, num_h3),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(num_h3, num_muscles),
            nn.Tanh(),
            nn.ReLU()
        )
        
        # Normalization constants
        self.std_muscle_tau = torch.zeros(self.num_total_muscle_related_dofs).to(device)
        self.std_tau = torch.zeros(self.num_dofs).to(device)

        for i in range(self.num_total_muscle_related_dofs):
            self.std_muscle_tau[i] = 200.0

        for i in range(self.num_dofs):
            self.std_tau[i] = 200.0

        self.to(device)
        self.fc.apply(weights_init)

    def forward(self, muscle_tau, tau):
        """Forward pass with input normalization."""
        muscle_tau = muscle_tau / self.std_muscle_tau
        tau = tau / self.std_tau
        out = self.fc.forward(torch.cat([muscle_tau, tau], dim=1))
        return out

    def load(self, path):
        """Load model weights from file."""
        print('load muscle nn {}'.format(path))
        self.load_state_dict(torch.load(path, weights_only=True))

    def save(self, path):
        """Save model weights to file."""
        print('save muscle nn {}'.format(path))
        torch.save(self.state_dict(), path)

    def get_activation(self, muscle_tau, tau):
        """Get muscle activations for given inputs (numpy interface)."""
        act = self.forward(
            Tensor(muscle_tau.reshape(1, -1).astype(np.float32)),
            Tensor(tau.reshape(1, -1).astype(np.float32))
        )
        return act.cpu().detach().numpy().squeeze()


# =============================================================================
# Simulation Neural Network (Policy + Value)
# =============================================================================

class SimulationNN(nn.Module):
    """
    Combined policy and value network for PPO.
    
    Policy head: Outputs mean of Gaussian distribution over actions
    Value head: Outputs state value estimate
    
    Both heads share no layers (separate networks).
    """
    
    def __init__(self, num_states, num_actions):
        super(SimulationNN, self).__init__()

        num_h1 = 256
        num_h2 = 256

        # Policy network
        self.p_fc1 = nn.Linear(num_states, num_h1)
        self.p_fc2 = nn.Linear(num_h1, num_h2)
        self.p_fc3 = nn.Linear(num_h2, num_actions)
        self.log_std = nn.Parameter(torch.zeros(num_actions))

        # Value network
        self.v_fc1 = nn.Linear(num_states, num_h1)
        self.v_fc2 = nn.Linear(num_h1, num_h2)
        self.v_fc3 = nn.Linear(num_h2, 1)

        # Initialize policy network
        torch.nn.init.xavier_uniform_(self.p_fc1.weight)
        torch.nn.init.xavier_uniform_(self.p_fc2.weight)
        torch.nn.init.xavier_uniform_(self.p_fc3.weight)
        self.p_fc1.bias.data.zero_()
        self.p_fc2.bias.data.zero_()
        self.p_fc3.bias.data.zero_()

        # Initialize value network
        torch.nn.init.xavier_uniform_(self.v_fc1.weight)
        torch.nn.init.xavier_uniform_(self.v_fc2.weight)
        torch.nn.init.xavier_uniform_(self.v_fc3.weight)
        self.v_fc1.bias.data.zero_()
        self.v_fc2.bias.data.zero_()
        self.v_fc3.bias.data.zero_()

        self.to(device)

    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: State tensor
            
        Returns:
            Tuple of (action_distribution, value)
        """
        # Policy forward
        p_out = F.relu(self.p_fc1(x))
        p_out = F.relu(self.p_fc2(p_out))
        p_out = self.p_fc3(p_out)
        p_out = MultiVariateNormal(p_out, self.log_std.exp())

        # Value forward
        v_out = F.relu(self.v_fc1(x))
        v_out = F.relu(self.v_fc2(v_out))
        v_out = self.v_fc3(v_out)
        
        return p_out, v_out

    def load(self, path):
        """Load model weights from file."""
        print('load simulation nn {}'.format(path))
        self.load_state_dict(torch.load(path, weights_only=True))

    def save(self, path):
        """Save model weights to file."""
        print('save simulation nn {}'.format(path))
        torch.save(self.state_dict(), path)

    def get_action(self, s):
        """Get deterministic action (mean) for given state (numpy interface)."""
        ts = Tensor(s.astype(np.float32))
        p, _ = self.forward(ts)
        return p.loc.cpu().detach().numpy().squeeze()

    def get_random_action(self, s):
        """Get stochastic action (sampled) for given state (numpy interface)."""
        ts = Tensor(s.astype(np.float32))
        p, _ = self.forward(ts)
        return p.sample().cpu().detach().numpy().squeeze()