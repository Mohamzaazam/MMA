"""
utils.py - Common utility functions and classes for MASS training

This module contains shared utilities used by:
- main.py (single motion training)
- main_multimodal.py (multimodal training)
- Model.py (neural network models)

Contents:
- Device setup and Tensor conversion
- Namedtuples for Episode, Transition, MuscleTransition
- Buffer classes: EpisodeBuffer, ReplayBuffer, MuscleBuffer
"""

import torch
import numpy as np
from collections import namedtuple, deque

# =============================================================================
# Device Configuration
# =============================================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def Tensor(x):
    """
    Convert input to PyTorch tensor on the appropriate device.
    
    Args:
        x: numpy array, torch tensor, or other numeric type
        
    Returns:
        torch.Tensor on the configured device
    """
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x).float()
    elif not isinstance(x, torch.Tensor):
        x = torch.tensor(x).float()
    return x.to(device)


# =============================================================================
# Namedtuples for Data Storage
# =============================================================================

# Episode data: state, action, reward, value, log probability
Episode = namedtuple('Episode', ('s', 'a', 'r', 'value', 'logprob'))

# Episode data with motion index (for multimodal training)
EpisodeWithMotion = namedtuple('EpisodeWithMotion', ('s', 'a', 'r', 'value', 'logprob', 'motion_idx'))

# Transition data for replay buffer: state, action, log prob, TD target, GAE advantage
Transition = namedtuple('Transition', ('s', 'a', 'logprob', 'TD', 'GAE'))

# Muscle transition data
MuscleTransition = namedtuple('MuscleTransition', ('JtA', 'tau_des', 'L', 'b'))


# =============================================================================
# Buffer Classes
# =============================================================================

class EpisodeBuffer:
    """
    Buffer to store episode data during rollout.
    
    Stores tuples of (state, action, reward, value, logprob) for a single episode.
    """
    
    def __init__(self):
        self.data = []

    def Push(self, *args):
        """Add a transition to the episode buffer."""
        self.data.append(Episode(*args))
        
    def Pop(self):
        """Remove the last transition (used when NaN occurs)."""
        if self.data:
            self.data.pop()
        
    def GetData(self):
        """Return all stored episode data."""
        return self.data
    
    def __len__(self):
        return len(self.data)


class EpisodeBufferWithMotion:
    """
    Buffer to store episode data with motion index for multimodal training.
    
    Stores tuples of (state, action, reward, value, logprob, motion_idx).
    """
    
    def __init__(self):
        self.data = []

    def Push(self, *args):
        """Add a transition to the episode buffer."""
        self.data.append(EpisodeWithMotion(*args))
        
    def Pop(self):
        """Remove the last transition (used when NaN occurs)."""
        if self.data:
            self.data.pop()
        
    def GetData(self):
        """Return all stored episode data."""
        return self.data
    
    def __len__(self):
        return len(self.data)


class ReplayBuffer:
    """
    Replay buffer for storing transitions for PPO training.
    
    Stores Transition tuples with a maximum capacity.
    """
    
    def __init__(self, buff_size=10000):
        super(ReplayBuffer, self).__init__()
        self.buffer = deque(maxlen=buff_size)

    def Push(self, *args):
        """Add a transition to the buffer."""
        self.buffer.append(Transition(*args))

    def Clear(self):
        """Clear all stored transitions."""
        self.buffer.clear()
        
    def __len__(self):
        return len(self.buffer)


class MuscleBuffer:
    """
    Buffer for storing muscle transition data.
    
    Used for training the muscle network.
    """
    
    def __init__(self, buff_size=10000):
        super(MuscleBuffer, self).__init__()
        self.buffer = deque(maxlen=buff_size)

    def Push(self, *args):
        """Add a muscle transition to the buffer."""
        self.buffer.append(MuscleTransition(*args))

    def Clear(self):
        """Clear all stored muscle transitions."""
        self.buffer.clear()
        
    def __len__(self):
        return len(self.buffer)


# =============================================================================
# Helper Functions
# =============================================================================

def generate_shuffle_indices(batch_size, minibatch_size):
    """
    Generate shuffled indices for minibatch training.
    
    Args:
        batch_size: Total number of samples
        minibatch_size: Size of each minibatch
        
    Returns:
        Array of shape (num_minibatches, minibatch_size) with shuffled indices
    """
    n = batch_size
    m = minibatch_size
    p = np.random.permutation(n)

    # Pad to make divisible by minibatch_size
    r = m - n % m
    if r > 0 and r < m:
        p = np.hstack([p, np.random.randint(0, n, r)])

    p = p.reshape(-1, m)
    return p


def format_time(seconds):
    """
    Format seconds into hours:minutes:seconds string.
    
    Args:
        seconds: Total elapsed seconds
        
    Returns:
        Formatted string "Hh:Mm:Ss"
    """
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h}h:{m}m:{s}s"