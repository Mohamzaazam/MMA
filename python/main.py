"""
main.py - Single Motion Training for MASS

This script trains a policy to imitate a single motion (e.g., walking)
using PPO with muscle-actuated control.

Usage:
    python main.py -d data/metadata.txt
    python main.py -d data/metadata.txt -m checkpoint_name
"""

import argparse
import os
import time
import numpy as np
from datetime import datetime

import torch
from torch.utils.tensorboard import SummaryWriter

import pymss
from PPO import BasePPO
from utils import device, Tensor, EpisodeBuffer

print(f"Using device: {device}")


class PPO(BasePPO):
    """
    PPO trainer for single motion imitation.
    
    Extends BasePPO with specific initialization for single-motion training.
    """
    
    def __init__(self, meta_file):
        """
        Initialize PPO trainer.
        
        Args:
            meta_file: Path to metadata file defining the environment
        """
        super().__init__()
        
        np.random.seed(seed=int(time.time()))
        
        # Create environment
        self.num_slaves = 32
        env = pymss.pymss(meta_file, self.num_slaves)
        
        # Initialize from environment (sets up networks, buffers, etc.)
        self._init_from_env(env, self.num_slaves)
        
        # Initialize TensorBoard
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._init_tensorboard(f'./runs/single_{timestamp}')
        
        print(f"Environment initialized:")
        print(f"  Num states: {self.num_state}")
        print(f"  Num actions: {self.num_action}")
        print(f"  Use muscle: {self.use_muscle}")
        print(f"  Num muscles: {self.num_muscles}")

    def SaveModel(self):
        """Save models with default paths for single-motion training."""
        nn_dir = './nn'
        os.makedirs(nn_dir, exist_ok=True)
        
        self.model.save(os.path.join(nn_dir, 'current.pt'))
        self.muscle_model.save(os.path.join(nn_dir, 'current_muscle.pt'))

        if self.max_return_epoch == self.num_evaluation:
            self.model.save(os.path.join(nn_dir, 'max.pt'))
            self.muscle_model.save(os.path.join(nn_dir, 'max_muscle.pt'))
            
        if self.num_evaluation % 100 == 0:
            epoch_num = self.num_evaluation // 100
            self.model.save(os.path.join(nn_dir, f'{epoch_num}.pt'))
            self.muscle_model.save(os.path.join(nn_dir, f'{epoch_num}_muscle.pt'))

    def LoadModel(self, path):
        """Load models from checkpoint."""
        nn_dir = './nn'
        self.model.load(os.path.join(nn_dir, f'{path}.pt'))
        self.muscle_model.load(os.path.join(nn_dir, f'{path}_muscle.pt'))

    def Evaluate(self):
        """
        Evaluate and log training progress.
        
        Overrides base to call SaveModel and return rewards array.
        """
        # Call base evaluation (prints stats, logs to tensorboard)
        rewards = super().Evaluate()
        
        # Save models
        self.SaveModel()
        
        return rewards


def main():
    parser = argparse.ArgumentParser(description='MASS Single Motion Training')
    parser.add_argument('-m', '--model', help='Model checkpoint to resume from')
    parser.add_argument('-d', '--meta', help='Metadata file path', required=True)

    args = parser.parse_args()

    # Create PPO trainer
    ppo = PPO(args.meta)
    
    # Create nn directory
    nn_dir = './nn'
    if not os.path.exists(nn_dir):
        os.makedirs(nn_dir)
        
    # Load or save initial model
    if args.model is not None:
        ppo.LoadModel(args.model)
    else:
        ppo.SaveModel()
        
    print(f'num states: {ppo.env.GetNumState()}, num actions: {ppo.env.GetNumAction()}')
    
    # Training loop
    for i in range(ppo.max_iteration - 5):
        ppo.Train()
        rewards = ppo.Evaluate()


if __name__ == "__main__":
    main()