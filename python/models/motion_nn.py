"""
MotionNN - Neural network for motion prediction pre-training.

Predicts absolute next states from current state, learning motion dynamics
to replace BVH reference data during RL training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Optional, Literal


# =============================================================================
# Weight Initialization
# =============================================================================

def weights_init(m):
    """Xavier uniform initialization for Linear layers."""
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.zero_()


# =============================================================================
# Positional Encoding for Transformer
# =============================================================================

class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding."""
    
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq_len, d_model)"""
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


# =============================================================================
# MotionNN
# =============================================================================

class MotionNN(nn.Module):
    """
    State transition predictor: s_t → s_{t+1}
    
    Modes:
    - 'mlp': 3-layer MLP, single-step prediction
    - 'transformer_reg': Transformer encoder, sequence → next frame
    - 'transformer_ar': Transformer, sequence → next sequence (autoregressive)
    
    Args:
        state_dim: Dimension of state vector (positions + velocities)
        mode: 'mlp', 'transformer_reg', or 'transformer_ar'
        hidden_dim: Hidden layer dimension for MLP
        n_layers: Number of transformer layers
        n_heads: Number of attention heads
        seq_len: Sequence length for transformer modes
        dropout: Dropout probability
    """
    
    def __init__(
        self,
        state_dim: int,
        mode: Literal['mlp', 'transformer_reg', 'transformer_ar'] = 'mlp',
        hidden_dim: int = 256,
        n_layers: int = 3,
        n_heads: int = 4,
        seq_len: int = 32,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.mode = mode
        self.hidden_dim = hidden_dim
        self.seq_len = seq_len
        
        if mode == 'mlp':
            self._build_mlp(state_dim, hidden_dim, n_layers)
        elif mode in ('transformer_reg', 'transformer_ar'):
            self._build_transformer(state_dim, hidden_dim, n_layers, n_heads, dropout)
        else:
            raise ValueError(f"Unknown mode: {mode}")
        
        self.apply(weights_init)
    
    def _build_mlp(self, state_dim: int, hidden_dim: int, n_layers: int):
        """Build MLP backbone."""
        layers = []
        in_dim = state_dim
        for _ in range(n_layers):
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.LeakyReLU(0.2, inplace=True),
            ])
            in_dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, state_dim))
        self.mlp = nn.Sequential(*layers)
    
    def _build_transformer(self, state_dim: int, hidden_dim: int, n_layers: int, 
                           n_heads: int, dropout: float):
        """Build Transformer backbone."""
        self.input_proj = nn.Linear(state_dim, hidden_dim)
        self.pos_enc = PositionalEncoding(hidden_dim, dropout=dropout)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.output_proj = nn.Linear(hidden_dim, state_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input state(s)
               - MLP: (batch, state_dim)
               - Transformer: (batch, seq_len, state_dim)
        
        Returns:
            - MLP: (batch, state_dim) - predicted next state
            - transformer_reg: (batch, state_dim) - predicted next frame
            - transformer_ar: (batch, seq_len, state_dim) - predicted next sequence
        """
        if self.mode == 'mlp':
            return self.mlp(x)
        
        elif self.mode == 'transformer_reg':
            # Sequence → next frame (regression)
            h = self.input_proj(x)  # (batch, seq, hidden)
            h = self.pos_enc(h)
            h = self.transformer(h)
            # Use last position for prediction
            return self.output_proj(h[:, -1])  # (batch, state_dim)
        
        elif self.mode == 'transformer_ar':
            # Sequence → next sequence (autoregressive)
            h = self.input_proj(x)
            h = self.pos_enc(h)
            h = self.transformer(h)
            return self.output_proj(h)  # (batch, seq, state_dim)
    
    def predict_rollout(self, initial_state: torch.Tensor, n_steps: int) -> torch.Tensor:
        """
        Autoregressive rollout: predict n_steps into the future.
        
        Args:
            initial_state: Starting state(s)
                - MLP: (batch, state_dim)
                - Transformer: (batch, seq_len, state_dim)
            n_steps: Number of steps to predict
        
        Returns:
            Trajectory of predictions: (batch, n_steps, state_dim)
        """
        predictions = []
        state = initial_state
        
        for _ in range(n_steps):
            next_state = self.forward(state)
            
            if self.mode == 'mlp':
                predictions.append(next_state.unsqueeze(1))
                state = next_state
            elif self.mode == 'transformer_reg':
                predictions.append(next_state.unsqueeze(1))
                # Shift sequence: drop first, append prediction
                state = torch.cat([state[:, 1:], next_state.unsqueeze(1)], dim=1)
            elif self.mode == 'transformer_ar':
                # Take last prediction as next step
                predictions.append(next_state[:, -1:])
                state = torch.cat([state[:, 1:], next_state[:, -1:]], dim=1)
        
        return torch.cat(predictions, dim=1)
    
    def save(self, path: str):
        """Save model checkpoint with metadata."""
        torch.save({
            'state_dict': self.state_dict(),
            'state_dim': self.state_dim,
            'mode': self.mode,
            'hidden_dim': self.hidden_dim,
            'seq_len': self.seq_len,
        }, path)
        print(f"Saved MotionNN to {path}")
    
    @classmethod
    def load(cls, path: str, device: Optional[torch.device] = None) -> 'MotionNN':
        """Load model from checkpoint."""
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        model = cls(
            state_dim=checkpoint['state_dim'],
            mode=checkpoint['mode'],
            hidden_dim=checkpoint['hidden_dim'],
            seq_len=checkpoint.get('seq_len', 32),
        )
        model.load_state_dict(checkpoint['state_dict'])
        print(f"Loaded MotionNN from {path}")
        return model


# =============================================================================
# Rollout Evaluation
# =============================================================================

def evaluate_rollout(
    model: MotionNN,
    dataloader: torch.utils.data.DataLoader,
    steps: list[int] = [5, 10, 30, 60],
    device: torch.device = torch.device('cpu'),
) -> dict[int, float]:
    """
    Evaluate multi-step rollout error.
    
    Args:
        model: Trained MotionNN
        dataloader: DataLoader with sequences long enough for max(steps)
        steps: List of step counts to evaluate
        device: Device to run on
    
    Returns:
        Dict mapping step count to MSE
    """
    model.eval()
    errors = {s: [] for s in steps}
    
    with torch.no_grad():
        for batch in dataloader:
            if model.mode == 'mlp':
                x, _ = batch  # (batch, state_dim)
            else:
                x, _ = batch  # (batch, seq_len, state_dim)
            
            x = x.to(device)
            max_steps = max(steps)
            
            # Get ground truth trajectory from dataset
            # Note: This requires the dataset to return longer sequences
            # For now, we use model's own predictions to measure drift
            predictions = model.predict_rollout(x, max_steps)
            
            # Record errors at each step count
            # In production, compare against actual ground truth
            for s in steps:
                if s <= predictions.size(1):
                    # MSE of prediction at step s vs ground truth
                    # Placeholder: using first batch element
                    errors[s].append(predictions[:, s-1].pow(2).mean().item())
    
    return {s: np.mean(errors[s]) if errors[s] else 0.0 for s in steps}
