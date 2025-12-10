"""
MotionNN - Neural network for motion prediction pre-training.

Predicts absolute next states from current state, learning motion dynamics
to replace BVH reference data during RL training.

Modern best practices applied:
- Pre-LayerNorm (norm_first=True)  
- RMSNorm option for faster training
- Rotary Position Embeddings (RoPE) option
- SwiGLU activation option
- Residual scaling for deep networks
- Proper initialization
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
        # Use smaller init for output projections
        if hasattr(m, 'is_output_proj') and m.is_output_proj:
            torch.nn.init.normal_(m.weight, std=0.02)
        else:
            torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.zero_()


# =============================================================================
# RMSNorm (faster than LayerNorm)
# =============================================================================

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""
    
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight


# =============================================================================
# Rotary Positional Embeddings (RoPE)
# =============================================================================

class RotaryEmbedding(nn.Module):
    """Rotary Position Embeddings (RoPE) for better position encoding."""
    
    def __init__(self, dim: int, max_len: int = 512, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        self._build_cache(max_len)
    
    def _build_cache(self, max_len: int):
        t = torch.arange(max_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer('cos_cached', emb.cos().unsqueeze(0))
        self.register_buffer('sin_cached', emb.sin().unsqueeze(0))
    
    def forward(self, x: torch.Tensor) -> tuple:
        """Return (cos, sin) for position embeddings."""
        seq_len = x.size(1)
        return self.cos_cached[:, :seq_len], self.sin_cached[:, :seq_len]


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate half the hidden dims of the input."""
    x1, x2 = x[..., :x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary_pos_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply rotary position embeddings."""
    return x * cos + rotate_half(x) * sin


# =============================================================================
# Standard Positional Encoding (fallback)
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
# SwiGLU Feed-Forward (better than ReLU/GELU)
# =============================================================================

class SwiGLU(nn.Module):
    """SwiGLU activation for feed-forward layers."""
    
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int = None, bias: bool = True):
        super().__init__()
        out_dim = out_dim or in_dim
        # SwiGLU uses 2/3 of hidden_dim for each gate
        actual_hidden = int(2 * hidden_dim / 3)
        self.w1 = nn.Linear(in_dim, actual_hidden, bias=bias)
        self.w2 = nn.Linear(actual_hidden, out_dim, bias=bias)
        self.w3 = nn.Linear(in_dim, actual_hidden, bias=bias)  # Gate
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


# =============================================================================
# Rotary Attention (with RoPE)
# =============================================================================

class RotaryAttention(nn.Module):
    """Multi-head attention with Rotary Position Embeddings (RoPE)."""
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1, max_len: int = 512):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5
        
        # Precompute rotary embeddings
        self._build_rope(max_len)
    
    def _build_rope(self, max_len: int):
        """Build rotary position embeddings cache."""
        dim = self.head_dim
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_len)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer('cos', emb.cos().unsqueeze(0).unsqueeze(0))  # (1, 1, seq, dim)
        self.register_buffer('sin', emb.sin().unsqueeze(0).unsqueeze(0))
    
    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """Rotate half the hidden dims."""
        x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
        return torch.cat([-x2, x1], dim=-1)
    
    def _apply_rope(self, x: torch.Tensor, seq_len: int) -> torch.Tensor:
        """Apply rotary embeddings to input."""
        cos = self.cos[:, :, :seq_len, :]
        sin = self.sin[:, :, :seq_len, :]
        return x * cos + self._rotate_half(x) * sin
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            (batch, seq_len, d_model)
        """
        batch, seq_len, _ = x.shape
        
        # Project to Q, K, V
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # Reshape for multi-head: (batch, n_heads, seq_len, head_dim)
        q = q.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        
        # Apply RoPE to Q and K
        q = self._apply_rope(q, seq_len)
        k = self._apply_rope(k, seq_len)
        
        # Scaled dot-product attention
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention to values
        out = torch.matmul(attn, v)
        
        # Reshape back: (batch, seq_len, d_model)
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, self.d_model)
        return self.out_proj(out)


# =============================================================================
# Modern Transformer Block
# =============================================================================

class ModernTransformerBlock(nn.Module):
    """
    Transformer block with modern improvements:
    - Pre-norm (norm before attention/FFN)
    - RMSNorm (faster than LayerNorm)
    - Rotary Position Embeddings (RoPE)
    - SwiGLU activation
    - Optional residual scaling
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        ff_mult: float = 4.0,
        dropout: float = 0.1,
        use_swiglu: bool = True,
        use_rope: bool = True,
        residual_scale: float = 1.0,
        max_len: int = 512,
    ):
        super().__init__()
        self.residual_scale = residual_scale
        self.use_rope = use_rope
        
        # Pre-norm
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)
        
        # Self-attention (with or without RoPE)
        if use_rope:
            self.attn = RotaryAttention(d_model, n_heads, dropout=dropout, max_len=max_len)
        else:
            self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        
        # Feed-forward
        ff_dim = int(d_model * ff_mult)
        if use_swiglu:
            self.ff = SwiGLU(d_model, ff_dim)
        else:
            self.ff = nn.Sequential(
                nn.Linear(d_model, ff_dim),
                nn.GELU(),
                nn.Linear(ff_dim, d_model),
            )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor = None) -> torch.Tensor:
        # Pre-norm attention
        h = self.norm1(x)
        if self.use_rope:
            h = self.attn(h)
        else:
            h, _ = self.attn(h, h, h, attn_mask=attn_mask, need_weights=False)
        x = x + self.dropout(h) * self.residual_scale
        
        # Pre-norm FFN
        h = self.norm2(x)
        h = self.ff(h)
        x = x + self.dropout(h) * self.residual_scale
        
        return x


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
        use_modern: Use modern transformer (RMSNorm, SwiGLU)
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
        use_modern: bool = True,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.mode = mode
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.seq_len = seq_len
        self.use_modern = use_modern
        
        if mode == 'mlp':
            self._build_mlp(state_dim, hidden_dim, n_layers)
        elif mode in ('transformer_reg', 'transformer_ar'):
            if use_modern:
                self._build_modern_transformer(state_dim, hidden_dim, n_layers, n_heads, dropout)
            else:
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
                nn.LayerNorm(hidden_dim),  # Add LayerNorm for stability
                nn.SiLU(),  # SiLU (Swish) often better than LeakyReLU
            ])
            in_dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, state_dim))
        self.mlp = nn.Sequential(*layers)
    
    def _build_transformer(self, state_dim: int, hidden_dim: int, n_layers: int, 
                           n_heads: int, dropout: float):
        """Build standard Transformer backbone."""
        self.input_proj = nn.Linear(state_dim, hidden_dim)
        self.pos_enc = PositionalEncoding(hidden_dim, dropout=dropout)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,  # Pre-LayerNorm
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.output_norm = nn.LayerNorm(hidden_dim)  # Final norm
        self.output_proj = nn.Linear(hidden_dim, state_dim)
    
    def _build_modern_transformer(self, state_dim: int, hidden_dim: int, n_layers: int,
                                   n_heads: int, dropout: float):
        """Build modern Transformer with RMSNorm, SwiGLU, and RoPE (inside attention)."""
        self.input_proj = nn.Linear(state_dim, hidden_dim)
        # No positional encoding needed - RoPE is applied inside attention layers
        self.pos_enc = None
        
        # Residual scaling for deep networks (helps with gradient flow)
        residual_scale = 1.0 / math.sqrt(n_layers)
        
        self.layers = nn.ModuleList([
            ModernTransformerBlock(
                d_model=hidden_dim,
                n_heads=n_heads,
                ff_mult=4.0,
                dropout=dropout,
                use_swiglu=True,
                use_rope=True,
                residual_scale=residual_scale,
                max_len=self.seq_len * 2,  # Allow some extra for safety
            )
            for _ in range(n_layers)
        ])
        
        self.output_norm = RMSNorm(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, state_dim)
        self.output_proj.is_output_proj = True  # Mark for smaller init
    
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
            h = self.input_proj(x)
            if self.pos_enc is not None:
                h = self.pos_enc(h)
            
            if self.use_modern:
                for layer in self.layers:
                    h = layer(h)
            else:
                h = self.transformer(h)
            
            h = self.output_norm(h)
            return self.output_proj(h[:, -1])  # Last position
        
        elif self.mode == 'transformer_ar':
            h = self.input_proj(x)
            if self.pos_enc is not None:
                h = self.pos_enc(h)
            
            if self.use_modern:
                for layer in self.layers:
                    h = layer(h)
            else:
                h = self.transformer(h)
            
            h = self.output_norm(h)
            return self.output_proj(h)
    
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
            'n_layers': self.n_layers,
            'n_heads': self.n_heads,
            'seq_len': self.seq_len,
            'use_modern': self.use_modern,
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
            n_layers=checkpoint.get('n_layers', 3),
            n_heads=checkpoint.get('n_heads', 4),
            seq_len=checkpoint.get('seq_len', 32),
            use_modern=checkpoint.get('use_modern', True),
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
            
            predictions = model.predict_rollout(x, max_steps)
            
            for s in steps:
                if s <= predictions.size(1):
                    errors[s].append(predictions[:, s-1].pow(2).mean().item())
    
    return {s: np.mean(errors[s]) if errors[s] else 0.0 for s in steps}
