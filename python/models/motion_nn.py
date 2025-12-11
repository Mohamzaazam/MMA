"""
MotionNN - Neural network for motion prediction.

Predicts next state(s) from current state(s) for motion pre-training.

Architecture options:
- MLP: Single frame → next frame
- Encoder: Sequence → next single frame  
- Seq2Seq: Sequence → next sequence

Modern implementations use RMSNorm, SwiGLU, and Rotary Position Embeddings.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Optional, Literal


# =============================================================================
# Building Blocks
# =============================================================================

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (faster than LayerNorm)."""
    
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return x / rms * self.scale


class SwiGLU(nn.Module):
    """SwiGLU activation (better than GELU for transformers)."""
    
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        hidden = int(2 * hidden_dim / 3)
        self.w1 = nn.Linear(dim, hidden)
        self.w2 = nn.Linear(hidden, dim)
        self.w3 = nn.Linear(dim, hidden)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class RotaryEmbedding(nn.Module):
    """Rotary Position Embeddings (RoPE)."""
    
    def __init__(self, dim: int, max_len: int = 512):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        self._build_cache(max_len)
    
    def _build_cache(self, max_len: int):
        t = torch.arange(max_len, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer('cos', emb.cos())
        self.register_buffer('sin', emb.sin())
    
    def forward(self, seq_len: int):
        return self.cos[:seq_len], self.sin[:seq_len]


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply rotary embeddings to x."""
    x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
    x_rotated = torch.cat([-x2, x1], dim=-1)
    return x * cos + x_rotated * sin


# =============================================================================
# Attention with RoPE
# =============================================================================

class Attention(nn.Module):
    """Multi-head self-attention with RoPE."""
    
    def __init__(self, dim: int, n_heads: int, dropout: float = 0.1, max_len: int = 512):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, 3 * dim)
        self.out = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        
        # RoPE
        self.rope = RotaryEmbedding(self.head_dim, max_len)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        
        # Compute Q, K, V
        qkv = self.qkv(x).reshape(B, L, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(2)  # (B, L, H, D)
        
        # Apply RoPE
        cos, sin = self.rope(L)
        cos = cos.view(1, L, 1, self.head_dim)
        sin = sin.view(1, L, 1, self.head_dim)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        
        # Attention
        q, k, v = [t.transpose(1, 2) for t in (q, k, v)]  # (B, H, L, D)
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(B, L, D)
        return self.out(out)


# =============================================================================
# Transformer Block
# =============================================================================

class TransformerBlock(nn.Module):
    """Pre-norm transformer block with RMSNorm and SwiGLU."""
    
    def __init__(self, dim: int, n_heads: int, dropout: float = 0.1, max_len: int = 512):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = Attention(dim, n_heads, dropout, max_len)
        self.norm2 = RMSNorm(dim)
        self.ffn = SwiGLU(dim, dim * 4)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.dropout(self.attn(self.norm1(x)))
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x


# =============================================================================
# MotionNN
# =============================================================================

class MotionNN(nn.Module):
    """
    Motion prediction network: state(s) → next state(s)
    
    Modes:
        'mlp': (batch, dim) → (batch, dim)
        'encoder': (batch, seq, dim) → (batch, dim)  
        'seq2seq': (batch, seq, dim) → (batch, seq, dim)
    
    Args:
        state_dim: Dimension of state vector (56 pos + 56 vel = 112)
        mode: 'mlp', 'encoder', or 'seq2seq'
        hidden_dim: Hidden dimension
        n_layers: Number of layers
        n_heads: Number of attention heads (for transformer modes)
        seq_len: Sequence length (for transformer modes)
        dropout: Dropout probability
    """
    
    MODES = Literal['mlp', 'encoder', 'seq2seq']
    
    # Legacy mode name mapping
    _LEGACY_MODES = {
        'transformer_reg': 'encoder',
        'transformer_ar': 'seq2seq',
    }
    
    def __init__(
        self,
        state_dim: int,
        mode: MODES = 'mlp',
        hidden_dim: int = 256,
        n_layers: int = 3,
        n_heads: int = 8,
        seq_len: int = 32,
        dropout: float = 0.1,
        use_modern: bool = True,  # Kept for backward compatibility
    ):
        super().__init__()
        
        # Handle legacy mode names
        mode = self._LEGACY_MODES.get(mode, mode)
        
        self.state_dim = state_dim
        self.mode = mode
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.seq_len = seq_len
        self.use_modern = use_modern
        
        if mode == 'mlp':
            self._build_mlp()
        elif mode in ('encoder', 'seq2seq'):
            self._build_transformer()
        else:
            raise ValueError(f"Unknown mode: {mode}. Use 'mlp', 'encoder', or 'seq2seq'")
        
        self._init_weights()
    
    def _build_mlp(self):
        """Build MLP: state → next_state."""
        layers = []
        dim = self.state_dim
        for _ in range(self.n_layers):
            layers.extend([
                nn.Linear(dim, self.hidden_dim),
                nn.LayerNorm(self.hidden_dim),
                nn.SiLU(),
            ])
            dim = self.hidden_dim
        layers.append(nn.Linear(self.hidden_dim, self.state_dim))
        self.net = nn.Sequential(*layers)
    
    def _build_transformer(self):
        """Build transformer encoder."""
        self.input_proj = nn.Linear(self.state_dim, self.hidden_dim)
        self.blocks = nn.ModuleList([
            TransformerBlock(
                self.hidden_dim, 
                self.n_heads, 
                self.dropout if hasattr(self, 'dropout') else 0.1,
                self.seq_len * 2
            )
            for _ in range(self.n_layers)
        ])
        self.norm = RMSNorm(self.hidden_dim)
        self.output_proj = nn.Linear(self.hidden_dim, self.state_dim)
    
    def _init_weights(self):
        """Initialize weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input state(s)
               - MLP: (batch, state_dim)
               - Transformer: (batch, seq_len, state_dim)
        
        Returns:
            - mlp: (batch, state_dim)
            - encoder: (batch, state_dim)
            - seq2seq: (batch, seq_len, state_dim)
        """
        x = x.float()
        if self.mode == 'mlp':
            return self.net(x)
        
        # Transformer modes
        h = self.input_proj(x)
        for block in self.blocks:
            h = block(h)
        h = self.norm(h)
        
        if self.mode == 'encoder':
            return self.output_proj(h[:, -1])  # Last position only
        else:  # seq2seq
            return self.output_proj(h)
    
    def rollout(self, state: torch.Tensor, steps: int) -> torch.Tensor:
        """
        Autoregressive rollout.
        
        Args:
            state: Initial state(s)
            steps: Number of steps to predict
        
        Returns:
            (batch, steps, state_dim) trajectory
        """
        preds = []
        current = state
        
        for _ in range(steps):
            pred = self.forward(current)
            
            if self.mode == 'mlp':
                preds.append(pred.unsqueeze(1))
                current = pred
            else:
                # Shift sequence
                if len(pred.shape) == 2:
                    pred = pred.unsqueeze(1)
                preds.append(pred if self.mode == 'seq2seq' else pred)
                if self.mode == 'encoder':
                    current = torch.cat([current[:, 1:], pred], dim=1)
                else:
                    preds[-1] = pred[:, -1:]
                    current = torch.cat([current[:, 1:], pred[:, -1:]], dim=1)
        
        return torch.cat(preds, dim=1)
    
    def save(self, path: str):
        """Save model checkpoint."""
        # Map new names back to legacy for compatibility
        legacy_mode = {
            'encoder': 'transformer_reg',
            'seq2seq': 'transformer_ar',
        }.get(self.mode, self.mode)
        
        torch.save({
            'state_dict': self.state_dict(),
            'state_dim': self.state_dim,
            'mode': legacy_mode,  # Save as legacy name for compatibility
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
        ckpt = torch.load(path, map_location=device, weights_only=False)
        model = cls(
            state_dim=ckpt['state_dim'],
            mode=ckpt['mode'],  # Will be auto-mapped if legacy
            hidden_dim=ckpt['hidden_dim'],
            n_layers=ckpt.get('n_layers', 3),
            n_heads=ckpt.get('n_heads', 4),
            seq_len=ckpt.get('seq_len', 32),
            use_modern=ckpt.get('use_modern', True),
        )
        model.load_state_dict(ckpt['state_dict'])
        print(f"Loaded MotionNN from {path}")
        return model


# =============================================================================
# Evaluation Helper
# =============================================================================

def evaluate_rollout(
    model: MotionNN,
    dataloader: torch.utils.data.DataLoader,
    steps: list[int] = [5, 10, 30],
    device: torch.device = torch.device('cpu'),
) -> dict[int, float]:
    """
    Evaluate multi-step rollout error.
    
    Returns:
        Dict mapping step count to MSE
    """
    model.eval()
    errors = {s: [] for s in steps}
    
    with torch.no_grad():
        for x, _ in dataloader:
            x = x.to(device)
            preds = model.rollout(x, max(steps))
            
            for s in steps:
                if s <= preds.size(1):
                    errors[s].append(preds[:, s-1].pow(2).mean().item())
    
    return {s: np.mean(e) if e else 0.0 for s, e in errors.items()}
