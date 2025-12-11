#!/usr/bin/env python3
"""
State Normalization Utilities.

Provides mean/std normalization with save/load support for inference.
"""

import numpy as np
from typing import Optional


class StateNormalizer:
    """
    Normalize states using (x - mean) / std.
    
    Supports save/load for inference.
    """
    
    def __init__(self, mean: Optional[np.ndarray] = None, std: Optional[np.ndarray] = None, eps: float = 1e-8):
        self.mean = mean
        self.std = std
        self.eps = eps
        self._fitted = mean is not None and std is not None
    
    def fit(self, data: np.ndarray) -> 'StateNormalizer':
        """Compute mean and std from data."""
        self.mean = np.mean(data, axis=0)
        self.std = np.std(data, axis=0)
        self.std = np.where(self.std < self.eps, 1.0, self.std)
        self._fitted = True
        return self
    
    def normalize(self, data: np.ndarray) -> np.ndarray:
        """Normalize: (x - mean) / std."""
        if not self._fitted:
            raise RuntimeError("Normalizer not fitted. Call fit() first.")
        return (data - self.mean) / self.std
    
    def denormalize(self, data: np.ndarray) -> np.ndarray:
        """Inverse: x * std + mean."""
        if not self._fitted:
            raise RuntimeError("Normalizer not fitted. Call fit() first.")
        return data * self.std + self.mean
    
    def save(self, filepath: str):
        """Save normalization parameters."""
        np.savez(filepath, mean=self.mean, std=self.std, eps=self.eps)
    
    @classmethod
    def load(cls, filepath: str) -> 'StateNormalizer':
        """Load normalization parameters."""
        data = np.load(filepath)
        return cls(mean=data['mean'], std=data['std'], eps=float(data['eps']))
    
    def __repr__(self) -> str:
        if self._fitted:
            return f"StateNormalizer(dim={len(self.mean)}, fitted=True)"
        return "StateNormalizer(fitted=False)"
