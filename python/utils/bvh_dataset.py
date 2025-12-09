#!/usr/bin/env python3
"""
BVH Dataset - PyTorch Dataset for motion prediction training.

This module provides:
- StateNormalizer: Mean/std normalization with save/load
- BVHDataset: In-memory dataset for small datasets
- LazyBVHDataset: Memory-efficient dataset loading from pre-extracted files
- Pre-extraction utilities for large datasets
- Train/val split utilities (by file or by subject)
"""

import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union
import re

from .state_extractor import BVHStateExtractor

# Optional PyTorch import
try:
    import torch
    from torch.utils.data import Dataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    Dataset = object


# =============================================================================
# NORMALIZATION
# =============================================================================

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
        return (data - self.mean) / (self.std + self.eps)
    
    def denormalize(self, data: np.ndarray) -> np.ndarray:
        """Inverse: x * std + mean."""
        if not self._fitted:
            raise RuntimeError("Normalizer not fitted. Call fit() first.")
        return data * (self.std + self.eps) + self.mean
    
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


# =============================================================================
# DATASET
# =============================================================================

class BVHDataset(Dataset if TORCH_AVAILABLE else object):
    """
    PyTorch Dataset for BVH motion data.
    
    Modes:
    - 'mlp': Returns (state_t, state_t+1) pairs
    - 'transformer': Returns (seq_t, seq_t+1) sequences
    - 'autoregressive': Returns (seq, target) for next-frame prediction
    """
    
    def __init__(
        self,
        bvh_files: List[str],
        metadata_template: str = "data/metadata.txt",
        build_dir: str = "build",
        mode: str = "mlp",
        seq_len: int = 32,
        include_phase: bool = False,
        normalize: bool = True,
        normalizer: Optional[StateNormalizer] = None,
        step_frames: int = 1,
    ):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required for BVHDataset")
        
        super().__init__()
        self.bvh_files = bvh_files
        self.mode = mode
        self.seq_len = seq_len
        self.include_phase = include_phase
        self.normalize = normalize
        self.step_frames = step_frames
        self.build_dir = build_dir
        
        # Parse metadata template
        self._skeleton_file = "data/human.xml"
        self._muscle_file = "data/muscle284.xml"
        self._parse_metadata_template(metadata_template)
        
        # Load states from all files
        self.all_states: List[np.ndarray] = []
        self.file_boundaries: List[int] = [0]
        self._load_all_bvh_files()
        
        # Concatenate states
        self.states = np.concatenate(self.all_states, axis=0)
        
        # Normalize
        if normalize:
            self.normalizer = normalizer if normalizer else StateNormalizer().fit(self.states)
            self.states = self.normalizer.normalize(self.states)
        else:
            self.normalizer = None
        
        # Build sampling indices
        self._build_indices()
        
        # Convert to tensors
        self.states_tensor = torch.from_numpy(self.states).float()
    
    def _parse_metadata_template(self, template_path: str):
        try:
            with open(template_path, 'r') as f:
                for line in f:
                    if line.startswith('skel_file'):
                        self._skeleton_file = line.split()[1].lstrip('/')
                    elif line.startswith('muscle_file'):
                        self._muscle_file = line.split()[1].lstrip('/')
        except FileNotFoundError:
            pass
    
    def _load_all_bvh_files(self):
        import tempfile
        import os
        
        # Get MASS_ROOT for relative paths
        mass_root = os.environ.get('MASS_ROOT', os.getcwd())
        
        for bvh_file in self.bvh_files:
            # Convert to path relative to MASS_ROOT
            bvh_path = Path(bvh_file)
            if bvh_path.is_absolute():
                try:
                    bvh_rel = bvh_path.relative_to(mass_root)
                except ValueError:
                    bvh_rel = bvh_path  # Use as-is if can't make relative
            else:
                bvh_rel = bvh_path
            
            metadata_content = f"""use_muscle true
con_hz 30
sim_hz 600
skel_file /{self._skeleton_file}
muscle_file /{self._muscle_file}
bvh_file /{bvh_rel} false
reward_param 0.75 0.1 0.0 0.15
"""
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
                f.write(metadata_content)
                temp_path = f.name
            
            try:
                extractor = BVHStateExtractor(temp_path, self.build_dir)
                states = extractor.extract_all_states(
                    include_phase=self.include_phase, step_frames=self.step_frames
                )
                self.all_states.append(states)
                self.file_boundaries.append(self.file_boundaries[-1] + len(states))
            except Exception as e:
                print(f"Warning: Failed to load {bvh_file}: {e}")
    
    def _build_indices(self):
        self.indices = []
        
        if self.mode == 'mlp':
            for i in range(len(self.file_boundaries) - 1):
                start, end = self.file_boundaries[i], self.file_boundaries[i + 1]
                self.indices.extend(range(start, end - 1))
        
        elif self.mode in ('transformer', 'autoregressive'):
            for i in range(len(self.file_boundaries) - 1):
                start, end = self.file_boundaries[i], self.file_boundaries[i + 1]
                self.indices.extend(range(start, end - self.seq_len - 1))
    
    def __len__(self) -> int:
        return len(self.indices)
    
    def __getitem__(self, idx: int):
        start = self.indices[idx]
        
        if self.mode == 'mlp':
            return self.states_tensor[start], self.states_tensor[start + 1]
        elif self.mode == 'transformer':
            return (self.states_tensor[start:start + self.seq_len],
                    self.states_tensor[start + 1:start + self.seq_len + 1])
        elif self.mode == 'autoregressive':
            return (self.states_tensor[start:start + self.seq_len],
                    self.states_tensor[start + self.seq_len])
        raise ValueError(f"Unknown mode: {self.mode}")
    
    @property
    def state_dim(self) -> int:
        return self.states.shape[1]
    
    @property
    def num_files(self) -> int:
        return len(self.bvh_files)
    
    @property
    def total_frames(self) -> int:
        return len(self.states)
    
    def save_normalizer(self, filepath: str):
        if self.normalizer:
            self.normalizer.save(filepath)
    
    def get_info(self) -> Dict:
        return {
            'num_files': self.num_files,
            'total_frames': self.total_frames,
            'num_samples': len(self),
            'state_dim': self.state_dim,
            'mode': self.mode,
            'seq_len': self.seq_len if self.mode != 'mlp' else None,
            'normalized': self.normalize,
        }


# =============================================================================
# TRAIN/VAL SPLIT
# =============================================================================

def train_val_split(
    bvh_dir: str,
    pattern: str = "**/*.bvh",
    train_ratio: float = 0.8,
    seed: int = 42,
    **dataset_kwargs
) -> Tuple[BVHDataset, BVHDataset]:
    """
    Create train/val datasets from a directory of BVH files.
    
    Args:
        bvh_dir: Directory containing BVH files
        pattern: Glob pattern for finding files
        train_ratio: Ratio for training (rest for validation)
        seed: Random seed for reproducibility
        **dataset_kwargs: Additional args for BVHDataset
        
    Returns:
        (train_dataset, val_dataset) with shared normalizer
    """
    import random
    
    # Find all BVH files
    bvh_files = sorted([str(p) for p in Path(bvh_dir).glob(pattern)])
    if not bvh_files:
        raise ValueError(f"No BVH files found in {bvh_dir}")
    
    # Shuffle with seed
    random.seed(seed)
    random.shuffle(bvh_files)
    
    # Split
    split_idx = int(len(bvh_files) * train_ratio)
    train_files = bvh_files[:split_idx]
    val_files = bvh_files[split_idx:]
    
    print(f"Train/val split: {len(train_files)} train, {len(val_files)} val files")
    
    # Create datasets
    train_dataset = BVHDataset(train_files, **dataset_kwargs)
    
    # Val uses train normalizer
    val_kwargs = dataset_kwargs.copy()
    val_kwargs['normalizer'] = train_dataset.normalizer
    val_dataset = BVHDataset(val_files, **val_kwargs)
    
    return train_dataset, val_dataset


def subject_split(
    bvh_dir: str,
    pattern: str = "**/*.bvh",
    train_subjects: Optional[List[int]] = None,
    val_subjects: Optional[List[int]] = None,
    train_ratio: float = 0.8,
    seed: int = 42,
    **dataset_kwargs
) -> Tuple[BVHDataset, BVHDataset]:
    """
    Create train/val datasets split by subject number.
    
    This ensures train and val sets have completely different subjects,
    which is better for testing generalization to new subjects.
    
    Subject IDs are extracted from filenames (e.g., '01_01.bvh' -> subject 1).
    
    Args:
        bvh_dir: Directory containing BVH files
        pattern: Glob pattern for finding files
        train_subjects: List of subject IDs for training (optional)
        val_subjects: List of subject IDs for validation (optional)
        train_ratio: If subjects not specified, ratio of subjects for training
        seed: Random seed for subject shuffling
        **dataset_kwargs: Additional args for BVHDataset
        
    Returns:
        (train_dataset, val_dataset) with shared normalizer
    """
    import random
    import re
    
    # Find all BVH files
    bvh_files = sorted([str(p) for p in Path(bvh_dir).glob(pattern)])
    if not bvh_files:
        raise ValueError(f"No BVH files found in {bvh_dir}")
    
    # Group files by subject
    subject_files: Dict[int, List[str]] = {}
    for f in bvh_files:
        # Extract subject ID from filename (e.g., '01_01.bvh' -> 1)
        match = re.search(r'(\d+)_\d+\.bvh$', f)
        if match:
            subject_id = int(match.group(1))
            if subject_id not in subject_files:
                subject_files[subject_id] = []
            subject_files[subject_id].append(f)
    
    all_subjects = sorted(subject_files.keys())
    
    # Determine train/val subjects
    if train_subjects is None and val_subjects is None:
        # Auto-split subjects
        random.seed(seed)
        shuffled = all_subjects.copy()
        random.shuffle(shuffled)
        split_idx = int(len(shuffled) * train_ratio)
        train_subjects = shuffled[:split_idx]
        val_subjects = shuffled[split_idx:]
    elif train_subjects is not None and val_subjects is None:
        val_subjects = [s for s in all_subjects if s not in train_subjects]
    elif val_subjects is not None and train_subjects is None:
        train_subjects = [s for s in all_subjects if s not in val_subjects]
    
    # Collect files for each split
    train_files = []
    val_files = []
    for s in train_subjects:
        if s in subject_files:
            train_files.extend(subject_files[s])
    for s in val_subjects:
        if s in subject_files:
            val_files.extend(subject_files[s])
    
    print(f"Subject split: {len(train_subjects)} train subjects, {len(val_subjects)} val subjects")
    print(f"  Train subjects: {sorted(train_subjects)[:10]}{'...' if len(train_subjects) > 10 else ''}")
    print(f"  Val subjects: {sorted(val_subjects)[:10]}{'...' if len(val_subjects) > 10 else ''}")
    print(f"  Train files: {len(train_files)}, Val files: {len(val_files)}")
    
    # Create datasets
    train_dataset = BVHDataset(train_files, **dataset_kwargs)
    
    val_kwargs = dataset_kwargs.copy()
    val_kwargs['normalizer'] = train_dataset.normalizer
    val_dataset = BVHDataset(val_files, **val_kwargs)
    
    return train_dataset, val_dataset


def load_bvh_dataset(
    bvh_files: List[str],
    **dataset_kwargs
) -> BVHDataset:
    """Create a BVHDataset from a list of BVH files."""
    return BVHDataset(bvh_files, **dataset_kwargs)


# =============================================================================
# LARGE DATASET SUPPORT (Pre-extraction + Lazy Loading)
# =============================================================================

def extract_to_disk(
    bvh_files: List[str],
    output_dir: str,
    build_dir: str = "build",
    skeleton_file: str = "data/human.xml",
    muscle_file: str = "data/muscle284.xml",
    include_phase: bool = False,
    verbose: bool = True
) -> Tuple[List[str], StateNormalizer]:
    """
    Extract states from BVH files and save to disk.
    
    Use this for large datasets that don't fit in memory.
    Run once as preprocessing, then use LazyBVHDataset.
    
    Args:
        bvh_files: List of BVH file paths
        output_dir: Directory to save extracted .npz files
        build_dir: Directory containing pymss module
        skeleton_file: Path to skeleton XML
        muscle_file: Path to muscle XML
        include_phase: Whether to include phase in states
        verbose: Print progress
        
    Returns:
        Tuple of (list of output files, fitted normalizer)
    """
    import tempfile
    import os
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # First pass: extract all states and compute normalization
    all_states = []
    output_files = []
    mass_root = os.environ.get('MASS_ROOT', os.getcwd())
    
    for i, bvh_file in enumerate(bvh_files):
        if verbose and (i + 1) % 50 == 0:
            print(f"Extracting {i+1}/{len(bvh_files)}...")
        
        bvh_path = Path(bvh_file)
        if bvh_path.is_absolute():
            try:
                bvh_rel = bvh_path.relative_to(mass_root)
            except ValueError:
                bvh_rel = bvh_path
        else:
            bvh_rel = bvh_path
        
        metadata_content = f"""use_muscle true
con_hz 30
sim_hz 600
skel_file /{skeleton_file}
muscle_file /{muscle_file}
bvh_file /{bvh_rel} false
reward_param 0.75 0.1 0.0 0.15
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(metadata_content)
            temp_path = f.name
        
        try:
            extractor = BVHStateExtractor(temp_path, build_dir)
            states = extractor.extract_all_states(include_phase=include_phase)
            
            # Save individual file
            out_file = output_path / f"{bvh_path.stem}.npz"
            np.savez_compressed(out_file, states=states)
            output_files.append(str(out_file))
            
            # Collect for normalization
            all_states.append(states)
        except Exception as e:
            if verbose:
                print(f"Warning: Failed {bvh_file}: {e}")
    
    # Compute normalizer from all data
    combined = np.concatenate(all_states, axis=0)
    normalizer = StateNormalizer().fit(combined)
    
    # Save normalizer
    norm_path = output_path / "normalizer.npz"
    normalizer.save(str(norm_path))
    
    if verbose:
        print(f"Extracted {len(output_files)} files, {len(combined)} total frames")
        print(f"Saved normalizer to {norm_path}")
    
    return output_files, normalizer


class LazyBVHDataset(Dataset if TORCH_AVAILABLE else object):
    """
    Memory-efficient dataset loading from pre-extracted .npz files.
    
    Use extract_to_disk() first to preprocess, then this dataset
    loads data lazily from disk.
    """
    
    def __init__(
        self,
        npz_files: List[str],
        normalizer: Optional[StateNormalizer] = None,
        normalizer_path: Optional[str] = None,
        mode: str = "mlp",
        seq_len: int = 32,
    ):
        """
        Args:
            npz_files: List of pre-extracted .npz file paths
            normalizer: Pre-fitted normalizer
            normalizer_path: Path to saved normalizer .npz
            mode: 'mlp', 'transformer', or 'autoregressive'
            seq_len: Sequence length for transformer/autoregressive
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required")
        
        super().__init__()
        self.npz_files = npz_files
        self.mode = mode
        self.seq_len = seq_len
        
        # Load normalizer
        if normalizer is not None:
            self.normalizer = normalizer
        elif normalizer_path is not None:
            self.normalizer = StateNormalizer.load(normalizer_path)
        else:
            self.normalizer = None
        
        # Build index: (file_idx, frame_idx) for each sample
        self.indices = []
        self.file_frame_counts = []
        
        for file_idx, npz_file in enumerate(npz_files):
            data = np.load(npz_file)
            n_frames = data['states'].shape[0]
            self.file_frame_counts.append(n_frames)
            
            if mode == 'mlp':
                for frame_idx in range(n_frames - 1):
                    self.indices.append((file_idx, frame_idx))
            else:
                for frame_idx in range(n_frames - seq_len - 1):
                    self.indices.append((file_idx, frame_idx))
        
        # Cache for current file
        self._cache_file_idx = -1
        self._cache_states = None
    
    def _load_file(self, file_idx: int) -> np.ndarray:
        """Load and cache a file's states."""
        if file_idx != self._cache_file_idx:
            data = np.load(self.npz_files[file_idx])
            states = data['states']
            if self.normalizer is not None:
                states = self.normalizer.normalize(states)
            self._cache_states = torch.from_numpy(states).float()
            self._cache_file_idx = file_idx
        return self._cache_states
    
    def __len__(self) -> int:
        return len(self.indices)
    
    def __getitem__(self, idx: int):
        file_idx, frame_idx = self.indices[idx]
        states = self._load_file(file_idx)
        
        if self.mode == 'mlp':
            return states[frame_idx], states[frame_idx + 1]
        elif self.mode == 'transformer':
            return (states[frame_idx:frame_idx + self.seq_len],
                    states[frame_idx + 1:frame_idx + self.seq_len + 1])
        elif self.mode == 'autoregressive':
            return (states[frame_idx:frame_idx + self.seq_len],
                    states[frame_idx + self.seq_len])
    
    @property
    def state_dim(self) -> int:
        data = np.load(self.npz_files[0])
        return data['states'].shape[1]

