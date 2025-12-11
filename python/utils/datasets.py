#!/usr/bin/env python3
"""
PyTorch Datasets for BVH Motion Data.

Provides:
- BVHDataset: In-memory dataset loading from BVH files (uses C++ extractor)
- ExtractedBVHDataset: Fast loading from pre-extracted NPZ files
"""

import numpy as np
from pathlib import Path
from typing import List, Dict, Optional

from .normalization import StateNormalizer
from .state_extractor import BVHStateExtractor

# Optional PyTorch import
try:
    import torch
    from torch.utils.data import Dataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    Dataset = object


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
        
        mass_root = os.environ.get('MASS_ROOT', os.getcwd())
        
        for bvh_file in self.bvh_files:
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


class ExtractedBVHDataset(Dataset if TORCH_AVAILABLE else object):
    """
    Memory-efficient dataset loading from pre-extracted NPZ files.
    
    Uses lazy loading with LRU cache - only keeps a subset of files in memory.
    This enables training on large datasets (100+ subjects) without memory issues.
    
    Use scripts/extract_states.py to extract BVH files first:
        pixi run python scripts/extract_states.py --bvh_dir data/cmu --output_dir data/extracted
    
    Then use this dataset for fast training:
        dataset = ExtractedBVHDataset(
            npz_files=['data/extracted/01/01_01.npz', ...],
            mode='mlp',
        )
    """
    
    def __init__(
        self,
        npz_files: List[str],
        mode: str = "mlp",
        seq_len: int = 32,
        max_horizon: int = 1,
        normalize: bool = True,
        normalizer: Optional[StateNormalizer] = None,
        cache_size: int = 50,
        preload: bool = True,  # Load all data into memory
    ):
        """
        Args:
            npz_files: List of NPZ file paths (from extract_states.py)
            mode: 'mlp', 'transformer', or 'autoregressive'
            seq_len: Sequence length for transformer modes
            max_horizon: Maximum lookahead frames (1=single-step, >1=multi-step)
            normalize: Whether to normalize states
            normalizer: Pre-fitted normalizer (for val/test sets)
            cache_size: Number of files to keep in LRU cache (if not preloading)
            preload: If True, load all data into memory upfront (faster, uses more RAM)
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required")
        
        super().__init__()
        self.npz_files = npz_files
        self.mode = mode
        self.seq_len = seq_len
        self.max_horizon = max_horizon
        self.normalize = normalize
        self.cache_size = cache_size
        self.preload = preload
        
        # Scan files to get frame counts (without loading full data)
        self._file_frame_counts = []
        self._file_boundaries = [0]
        self._state_dim = None
        
        for npz_file in npz_files:
            try:
                # Load only to get shape, then close
                data = np.load(npz_file)
                n_frames = data['states'].shape[0]
                if self._state_dim is None:
                    self._state_dim = data['states'].shape[1]
                self._file_frame_counts.append(n_frames)
                self._file_boundaries.append(self._file_boundaries[-1] + n_frames)
                data.close()
            except Exception as e:
                print(f"Warning: failed to scan {npz_file}: {e}")
                self._file_frame_counts.append(0)
                self._file_boundaries.append(self._file_boundaries[-1])
        
        if self._state_dim is None:
            raise ValueError("Failed to load any NPZ files")
        
        # Build valid sample indices: (file_idx, frame_idx_within_file)
        # Ensure enough frames for max_horizon lookahead
        self._samples = []
        if mode == 'mlp':
            for file_idx, n_frames in enumerate(self._file_frame_counts):
                # Need at least max_horizon frames after current
                for frame_idx in range(n_frames - max_horizon):
                    self._samples.append((file_idx, frame_idx))
        else:
            for file_idx, n_frames in enumerate(self._file_frame_counts):
                # Need seq_len + max_horizon frames
                if n_frames > seq_len + max_horizon:
                    for frame_idx in range(n_frames - seq_len - max_horizon):
                        self._samples.append((file_idx, frame_idx))
        
        # Fit normalizer from sample of files (not all data)
        if normalizer is not None:
            self.normalizer = normalizer
        elif normalize:
            self.normalizer = self._fit_normalizer_from_sample()
        else:
            self.normalizer = None
        
        # Preload all data into memory for fast access
        if preload:
            print(f"  Preloading {len(npz_files)} files into memory...")
            self._all_data: List[torch.Tensor] = []
            for i, npz_file in enumerate(npz_files):
                try:
                    data = np.load(npz_file)
                    states = data['states'].astype(np.float32)
                    data.close()
                    if self.normalize and self.normalizer is not None:
                        states = self.normalizer.normalize(states)
                    self._all_data.append(torch.from_numpy(states))
                except Exception:
                    self._all_data.append(torch.zeros(0, self._state_dim))
            total_frames = sum(t.shape[0] for t in self._all_data)
            mem_gb = total_frames * self._state_dim * 4 / 1e9
            print(f"  Loaded {total_frames:,} frames ({mem_gb:.2f} GB)")
            self._cache = {}
            self._cache_order = []
        else:
            self._all_data = None
            self._cache: Dict[int, torch.Tensor] = {}
            self._cache_order: List[int] = []

    
    def _fit_normalizer_from_sample(self, sample_size: int = 30) -> StateNormalizer:
        """Fit normalizer from a sample of files."""
        import random
        # Use fixed seed for reproducible normalization
        rng = random.Random(42)
        sample_indices = rng.sample(range(len(self.npz_files)), min(sample_size, len(self.npz_files)))
        
        all_states = []
        for idx in sample_indices:
            try:
                data = np.load(self.npz_files[idx])
                all_states.append(data['states'])
                data.close()
            except Exception as e:
                print(f"Warning: Failed to load {self.npz_files[idx]}: {e}")
        
        if not all_states:
            raise ValueError("Failed to load any files for normalization")
        
        combined = np.concatenate(all_states, axis=0)
        return StateNormalizer().fit(combined)
    
    def _load_file(self, file_idx: int) -> torch.Tensor:
        """Load file data (from preloaded memory or cache)."""
        # Use preloaded data if available
        if self._all_data is not None:
            return self._all_data[file_idx]
        
        # Otherwise use LRU cache
        if file_idx in self._cache:
            # Move to end of LRU order
            self._cache_order.remove(file_idx)
            self._cache_order.append(file_idx)
            return self._cache[file_idx]
        
        # Load the file
        data = np.load(self.npz_files[file_idx])
        states = data['states'].astype(np.float32)
        data.close()
        
        # Normalize if needed
        if self.normalize and self.normalizer is not None:
            states = self.normalizer.normalize(states)
        
        states_tensor = torch.from_numpy(states).float()
        
        # Evict oldest if cache is full
        while len(self._cache) >= self.cache_size and self._cache_order:
            oldest_idx = self._cache_order.pop(0)
            del self._cache[oldest_idx]
        
        # Add to cache
        self._cache[file_idx] = states_tensor
        self._cache_order.append(file_idx)
        
        return states_tensor
    
    def __len__(self) -> int:
        return len(self._samples)
    
    def __getitem__(self, idx: int):
        file_idx, frame_idx = self._samples[idx]
        states = self._load_file(file_idx)
        
        if self.mode == 'mlp':
            x = states[frame_idx]
            if self.max_horizon == 1:
                y = states[frame_idx + 1]
            else:
                # Multi-step: return (max_horizon, state_dim) tensor
                y = states[frame_idx + 1 : frame_idx + 1 + self.max_horizon]
            return x, y
        
        elif self.mode == 'transformer':
            seq = states[frame_idx : frame_idx + self.seq_len]
            if self.max_horizon == 1:
                target = states[frame_idx + 1 : frame_idx + self.seq_len + 1]
            else:
                # Multi-step: return (seq_len, max_horizon, state_dim) tensor
                targets = []
                for i in range(self.seq_len):
                    targets.append(states[frame_idx + i + 1 : frame_idx + i + 1 + self.max_horizon])
                target = torch.stack(targets, dim=0)
            return seq, target
        
        elif self.mode == 'autoregressive':
            seq = states[frame_idx : frame_idx + self.seq_len]
            if self.max_horizon == 1:
                target = states[frame_idx + self.seq_len]
            else:
                # Multi-step: return (max_horizon, state_dim) tensor
                target = states[frame_idx + self.seq_len : frame_idx + self.seq_len + self.max_horizon]
            return seq, target
        
        raise ValueError(f"Unknown mode: {self.mode}")
    
    @property
    def state_dim(self) -> int:
        return self._state_dim
    
    def save_normalizer(self, filepath: str):
        if self.normalizer:
            self.normalizer.save(filepath)
    
    @property
    def total_frames(self) -> int:
        return sum(self._file_frame_counts)


def npz_from_bvh_path(bvh_path: str, bvh_root: str, npz_root: str) -> str:
    """Convert a BVH file path to its corresponding NPZ path."""
    bvh_rel = Path(bvh_path).relative_to(bvh_root)
    return str(Path(npz_root) / bvh_rel.with_suffix('.npz'))
