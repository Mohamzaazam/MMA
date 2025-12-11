#!/usr/bin/env python3
"""
Python Utils Package for Motion Capture Processing.

Core Modules:
- normalization: StateNormalizer for mean/std normalization
- datasets: PyTorch datasets (BVHDataset, ExtractedBVHDataset)
- splits: Data splitting utilities (subject_disjoint_activity_split)
- state_extractor: BVHStateExtractor for C++ state extraction

Rarely-Used Tools:
- scrape_mocap_metadata: CMU mocap web scraper
- truncate_mocap: AMC file truncation
- verify_conversion: BVH conversion verification
"""

# Core exports
from .normalization import StateNormalizer
from .datasets import BVHDataset, ExtractedBVHDataset, npz_from_bvh_path
from .splits import (
    subject_disjoint_activity_split,
    print_subject_disjoint_split_info,
    load_trial_info,
    infer_activity,
    get_activity_files,
)
from .state_extractor import BVHStateExtractor

__all__ = [
    # Normalization
    'StateNormalizer',
    # Datasets
    'BVHDataset',
    'ExtractedBVHDataset', 
    'npz_from_bvh_path',
    # Splits
    'subject_disjoint_activity_split',
    'print_subject_disjoint_split_info',
    'load_trial_info',
    'infer_activity',
    'get_activity_files',
    # State extraction
    'BVHStateExtractor',
]
