#!/usr/bin/env python3
"""
Data Splitting Utilities for CMU Mocap Dataset.

Provides:
- subject_disjoint_activity_split: Split by subject with activity coverage reporting
- load_trial_info: Load trial descriptions from CMU trials.txt files
- infer_activity: Infer activity category from description
"""

import re
import json
import random
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from collections import defaultdict


def load_trial_info(cmu_dir: str) -> Dict[str, Dict]:
    """
    Load trial-level information from trials.txt files.
    
    Returns:
        Dict mapping file_path -> {
            'subject': int,
            'trial_id': str,
            'description': str,
            'activity': str,  # Inferred activity category
        }
    """
    cmu_path = Path(cmu_dir)
    trials = {}
    
    for subject_dir in cmu_path.iterdir():
        if not subject_dir.is_dir():
            continue
        
        try:
            subject_id = int(subject_dir.name)
        except ValueError:
            continue
        
        trials_file = subject_dir / 'trials.txt'
        trial_descriptions = {}
        
        if trials_file.exists():
            with open(trials_file, 'r') as f:
                content = f.read()
            
            for match in re.finditer(r'(\d+_\d+):\s*(.+)', content):
                trial_id, desc = match.groups()
                trial_descriptions[trial_id] = desc.strip()
        
        for bvh_file in subject_dir.glob('*.bvh'):
            trial_id = bvh_file.stem
            description = trial_descriptions.get(trial_id, 'unknown')
            activity = infer_activity(description)
            
            trials[str(bvh_file)] = {
                'subject': subject_id,
                'trial_id': trial_id,
                'description': description,
                'activity': activity,
            }
    
    return trials


def infer_activity(description: str) -> str:
    """Infer activity category from trial description."""
    desc_lower = description.lower()
    
    patterns = {
        'walk': ['walk', 'stroll', 'stride', 'step', 'march', 'pace'],
        'run': ['run', 'jog', 'sprint', 'dash'],
        'jump': ['jump', 'leap', 'hop', 'bound', 'hopscotch'],
        # 'dance': ['dance', 'salsa', 'charleston', 'ballet', 'sway', 'pirouette'],
        # 'basketball': ['basketball', 'dribble', 'shoot', 'layup'],
        # 'soccer': ['soccer', 'kick ball', 'kick soccer'],
        # 'golf': ['golf', 'swing club'],
        # 'climb': ['climb', 'hang', 'swing', 'playground', 'pull up'],
        # 'sit': ['sit', 'seated', 'chair'],
        # 'stand': ['stand', 'idle', 'rest'],
        # 'turn': ['turn', 'rotate', 'spin', 'pivot'],
        # 'bend': ['bend', 'lean', 'crouch', 'duck', 'bow'],
        # 'throw': ['throw', 'catch', 'toss'],
        # 'punch': ['punch', 'hit', 'strike', 'box'],
        # 'kick': ['kick'],
        # 'flip': ['flip', 'cartwheel', 'somersault', 'acrobat', 'tumble'],
        # 'swim': ['swim', 'stroke'],
        # 'gesture': ['gesture', 'point', 'wave', 'signal'],
        # 'talk': ['talk', 'convers', 'speak'],
        # 'carry': ['carry', 'lift', 'hold', 'suitcase', 'box'],
        # 'push': ['push', 'shove'],
        # 'pull': ['pull', 'drag'],
        # 'crawl': ['crawl', 'creep'],
    }
    
    for activity, keywords in patterns.items():
        for kw in keywords:
            if kw in desc_lower:
                return activity
    
    return 'general'


def get_activity_files(cmu_dir: str) -> Dict[str, List[str]]:
    """Group files by their inferred activity."""
    trials = load_trial_info(cmu_dir)
    
    activity_files = defaultdict(list)
    for filepath, info in trials.items():
        activity_files[info['activity']].append(filepath)
    
    return dict(activity_files)


def subject_disjoint_activity_split(
    bvh_files: List[str],
    cmu_dir: str,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    save_split: Optional[str] = None,
) -> Tuple[List[str], List[str], List[str], Dict]:
    """
    Split by subject first, then report activity coverage.
    
    Key guarantee: If subject 35 is in train, NONE of their trials 
    appear in val or test.
    
    Args:
        bvh_files: List of BVH/NPZ file paths
        cmu_dir: CMU data directory (for trial info lookup)
        train_ratio: Ratio of subjects for training (default 0.7)
        val_ratio: Ratio of subjects for validation (default 0.15)
        test_ratio: Ratio of subjects for testing (default 0.15)
        seed: Random seed for reproducibility
        save_split: Path to save split JSON (optional)
        
    Returns:
        (train_files, val_files, test_files, split_info)
    """
    random.seed(seed)
    
    # Validate ratios
    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 0.01:
        print(f"Warning: Ratios sum to {total_ratio}, normalizing...")
        train_ratio /= total_ratio
        val_ratio /= total_ratio
        test_ratio /= total_ratio
    
    # Load trial info
    all_trials = load_trial_info(cmu_dir)
    
    # Helper to convert NPZ path to BVH path for lookup
    def to_bvh_key(filepath: str) -> str:
        """Convert NPZ path to BVH path for trial info lookup."""
        match = re.search(r'(\d+)/(\d+_\d+)\.(bvh|npz)$', filepath)
        if match:
            subject, trial, _ = match.groups()
            return str(Path(cmu_dir) / subject / f"{trial}.bvh")
        return filepath
    
    # Group files by subject
    subject_files: Dict[int, List[str]] = defaultdict(list)
    for f in bvh_files:
        bvh_key = to_bvh_key(f)
        if bvh_key in all_trials:
            subject_id = all_trials[bvh_key]['subject']
            subject_files[subject_id].append(f)
        else:
            match = re.search(r'/(\d+)/\d+_\d+\.(bvh|npz)$', f) or re.search(r'(\d+)_\d+\.(bvh|npz)$', f)
            if match:
                subject_id = int(match.group(1))
                subject_files[subject_id].append(f)
    
    all_subjects = sorted(subject_files.keys())
    n_subjects = len(all_subjects)
    
    if n_subjects < 3:
        raise ValueError(f"Need at least 3 subjects for 3-way split, got {n_subjects}")
    
    # Shuffle and split subjects
    random.shuffle(all_subjects)
    
    n_train = max(1, int(n_subjects * train_ratio))
    n_val = max(1, int(n_subjects * val_ratio))
    n_test = n_subjects - n_train - n_val
    
    if n_test < 1:
        n_test = 1
        n_train = n_subjects - n_val - n_test
    
    train_subjects = all_subjects[:n_train]
    val_subjects = all_subjects[n_train:n_train + n_val]
    test_subjects = all_subjects[n_train + n_val:]
    
    # Verify no overlap
    assert not (set(train_subjects) & set(val_subjects)), "Train/val subject overlap!"
    assert not (set(train_subjects) & set(test_subjects)), "Train/test subject overlap!"
    assert not (set(val_subjects) & set(test_subjects)), "Val/test subject overlap!"
    
    # Collect files for each split
    train_files = [f for s in train_subjects for f in subject_files[s]]
    val_files = [f for s in val_subjects for f in subject_files[s]]
    test_files = [f for s in test_subjects for f in subject_files[s]]
    
    # Compute activity coverage per split
    def get_activities(files: List[str]) -> set:
        activities = set()
        for f in files:
            bvh_key = to_bvh_key(f)
            if bvh_key in all_trials:
                activities.add(all_trials[bvh_key]['activity'])
        return activities
    
    train_activities = get_activities(train_files)
    val_activities = get_activities(val_files)
    test_activities = get_activities(test_files)
    
    # Build split info
    split_info = {
        'train_subjects': sorted(train_subjects),
        'val_subjects': sorted(val_subjects),
        'test_subjects': sorted(test_subjects),
        'train_files': train_files,
        'val_files': val_files,
        'test_files': test_files,
        'n_train_files': len(train_files),
        'n_val_files': len(val_files),
        'n_test_files': len(test_files),
        'activity_coverage': {
            'train': sorted(train_activities),
            'val': sorted(val_activities),
            'test': sorted(test_activities),
        },
        'activity_overlap': {
            'train_val': sorted(train_activities & val_activities),
            'train_test': sorted(train_activities & test_activities),
            'all': sorted(train_activities & val_activities & test_activities),
        },
        'seed': seed,
        'timestamp': datetime.now().isoformat(),
    }
    
    # Save to JSON if requested
    if save_split:
        split_path = Path(save_split)
        split_path.parent.mkdir(parents=True, exist_ok=True)
        with open(split_path, 'w') as f:
            json.dump(split_info, f, indent=2)
        print(f"Saved split info to {save_split}")
    
    return train_files, val_files, test_files, split_info


def print_subject_disjoint_split_info(split_info: dict):
    """Print subject-disjoint split information."""
    print(f"\n✓ Subject-disjoint split (no subject overlap):")
    print(f"  Train: {len(split_info['train_subjects'])} subjects, {split_info['n_train_files']} files")
    print(f"  Val:   {len(split_info['val_subjects'])} subjects, {split_info['n_val_files']} files")
    print(f"  Test:  {len(split_info['test_subjects'])} subjects, {split_info['n_test_files']} files")
    print(f"  Activity coverage:")
    print(f"    Train: {split_info['activity_coverage']['train']}")
    print(f"    Val:   {split_info['activity_coverage']['val']}")
    print(f"    Test:  {split_info['activity_coverage']['test']}")
    print(f"  Activities in all splits: {split_info['activity_overlap']['all']}")
