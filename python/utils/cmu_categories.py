"""
CMU Mocap Activity Categories.

Dynamically loads activity categories from trials.txt files in CMU subject directories.
Supports trial-level activity splitting for balanced train/val sets.
"""

import re
from pathlib import Path
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
            
            # Parse trials: trial_id: description
            for match in re.finditer(r'(\d+_\d+):\s*(.+)', content):
                trial_id, desc = match.groups()
                trial_descriptions[trial_id] = desc.strip()
        
        # Process BVH files
        for bvh_file in subject_dir.glob('*.bvh'):
            trial_id = bvh_file.stem  # e.g., "16_01"
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
    
    # Priority order matters - check more specific patterns first
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


def activity_stratified_split(
    bvh_files: List[str],
    cmu_dir: str,
    train_ratio: float = 0.8,
    seed: int = 42,
    max_files: Optional[int] = None,
    min_activity_files: int = 2,
) -> Tuple[List[str], List[str], Dict]:
    """
    Split files by activity category, ensuring each activity
    is represented in both train and val sets.
    
    NOTE: This may have subject overlap between train/val.
    For no-subject-overlap, use subject_split() instead.
    
    Args:
        bvh_files: List of BVH file paths
        cmu_dir: CMU data directory
        train_ratio: Ratio for train split
        seed: Random seed
        max_files: Max files per activity (for testing)
        min_activity_files: Min files needed for an activity to be included
        
    Returns:
        (train_files, val_files, split_info)
    """
    import random
    random.seed(seed)
    
    # Load trial info
    all_trials = load_trial_info(cmu_dir)
    
    # Filter to requested files and group by activity
    activity_files = defaultdict(list)
    for f in bvh_files:
        if f in all_trials:
            activity = all_trials[f]['activity']
            activity_files[activity].append(f)
        else:
            activity_files['general'].append(f)
    
    # Filter activities with too few files
    valid_activities = {
        act: files for act, files in activity_files.items()
        if len(files) >= min_activity_files
    }
    
    print(f"Found {len(valid_activities)} activities with >= {min_activity_files} files")
    
    # Split each activity
    train_files = []
    val_files = []
    activity_distribution = {}
    
    for activity, files in sorted(valid_activities.items()):
        random.shuffle(files)
        
        # Limit files if requested
        if max_files:
            files = files[:max_files]
        
        # Split ensuring at least 1 in each set
        n = len(files)
        n_train = max(1, int(n * train_ratio))
        n_train = min(n_train, n - 1) if n > 1 else n  # Leave at least 1 for val
        
        train_files.extend(files[:n_train])
        val_files.extend(files[n_train:])
        
        activity_distribution[activity] = {
            'total': n,
            'train': n_train,
            'val': n - n_train,
        }
    
    # Build split info
    train_activities = set()
    val_activities = set()
    for f in train_files:
        if f in all_trials:
            train_activities.add(all_trials[f]['activity'])
    for f in val_files:
        if f in all_trials:
            val_activities.add(all_trials[f]['activity'])
    
    split_info = {
        'train_files': len(train_files),
        'val_files': len(val_files),
        'activities': list(activity_distribution.keys()),
        'activity_distribution': activity_distribution,
        'train_activities': sorted(train_activities),
        'val_activities': sorted(val_activities),
        'activity_overlap': sorted(train_activities & val_activities),
    }
    
    return train_files, val_files, split_info


def print_activity_split_info(split_info: dict):
    """Print activity split information."""
    print(f"\n✓ Activity-based split:")
    print(f"  Activities: {len(split_info['activities'])}")
    for act, dist in sorted(split_info['activity_distribution'].items()):
        print(f"    {act}: {dist['train']} train / {dist['val']} val")
    print(f"  Activity overlap: {len(split_info['activity_overlap'])}/{len(split_info['activities'])}")
    print(f"  Train: {split_info['train_files']} files, Val: {split_info['val_files']} files")


# Keep old functions for backward compatibility
def stratified_split(*args, **kwargs):
    """Alias for activity_stratified_split (backward compatibility)."""
    return activity_stratified_split(*args, **kwargs)

def print_split_info(split_info: dict):
    """Alias for print_activity_split_info (backward compatibility)."""
    return print_activity_split_info(split_info)
