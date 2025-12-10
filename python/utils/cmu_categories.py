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
        bvh_files: List of BVH file paths
        cmu_dir: CMU data directory
        train_ratio: Ratio of subjects for training (default 0.7)
        val_ratio: Ratio of subjects for validation (default 0.15)
        test_ratio: Ratio of subjects for testing (default 0.15)
        seed: Random seed for reproducibility
        save_split: Path to save split JSON (optional)
        
    Returns:
        (train_files, val_files, test_files, split_info)
    """
    import random
    import json
    from datetime import datetime
    
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
    
    # Group files by subject
    subject_files: Dict[int, List[str]] = defaultdict(list)
    for f in bvh_files:
        if f in all_trials:
            subject_id = all_trials[f]['subject']
            subject_files[subject_id].append(f)
        else:
            # Try to extract subject from filename
            match = re.search(r'/(\d+)/\d+_\d+\.bvh$', f) or re.search(r'(\d+)_\d+\.bvh$', f)
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
    train_set = set(train_subjects)
    val_set = set(val_subjects)
    test_set = set(test_subjects)
    
    assert not (train_set & val_set), "Train/val subject overlap!"
    assert not (train_set & test_set), "Train/test subject overlap!"
    assert not (val_set & test_set), "Val/test subject overlap!"
    
    # Collect files for each split
    train_files = []
    val_files = []
    test_files = []
    
    for s in train_subjects:
        train_files.extend(subject_files[s])
    for s in val_subjects:
        val_files.extend(subject_files[s])
    for s in test_subjects:
        test_files.extend(subject_files[s])
    
    # Compute activity coverage per split
    def get_activities(files: List[str]) -> set:
        activities = set()
        for f in files:
            if f in all_trials:
                activities.add(all_trials[f]['activity'])
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
