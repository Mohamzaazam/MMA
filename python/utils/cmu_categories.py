"""
CMU Mocap Activity Categories.

Dynamically loads activity categories from trials.txt files in CMU subject directories.
Source: http://mocap.cs.cmu.edu/
"""

import re
from pathlib import Path
from typing import Dict, List, Optional, Set
from collections import defaultdict


def load_subject_info(cmu_dir: str) -> Dict[int, Dict]:
    """
    Load subject information from trials.txt files.
    
    Returns:
        Dict mapping subject_id -> {
            'description': str,  # Main activity description
            'trials': Dict[str, str],  # trial_id -> trial description
            'category': str,  # Inferred category
        }
    """
    cmu_path = Path(cmu_dir)
    subjects = {}
    
    for subject_dir in cmu_path.iterdir():
        if not subject_dir.is_dir():
            continue
        
        try:
            subject_id = int(subject_dir.name)
        except ValueError:
            continue
        
        trials_file = subject_dir / 'trials.txt'
        if not trials_file.exists():
            # Fallback: infer from BVH filenames
            bvh_files = list(subject_dir.glob('*.bvh'))
            if bvh_files:
                subjects[subject_id] = {
                    'description': 'unknown',
                    'trials': {},
                    'category': 'general',
                }
            continue
        
        with open(trials_file, 'r') as f:
            content = f.read()
        
        # Parse header: # Subject XX: description
        header_match = re.search(r'# Subject \d+:\s*(.+)', content)
        description = header_match.group(1).strip() if header_match else 'unknown'
        
        # Parse trials
        trials = {}
        for match in re.finditer(r'(\d+_\d+):\s*(.+)', content):
            trial_id, trial_desc = match.groups()
            trials[trial_id] = trial_desc.strip()
        
        # Infer category from description
        category = infer_category(description)
        
        subjects[subject_id] = {
            'description': description,
            'trials': trials,
            'category': category,
        }
    
    return subjects


def infer_category(description: str) -> str:
    """Infer activity category from description text."""
    desc_lower = description.lower()
    
    # Keywords for each category
    categories = {
        'locomotion': ['walk', 'run', 'jog', 'step', 'gait', 'stride'],
        'dance': ['dance', 'salsa', 'charleston', 'ballet', 'indian dance'],
        'sports': ['basketball', 'soccer', 'football', 'golf', 'kick', 'throw', 'catch', 
                   'swimming', 'sports', 'athletic'],
        'acrobatics': ['jump', 'flip', 'cartwheel', 'acrobatic', 'gymnastic', 
                       'breakdance', 'hopscotch'],
        'playground': ['playground', 'climb', 'swing', 'hang'],
        'interaction': ['interaction', 'human interaction', 'communication', 
                        'two subjects', '2 subjects'],
        'pantomime': ['pantomime', 'animal behavior', 'nursery rhyme', 'recreation'],
        'everyday': ['everyday', 'behavior', 'expression', 'general', 'careful', 
                     'assorted', 'various'],
        'stylized': ['stylized', 'weird walk', 'martial art', 'action', 
                     'michael jackson', 'baby'],
        'terrain': ['terrain', 'slope', 'obstacle', 'uneven', 'stairs'],
    }
    
    for cat, keywords in categories.items():
        for kw in keywords:
            if kw in desc_lower:
                return cat
    
    return 'general'


def get_category_subjects(cmu_dir: str) -> Dict[str, List[int]]:
    """Get subjects grouped by activity category."""
    subjects = load_subject_info(cmu_dir)
    
    category_subjects = defaultdict(list)
    for subj_id, info in subjects.items():
        category_subjects[info['category']].append(subj_id)
    
    return dict(category_subjects)


def get_subject_category(subject_id: int, cmu_dir: str = "data/cmu") -> str:
    """Get activity category for a subject."""
    subjects = load_subject_info(cmu_dir)
    if subject_id in subjects:
        return subjects[subject_id]['category']
    return 'general'


def stratified_split(
    bvh_files: List[str],
    cmu_dir: str,
    train_ratio: float = 0.8,
    seed: int = 42,
    max_subjects: Optional[int] = None,
) -> tuple:
    """
    Stratified split ensuring activity category overlap between train and val.
    
    Returns:
        (train_files, val_files, split_info)
    """
    import random
    random.seed(seed)
    
    # Load subject info
    subjects_info = load_subject_info(cmu_dir)
    
    # Group files by subject
    subject_files = defaultdict(list)
    for f in bvh_files:
        match = re.search(r'/(\d+)/\d+_\d+\.bvh$', f) or re.search(r'(\d+)_\d+\.bvh$', f)
        if match:
            subject_id = int(match.group(1))
            subject_files[subject_id].append(f)
    
    # Group subjects by category
    category_subjects = defaultdict(list)
    for subj_id in subject_files.keys():
        if subj_id in subjects_info:
            cat = subjects_info[subj_id]['category']
        else:
            cat = 'general'
        category_subjects[cat].append(subj_id)
    
    # Limit subjects if specified (proportionally from each category)
    if max_subjects and max_subjects < len(subject_files):
        selected = []
        n_cats = len(category_subjects)
        per_cat = max(1, max_subjects // n_cats)
        
        for cat, subjs in category_subjects.items():
            random.shuffle(subjs)
            selected.extend(subjs[:per_cat])
        
        # Fill remaining
        remaining = max_subjects - len(selected)
        all_subjs = list(subject_files.keys())
        random.shuffle(all_subjs)
        for s in all_subjs:
            if s not in selected and remaining > 0:
                selected.append(s)
                remaining -= 1
        
        # Filter to selected
        subject_files = {s: subject_files[s] for s in selected if s in subject_files}
        
        # Rebuild category mapping
        category_subjects = defaultdict(list)
        for subj_id in subject_files.keys():
            cat = subjects_info.get(subj_id, {}).get('category', 'general')
            category_subjects[cat].append(subj_id)
    
    # Stratified split: from each category, split subjects
    train_subjects = []
    val_subjects = []
    
    for cat, subjs in category_subjects.items():
        if not subjs:
            continue
        random.shuffle(subjs)
        split_idx = max(1, int(len(subjs) * train_ratio))
        
        # Ensure at least one in val if possible
        if len(subjs) > 1 and split_idx == len(subjs):
            split_idx -= 1
        
        train_subjects.extend(subjs[:split_idx])
        val_subjects.extend(subjs[split_idx:])
    
    # Collect files
    train_files = []
    val_files = []
    for s in train_subjects:
        train_files.extend(subject_files.get(s, []))
    for s in val_subjects:
        val_files.extend(subject_files.get(s, []))
    
    # Build split info
    train_cats = set(subjects_info.get(s, {}).get('category', 'general') for s in train_subjects)
    val_cats = set(subjects_info.get(s, {}).get('category', 'general') for s in val_subjects)
    
    split_info = {
        'train_subjects': sorted(train_subjects),
        'val_subjects': sorted(val_subjects),
        'train_categories': sorted(train_cats),
        'val_categories': sorted(val_cats),
        'category_overlap': sorted(train_cats & val_cats),
        'category_distribution': {cat: len(subjs) for cat, subjs in category_subjects.items()},
    }
    
    return train_files, val_files, split_info


def print_split_info(split_info: dict):
    """Print stratified split information."""
    print(f"\n✓ Activity-stratified split:")
    print(f"  Categories: {list(split_info['category_distribution'].keys())}")
    for cat, count in split_info['category_distribution'].items():
        print(f"    {cat}: {count} subjects")
    print(f"  Train subjects ({len(split_info['train_subjects'])}): {split_info['train_subjects'][:10]}...")
    print(f"  Val subjects ({len(split_info['val_subjects'])}): {split_info['val_subjects']}")
    print(f"  Category overlap: {split_info['category_overlap']}")
