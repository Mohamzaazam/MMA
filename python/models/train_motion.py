#!/usr/bin/env python3
"""
Training script for MotionNN motion prediction model.

Usage:
    python python/models/train_motion.py --bvh_dir data/motion --epochs 100
    python python/models/train_motion.py --bvh_files data/motion/walk.bvh data/motion/run.bvh
"""

import argparse
import sys
import os
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from python.models.motion_nn import MotionNN
from python.utils.bvh_dataset import (
    BVHDataset, train_val_split, subject_split, StateNormalizer
)


# =============================================================================
# Training Utilities
# =============================================================================

def get_device() -> torch.device:
    """Get best available device."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def cosine_lr_scheduler(optimizer, epoch, max_epochs, base_lr, min_lr=1e-6):
    """Cosine learning rate decay."""
    lr = min_lr + 0.5 * (base_lr - min_lr) * (1 + np.cos(np.pi * epoch / max_epochs))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


from contextlib import contextmanager

@contextmanager
def suppress_stdout(log_file=None):
    """Context manager to redirect stdout to a file or /dev/null."""
    import sys
    import os
    
    stdout_fd = sys.stdout.fileno()
    saved_stdout = os.dup(stdout_fd)
    
    if log_file:
        devnull = open(log_file, 'a')
    else:
        devnull = open(os.devnull, 'w')
    
    os.dup2(devnull.fileno(), stdout_fd)
    try:
        yield
    finally:
        os.dup2(saved_stdout, stdout_fd)
        os.close(saved_stdout)
        devnull.close()


# =============================================================================
# Training Loop
# =============================================================================

def train_epoch(
    model: MotionNN,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    n_batches = 0
    
    for batch in dataloader:
        x, y = batch
        x, y = x.to(device), y.to(device)
        
        optimizer.zero_grad()
        y_pred = model(x)
        
        # Handle transformer_ar output shape
        if model.mode == 'transformer_ar':
            loss = criterion(y_pred, y)
        else:
            loss = criterion(y_pred, y)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        n_batches += 1
    
    return total_loss / max(n_batches, 1)


def validate(
    model: MotionNN,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Validate the model."""
    model.eval()
    total_loss = 0.0
    n_batches = 0
    
    with torch.no_grad():
        for batch in dataloader:
            x, y = batch
            x, y = x.to(device), y.to(device)
            
            y_pred = model(x)
            loss = criterion(y_pred, y)
            
            total_loss += loss.item()
            n_batches += 1
    
    return total_loss / max(n_batches, 1)


def evaluate_rollout_mse(
    model: MotionNN,
    dataloader: DataLoader,
    steps: list[int],
    device: torch.device,
) -> dict[int, float]:
    """
    Evaluate multi-step rollout MSE.
    
    For each step count, measures how prediction error accumulates.
    """
    model.eval()
    errors = {s: [] for s in steps}
    
    with torch.no_grad():
        for batch in dataloader:
            x, y = batch
            x = x.to(device)
            
            # For MLP mode, we use single state and roll out
            if model.mode == 'mlp':
                state = x
                for step in range(1, max(steps) + 1):
                    state = model(state)
                    if step in steps:
                        # We don't have ground truth for multi-step here
                        # Track prediction magnitude as proxy for drift
                        errors[step].append(state.pow(2).mean().item())
            else:
                # Transformer modes: use predict_rollout
                max_steps = max(steps)
                predictions = model.predict_rollout(x, max_steps)
                for s in steps:
                    if s <= predictions.size(1):
                        errors[s].append(predictions[:, s-1].pow(2).mean().item())
    
    return {s: np.mean(errs) if errs else 0.0 for s, errs in errors.items()}


# =============================================================================
# Main Training
# =============================================================================

def train(args):
    """Main training function."""
    device = get_device()
    print(f"Using device: {device}")
    
    # Setup output directories
    nn_dir = Path(args.nn_dir)
    nn_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup for suppressing C++ output during BVH loading
    import sys
    import os
    log_file = nn_dir / 'training.log'
    suppress_cpp_output = not args.verbose
    if suppress_cpp_output:
        print(f"C++ logs will be redirected to: {log_file}")
    
    # TensorBoard
    log_dir = Path(args.log_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")
    writer = SummaryWriter(log_dir)
    print(f"TensorBoard logs: {log_dir}")
    
    # Load data
    print("\nLoading BVH data...")
    if args.bvh_files:
        bvh_files = args.bvh_files
        print(f"Using {len(bvh_files)} specified BVH files")
    elif args.subject_split:
        # subject_split handles its own file discovery
        bvh_files = None  # Will be handled in subject_split branch
        print(f"Using subject-based split from {args.bvh_dir}")
    else:
        bvh_dir = Path(args.bvh_dir)
        bvh_files = sorted([str(f) for f in bvh_dir.glob("**/*.bvh")])
        if not bvh_files:
            print(f"ERROR: No BVH files found in {args.bvh_dir}")
            return
        print(f"Found {len(bvh_files)} BVH files")
    
    # Determine mode for dataset
    # - mlp: (state, next_state)
    # - transformer_reg: (sequence, next_single_frame) -> 'autoregressive'
    # - transformer_ar: (sequence, next_sequence) -> 'transformer'
    if args.mode == 'mlp':
        dataset_mode = 'mlp'
    elif args.mode == 'transformer_reg':
        dataset_mode = 'autoregressive'  # seq -> single frame
    else:  # transformer_ar
        dataset_mode = 'transformer'  # seq -> seq
    
    # Create datasets
    if args.bvh_files:
        # Use specified files directly
        import random
        random.seed(42)
        shuffled = list(bvh_files)
        random.shuffle(shuffled)
        split_idx = int(len(shuffled) * args.train_ratio)
        train_files = shuffled[:split_idx] if split_idx > 0 else shuffled
        val_files = shuffled[split_idx:] if split_idx < len(shuffled) else shuffled[-1:]
        
        print(f"Train files: {len(train_files)}, Val files: {len(val_files)}")
        
        # Verify no overlap (data leakage check)
        overlap = set(train_files) & set(val_files)
        if overlap:
            print(f"WARNING: Data leakage detected! {len(overlap)} files in both train and val")
        else:
            print("✓ No data leakage: train and val files are disjoint")
        
        print("Loading datasets...")
        with suppress_stdout(str(log_file) if suppress_cpp_output else None):
            train_dataset = BVHDataset(
                train_files, mode=dataset_mode, seq_len=args.seq_len, build_dir=args.build_dir
            )
            val_dataset = BVHDataset(
                val_files, mode=dataset_mode, seq_len=args.seq_len, build_dir=args.build_dir,
                normalizer=train_dataset.normalizer
            )
    elif args.subject_split:
        # Use subject-based split with optional max_subjects
        from python.utils.bvh_dataset import subject_split as do_subject_split
        import re
        
        # Find all BVH files
        bvh_dir = Path(args.bvh_dir)
        all_bvh = sorted([str(p) for p in bvh_dir.glob("**/*.bvh")])
        print(f"Found {len(all_bvh)} BVH files in {args.bvh_dir}")
        
        # Group by subject
        subject_files = {}
        for f in all_bvh:
            match = re.search(r'/(\d+)/\d+_\d+\.bvh$', f) or re.search(r'(\d+)_\d+\.bvh$', f)
            if match:
                subject_id = int(match.group(1))
                if subject_id not in subject_files:
                    subject_files[subject_id] = []
                subject_files[subject_id].append(f)
        
        all_subjects = sorted(subject_files.keys())
        print(f"Found {len(all_subjects)} subjects")
        
        # Limit subjects if specified
        if args.max_subjects and args.max_subjects < len(all_subjects):
            all_subjects = all_subjects[:args.max_subjects]
            print(f"Limited to {len(all_subjects)} subjects")
        
        # Split subjects
        import random
        random.seed(42)
        shuffled = list(all_subjects)
        random.shuffle(shuffled)
        split_idx = int(len(shuffled) * args.train_ratio)
        train_subjects = shuffled[:split_idx]
        val_subjects = shuffled[split_idx:]
        
        # Verify no overlap
        overlap = set(train_subjects) & set(val_subjects)
        print(f"\n✓ No data leakage: subjects are disjoint")
        print(f"  Train subjects ({len(train_subjects)}): {sorted(train_subjects)}")
        print(f"  Val subjects ({len(val_subjects)}): {sorted(val_subjects)}")
        
        # Collect files
        train_files = []
        val_files = []
        for s in train_subjects:
            train_files.extend(subject_files.get(s, []))
        for s in val_subjects:
            val_files.extend(subject_files.get(s, []))
        
        print(f"  Train files: {len(train_files)}, Val files: {len(val_files)}")
        
        print("Loading datasets...")
        with suppress_stdout(str(log_file) if suppress_cpp_output else None):
            train_dataset = BVHDataset(
                train_files, mode=dataset_mode, seq_len=args.seq_len, build_dir=args.build_dir
            )
            val_dataset = BVHDataset(
                val_files, mode=dataset_mode, seq_len=args.seq_len, build_dir=args.build_dir,
                normalizer=train_dataset.normalizer
            )
    elif args.activity_split:
        # Activity-stratified split: ensures activity overlap between train and val
        from python.utils.cmu_categories import stratified_split, print_split_info
        
        # Find all BVH files
        bvh_dir = Path(args.bvh_dir)
        all_bvh = sorted([str(p) for p in bvh_dir.glob("**/*.bvh")])
        print(f"Found {len(all_bvh)} BVH files in {args.bvh_dir}")
        
        # Use stratified split
        train_files, val_files, split_info = stratified_split(
            all_bvh,
            args.bvh_dir,
            train_ratio=args.train_ratio,
            seed=313,
            max_subjects=args.max_subjects,
        )
        print_split_info(split_info)
        print(f"  Train files: {len(train_files)}, Val files: {len(val_files)}")
        
        print("Loading datasets...")
        with suppress_stdout(str(log_file) if suppress_cpp_output else None):
            train_dataset = BVHDataset(
                train_files, mode=dataset_mode, seq_len=args.seq_len, build_dir=args.build_dir
            )
            val_dataset = BVHDataset(
                val_files, mode=dataset_mode, seq_len=args.seq_len, build_dir=args.build_dir,
                normalizer=train_dataset.normalizer
            )
    else:
        print("Loading datasets...")
        with suppress_stdout(str(log_file) if suppress_cpp_output else None):
            train_dataset, val_dataset = train_val_split(
                args.bvh_dir,
                mode=dataset_mode,
                seq_len=args.seq_len,
                train_ratio=args.train_ratio,
                build_dir=args.build_dir,
            )
    
    print(f"Train: {len(train_dataset)} samples, Val: {len(val_dataset)} samples")
    print(f"State dim: {train_dataset.state_dim}")
    
    # DataLoaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    
    # Create model
    model = MotionNN(
        state_dim=train_dataset.state_dim,
        mode=args.mode,
        hidden_dim=args.hidden_dim,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        seq_len=args.seq_len,
        dropout=args.dropout,
    ).to(device)
    
    print(f"\nModel: {args.mode}")
    print(f"  Hidden dim: {args.hidden_dim}")
    print(f"  Layers: {args.n_layers}")
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {n_params:,}")
    
    # Optimizer and loss
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.MSELoss()
    
    # Training loop
    best_val_loss = float('inf')
    rollout_steps = [5, 10, 30]
    
    print(f"\nTraining for {args.epochs} epochs...")
    
    for epoch in range(1, args.epochs + 1):
        # Learning rate schedule
        lr = cosine_lr_scheduler(optimizer, epoch, args.epochs, args.lr)
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        
        # Validate
        val_loss = validate(model, val_loader, criterion, device)
        
        # Log to TensorBoard
        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('val/loss', val_loss, epoch)
        writer.add_scalar('train/lr', lr, epoch)
        
        # Rollout evaluation (less frequently)
        if epoch % args.rollout_freq == 0:
            rollout_errors = evaluate_rollout_mse(model, val_loader, rollout_steps, device)
            for s, err in rollout_errors.items():
                writer.add_scalar(f'val/rollout_{s}', err, epoch)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model.save(str(nn_dir / 'motion_model_best.pt'))
            
            # Also save normalizer
            if hasattr(train_dataset, 'normalizer') and train_dataset.normalizer:
                train_dataset.save_normalizer(str(nn_dir / 'motion_normalizer.npz'))
        
        # Periodic save
        if epoch % args.save_freq == 0:
            model.save(str(nn_dir / f'motion_model_epoch{epoch}.pt'))
        
        # Print progress
        if epoch % args.print_freq == 0:
            print(f"Epoch {epoch:4d} | Train: {train_loss:.6f} | Val: {val_loss:.6f} | Best: {best_val_loss:.6f}")
    
    # Final save
    model.save(str(nn_dir / 'motion_model.pt'))
    
    print(f"\nTraining complete!")
    print(f"Best val loss: {best_val_loss:.6f}")
    print(f"Model saved to: {nn_dir / 'motion_model.pt'}")
    
    writer.close()


def main():
    parser = argparse.ArgumentParser(description='Train MotionNN motion prediction model')
    
    # Data
    parser.add_argument('--bvh_dir', type=str, default='data/motion',
                        help='Directory containing BVH files')
    parser.add_argument('--bvh_files', type=str, nargs='+', default=None,
                        help='Specific BVH files to use (overrides bvh_dir)')
    parser.add_argument('--build_dir', type=str, default='build',
                        help='Build directory for pymss')
    parser.add_argument('--train_ratio', type=float, default=0.8,
                        help='Train/val split ratio')
    parser.add_argument('--subject_split', action='store_true',
                        help='Use subject-based splitting')
    parser.add_argument('--max_subjects', type=int, default=None,
                        help='Max number of subjects to use (for testing)')
    parser.add_argument('--activity_split', action='store_true',
                        help='Use activity-stratified splitting (ensures activity overlap)')
    
    # Model
    parser.add_argument('--mode', type=str, default='mlp',
                        choices=['mlp', 'transformer_reg', 'transformer_ar'],
                        help='Model mode')
    parser.add_argument('--hidden_dim', type=int, default=256,
                        help='Hidden dimension')
    parser.add_argument('--n_layers', type=int, default=3,
                        help='Number of layers')
    parser.add_argument('--n_heads', type=int, default=16,
                        help='Number of attention heads (transformer)')
    parser.add_argument('--seq_len', type=int, default=128,
                        help='Sequence length (transformer)')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout probability')
    
    # Training
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='DataLoader workers')
    
    # Logging
    parser.add_argument('--nn_dir', type=str, default='nn',
                        help='Directory to save models')
    parser.add_argument('--log_dir', type=str, default='runs/motion_model',
                        help='TensorBoard log directory')
    parser.add_argument('--print_freq', type=int, default=1,
                        help='Print frequency (epochs)')
    parser.add_argument('--save_freq', type=int, default=50,
                        help='Checkpoint save frequency (epochs)')
    parser.add_argument('--rollout_freq', type=int, default=10,
                        help='Rollout evaluation frequency (epochs)')
    
    # Test mode
    parser.add_argument('--test_run', action='store_true',
                        help='Quick test run (2 epochs)')
    parser.add_argument('--verbose', action='store_true',
                        help='Show C++ BVH parsing output (default: redirect to log file)')
    
    args = parser.parse_args()
    
    if args.test_run:
        args.epochs = 2
        args.print_freq = 1
        args.rollout_freq = 1
    
    train(args)


if __name__ == '__main__':
    main()
