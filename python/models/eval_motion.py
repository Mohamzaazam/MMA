"""
Evaluate MotionNN model: test predictions and plot input vs output.

Usage:
    pixi run python python/models/eval_motion.py --model nn/motion_model_best.pt --bvh data/motion/walk.bvh
"""

import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
import sys
sys.path.append('.')

from python.models.motion_nn import MotionNN
from python.utils.bvh_dataset import BVHDataset, StateNormalizer


def load_model_and_normalizer(model_path: str, normalizer_path: str = None):
    """Load trained model and normalizer."""
    model = MotionNN.load(model_path)
    model.eval()
    
    if normalizer_path is None:
        normalizer_path = Path(model_path).parent / 'motion_normalizer.npz'
    
    normalizer = StateNormalizer.load(str(normalizer_path))
    return model, normalizer


def evaluate_single_step(model, dataset, device, num_samples=100):
    """Evaluate single-step prediction accuracy."""
    model.eval()
    
    errors = []
    predictions = []
    targets = []
    
    indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)
    
    with torch.no_grad():
        for idx in indices:
            x, y = dataset[idx]
            x = x.unsqueeze(0).to(device)  # Add batch dim
            y = y.numpy() if isinstance(y, torch.Tensor) else y
            
            pred = model(x).cpu().numpy().squeeze()
            
            predictions.append(pred)
            targets.append(y)
            errors.append(np.mean((pred - y) ** 2))
    
    return {
        'predictions': np.array(predictions),
        'targets': np.array(targets),
        'mse': np.mean(errors),
        'std': np.std(errors),
    }


def evaluate_sequential(model, dataset, device, start_idx=0, num_frames=200):
    """
    Evaluate on sequential frames for trajectory visualization.
    
    This gives continuous trajectory data for proper time-series plotting.
    """
    model.eval()
    
    predictions = []
    targets = []
    
    # Ensure we don't go out of bounds
    end_idx = min(start_idx + num_frames, len(dataset))
    
    with torch.no_grad():
        for idx in range(start_idx, end_idx):
            x, y = dataset[idx]
            x = x.unsqueeze(0).to(device)
            y = y.numpy() if isinstance(y, torch.Tensor) else y
            
            pred = model(x).cpu().numpy().squeeze()
            
            predictions.append(pred)
            targets.append(y)
    
    return {
        'predictions': np.array(predictions),
        'targets': np.array(targets),
    }


def evaluate_rollout(model, dataset, device, rollout_steps=30, num_rollouts=10):
    """Evaluate multi-step rollout prediction."""
    model.eval()
    
    rollout_errors = []
    
    indices = np.random.choice(len(dataset), min(num_rollouts, len(dataset)), replace=False)
    
    with torch.no_grad():
        for idx in indices:
            x, y = dataset[idx]
            
            # Get initial state
            if len(x.shape) == 1:  # MLP mode
                state = x.unsqueeze(0).to(device)
            else:  # Transformer mode
                state = x.unsqueeze(0).to(device)
            
            # Rollout
            errors = []
            current = state
            for step in range(rollout_steps):
                pred = model(current)
                
                # For MLP, just use prediction as next input
                if model.mode == 'mlp':
                    current = pred
                else:
                    # For transformer, shift sequence and append prediction
                    current = torch.cat([current[:, 1:, :], pred.unsqueeze(1)], dim=1)
                
                # Calculate error (using prediction magnitude as proxy)
                errors.append(torch.mean(pred ** 2).item())
            
            rollout_errors.append(errors)
    
    return np.array(rollout_errors)


def plot_predictions(predictions, targets, normalizer, output_path=None, num_samples=5):
    """Plot input vs output comparison."""
    # Denormalize
    preds_denorm = normalizer.denormalize(predictions)
    targs_denorm = normalizer.denormalize(targets)
    
    state_dim = predictions.shape[1]
    
    # Plot first few state dimensions
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    # Plot 6 different state dimensions
    dims_to_plot = [0, 1, 2, 56, 57, 58]  # Root pos, root vel
    dim_names = ['Root X', 'Root Y', 'Root Z', 'Root VelX', 'Root VelY', 'Root VelZ']
    
    for ax, dim, name in zip(axes, dims_to_plot, dim_names):
        if dim < state_dim:
            ax.scatter(targs_denorm[:num_samples, dim], preds_denorm[:num_samples, dim], 
                      alpha=0.6, label='Predictions')
            
            # Perfect prediction line
            min_val = min(targs_denorm[:num_samples, dim].min(), preds_denorm[:num_samples, dim].min())
            max_val = max(targs_denorm[:num_samples, dim].max(), preds_denorm[:num_samples, dim].max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect')
            
            ax.set_xlabel('Ground Truth')
            ax.set_ylabel('Prediction')
            ax.set_title(f'{name} (dim {dim})')
            ax.legend()
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved prediction plot to {output_path}")
    
    plt.show()
    return fig


def plot_trajectory(predictions, targets, normalizer, output_path=None):
    """Plot trajectory comparison over time."""
    # Denormalize
    preds_denorm = normalizer.denormalize(predictions)
    targs_denorm = normalizer.denormalize(targets)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Time series of first few samples
    n_samples = min(50, len(predictions))
    
    # Plot root position components
    for i, (ax, dim, name) in enumerate(zip(
        axes.flatten(),
        [0, 1, 2, 56],
        ['Root X', 'Root Y', 'Root Z', 'Root VelX']
    )):
        ax.plot(targs_denorm[:n_samples, dim], 'b-', label='Ground Truth', linewidth=2)
        ax.plot(preds_denorm[:n_samples, dim], 'r--', label='Prediction', linewidth=2)
        ax.set_xlabel('Sample')
        ax.set_ylabel('Value')
        ax.set_title(f'{name} Over Time')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved trajectory plot to {output_path}")
    
    plt.show()
    return fig


def plot_error_distribution(predictions, targets, output_path=None):
    """Plot error distribution."""
    errors = predictions - targets
    mse_per_sample = np.mean(errors ** 2, axis=1)
    mse_per_dim = np.mean(errors ** 2, axis=0)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Error per sample
    axes[0].hist(mse_per_sample, bins=50, edgecolor='black', alpha=0.7)
    axes[0].axvline(np.mean(mse_per_sample), color='r', linestyle='--', 
                    label=f'Mean: {np.mean(mse_per_sample):.4f}')
    axes[0].set_xlabel('MSE')
    axes[0].set_ylabel('Count')
    axes[0].set_title('MSE Distribution (per sample)')
    axes[0].legend()
    
    # Error per dimension
    axes[1].bar(range(len(mse_per_dim)), mse_per_dim, alpha=0.7)
    axes[1].set_xlabel('State Dimension')
    axes[1].set_ylabel('MSE')
    axes[1].set_title('MSE by State Dimension')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved error plot to {output_path}")
    
    plt.show()
    return fig


# Lower limb joint angle indices for the CMU/DART skeleton
# State layout: [pos (3) + joint_angles (n_joints * 3) + vel (...)]
# These are approximate indices - adjust based on your skeleton
LOWER_LIMB_JOINTS = {
    # Right leg
    'RHip': {'dims': [9, 10, 11], 'label': 'Right Hip'},
    'RKnee': {'dims': [12, 13, 14], 'label': 'Right Knee'},
    'RAnkle': {'dims': [15, 16, 17], 'label': 'Right Ankle'},
    # Left leg
    'LHip': {'dims': [18, 19, 20], 'label': 'Left Hip'},
    'LKnee': {'dims': [21, 22, 23], 'label': 'Left Knee'},
    'LAnkle': {'dims': [24, 25, 26], 'label': 'Left Ankle'},
}


def plot_joint_angles(predictions, targets, normalizer, output_path=None, joints=None):
    """
    Plot lower limb joint angles: ground truth vs prediction.
    
    Args:
        predictions: Predicted states (N, state_dim)
        targets: Ground truth states (N, state_dim)
        normalizer: StateNormalizer for denormalization
        output_path: Path to save figure
        joints: Dict of joint definitions, defaults to LOWER_LIMB_JOINTS
    """
    # Denormalize
    preds_denorm = normalizer.denormalize(predictions)
    targs_denorm = normalizer.denormalize(targets)
    
    state_dim = predictions.shape[1]
    joints = joints or LOWER_LIMB_JOINTS
    
    # Filter joints that fit in state dim
    valid_joints = {k: v for k, v in joints.items() 
                    if max(v['dims']) < state_dim}
    
    if not valid_joints:
        print(f"Warning: No valid joint indices for state_dim={state_dim}")
        # Fallback: use generic indices
        valid_joints = {
            f'Joint_{i}': {'dims': [i*3, i*3+1, i*3+2], 'label': f'Joint {i}'}
            for i in range(3, min(9, state_dim // 3))
        }
    
    n_joints = len(valid_joints)
    n_samples = min(100, len(predictions))
    
    fig, axes = plt.subplots(n_joints, 3, figsize=(15, 3 * n_joints))
    if n_joints == 1:
        axes = axes.reshape(1, -1)
    
    axis_labels = ['X (flexion)', 'Y (abduction)', 'Z (rotation)']
    
    for row, (joint_name, joint_info) in enumerate(valid_joints.items()):
        dims = joint_info['dims']
        label = joint_info['label']
        
        for col, (dim, axis_label) in enumerate(zip(dims, axis_labels)):
            if dim < state_dim:
                ax = axes[row, col]
                
                # Time series comparison
                gt = targs_denorm[:n_samples, dim]
                pred = preds_denorm[:n_samples, dim]
                
                ax.plot(gt, 'b-', label='Ground Truth', linewidth=1.5, alpha=0.8)
                ax.plot(pred, 'r--', label='Prediction', linewidth=1.5, alpha=0.8)
                
                # Error shading
                ax.fill_between(range(len(gt)), gt, pred, alpha=0.2, color='gray')
                
                ax.set_xlabel('Frame')
                ax.set_ylabel('Angle (rad)')
                ax.set_title(f'{label} - {axis_label}')
                ax.legend(loc='upper right', fontsize=8)
                ax.grid(True, alpha=0.3)
                
                # Add error metric
                mse = np.mean((gt - pred) ** 2)
                ax.text(0.02, 0.98, f'MSE: {mse:.4f}', transform=ax.transAxes,
                       fontsize=8, verticalalignment='top', 
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle('Lower Limb Joint Angles: Ground Truth vs Prediction', fontsize=14, y=1.02)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved joint angle plot to {output_path}")
    
    plt.show()
    return fig


def plot_joint_scatter(predictions, targets, normalizer, output_path=None, joints=None):
    """
    Scatter plot of joint angle predictions vs ground truth.
    """
    preds_denorm = normalizer.denormalize(predictions)
    targs_denorm = normalizer.denormalize(targets)
    
    state_dim = predictions.shape[1]
    joints = joints or LOWER_LIMB_JOINTS
    
    valid_joints = {k: v for k, v in joints.items() 
                    if max(v['dims']) < state_dim}
    
    if not valid_joints:
        valid_joints = {
            f'Joint_{i}': {'dims': [i*3, i*3+1, i*3+2], 'label': f'Joint {i}'}
            for i in range(3, min(9, state_dim // 3))
        }
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    joint_items = list(valid_joints.items())[:6]  # Max 6 joints
    
    for ax, (joint_name, joint_info) in zip(axes, joint_items):
        dims = joint_info['dims']
        label = joint_info['label']
        
        # Average across XYZ for this joint
        gt_mean = np.mean(targs_denorm[:, dims], axis=1)
        pred_mean = np.mean(preds_denorm[:, dims], axis=1)
        
        ax.scatter(gt_mean, pred_mean, alpha=0.3, s=10)
        
        # Perfect line
        min_v, max_v = min(gt_mean.min(), pred_mean.min()), max(gt_mean.max(), pred_mean.max())
        ax.plot([min_v, max_v], [min_v, max_v], 'r--', label='Perfect', linewidth=2)
        
        ax.set_xlabel('Ground Truth (rad)')
        ax.set_ylabel('Prediction (rad)')
        ax.set_title(label)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Correlation
        corr = np.corrcoef(gt_mean, pred_mean)[0, 1]
        ax.text(0.05, 0.95, f'r = {corr:.3f}', transform=ax.transAxes,
               fontsize=10, verticalalignment='top')
    
    plt.suptitle('Joint Angle Prediction Accuracy', fontsize=14)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved joint scatter plot to {output_path}")
    
    plt.show()
    return fig


def main():
    parser = argparse.ArgumentParser(description='Evaluate MotionNN model')
    parser.add_argument('--model', type=str, default='nn/motion_model_best.pt',
                        help='Path to trained model')
    parser.add_argument('--normalizer', type=str, default=None,
                        help='Path to normalizer (default: same dir as model)')
    parser.add_argument('--bvh', type=str, nargs='+', default=None,
                        help='BVH files to evaluate on')
    parser.add_argument('--bvh_dir', type=str, default='data/motion',
                        help='Directory containing BVH files')
    parser.add_argument('--build_dir', type=str, default='build',
                        help='Build directory')
    parser.add_argument('--output_dir', type=str, default='eval_results',
                        help='Output directory for plots')
    parser.add_argument('--num_samples', type=int, default=500,
                        help='Number of samples to evaluate')
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    print(f"\nLoading model from {args.model}")
    model, normalizer = load_model_and_normalizer(args.model, args.normalizer)
    model = model.to(device)
    print(f"Model mode: {model.mode}")
    print(f"State dim: {model.state_dim}")
    print(f"Hidden dim: {model.hidden_dim}")
    print(f"Seq len: {model.seq_len}")
    if hasattr(model, 'use_modern'):
        print(f"Modern (RMSNorm/SwiGLU/RoPE): {model.use_modern}")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")
    
    # Determine dataset mode
    if model.mode == 'mlp':
        dataset_mode = 'mlp'
    elif model.mode == 'transformer_reg':
        dataset_mode = 'autoregressive'
    else:
        dataset_mode = 'transformer'
    
    # Load dataset
    if args.bvh:
        bvh_files = args.bvh
    else:
        bvh_files = list(Path(args.bvh_dir).glob('**/*.bvh'))
        bvh_files = [str(f) for f in bvh_files[:10]]  # Limit for eval
    
    print(f"\nLoading {len(bvh_files)} BVH files...")
    dataset = BVHDataset(
        bvh_files, 
        mode=dataset_mode, 
        seq_len=model.seq_len if hasattr(model, 'seq_len') else 32,
        build_dir=args.build_dir,
        normalizer=normalizer
    )
    print(f"Dataset: {len(dataset)} samples")
    
    # Evaluate
    print("\n" + "="*50)
    print("Evaluating single-step prediction...")
    results = evaluate_single_step(model, dataset, device, args.num_samples)
    print(f"MSE: {results['mse']:.6f} ± {results['std']:.6f}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot results
    print("\n" + "="*50)
    print("Generating plots...")
    
    plot_predictions(
        results['predictions'], 
        results['targets'], 
        normalizer,
        output_dir / 'prediction_scatter.png',
        num_samples=args.num_samples
    )
    
    # Use sequential data for trajectory plots (not random samples)
    print("\nEvaluating sequential frames for trajectory plots...")
    seq_results = evaluate_sequential(model, dataset, device, start_idx=0, num_frames=200)
    
    plot_trajectory(
        seq_results['predictions'],
        seq_results['targets'],
        normalizer,
        output_dir / 'trajectory.png'
    )
    
    plot_error_distribution(
        results['predictions'],
        results['targets'],
        output_dir / 'error_distribution.png'
    )
    
    # Joint angle plots (especially lower limb) - use sequential data
    print("\nGenerating joint angle plots...")
    plot_joint_angles(
        seq_results['predictions'],
        seq_results['targets'],
        normalizer,
        output_dir / 'joint_angles.png'
    )
    
    plot_joint_scatter(
        results['predictions'],  # scatter can use random samples
        results['targets'],
        normalizer,
        output_dir / 'joint_scatter.png'
    )
    
    print(f"\nResults saved to {output_dir}/")


if __name__ == '__main__':
    main()
