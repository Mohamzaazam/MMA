"""
Evaluate MotionNN model: test predictions and plot input vs output.

Usage:
    pixi run python python/models/eval_motion.py --model nn/motion_model_best.pt --bvh data/motion/walk.bvh
    
    # Evaluate on test set from training split
    pixi run python python/models/eval_motion.py --model nn/motion_model_best.pt --split_file nn/split_info.json --use_test_set
"""

import argparse
import json
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
    n_samples = min(256, len(predictions))
    
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


# =============================================================================
# Per-Component Error Analysis
# =============================================================================

# State layout for CMU/DART skeleton (112 dims = 56 pos + 56 vel)
STATE_COMPONENTS = {
    'root_pos': {'dims': [0, 1, 2], 'label': 'Root Position', 'axes': ['X', 'Y', 'Z']},
    'root_rot': {'dims': [3, 4, 5], 'label': 'Root Rotation', 'axes': ['X', 'Y', 'Z']},
    # Spine
    'spine': {'dims': [6, 7, 8], 'label': 'Spine', 'axes': ['X', 'Y', 'Z']},
    # Right leg
    'r_hip': {'dims': [9, 10, 11], 'label': 'Right Hip', 'axes': ['Flex', 'Abd', 'Rot']},
    'r_knee': {'dims': [12, 13, 14], 'label': 'Right Knee', 'axes': ['Flex', 'Abd', 'Rot']},
    'r_ankle': {'dims': [15, 16, 17], 'label': 'Right Ankle', 'axes': ['Flex', 'Abd', 'Rot']},
    # Left leg
    'l_hip': {'dims': [18, 19, 20], 'label': 'Left Hip', 'axes': ['Flex', 'Abd', 'Rot']},
    'l_knee': {'dims': [21, 22, 23], 'label': 'Left Knee', 'axes': ['Flex', 'Abd', 'Rot']},
    'l_ankle': {'dims': [24, 25, 26], 'label': 'Left Ankle', 'axes': ['Flex', 'Abd', 'Rot']},
    # Arms (approximate)
    'r_shoulder': {'dims': [27, 28, 29], 'label': 'Right Shoulder', 'axes': ['X', 'Y', 'Z']},
    'r_elbow': {'dims': [30, 31, 32], 'label': 'Right Elbow', 'axes': ['X', 'Y', 'Z']},
    'l_shoulder': {'dims': [33, 34, 35], 'label': 'Left Shoulder', 'axes': ['X', 'Y', 'Z']},
    'l_elbow': {'dims': [36, 37, 38], 'label': 'Left Elbow', 'axes': ['X', 'Y', 'Z']},
}


def compute_per_component_errors(predictions, targets, normalizer, state_dim=112):
    """
    Compute MSE for each state component.
    
    Args:
        predictions: Predicted states (N, state_dim) - normalized
        targets: Ground truth states (N, state_dim) - normalized
        normalizer: StateNormalizer for denormalization
        state_dim: Total state dimension (default 112 = 56 pos + 56 vel)
        
    Returns:
        Dict with per-component errors
    """
    # Denormalize for interpretable errors
    preds = normalizer.denormalize(predictions)
    targs = normalizer.denormalize(targets)
    
    errors = preds - targs
    squared_errors = errors ** 2
    
    result = {
        'per_dim': {},
        'joints': {},
        'groups': {},
        'summary': {},
    }
    
    # Per-dimension MSE
    mse_per_dim = np.mean(squared_errors, axis=0)
    for i, mse in enumerate(mse_per_dim):
        result['per_dim'][i] = float(mse)
    
    # Determine position/velocity split
    pos_dims = state_dim // 2  # 56 for standard skeleton
    
    # Position vs velocity
    result['summary']['position_mse'] = float(np.mean(squared_errors[:, :pos_dims]))
    result['summary']['velocity_mse'] = float(np.mean(squared_errors[:, pos_dims:]))
    result['summary']['total_mse'] = float(np.mean(squared_errors))
    result['summary']['total_mae'] = float(np.mean(np.abs(errors)))
    result['summary']['total_rmse'] = float(np.sqrt(result['summary']['total_mse']))
    
    # R² (coefficient of determination)
    ss_res = np.sum(squared_errors)
    ss_tot = np.sum((targs - np.mean(targs, axis=0)) ** 2)
    result['summary']['r2'] = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0
    
    # Per-joint errors
    for comp_name, comp_info in STATE_COMPONENTS.items():
        dims = comp_info['dims']
        valid_dims = [d for d in dims if d < predictions.shape[1]]
        if valid_dims:
            joint_mse = float(np.mean(squared_errors[:, valid_dims]))
            result['joints'][comp_name] = {
                'mse': joint_mse,
                'rmse': float(np.sqrt(joint_mse)),
                'label': comp_info['label'],
            }
            # Per-axis
            for i, d in enumerate(valid_dims):
                if i < len(comp_info['axes']):
                    axis_name = comp_info['axes'][i]
                    result['joints'][comp_name][f'{axis_name}_mse'] = float(mse_per_dim[d])
    
    # Group errors
    lower_limb_dims = []
    for key in ['r_hip', 'r_knee', 'r_ankle', 'l_hip', 'l_knee', 'l_ankle']:
        if key in STATE_COMPONENTS:
            lower_limb_dims.extend(STATE_COMPONENTS[key]['dims'])
    valid_lower = [d for d in lower_limb_dims if d < predictions.shape[1]]
    if valid_lower:
        result['groups']['lower_limb'] = float(np.mean(squared_errors[:, valid_lower]))
    
    upper_limb_dims = []
    for key in ['r_shoulder', 'r_elbow', 'l_shoulder', 'l_elbow']:
        if key in STATE_COMPONENTS:
            upper_limb_dims.extend(STATE_COMPONENTS[key]['dims'])
    valid_upper = [d for d in upper_limb_dims if d < predictions.shape[1]]
    if valid_upper:
        result['groups']['upper_limb'] = float(np.mean(squared_errors[:, valid_upper]))
    
    spine_dims = STATE_COMPONENTS.get('spine', {}).get('dims', [])
    valid_spine = [d for d in spine_dims if d < predictions.shape[1]]
    if valid_spine:
        result['groups']['spine'] = float(np.mean(squared_errors[:, valid_spine]))
    
    return result


def evaluate_rollout_horizons(model, dataset, device, horizons=[1, 5, 10, 30, 60], num_rollouts=50):
    """
    Evaluate prediction quality at multiple rollout horizons.
    
    Args:
        model: Trained MotionNN model
        dataset: BVHDataset
        device: torch device
        horizons: List of horizon steps to evaluate
        num_rollouts: Number of rollout trajectories to average
        
    Returns:
        Dict mapping horizon -> MSE
    """
    model.eval()
    max_horizon = max(horizons)
    
    # Need enough sequential data
    valid_starts = [i for i in range(len(dataset) - max_horizon)]
    if len(valid_starts) == 0:
        print(f"Warning: Dataset too small for horizon {max_horizon}")
        return {h: None for h in horizons}
    
    indices = np.random.choice(valid_starts, min(num_rollouts, len(valid_starts)), replace=False)
    
    horizon_errors = {h: [] for h in horizons}
    
    with torch.no_grad():
        for start_idx in indices:
            # Get initial state
            x, _ = dataset[start_idx]
            state = x.unsqueeze(0).to(device)
            
            # Rollout
            for step in range(1, max_horizon + 1):
                pred = model(state)
                
                # Get ground truth for this step
                if start_idx + step < len(dataset):
                    _, gt = dataset[start_idx + step]
                    gt = gt.numpy() if isinstance(gt, torch.Tensor) else gt
                    pred_np = pred.cpu().numpy().squeeze()
                    
                    mse = np.mean((pred_np - gt) ** 2)
                    
                    if step in horizons:
                        horizon_errors[step].append(mse)
                
                # Update state for next prediction
                if model.mode == 'mlp':
                    state = pred
                else:
                    # Transformer: shift sequence
                    state = torch.cat([state[:, 1:, :], pred.unsqueeze(1)], dim=1)
    
    return {h: np.mean(errs) if errs else None for h, errs in horizon_errors.items()}


def plot_per_component_errors(component_errors, output_path=None):
    """
    Plot horizontal bar chart of per-component MSE.
    """
    joints = component_errors.get('joints', {})
    if not joints:
        print("No joint data for per-component plot")
        return None
    
    # Prepare data
    labels = []
    mses = []
    colors = []
    
    # Color coding by body region
    color_map = {
        'root': '#1f77b4',
        'spine': '#ff7f0e',
        'r_hip': '#2ca02c', 'r_knee': '#2ca02c', 'r_ankle': '#2ca02c',
        'l_hip': '#d62728', 'l_knee': '#d62728', 'l_ankle': '#d62728',
        'r_shoulder': '#9467bd', 'r_elbow': '#9467bd',
        'l_shoulder': '#8c564b', 'l_elbow': '#8c564b',
    }
    
    for comp_name, comp_data in sorted(joints.items(), key=lambda x: x[1]['mse'], reverse=True):
        labels.append(comp_data['label'])
        mses.append(comp_data['mse'])
        colors.append(color_map.get(comp_name, '#888888'))
    
    fig, ax = plt.subplots(figsize=(10, max(6, len(labels) * 0.4)))
    
    y_pos = np.arange(len(labels))
    ax.barh(y_pos, mses, color=colors, alpha=0.8)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()  # Highest error at top
    ax.set_xlabel('MSE')
    ax.set_title('Per-Component Prediction Error')
    ax.grid(True, axis='x', alpha=0.3)
    
    # Add value labels
    for i, (label, mse) in enumerate(zip(labels, mses)):
        ax.text(mse + max(mses) * 0.01, i, f'{mse:.4f}', va='center', fontsize=9)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved per-component error plot to {output_path}")
    
    plt.show()
    return fig


def plot_rollout_horizon(horizon_errors, output_path=None):
    """
    Plot MSE vs rollout horizon.
    """
    horizons = sorted([h for h, e in horizon_errors.items() if e is not None])
    errors = [horizon_errors[h] for h in horizons]
    
    if not horizons:
        print("No valid horizon data to plot")
        return None
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(horizons, errors, 'b-o', linewidth=2, markersize=8)
    ax.fill_between(horizons, 0, errors, alpha=0.2)
    
    ax.set_xlabel('Prediction Horizon (steps)')
    ax.set_ylabel('MSE')
    ax.set_title('Prediction Error vs Rollout Horizon')
    ax.grid(True, alpha=0.3)
    
    # Log scale if range is large
    if max(errors) / (min(errors) + 1e-8) > 100:
        ax.set_yscale('log')
    
    # Annotate points
    for h, e in zip(horizons, errors):
        ax.annotate(f'{e:.4f}', (h, e), textcoords="offset points", 
                   xytext=(0, 10), ha='center', fontsize=9)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved rollout horizon plot to {output_path}")
    
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
    n_samples = min(256, len(predictions))
    
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
    parser.add_argument('--num_frames', type=int, default=500,
                        help='Number of frames for trajectory visualization (default: 500)')
    
    # Test set support
    parser.add_argument('--split_file', type=str, default=None,
                        help='Load files from split JSON (use with --use_test_set)')
    parser.add_argument('--use_test_set', action='store_true',
                        help='Evaluate on test set from split file')
    parser.add_argument('--use_val_set', action='store_true',
                        help='Evaluate on validation set from split file')
    
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
    
    # Load dataset - from split file or directly
    if args.split_file:
        print(f"\nLoading split info from {args.split_file}")
        with open(args.split_file, 'r') as f:
            split_info = json.load(f)
        
        if args.use_test_set:
            bvh_files = split_info.get('test_files', [])
            print(f"Using TEST set: {len(bvh_files)} files from {len(split_info.get('test_subjects', []))} subjects")
        elif args.use_val_set:
            bvh_files = split_info.get('val_files', [])
            print(f"Using VAL set: {len(bvh_files)} files from {len(split_info.get('val_subjects', []))} subjects")
        else:
            # Default to test set if split file provided but no flag
            bvh_files = split_info.get('test_files', [])
            print(f"Using TEST set (default): {len(bvh_files)} files")
        
        if not bvh_files:
            print("ERROR: No files found in split. Check split_file contents.")
            return
    elif args.bvh:
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
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Evaluate single-step
    print("\n" + "="*50)
    print("Evaluating single-step prediction...")
    results = evaluate_single_step(model, dataset, device, args.num_samples)
    print(f"MSE: {results['mse']:.6f} ± {results['std']:.6f}")
    
    # Per-component error analysis
    print("\n" + "="*50)
    print("Computing per-component errors...")
    component_errors = compute_per_component_errors(
        results['predictions'], 
        results['targets'], 
        normalizer,
        state_dim=model.state_dim
    )
    
    print(f"\nSummary Metrics:")
    print(f"  Total MSE:    {component_errors['summary']['total_mse']:.6f}")
    print(f"  Total RMSE:   {component_errors['summary']['total_rmse']:.6f}")
    print(f"  Total MAE:    {component_errors['summary']['total_mae']:.6f}")
    print(f"  R²:           {component_errors['summary']['r2']:.4f}")
    print(f"  Position MSE: {component_errors['summary']['position_mse']:.6f}")
    print(f"  Velocity MSE: {component_errors['summary']['velocity_mse']:.6f}")
    
    print(f"\nGroup Errors:")
    for group, mse in component_errors.get('groups', {}).items():
        print(f"  {group}: {mse:.6f}")
    
    # Save metrics to JSON
    metrics_path = output_dir / 'metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(component_errors, f, indent=2)
    print(f"\nSaved metrics to {metrics_path}")
    
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
    
    # Per-component error bar chart
    plot_per_component_errors(
        component_errors,
        output_dir / 'per_component_errors.png'
    )
    
    # Use sequential data for trajectory plots with configurable frame count
    print(f"\nEvaluating sequential frames for trajectory plots ({args.num_frames} frames)...")
    seq_results = evaluate_sequential(model, dataset, device, start_idx=0, num_frames=args.num_frames)
    
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
    
    # Rollout horizon analysis
    print("\n" + "="*50)
    print("Evaluating rollout horizons...")
    horizon_errors = evaluate_rollout_horizons(
        model, dataset, device, 
        horizons=[1, 5, 10, 30, 60],
        num_rollouts=50
    )
    print("Horizon -> MSE:")
    for h, e in sorted(horizon_errors.items()):
        if e is not None:
            print(f"  {h:3d} steps: {e:.6f}")
    
    plot_rollout_horizon(
        horizon_errors,
        output_dir / 'rollout_horizon.png'
    )
    
    print(f"\n{'='*50}")
    print(f"Results saved to {output_dir}/")
    print(f"  - metrics.json (all computed metrics)")
    print(f"  - per_component_errors.png")
    print(f"  - prediction_scatter.png")
    print(f"  - trajectory.png")
    print(f"  - error_distribution.png")
    print(f"  - joint_angles.png")
    print(f"  - joint_scatter.png")
    print(f"  - rollout_horizon.png")


if __name__ == '__main__':
    main()

