#!/usr/bin/env python3
"""
BVH Motion Capture Visualization Tool
=====================================
Comprehensive analysis and visualization of BVH motion capture files.
Features:
- Automatic exclusion of rigging joints (_RIGMESH, _RIG, etc.)
- Multiple smoothing techniques (Moving Average, Gaussian, Savitzky-Golay, Exponential)
- Side-by-side comparison between two BVH files
- Joint hierarchy visualization
- Statistical analysis
- Export capabilities
"""

import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy import signal
from scipy.ndimage import gaussian_filter1d
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# Import shared parser (can also use local Joint/BVHData/BVHParser for visualization-specific needs)
from . import bvh_parser as shared_parser


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class Joint:
    """Represents a single joint in the skeleton hierarchy."""
    name: str
    offset: np.ndarray
    channels: List[str]
    channel_indices: List[int] = field(default_factory=list)
    children: List['Joint'] = field(default_factory=list)
    parent: Optional['Joint'] = None
    is_end_site: bool = False
    is_rigging: bool = False

@dataclass
class BVHData:
    """Complete BVH file data structure."""
    root: Joint
    joints: Dict[str, Joint]
    frame_count: int
    frame_time: float
    motion_data: np.ndarray
    filename: str = ""
    
    @property
    def duration(self) -> float:
        return self.frame_count * self.frame_time
    
    @property
    def fps(self) -> float:
        return 1.0 / self.frame_time if self.frame_time > 0 else 0

# =============================================================================
# RIGGING JOINT DETECTION
# =============================================================================

# Patterns that indicate rigging/mesh joints (not actual skeleton joints)
RIGGING_PATTERNS = [
    r'_RIGMESH$',
    r'_RIG$',
    r'_MESH$',
    r'_IK$',
    r'_FK$',
    r'_CTRL$',
    r'_ctrl$',
    r'_Helper$',
    r'_helper$',
    r'_Twist$',
    r'_twist$',
    r'_Roll$',
    r'_roll$',
    r'Bone\d+$',
    r'_proxy$',
    r'_PROXY$',
    r'_deform$',
    r'_DEFORM$',
]

def is_rigging_joint(joint_name: str) -> bool:
    """Check if a joint is a rigging/helper joint based on naming conventions."""
    for pattern in RIGGING_PATTERNS:
        if re.search(pattern, joint_name):
            return True
    return False

def get_analysis_joints(bvh: BVHData) -> Dict[str, Joint]:
    """Get only the joints suitable for analysis (excluding rigging joints)."""
    return {name: joint for name, joint in bvh.joints.items() 
            if not joint.is_rigging and not joint.is_end_site}

# =============================================================================
# BVH PARSER
# =============================================================================

class BVHParser:
    """Parser for BVH motion capture files."""
    
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.joints: Dict[str, Joint] = {}
        self.channel_count = 0
        
    def parse(self) -> BVHData:
        """Parse the BVH file and return structured data."""
        with open(self.filepath, 'r') as f:
            content = f.read()
        
        # Split into HIERARCHY and MOTION sections
        parts = content.split('MOTION')
        if len(parts) != 2:
            raise ValueError("Invalid BVH file format")
        
        hierarchy_section = parts[0]
        motion_section = 'MOTION' + parts[1]
        
        # Parse hierarchy
        root = self._parse_hierarchy(hierarchy_section)
        
        # Parse motion data
        frame_count, frame_time, motion_data = self._parse_motion(motion_section)
        
        return BVHData(
            root=root,
            joints=self.joints,
            frame_count=frame_count,
            frame_time=frame_time,
            motion_data=motion_data,
            filename=self.filepath.split('/')[-1]
        )
    
    def _parse_hierarchy(self, content: str) -> Joint:
        """Parse the HIERARCHY section."""
        lines = [l.strip() for l in content.split('\n') if l.strip()]
        
        root = None
        joint_stack = []
        current_joint = None
        
        i = 0
        while i < len(lines):
            line = lines[i]
            tokens = line.split()
            
            if not tokens:
                i += 1
                continue
            
            if tokens[0] == 'HIERARCHY':
                pass
            
            elif tokens[0] in ('ROOT', 'JOINT'):
                joint_name = tokens[1]
                is_rigging = is_rigging_joint(joint_name)
                
                new_joint = Joint(
                    name=joint_name,
                    offset=np.zeros(3),
                    channels=[],
                    is_rigging=is_rigging
                )
                
                if tokens[0] == 'ROOT':
                    root = new_joint
                else:
                    if current_joint:
                        current_joint.children.append(new_joint)
                        new_joint.parent = current_joint
                
                self.joints[joint_name] = new_joint
                joint_stack.append(new_joint)
                current_joint = new_joint
            
            elif tokens[0] == 'End' and len(tokens) > 1 and tokens[1] == 'Site':
                end_joint = Joint(
                    name=f"{current_joint.name}_End",
                    offset=np.zeros(3),
                    channels=[],
                    is_end_site=True
                )
                current_joint.children.append(end_joint)
                end_joint.parent = current_joint
                joint_stack.append(end_joint)
                current_joint = end_joint
            
            elif tokens[0] == 'OFFSET':
                if current_joint:
                    current_joint.offset = np.array([float(tokens[1]), float(tokens[2]), float(tokens[3])])
            
            elif tokens[0] == 'CHANNELS':
                if current_joint:
                    num_channels = int(tokens[1])
                    current_joint.channels = tokens[2:2+num_channels]
                    current_joint.channel_indices = list(range(self.channel_count, self.channel_count + num_channels))
                    self.channel_count += num_channels
            
            elif tokens[0] == '{':
                pass
            
            elif tokens[0] == '}':
                if joint_stack:
                    joint_stack.pop()
                    current_joint = joint_stack[-1] if joint_stack else None
            
            i += 1
        
        return root
    
    def _parse_motion(self, content: str) -> Tuple[int, float, np.ndarray]:
        """Parse the MOTION section."""
        lines = [l.strip() for l in content.split('\n') if l.strip()]
        
        frame_count = 0
        frame_time = 0.0
        motion_lines = []
        
        parsing_frames = False
        for line in lines:
            if line.startswith('Frames:'):
                frame_count = int(line.split(':')[1].strip())
            elif line.startswith('Frame Time:'):
                frame_time = float(line.split(':')[1].strip())
                parsing_frames = True
            elif parsing_frames:
                values = [float(v) for v in line.split()]
                motion_lines.append(values)
        
        motion_data = np.array(motion_lines)
        return frame_count, frame_time, motion_data

# =============================================================================
# SMOOTHING TECHNIQUES
# =============================================================================

class SmoothingMethods:
    """Collection of smoothing algorithms for motion data."""
    
    @staticmethod
    def moving_average(data: np.ndarray, window_size: int = 5) -> np.ndarray:
        """Apply simple moving average smoothing."""
        if len(data) < window_size:
            return data.copy()
        kernel = np.ones(window_size) / window_size
        return np.convolve(data, kernel, mode='same')
    
    @staticmethod
    def gaussian(data: np.ndarray, sigma: float = 2.0) -> np.ndarray:
        """Apply Gaussian smoothing."""
        return gaussian_filter1d(data, sigma=sigma)
    
    @staticmethod
    def savitzky_golay(data: np.ndarray, window_size: int = 7, poly_order: int = 3) -> np.ndarray:
        """Apply Savitzky-Golay filter for smoothing while preserving features."""
        if len(data) < window_size:
            return data.copy()
        # Ensure window_size is odd
        if window_size % 2 == 0:
            window_size += 1
        return signal.savgol_filter(data, window_size, poly_order)
    
    @staticmethod
    def exponential(data: np.ndarray, alpha: float = 0.3) -> np.ndarray:
        """Apply exponential moving average smoothing."""
        result = np.zeros_like(data)
        result[0] = data[0]
        for i in range(1, len(data)):
            result[i] = alpha * data[i] + (1 - alpha) * result[i-1]
        return result
    
    @staticmethod
    def butterworth(data: np.ndarray, cutoff: float = 0.1, order: int = 4) -> np.ndarray:
        """Apply Butterworth low-pass filter."""
        if len(data) < 15:  # Need sufficient data for filter
            return data.copy()
        b, a = signal.butter(order, cutoff, btype='low')
        # Use filtfilt for zero-phase filtering
        return signal.filtfilt(b, a, data, padlen=min(len(data)-1, 3*max(len(a), len(b))))
    
    @staticmethod
    def median(data: np.ndarray, kernel_size: int = 5) -> np.ndarray:
        """Apply median filter (good for removing outliers)."""
        return signal.medfilt(data, kernel_size=kernel_size)

# =============================================================================
# DATA EXTRACTION
# =============================================================================

def get_joint_data(bvh: BVHData, joint_name: str) -> Dict[str, np.ndarray]:
    """Extract rotation/position data for a specific joint."""
    if joint_name not in bvh.joints:
        raise ValueError(f"Joint '{joint_name}' not found")
    
    joint = bvh.joints[joint_name]
    data = {}
    
    for i, channel in enumerate(joint.channels):
        idx = joint.channel_indices[i]
        if idx < bvh.motion_data.shape[1]:
            data[channel] = bvh.motion_data[:, idx]
    
    return data

def get_all_joint_rotations(bvh: BVHData, exclude_rigging: bool = True) -> Dict[str, Dict[str, np.ndarray]]:
    """Get rotation data for all joints."""
    result = {}
    for name, joint in bvh.joints.items():
        if exclude_rigging and (joint.is_rigging or joint.is_end_site):
            continue
        try:
            data = get_joint_data(bvh, name)
            # Only include rotation channels
            rot_data = {k: v for k, v in data.items() if 'rotation' in k.lower()}
            if rot_data:
                result[name] = rot_data
        except (ValueError, IndexError):
            continue
    return result

# =============================================================================
# STATISTICAL ANALYSIS
# =============================================================================

def compute_joint_statistics(data: np.ndarray) -> Dict[str, float]:
    """Compute statistical measures for joint data."""
    return {
        'mean': np.mean(data),
        'std': np.std(data),
        'min': np.min(data),
        'max': np.max(data),
        'range': np.ptp(data),
        'variance': np.var(data),
        'rms': np.sqrt(np.mean(data**2)),
    }

def compute_motion_velocity(data: np.ndarray, frame_time: float) -> np.ndarray:
    """Compute angular velocity from rotation data."""
    velocity = np.diff(data) / frame_time
    return np.concatenate([[velocity[0]], velocity])

def compute_motion_acceleration(data: np.ndarray, frame_time: float) -> np.ndarray:
    """Compute angular acceleration from rotation data."""
    velocity = compute_motion_velocity(data, frame_time)
    acceleration = np.diff(velocity) / frame_time
    return np.concatenate([[acceleration[0]], acceleration])

# =============================================================================
# VISUALIZATION
# =============================================================================

class BVHVisualizer:
    """Comprehensive visualization toolkit for BVH data."""
    
    def __init__(self, bvh1: BVHData, bvh2: Optional[BVHData] = None):
        self.bvh1 = bvh1
        self.bvh2 = bvh2
        self.smoother = SmoothingMethods()
        
        # Set style
        plt.style.use('seaborn-v0_8-whitegrid')
        self.colors = plt.cm.tab10.colors
    
    def plot_joint_rotations(self, joint_name: str, apply_smoothing: Optional[str] = None,
                             smoothing_params: dict = None, save_path: str = None):
        """Plot rotation data for a specific joint with optional smoothing."""
        fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
        fig.suptitle(f'Joint Rotations: {joint_name}', fontsize=14, fontweight='bold')
        
        data1 = get_joint_data(self.bvh1, joint_name)
        time1 = np.arange(self.bvh1.frame_count) * self.bvh1.frame_time
        
        rotation_channels = ['Zrotation', 'Xrotation', 'Yrotation']
        axis_labels = ['Z Rotation', 'X Rotation', 'Y Rotation']
        
        for i, (channel, label) in enumerate(zip(rotation_channels, axis_labels)):
            ax = axes[i]
            
            if channel in data1:
                raw_data = data1[channel]
                ax.plot(time1, raw_data, 'b-', alpha=0.5, label=f'{self.bvh1.filename} (raw)', linewidth=1)
                
                if apply_smoothing and smoothing_params:
                    smoothed = self._apply_smoothing(raw_data, apply_smoothing, smoothing_params)
                    ax.plot(time1, smoothed, 'b-', label=f'{self.bvh1.filename} (smoothed)', linewidth=2)
            
            if self.bvh2 and joint_name in self.bvh2.joints:
                data2 = get_joint_data(self.bvh2, joint_name)
                time2 = np.arange(self.bvh2.frame_count) * self.bvh2.frame_time
                
                if channel in data2:
                    raw_data2 = data2[channel]
                    ax.plot(time2, raw_data2, 'r-', alpha=0.5, label=f'{self.bvh2.filename} (raw)', linewidth=1)
                    
                    if apply_smoothing and smoothing_params:
                        smoothed2 = self._apply_smoothing(raw_data2, apply_smoothing, smoothing_params)
                        ax.plot(time2, smoothed2, 'r-', label=f'{self.bvh2.filename} (smoothed)', linewidth=2)
            
            ax.set_ylabel(f'{label} (°)')
            ax.legend(loc='upper right', fontsize=8)
            ax.grid(True, alpha=0.3)
        
        axes[-1].set_xlabel('Time (s)')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        return fig
    
    def plot_smoothing_comparison(self, joint_name: str, channel: str = 'Zrotation',
                                  save_path: str = None):
        """Compare different smoothing techniques on the same data."""
        data = get_joint_data(self.bvh1, joint_name)
        if channel not in data:
            raise ValueError(f"Channel '{channel}' not found for joint '{joint_name}'")
        
        raw = data[channel]
        time = np.arange(len(raw)) * self.bvh1.frame_time
        
        fig, axes = plt.subplots(3, 2, figsize=(16, 12))
        fig.suptitle(f'Smoothing Comparison: {joint_name} - {channel}', fontsize=14, fontweight='bold')
        
        smoothing_methods = [
            ('Raw Data', raw),
            ('Moving Average (w=5)', self.smoother.moving_average(raw, 5)),
            ('Gaussian (σ=4)', self.smoother.gaussian(raw, 4)),
            ('Savitzky-Golay', self.smoother.savitzky_golay(raw, 7, 3)),
            ('Exponential (α=0.3)', self.smoother.exponential(raw, 0.3)),
            ('Butterworth', self.smoother.butterworth(raw, 0.1, 4)),
        ]
        
        for ax, (name, smoothed_data) in zip(axes.flat, smoothing_methods):
            ax.plot(time, raw, 'lightgray', alpha=0.7, label='Raw', linewidth=1)
            ax.plot(time, smoothed_data, 'b-', label=name, linewidth=2)
            ax.set_title(name)
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Rotation (°)')
            ax.legend(loc='upper right', fontsize=8)
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        return fig
    
    def plot_velocity_acceleration(self, joint_name: str, channel: str = 'Zrotation',
                                   apply_smoothing: bool = True, save_path: str = None):
        """Plot rotation, velocity, and acceleration for a joint."""
        data = get_joint_data(self.bvh1, joint_name)
        if channel not in data:
            raise ValueError(f"Channel '{channel}' not found")
        
        raw = data[channel]
        if apply_smoothing:
            rotation = self.smoother.savitzky_golay(raw, 7, 3)
        else:
            rotation = raw
        
        velocity = compute_motion_velocity(rotation, self.bvh1.frame_time)
        acceleration = compute_motion_acceleration(rotation, self.bvh1.frame_time)
        time = np.arange(len(rotation)) * self.bvh1.frame_time
        
        fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
        fig.suptitle(f'Motion Analysis: {joint_name} - {channel}', fontsize=14, fontweight='bold')
        
        axes[0].plot(time, rotation, 'b-', linewidth=2)
        axes[0].set_ylabel('Rotation (°)')
        axes[0].set_title('Angular Position')
        axes[0].grid(True, alpha=0.3)
        
        axes[1].plot(time, velocity, 'g-', linewidth=2)
        axes[1].set_ylabel('Angular Velocity (°/s)')
        axes[1].set_title('Angular Velocity')
        axes[1].grid(True, alpha=0.3)
        
        axes[2].plot(time, acceleration, 'r-', linewidth=2)
        axes[2].set_ylabel('Angular Acceleration (°/s²)')
        axes[2].set_title('Angular Acceleration')
        axes[2].set_xlabel('Time (s)')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        return fig
    
    def plot_joint_comparison_heatmap(self, save_path: str = None):
        """Create a heatmap comparing joint motion ranges between two files."""
        if not self.bvh2:
            raise ValueError("Second BVH file required for comparison")
        
        joints1 = get_analysis_joints(self.bvh1)
        joints2 = get_analysis_joints(self.bvh2)
        common_joints = set(joints1.keys()) & set(joints2.keys())
        
        if not common_joints:
            raise ValueError("No common joints found between files")
        
        joint_names = sorted(common_joints)
        channels = ['Zrotation', 'Xrotation', 'Yrotation']
        
        # Compute range differences
        diff_matrix = np.zeros((len(joint_names), len(channels)))
        
        for i, joint in enumerate(joint_names):
            data1 = get_joint_data(self.bvh1, joint)
            data2 = get_joint_data(self.bvh2, joint)
            
            for j, ch in enumerate(channels):
                if ch in data1 and ch in data2:
                    range1 = np.ptp(data1[ch])
                    range2 = np.ptp(data2[ch])
                    diff_matrix[i, j] = range2 - range1
        
        fig, ax = plt.subplots(figsize=(10, max(8, len(joint_names) * 0.4)))
        
        im = ax.imshow(diff_matrix, cmap='RdBu_r', aspect='auto')
        ax.set_xticks(range(len(channels)))
        ax.set_xticklabels(['Z', 'X', 'Y'])
        ax.set_yticks(range(len(joint_names)))
        ax.set_yticklabels([j.replace('Character1_', '') for j in joint_names])
        
        plt.colorbar(im, ax=ax, label='Range Difference (°)')
        ax.set_title(f'Motion Range Comparison\n{self.bvh2.filename} - {self.bvh1.filename}',
                    fontsize=12, fontweight='bold')
        ax.set_xlabel('Rotation Axis')
        ax.set_ylabel('Joint')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        return fig
    
    def plot_statistics_summary(self, save_path: str = None):
        """Create a comprehensive statistics summary plot."""
        joints = get_analysis_joints(self.bvh1)
        joint_names = list(joints.keys())
        
        # Compute statistics for each joint
        stats_data = {'range': [], 'std': [], 'rms': []}
        
        for joint in joint_names:
            data = get_joint_data(self.bvh1, joint)
            all_rot = []
            for ch in ['Zrotation', 'Xrotation', 'Yrotation']:
                if ch in data:
                    all_rot.extend(data[ch])
            
            if all_rot:
                all_rot = np.array(all_rot)
                stats_data['range'].append(np.ptp(all_rot))
                stats_data['std'].append(np.std(all_rot))
                stats_data['rms'].append(np.sqrt(np.mean(all_rot**2)))
        
        fig, axes = plt.subplots(1, 3, figsize=(16, 6))
        fig.suptitle(f'Joint Statistics Summary: {self.bvh1.filename}', fontsize=14, fontweight='bold')
        
        short_names = [j.replace('Character1_', '') for j in joint_names]
        x = np.arange(len(joint_names))
        
        # Range plot
        axes[0].barh(x, stats_data['range'], color=self.colors[0])
        axes[0].set_yticks(x)
        axes[0].set_yticklabels(short_names)
        axes[0].set_xlabel('Range (°)')
        axes[0].set_title('Motion Range')
        
        # Standard deviation plot
        axes[1].barh(x, stats_data['std'], color=self.colors[1])
        axes[1].set_yticks(x)
        axes[1].set_yticklabels(short_names)
        axes[1].set_xlabel('Std Dev (°)')
        axes[1].set_title('Motion Variability')
        
        # RMS plot
        axes[2].barh(x, stats_data['rms'], color=self.colors[2])
        axes[2].set_yticks(x)
        axes[2].set_yticklabels(short_names)
        axes[2].set_xlabel('RMS (°)')
        axes[2].set_title('RMS Rotation')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        return fig
    
    def plot_all_joints_overview(self, save_path: str = None):
        """Create an overview plot of all joint rotations."""
        joints = get_analysis_joints(self.bvh1)
        joint_names = list(joints.keys())
        n_joints = len(joint_names)
        
        # Calculate grid dimensions
        n_cols = 3
        n_rows = (n_joints + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 3 * n_rows), sharex=True)
        fig.suptitle(f'All Joints Overview: {self.bvh1.filename}', fontsize=14, fontweight='bold')
        
        axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes
        time = np.arange(self.bvh1.frame_count) * self.bvh1.frame_time
        
        for i, joint in enumerate(joint_names):
            ax = axes[i]
            data = get_joint_data(self.bvh1, joint)
            
            for ch, color in zip(['Zrotation', 'Xrotation', 'Yrotation'], ['b', 'g', 'r']):
                if ch in data:
                    ax.plot(time, data[ch], color, alpha=0.7, linewidth=1, label=ch[0])
            
            ax.set_title(joint.replace('Character1_', ''), fontsize=10)
            ax.legend(loc='upper right', fontsize=7)
            ax.grid(True, alpha=0.3)
        
        # Hide unused axes
        for i in range(n_joints, len(axes)):
            axes[i].set_visible(False)
        
        # Add common x-label
        fig.text(0.5, 0.02, 'Time (s)', ha='center', fontsize=12)
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.97])
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        return fig
    
    def plot_hierarchy_tree(self, save_path: str = None):
        """Visualize the joint hierarchy as a tree structure."""
        fig, ax = plt.subplots(figsize=(14, 10))
        ax.set_title(f'Joint Hierarchy: {self.bvh1.filename}', fontsize=14, fontweight='bold')
        
        # BFS to assign positions
        positions = {}
        level_counts = {}
        
        def assign_positions(joint, level=0, parent_x=0):
            if level not in level_counts:
                level_counts[level] = 0
            
            x = level_counts[level]
            level_counts[level] += 1
            positions[joint.name] = (x, -level)
            
            for child in joint.children:
                if not child.is_end_site:
                    assign_positions(child, level + 1, x)
        
        assign_positions(self.bvh1.root)
        
        # Normalize x positions
        max_count = max(level_counts.values()) if level_counts else 1
        for name in positions:
            x, y = positions[name]
            positions[name] = (x / max_count * 10, y)
        
        # Draw connections
        def draw_connections(joint):
            if joint.name in positions:
                x1, y1 = positions[joint.name]
                for child in joint.children:
                    if not child.is_end_site and child.name in positions:
                        x2, y2 = positions[child.name]
                        ax.plot([x1, x2], [y1, y2], 'k-', linewidth=1, alpha=0.5)
                        draw_connections(child)
        
        draw_connections(self.bvh1.root)
        
        # Draw nodes
        for name, (x, y) in positions.items():
            joint = self.bvh1.joints.get(name)
            color = 'lightcoral' if (joint and joint.is_rigging) else 'lightblue'
            ax.scatter(x, y, s=200, c=color, edgecolors='black', zorder=5)
            short_name = name.replace('Character1_', '')
            ax.annotate(short_name, (x, y), ha='center', va='bottom', fontsize=8)
        
        # Legend
        ax.scatter([], [], c='lightblue', edgecolors='black', label='Analysis Joints')
        ax.scatter([], [], c='lightcoral', edgecolors='black', label='Rigging Joints')
        ax.legend(loc='upper right')
        
        ax.set_xlim(-1, 11)
        ax.axis('off')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        return fig
    
    def _apply_smoothing(self, data: np.ndarray, method: str, params: dict) -> np.ndarray:
        """Apply specified smoothing method with parameters."""
        if method == 'moving_average':
            return self.smoother.moving_average(data, params.get('window_size', 5))
        elif method == 'gaussian':
            return self.smoother.gaussian(data, params.get('sigma', 2.0))
        elif method == 'savgol':
            return self.smoother.savitzky_golay(data, params.get('window_size', 7), params.get('poly_order', 3))
        elif method == 'exponential':
            return self.smoother.exponential(data, params.get('alpha', 0.3))
        elif method == 'butterworth':
            return self.smoother.butterworth(data, params.get('cutoff', 0.1), params.get('order', 4))
        elif method == 'median':
            return self.smoother.median(data, params.get('kernel_size', 5))
        else:
            return data

# =============================================================================
# REPORT GENERATION
# =============================================================================

def generate_analysis_report(bvh1: BVHData, bvh2: Optional[BVHData] = None) -> str:
    """Generate a text-based analysis report."""
    lines = []
    lines.append("=" * 70)
    lines.append("BVH MOTION CAPTURE ANALYSIS REPORT")
    lines.append("=" * 70)
    lines.append("")
    
    # File 1 info
    lines.append(f"File 1: {bvh1.filename}")
    lines.append(f"  Frames: {bvh1.frame_count}")
    lines.append(f"  Frame Time: {bvh1.frame_time:.6f}s ({bvh1.fps:.2f} FPS)")
    lines.append(f"  Duration: {bvh1.duration:.3f}s")
    
    analysis_joints1 = get_analysis_joints(bvh1)
    rigging_joints1 = [j for j in bvh1.joints.values() if j.is_rigging]
    lines.append(f"  Total Joints: {len(bvh1.joints)}")
    lines.append(f"  Analysis Joints: {len(analysis_joints1)}")
    lines.append(f"  Rigging Joints (excluded): {len(rigging_joints1)}")
    lines.append("")
    
    if bvh2:
        lines.append(f"File 2: {bvh2.filename}")
        lines.append(f"  Frames: {bvh2.frame_count}")
        lines.append(f"  Frame Time: {bvh2.frame_time:.6f}s ({bvh2.fps:.2f} FPS)")
        lines.append(f"  Duration: {bvh2.duration:.3f}s")
        
        analysis_joints2 = get_analysis_joints(bvh2)
        rigging_joints2 = [j for j in bvh2.joints.values() if j.is_rigging]
        lines.append(f"  Total Joints: {len(bvh2.joints)}")
        lines.append(f"  Analysis Joints: {len(analysis_joints2)}")
        lines.append(f"  Rigging Joints (excluded): {len(rigging_joints2)}")
        lines.append("")
    
    # Joint statistics for file 1
    lines.append("-" * 70)
    lines.append("JOINT STATISTICS (File 1)")
    lines.append("-" * 70)
    lines.append(f"{'Joint':<25} {'Z Range':>10} {'X Range':>10} {'Y Range':>10} {'Total STD':>10}")
    lines.append("-" * 70)
    
    for joint_name in sorted(analysis_joints1.keys()):
        data = get_joint_data(bvh1, joint_name)
        z_range = np.ptp(data.get('Zrotation', [0]))
        x_range = np.ptp(data.get('Xrotation', [0]))
        y_range = np.ptp(data.get('Yrotation', [0]))
        
        all_rot = []
        for ch in ['Zrotation', 'Xrotation', 'Yrotation']:
            if ch in data:
                all_rot.extend(data[ch])
        total_std = np.std(all_rot) if all_rot else 0
        
        short_name = joint_name.replace('Character1_', '')
        lines.append(f"{short_name:<25} {z_range:>10.2f} {x_range:>10.2f} {y_range:>10.2f} {total_std:>10.2f}")
    
    if rigging_joints1:
        lines.append("")
        lines.append("-" * 70)
        lines.append("EXCLUDED RIGGING JOINTS")
        lines.append("-" * 70)
        for joint in rigging_joints1:
            lines.append(f"  - {joint.name}")
    
    lines.append("")
    lines.append("=" * 70)
    lines.append("END OF REPORT")
    lines.append("=" * 70)
    
    return "\n".join(lines)

# =============================================================================
# CLI ARGUMENT PARSING
# =============================================================================

def create_argument_parser():
    """Create and configure the argument parser with all available options."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='BVH Motion Capture Visualization Tool - Analyze and compare BVH files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with one BVH file
  python bvh_visualizer.py data/motion/29_01.bvh
  
  # Compare two BVH files
  python bvh_visualizer.py data/motion/29_01.bvh --compare data/motion/walk.bvh
  
  # Analyze specific joints with Gaussian smoothing
  python bvh_visualizer.py data/motion/29_01.bvh --joints Hips Spine --smoothing gaussian --sigma 4
  
  # Generate only specific visualizations
  python bvh_visualizer.py data/motion/29_01.bvh --viz hierarchy statistics smoothing
  
  # Full analysis with custom output directory
  python bvh_visualizer.py data/motion/29_01.bvh -o results/ --smoothing savgol --window-size 11

Available smoothing methods:
  moving_average  - Simple moving average (use --window-size)
  gaussian        - Gaussian filter (use --sigma)
  savgol          - Savitzky-Golay filter (use --window-size, --poly-order)
  exponential     - Exponential moving average (use --alpha)
  butterworth     - Butterworth low-pass filter (use --cutoff, --order)
  median          - Median filter (use --kernel-size)
        """
    )
    
    # Input files
    parser.add_argument('bvh_file', type=str,
                        help='Primary BVH file to analyze')
    parser.add_argument('-c', '--compare', type=str, metavar='BVH_FILE',
                        help='Second BVH file for comparison')
    
    # Output options
    parser.add_argument('-o', '--output-dir', type=str, default='data/outputs',
                        help='Output directory for generated files (default: data/outputs)')
    parser.add_argument('--no-save', action='store_true',
                        help='Display plots without saving to files')
    
    # Joint selection
    parser.add_argument('-j', '--joints', type=str, nargs='+', metavar='JOINT',
                        help='Specific joints to analyze (e.g., Hips Spine LeftUpLeg). '
                             'Partial matches are supported.')
    parser.add_argument('--list-joints', action='store_true',
                        help='List all available joints in the BVH file and exit')
    
    # Smoothing options
    parser.add_argument('-s', '--smoothing', type=str, 
                        choices=['none', 'moving_average', 'gaussian', 'savgol', 
                                 'exponential', 'butterworth', 'median'],
                        default='none',
                        help='Smoothing method to apply (default: none)')
    
    # Smoothing parameters
    smooth_params = parser.add_argument_group('Smoothing Parameters')
    smooth_params.add_argument('--window-size', type=int, default=5,
                               help='Window size for moving_average/savgol/median (default: 5)')
    smooth_params.add_argument('--sigma', type=float, default=2.0,
                               help='Sigma for Gaussian smoothing (default: 2.0)')
    smooth_params.add_argument('--poly-order', type=int, default=3,
                               help='Polynomial order for Savitzky-Golay filter (default: 3)')
    smooth_params.add_argument('--alpha', type=float, default=0.3,
                               help='Alpha for exponential smoothing (default: 0.3)')
    smooth_params.add_argument('--cutoff', type=float, default=0.1,
                               help='Cutoff frequency for Butterworth filter (default: 0.1)')
    smooth_params.add_argument('--order', type=int, default=4,
                               help='Filter order for Butterworth filter (default: 4)')
    smooth_params.add_argument('--kernel-size', type=int, default=5,
                               help='Kernel size for median filter (default: 5)')
    
    # Visualization selection
    viz_group = parser.add_argument_group('Visualization Options')
    viz_group.add_argument('-v', '--viz', type=str, nargs='+',
                           choices=['all', 'hierarchy', 'overview', 'statistics', 
                                   'smoothing', 'velocity', 'rotations', 'heatmap', 'report'],
                           default=['all'],
                           help='Visualizations to generate (default: all)')
    viz_group.add_argument('--channel', type=str, 
                           choices=['Xrotation', 'Yrotation', 'Zrotation'],
                           default='Zrotation',
                           help='Rotation channel to analyze (default: Zrotation)')
    viz_group.add_argument('--no-display', action='store_true',
                           help='Do not display plots (only save)')
    
    # Report options
    report_group = parser.add_argument_group('Report Options')
    report_group.add_argument('--report-only', action='store_true',
                              help='Generate only the text report, no plots')
    report_group.add_argument('--quiet', '-q', action='store_true',
                              help='Suppress progress output')
    
    return parser


def get_smoothing_params(args) -> dict:
    """Extract smoothing parameters from parsed arguments."""
    params = {
        'window_size': args.window_size,
        'sigma': args.sigma,
        'poly_order': args.poly_order,
        'alpha': args.alpha,
        'cutoff': args.cutoff,
        'order': args.order,
        'kernel_size': args.kernel_size,
    }
    return params


def find_matching_joints(bvh: BVHData, joint_patterns: List[str]) -> List[str]:
    """Find joints that match the given patterns (partial matching supported)."""
    analysis_joints = get_analysis_joints(bvh)
    matched = []
    
    for pattern in joint_patterns:
        pattern_lower = pattern.lower()
        for joint_name in analysis_joints.keys():
            # Support partial matching (case-insensitive)
            if pattern_lower in joint_name.lower():
                if joint_name not in matched:
                    matched.append(joint_name)
    
    return matched


def list_available_joints(bvh: BVHData):
    """Print all available joints in the BVH file."""
    print(f"\nJoints in {bvh.filename}:")
    print("-" * 50)
    
    analysis_joints = get_analysis_joints(bvh)
    rigging_joints = [j for j in bvh.joints.values() if j.is_rigging]
    end_sites = [j for j in bvh.joints.values() if j.is_end_site]
    
    print(f"\n📊 Analysis Joints ({len(analysis_joints)}):")
    for name in sorted(analysis_joints.keys()):
        joint = analysis_joints[name]
        channels = ', '.join(joint.channels) if joint.channels else 'none'
        print(f"  • {name}")
        print(f"      Channels: {channels}")
    
    if rigging_joints:
        print(f"\n🔧 Rigging Joints (excluded, {len(rigging_joints)}):")
        for joint in sorted(rigging_joints, key=lambda j: j.name):
            print(f"  • {joint.name}")
    
    if end_sites:
        print(f"\n🔚 End Sites ({len(end_sites)}):")
        for joint in sorted(end_sites, key=lambda j: j.name):
            print(f"  • {joint.name}")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main execution function with CLI argument parsing."""
    import os
    
    parser = create_argument_parser()
    args = parser.parse_args()
    
    # Parse primary BVH file
    if not args.quiet:
        print("Loading BVH files...")
    
    try:
        bvh1 = BVHParser(args.bvh_file).parse()
    except FileNotFoundError:
        print(f"Error: File not found: {args.bvh_file}")
        return 1
    except Exception as e:
        print(f"Error parsing {args.bvh_file}: {e}")
        return 1
    
    if not args.quiet:
        print(f"  File 1: {bvh1.filename} - {bvh1.frame_count} frames, {bvh1.fps:.2f} FPS")
    
    # Handle --list-joints
    if args.list_joints:
        list_available_joints(bvh1)
        return 0
    
    # Parse comparison file if provided
    bvh2 = None
    if args.compare:
        try:
            bvh2 = BVHParser(args.compare).parse()
            if not args.quiet:
                print(f"  File 2: {bvh2.filename} - {bvh2.frame_count} frames, {bvh2.fps:.2f} FPS")
        except FileNotFoundError:
            print(f"Error: Comparison file not found: {args.compare}")
            return 1
        except Exception as e:
            print(f"Error parsing {args.compare}: {e}")
            return 1
    
    # Determine which joints to analyze
    if args.joints:
        target_joints = find_matching_joints(bvh1, args.joints)
        if not target_joints:
            print(f"Warning: No joints matched patterns: {args.joints}")
            print("Use --list-joints to see available joints")
            return 1
        if not args.quiet:
            print(f"\nSelected joints: {', '.join(target_joints)}")
    else:
        # Default to first few analysis joints
        analysis_joints = get_analysis_joints(bvh1)
        target_joints = list(analysis_joints.keys())[:3] if analysis_joints else []
    
    # Get smoothing parameters
    smoothing_method = None if args.smoothing == 'none' else args.smoothing
    smoothing_params = get_smoothing_params(args)
    
    # Report-only mode
    if args.report_only:
        report = generate_analysis_report(bvh1, bvh2)
        print(report)
        if not args.no_save:
            os.makedirs(args.output_dir, exist_ok=True)
            report_path = os.path.join(args.output_dir, 'analysis_report.txt')
            with open(report_path, 'w') as f:
                f.write(report)
            if not args.quiet:
                print(f"\n✓ Report saved to {report_path}")
        return 0
    
    # Create output directory
    if not args.no_save:
        os.makedirs(args.output_dir, exist_ok=True)
    
    # Create visualizer
    viz = BVHVisualizer(bvh1, bvh2)
    
    # Determine which visualizations to generate
    viz_types = set(args.viz)
    if 'all' in viz_types:
        viz_types = {'hierarchy', 'overview', 'statistics', 'smoothing', 
                     'velocity', 'rotations', 'heatmap', 'report'}
    
    if not args.quiet:
        print("\nGenerating visualizations...")
    
    generated_files = []
    
    def get_save_path(filename):
        if args.no_save:
            return None
        return os.path.join(args.output_dir, filename)
    
    # 1. Joint hierarchy
    if 'hierarchy' in viz_types:
        if not args.quiet:
            print("  - Joint hierarchy tree...")
        save_path = get_save_path('01_hierarchy_tree.png')
        viz.plot_hierarchy_tree(save_path)
        if save_path:
            generated_files.append(save_path)
        if args.no_display:
            plt.close()
    
    # 2. All joints overview
    if 'overview' in viz_types:
        if not args.quiet:
            print("  - All joints overview...")
        save_path = get_save_path('02_all_joints_overview.png')
        viz.plot_all_joints_overview(save_path)
        if save_path:
            generated_files.append(save_path)
        if args.no_display:
            plt.close()
    
    # 3. Statistics summary
    if 'statistics' in viz_types:
        if not args.quiet:
            print("  - Statistics summary...")
        save_path = get_save_path('03_statistics_summary.png')
        viz.plot_statistics_summary(save_path)
        if save_path:
            generated_files.append(save_path)
        if args.no_display:
            plt.close()
    
    # 4. Smoothing comparison
    if 'smoothing' in viz_types and target_joints:
        if not args.quiet:
            print("  - Smoothing comparison...")
        for i, joint in enumerate(target_joints[:2]):  # Limit to first 2 joints
            save_path = get_save_path(f'04_smoothing_comparison_{i+1}_{joint.split("_")[-1]}.png')
            try:
                viz.plot_smoothing_comparison(joint, args.channel, save_path)
                if save_path:
                    generated_files.append(save_path)
            except ValueError as e:
                if not args.quiet:
                    print(f"    Warning: {e}")
            if args.no_display:
                plt.close()
    
    # 5. Velocity and acceleration analysis
    if 'velocity' in viz_types and target_joints:
        if not args.quiet:
            print("  - Velocity/acceleration analysis...")
        for i, joint in enumerate(target_joints[:2]):  # Limit to first 2 joints
            save_path = get_save_path(f'05_velocity_acceleration_{i+1}_{joint.split("_")[-1]}.png')
            try:
                viz.plot_velocity_acceleration(joint, args.channel,
                                               apply_smoothing=(smoothing_method is not None),
                                               save_path=save_path)
                if save_path:
                    generated_files.append(save_path)
            except ValueError as e:
                if not args.quiet:
                    print(f"    Warning: {e}")
            if args.no_display:
                plt.close()
    
    # 6. Joint rotation comparison
    if 'rotations' in viz_types and target_joints:
        if not args.quiet:
            print("  - Joint rotation comparison...")
        for i, joint in enumerate(target_joints[:3]):  # Limit to first 3 joints
            save_path = get_save_path(f'06_joint_rotations_{i+1}_{joint.split("_")[-1]}.png')
            try:
                viz.plot_joint_rotations(joint,
                                        apply_smoothing=smoothing_method,
                                        smoothing_params=smoothing_params,
                                        save_path=save_path)
                if save_path:
                    generated_files.append(save_path)
            except ValueError as e:
                if not args.quiet:
                    print(f"    Warning: {e}")
            if args.no_display:
                plt.close()
    
    # 7. Heatmap comparison
    if 'heatmap' in viz_types and bvh2:
        if not args.quiet:
            print("  - Heatmap comparison...")
        save_path = get_save_path('07_heatmap_comparison.png')
        try:
            viz.plot_joint_comparison_heatmap(save_path)
            if save_path:
                generated_files.append(save_path)
        except ValueError as e:
            if not args.quiet:
                print(f"    Warning: {e}")
        if args.no_display:
            plt.close()
    elif 'heatmap' in viz_types and not bvh2:
        if not args.quiet:
            print("  - Skipping heatmap (requires --compare)")
    
    # 8. Generate text report
    if 'report' in viz_types:
        if not args.quiet:
            print("  - Analysis report...")
        report = generate_analysis_report(bvh1, bvh2)
        if not args.no_save:
            report_path = os.path.join(args.output_dir, 'analysis_report.txt')
            with open(report_path, 'w') as f:
                f.write(report)
            generated_files.append(report_path)
        if not args.quiet:
            print(report)
    
    # Summary
    if not args.quiet and generated_files:
        print(f"\n✓ All outputs saved to {args.output_dir}/")
        print("\nGenerated files:")
        for f in sorted(generated_files):
            print(f"  - {os.path.basename(f)}")
    
    # Show plots if not suppressed
    if not args.no_display and not args.no_save:
        plt.show()
    
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main() or 0)