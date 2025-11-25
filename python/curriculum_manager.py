"""
curriculum_manager.py - Adaptive Curriculum Learning for Multimodal Motion Training

This module implements curriculum learning strategies that adaptively adjust
motion sampling weights based on training performance.

Key Strategies:
1. Performance-based: Sample more from motions with lower rewards (harder motions)
2. Progress-based: Sample more from motions showing improvement
3. Balanced: Ensure all motions get minimum coverage

Usage:
    curriculum = CurriculumManager(motion_names, strategy='performance')
    
    # During training loop:
    curriculum.update(motion_returns_dict)
    new_weights = curriculum.get_weights()
    env.set_motion_weights(new_weights)
"""

import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from collections import deque
import json


@dataclass
class MotionPerformanceTracker:
    """Tracks performance history for a single motion"""
    name: str
    window_size: int = 50  # Rolling window for averaging
    
    # Performance metrics
    recent_returns: deque = field(default_factory=lambda: deque(maxlen=50))
    recent_steps: deque = field(default_factory=lambda: deque(maxlen=50))
    
    # Cumulative stats
    total_episodes: int = 0
    total_return: float = 0.0
    
    # Progress tracking (for progress-based curriculum)
    baseline_return: Optional[float] = None
    best_return: float = float('-inf')
    
    def __post_init__(self):
        self.recent_returns = deque(maxlen=self.window_size)
        self.recent_steps = deque(maxlen=self.window_size)
    
    def update(self, returns: List[float], steps: Optional[List[int]] = None):
        """Update with new episode returns"""
        for r in returns:
            self.recent_returns.append(r)
            self.total_return += r
            self.total_episodes += 1
            
            if self.baseline_return is None:
                self.baseline_return = r
            self.best_return = max(self.best_return, r)
        
        if steps:
            for s in steps:
                self.recent_steps.append(s)
    
    @property
    def avg_return(self) -> float:
        """Average return over recent window"""
        if not self.recent_returns:
            return 0.0
        return np.mean(self.recent_returns)
    
    @property
    def return_std(self) -> float:
        """Standard deviation of recent returns"""
        if len(self.recent_returns) < 2:
            return 1.0
        return np.std(self.recent_returns) + 1e-6
    
    @property
    def improvement(self) -> float:
        """How much this motion has improved from baseline"""
        if self.baseline_return is None or self.baseline_return == 0:
            return 0.0
        return (self.avg_return - self.baseline_return) / abs(self.baseline_return)
    
    @property
    def learning_potential(self) -> float:
        """Estimate of how much more this motion could improve"""
        # Gap between current performance and best seen
        if self.best_return == float('-inf'):
            return 1.0
        gap = self.best_return - self.avg_return
        return max(0.0, gap / (abs(self.best_return) + 1e-6))


class CurriculumManager:
    """
    Manages adaptive curriculum learning for multimodal motion training.
    
    Strategies:
        - 'uniform': Equal weights for all motions (baseline)
        - 'performance': Higher weights for lower-performing motions
        - 'progress': Higher weights for motions showing less improvement
        - 'balanced': Combination of performance + minimum coverage guarantee
        - 'ucb': Upper Confidence Bound - balance exploration vs exploitation
    """
    
    def __init__(self, 
                 motion_names: List[str],
                 strategy: str = 'balanced',
                 min_weight: float = 0.1,
                 max_weight: float = 3.0,
                 temperature: float = 1.0,
                 update_frequency: int = 10,
                 warmup_epochs: int = 20):
        """
        Initialize curriculum manager.
        
        Args:
            motion_names: List of motion names to track
            strategy: Curriculum strategy ('uniform', 'performance', 'progress', 'balanced', 'ucb')
            min_weight: Minimum weight for any motion (ensures coverage)
            max_weight: Maximum weight for any motion (prevents over-focusing)
            temperature: Controls how aggressive weight adjustments are (higher = more uniform)
            update_frequency: How often to update weights (in epochs)
            warmup_epochs: Number of epochs before starting curriculum (use uniform during warmup)
        """
        self.motion_names = motion_names
        self.strategy = strategy
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.temperature = temperature
        self.update_frequency = update_frequency
        self.warmup_epochs = warmup_epochs
        
        # Initialize trackers
        self.trackers: Dict[str, MotionPerformanceTracker] = {
            name: MotionPerformanceTracker(name=name)
            for name in motion_names
        }
        
        # Current weights
        self.weights: Dict[str, float] = {name: 1.0 for name in motion_names}
        
        # Tracking
        self.epoch = 0
        self.weight_history: List[Dict[str, float]] = []
        
    def update(self, motion_returns: Dict[str, List[float]], 
               motion_steps: Optional[Dict[str, List[int]]] = None):
        """
        Update trackers with new episode data and potentially adjust weights.
        
        Args:
            motion_returns: Dict mapping motion name to list of episode returns
            motion_steps: Optional dict mapping motion name to list of episode steps
        """
        # Update trackers
        for name, returns in motion_returns.items():
            if name in self.trackers and returns:
                steps = motion_steps.get(name) if motion_steps else None
                self.trackers[name].update(returns, steps)
        
        self.epoch += 1
        
        # Check if we should update weights
        if self.epoch <= self.warmup_epochs:
            # During warmup, use uniform weights
            return
        
        if self.epoch % self.update_frequency == 0:
            self._update_weights()
    
    def _update_weights(self):
        """Update weights based on current strategy"""
        if self.strategy == 'uniform':
            weights = self._uniform_weights()
        elif self.strategy == 'performance':
            weights = self._performance_weights()
        elif self.strategy == 'progress':
            weights = self._progress_weights()
        elif self.strategy == 'balanced':
            weights = self._balanced_weights()
        elif self.strategy == 'ucb':
            weights = self._ucb_weights()
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
        
        # Apply min/max bounds
        for name in weights:
            weights[name] = np.clip(weights[name], self.min_weight, self.max_weight)
        
        self.weights = weights
        self.weight_history.append(weights.copy())
    
    def _uniform_weights(self) -> Dict[str, float]:
        """Uniform weights - baseline"""
        return {name: 1.0 for name in self.motion_names}
    
    def _performance_weights(self) -> Dict[str, float]:
        """
        Performance-based weights: harder motions (lower returns) get higher weights.
        
        Uses softmax with temperature to convert returns to weights.
        """
        returns = []
        for name in self.motion_names:
            avg = self.trackers[name].avg_return
            returns.append(avg if avg != 0 else 0.0)
        
        returns = np.array(returns)
        
        # If all returns are 0, use uniform
        if np.all(returns == 0):
            return {name: 1.0 for name in self.motion_names}
        
        # Invert returns (lower return = higher weight)
        # Normalize to prevent numerical issues
        returns_norm = (returns - returns.mean()) / (returns.std() + 1e-6)
        
        # Softmax with negative (to invert - lower performance = higher weight)
        weights = np.exp(-returns_norm / self.temperature)
        weights = weights / weights.sum() * len(self.motion_names)  # Scale to sum to N
        
        return {name: w for name, w in zip(self.motion_names, weights)}
    
    def _progress_weights(self) -> Dict[str, float]:
        """
        Progress-based weights: motions showing less improvement get higher weights.
        """
        improvements = []
        for name in self.motion_names:
            improvements.append(self.trackers[name].improvement)
        
        improvements = np.array(improvements)
        
        # Invert improvements (less improvement = higher weight)
        improvements_norm = (improvements - improvements.mean()) / (improvements.std() + 1e-6)
        weights = np.exp(-improvements_norm / self.temperature)
        weights = weights / weights.sum() * len(self.motion_names)
        
        return {name: w for name, w in zip(self.motion_names, weights)}
    
    def _balanced_weights(self) -> Dict[str, float]:
        """
        Balanced strategy: combines performance-based with coverage guarantee.
        
        - 70% weight from performance (harder motions)
        - 30% from episode count (less-sampled motions)
        """
        perf_weights = self._performance_weights()
        
        # Count-based component
        counts = np.array([self.trackers[name].total_episodes for name in self.motion_names])
        if counts.sum() > 0:
            # Inverse count weighting (less episodes = higher weight)
            count_weights = 1.0 / (counts + 1)
            count_weights = count_weights / count_weights.sum() * len(self.motion_names)
        else:
            count_weights = np.ones(len(self.motion_names))
        
        # Combine
        combined = {}
        for i, name in enumerate(self.motion_names):
            combined[name] = 0.7 * perf_weights[name] + 0.3 * count_weights[i]
        
        return combined
    
    def _ucb_weights(self) -> Dict[str, float]:
        """
        Upper Confidence Bound strategy: balance exploitation (train on hard motions)
        with exploration (ensure all motions are tried).
        
        UCB = mean_reward + c * sqrt(log(total) / count)
        We invert this for weighting (lower UCB = higher weight)
        """
        total_episodes = sum(t.total_episodes for t in self.trackers.values()) + 1
        
        ucb_scores = []
        for name in self.motion_names:
            tracker = self.trackers[name]
            count = tracker.total_episodes + 1
            
            # UCB formula (inverted for weighting)
            exploration = np.sqrt(2 * np.log(total_episodes) / count)
            ucb = tracker.avg_return - self.temperature * exploration
            ucb_scores.append(ucb)
        
        ucb_scores = np.array(ucb_scores)
        
        # Lower UCB = higher weight (those we're less confident about)
        ucb_norm = (ucb_scores - ucb_scores.mean()) / (ucb_scores.std() + 1e-6)
        weights = np.exp(-ucb_norm / self.temperature)
        weights = weights / weights.sum() * len(self.motion_names)
        
        return {name: w for name, w in zip(self.motion_names, weights)}
    
    def get_weights(self) -> Dict[str, float]:
        """Get current motion weights"""
        return self.weights.copy()
    
    def get_stats(self) -> Dict[str, Dict]:
        """Get detailed statistics for all motions"""
        stats = {}
        for name, tracker in self.trackers.items():
            stats[name] = {
                'avg_return': tracker.avg_return,
                'return_std': tracker.return_std,
                'total_episodes': tracker.total_episodes,
                'improvement': tracker.improvement,
                'best_return': tracker.best_return,
                'current_weight': self.weights.get(name, 1.0)
            }
        return stats
    
    def print_summary(self):
        """Print curriculum summary"""
        print(f"\n{'='*60}")
        print(f"Curriculum Summary (Epoch {self.epoch}, Strategy: {self.strategy})")
        print(f"{'='*60}")
        print(f"{'Motion':<20} {'AvgRet':>8} {'Episodes':>10} {'Weight':>8}")
        print(f"{'-'*60}")
        
        for name in self.motion_names:
            tracker = self.trackers[name]
            weight = self.weights.get(name, 1.0)
            print(f"{name:<20} {tracker.avg_return:>8.3f} {tracker.total_episodes:>10} {weight:>8.3f}")
        
        print(f"{'='*60}\n")
    
    def save_history(self, path: str):
        """Save weight history to JSON"""
        with open(path, 'w') as f:
            json.dump({
                'strategy': self.strategy,
                'weight_history': self.weight_history,
                'stats': self.get_stats()
            }, f, indent=2)


# =============================================================================
# Testing utilities
# =============================================================================

def test_curriculum_manager():
    """Test curriculum manager with synthetic data"""
    print("Testing CurriculumManager...")
    
    motion_names = ['walk', 'run', 'jump']
    
    # Test each strategy
    for strategy in ['uniform', 'performance', 'progress', 'balanced', 'ucb']:
        print(f"\n--- Testing strategy: {strategy} ---")
        
        curriculum = CurriculumManager(
            motion_names=motion_names,
            strategy=strategy,
            warmup_epochs=5,
            update_frequency=2
        )
        
        # Simulate training
        for epoch in range(20):
            # Simulate different performance levels
            motion_returns = {
                'walk': [0.8 + np.random.randn() * 0.1 for _ in range(3)],  # Easy
                'run': [0.5 + np.random.randn() * 0.15 for _ in range(3)],  # Medium
                'jump': [0.3 + np.random.randn() * 0.2 for _ in range(3)],  # Hard
            }
            
            curriculum.update(motion_returns)
            
            if epoch % 5 == 0:
                weights = curriculum.get_weights()
                print(f"Epoch {epoch}: weights = {weights}")
        
        curriculum.print_summary()
        
        # Verify weights
        weights = curriculum.get_weights()
        assert all(w >= curriculum.min_weight for w in weights.values()), "Min weight violated"
        assert all(w <= curriculum.max_weight for w in weights.values()), "Max weight violated"
        
        if strategy == 'performance':
            # Jump should have highest weight (lowest return)
            assert weights['jump'] >= weights['walk'], \
                f"Performance strategy should weight harder motions more: {weights}"
    
    print("\n✓ All tests passed!")


if __name__ == "__main__":
    test_curriculum_manager()