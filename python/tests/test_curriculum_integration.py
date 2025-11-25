"""
test_curriculum_integration.py - Test curriculum learning integration

This script tests the curriculum learning components without requiring
the full C++ MASS backend. Run this to verify the logic before training.

Usage:
    python test_curriculum_integration.py
"""

import numpy as np
import sys
import os

# Add the python directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from curriculum_manager import CurriculumManager, MotionPerformanceTracker


def test_performance_tracker():
    """Test individual motion performance tracker"""
    print("\n" + "="*60)
    print("Testing MotionPerformanceTracker")
    print("="*60)
    
    tracker = MotionPerformanceTracker(name="walk", window_size=10)
    
    # Simulate some returns
    returns = [0.5, 0.6, 0.55, 0.7, 0.65, 0.75, 0.8, 0.72, 0.78, 0.85]
    for r in returns:
        tracker.update([r])
    
    print(f"Motion: {tracker.name}")
    print(f"Total episodes: {tracker.total_episodes}")
    print(f"Average return: {tracker.avg_return:.3f}")
    print(f"Baseline return: {tracker.baseline_return:.3f}")
    print(f"Best return: {tracker.best_return:.3f}")
    print(f"Improvement: {tracker.improvement:.3f}")
    print(f"Learning potential: {tracker.learning_potential:.3f}")
    
    assert tracker.total_episodes == 10
    assert tracker.best_return == 0.85
    assert tracker.baseline_return == 0.5
    print("✓ MotionPerformanceTracker tests passed!")


def test_curriculum_weight_adjustment():
    """Test that curriculum properly adjusts weights"""
    print("\n" + "="*60)
    print("Testing Curriculum Weight Adjustment")
    print("="*60)
    
    motion_names = ['walk', 'run', 'jump']
    
    curriculum = CurriculumManager(
        motion_names=motion_names,
        strategy='performance',
        warmup_epochs=5,
        update_frequency=2,
        min_weight=0.1,
        max_weight=3.0,
        temperature=1.0
    )
    
    # Simulate training where:
    # - 'walk' is easy (high returns)
    # - 'run' is medium
    # - 'jump' is hard (low returns)
    
    print("\nSimulating 30 epochs of training...")
    for epoch in range(30):
        # Simulate different performance levels
        motion_returns = {
            'walk': [0.8 + np.random.randn() * 0.1 for _ in range(5)],
            'run': [0.5 + np.random.randn() * 0.1 for _ in range(5)],
            'jump': [0.3 + np.random.randn() * 0.1 for _ in range(5)],
        }
        
        curriculum.update(motion_returns)
        
        if epoch % 10 == 0:
            weights = curriculum.get_weights()
            print(f"\nEpoch {epoch}:")
            print(f"  Weights: walk={weights['walk']:.3f}, run={weights['run']:.3f}, jump={weights['jump']:.3f}")
    
    # After training, jump should have highest weight (it's hardest)
    final_weights = curriculum.get_weights()
    print(f"\nFinal weights: {final_weights}")
    
    # The hardest motion (jump) should get more weight
    assert final_weights['jump'] > final_weights['walk'], \
        f"Expected jump weight > walk weight, got {final_weights}"
    
    # All weights should be within bounds
    for name, weight in final_weights.items():
        assert curriculum.min_weight <= weight <= curriculum.max_weight, \
            f"Weight {weight} for {name} out of bounds"
    
    print("✓ Curriculum weight adjustment tests passed!")


def test_curriculum_strategies():
    """Test different curriculum strategies"""
    print("\n" + "="*60)
    print("Testing Different Curriculum Strategies")
    print("="*60)
    
    motion_names = ['walk', 'run', 'jump']
    strategies = ['uniform', 'performance', 'progress', 'balanced', 'ucb']
    
    for strategy in strategies:
        print(f"\nTesting strategy: {strategy}")
        
        curriculum = CurriculumManager(
            motion_names=motion_names,
            strategy=strategy,
            warmup_epochs=3,
            update_frequency=1
        )
        
        # Run a few epochs
        for _ in range(10):
            motion_returns = {
                'walk': [0.8, 0.82, 0.79],
                'run': [0.5, 0.52, 0.48],
                'jump': [0.3, 0.28, 0.32],
            }
            curriculum.update(motion_returns)
        
        weights = curriculum.get_weights()
        print(f"  Weights: walk={weights['walk']:.3f}, run={weights['run']:.3f}, jump={weights['jump']:.3f}")
        
        # Verify weights are valid
        for w in weights.values():
            assert 0 < w < 10, f"Invalid weight: {w}"
    
    print("\n✓ All curriculum strategy tests passed!")


def test_warmup_period():
    """Test that warmup period uses uniform weights"""
    print("\n" + "="*60)
    print("Testing Warmup Period")
    print("="*60)
    
    curriculum = CurriculumManager(
        motion_names=['walk', 'run', 'jump'],
        strategy='performance',
        warmup_epochs=10,
        update_frequency=1
    )
    
    # During warmup, weights should stay uniform
    for epoch in range(10):
        motion_returns = {
            'walk': [0.9],  # Very different returns
            'run': [0.5],
            'jump': [0.1],
        }
        curriculum.update(motion_returns)
        
        weights = curriculum.get_weights()
        
        # During warmup, all weights should be 1.0
        assert all(w == 1.0 for w in weights.values()), \
            f"Epoch {epoch}: Expected uniform weights during warmup, got {weights}"
    
    print("  Warmup period maintained uniform weights ✓")
    
    # After warmup, weights should start adjusting
    for _ in range(5):
        curriculum.update({
            'walk': [0.9],
            'run': [0.5],
            'jump': [0.1],
        })
    
    post_warmup_weights = curriculum.get_weights()
    assert not all(w == 1.0 for w in post_warmup_weights.values()), \
        f"Expected non-uniform weights after warmup, got {post_warmup_weights}"
    
    print(f"  Post-warmup weights: {post_warmup_weights}")
    print("✓ Warmup period tests passed!")


def test_stats_tracking():
    """Test that statistics are properly tracked"""
    print("\n" + "="*60)
    print("Testing Statistics Tracking")
    print("="*60)
    
    curriculum = CurriculumManager(
        motion_names=['walk', 'run'],
        strategy='balanced',
        warmup_epochs=0,
        update_frequency=1
    )
    
    # Add some data
    for i in range(5):
        curriculum.update({
            'walk': [0.5 + i*0.1],  # Improving
            'run': [0.5],           # Constant
        })
    
    stats = curriculum.get_stats()
    
    print(f"\nWalk stats: {stats['walk']}")
    print(f"Run stats: {stats['run']}")
    
    assert stats['walk']['total_episodes'] == 5
    assert stats['run']['total_episodes'] == 5
    assert stats['walk']['improvement'] > 0, "Walk should show improvement"
    
    print("✓ Statistics tracking tests passed!")


def simulate_multimodal_training():
    """Simulate a complete multimodal training scenario"""
    print("\n" + "="*60)
    print("Simulating Complete Multimodal Training Scenario")
    print("="*60)
    
    motion_names = ['balance', 'dance', 'kick', 'run', 'walk', 'walk_fullbody']
    
    # Different difficulty levels for each motion
    motion_difficulty = {
        'balance': 0.4,    # Hard
        'dance': 0.35,     # Hardest
        'kick': 0.5,       # Medium-hard
        'run': 0.6,        # Medium
        'walk': 0.75,      # Easy
        'walk_fullbody': 0.7  # Easy-medium
    }
    
    curriculum = CurriculumManager(
        motion_names=motion_names,
        strategy='balanced',
        warmup_epochs=20,
        update_frequency=5,
        temperature=1.0
    )
    
    print(f"\nMotion difficulty levels: {motion_difficulty}")
    print("\nSimulating 100 training epochs...\n")
    
    # Simulate training
    for epoch in range(100):
        # Simulate returns based on difficulty + learning progress
        learning_factor = min(1.0, epoch / 50.0) * 0.2  # Slow improvement
        
        motion_returns = {}
        for name, difficulty in motion_difficulty.items():
            # Base return from difficulty + some learning + noise
            base_return = difficulty + learning_factor
            returns = [base_return + np.random.randn() * 0.1 for _ in range(3)]
            motion_returns[name] = returns
        
        curriculum.update(motion_returns)
        
        if epoch % 25 == 0:
            weights = curriculum.get_weights()
            print(f"Epoch {epoch}:")
            for name in sorted(motion_names):
                w = weights[name]
                print(f"  {name:15} weight={w:.3f}")
            print()
    
    # Final analysis
    print("="*60)
    print("Final Curriculum State")
    print("="*60)
    curriculum.print_summary()
    
    # Verify that harder motions get higher weights
    final_weights = curriculum.get_weights()
    
    # dance is hardest, walk is easiest
    if final_weights['dance'] > final_weights['walk']:
        print("✓ Harder motion (dance) has higher weight than easier motion (walk)")
    else:
        print(f"! Warning: Expected dance weight > walk weight")
        print(f"  dance: {final_weights['dance']:.3f}, walk: {final_weights['walk']:.3f}")
    
    print("\n✓ Multimodal training simulation complete!")


def main():
    """Run all tests"""
    print("="*60)
    print("Curriculum Learning Integration Tests")
    print("="*60)
    
    test_performance_tracker()
    test_curriculum_weight_adjustment()
    test_curriculum_strategies()
    test_warmup_period()
    test_stats_tracking()
    simulate_multimodal_training()
    
    print("\n" + "="*60)
    print("ALL TESTS PASSED!")
    print("="*60)
    print("\nThe curriculum learning module is ready for integration.")
    print("You can now run the full multimodal training with:")
    print("  python main_multimodal.py --motion_list data/motion_list.txt \\")
    print("         --template data/metadata.txt --curriculum balanced")


if __name__ == "__main__":
    main()