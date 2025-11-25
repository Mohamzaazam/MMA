#!/usr/bin/env python3
"""
Headless Model Evaluation
==========================
This script evaluates trained models without visualization.
Useful for comparing performance of different models and motions.

Run from project root:
    pixi shell
    
    # Evaluate single-motion model
    python python/evaluate_models.py --mode single
    
    # Evaluate multi-modal model on all motions
    python python/evaluate_models.py --mode multi
    
    # Evaluate multi-modal model on specific motion
    python python/evaluate_models.py --mode multi --motion walk
    
    # Compare all available models
    python python/evaluate_models.py --mode compare
"""

import os
import sys
import time
import argparse
import numpy as np
from collections import defaultdict

sys.path.insert(0, 'build')
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class ModelEvaluator:
    """Evaluator for trained MASS models"""
    
    def __init__(self):
        self.results = {}
        
    def load_models(self, sim_path, muscle_path):
        """Load neural network models"""
        import torch
        from Model import SimulationNN, MuscleNN, Tensor
        
        # We need to get dimensions from an environment first
        # So we load them lazily
        self.sim_path = sim_path
        self.muscle_path = muscle_path
        self.models_loaded = False
        
    def _ensure_models_loaded(self, num_state, num_action, num_dofs, num_muscles):
        """Actually load models once we know dimensions"""
        if self.models_loaded:
            return
            
        from Model import SimulationNN, MuscleNN
        
        self.sim_nn = SimulationNN(num_state, num_action)
        self.muscle_nn = MuscleNN(num_dofs, num_action, num_muscles)
        
        self.sim_nn.load(self.sim_path)
        self.muscle_nn.load(self.muscle_path)
        
        self.models_loaded = True
        
    def evaluate_single(self, metadata_path, num_episodes=10, max_steps=300, 
                        num_slaves=4, verbose=True):
        """Evaluate model on single-motion environment"""
        import pymss
        from Model import Tensor
        import torch
        
        if verbose:
            print(f"\n--- Evaluating on: {metadata_path} ---")
        
        # Create environment
        env = pymss.pymss(metadata_path, num_slaves)
        
        # Load models with correct dimensions
        num_state = env.GetNumState()
        num_action = env.GetNumAction()
        num_dofs = env.GetNumTotalMuscleRelatedDofs()
        num_muscles = env.GetNumMuscles()
        use_muscle = env.UseMuscle()
        num_steps_per_control = env.GetNumSteps()
        
        self._ensure_models_loaded(num_state, num_action, num_dofs, num_muscles)
        
        # Run evaluation episodes
        episode_returns = []
        episode_lengths = []
        
        for ep in range(num_episodes):
            env.Resets(True)
            
            total_rewards = np.zeros(num_slaves)
            steps = 0
            active = np.ones(num_slaves, dtype=bool)
            
            while steps < max_steps and np.any(active):
                # Get states
                states = env.GetStates()
                
                # Policy inference
                with torch.no_grad():
                    state_tensor = Tensor(states.astype(np.float32))
                    action_dist, _ = self.sim_nn(state_tensor)
                    actions = action_dist.loc.cpu().numpy()  # Deterministic
                
                env.SetActions(actions)
                
                # Muscle inference
                if use_muscle:
                    mt = env.GetMuscleTorques()
                    dt = env.GetDesiredTorques()
                    with torch.no_grad():
                        activations = self.muscle_nn(
                            Tensor(mt.astype(np.float32)),
                            Tensor(dt.astype(np.float32))
                        ).cpu().numpy()
                    env.SetActivationLevels(activations)
                
                env.StepsAtOnce()
                steps += 1
                
                # Get rewards
                rewards = env.GetRewards()
                total_rewards += rewards * active
                
                # Check termination
                for j in range(num_slaves):
                    if active[j] and env.IsEndOfEpisode(j):
                        episode_returns.append(total_rewards[j])
                        episode_lengths.append(steps)
                        active[j] = False
                        
                        if len(episode_returns) >= num_episodes:
                            break
                
                if len(episode_returns) >= num_episodes:
                    break
            
            # Handle any remaining active episodes
            for j in range(num_slaves):
                if active[j]:
                    episode_returns.append(total_rewards[j])
                    episode_lengths.append(steps)
            
            if len(episode_returns) >= num_episodes:
                break
        
        # Compute statistics
        returns = np.array(episode_returns[:num_episodes])
        lengths = np.array(episode_lengths[:num_episodes])
        
        result = {
            'mean_return': np.mean(returns),
            'std_return': np.std(returns),
            'min_return': np.min(returns),
            'max_return': np.max(returns),
            'mean_length': np.mean(lengths),
            'std_length': np.std(lengths),
            'num_episodes': len(returns)
        }
        
        if verbose:
            print(f"   Episodes: {result['num_episodes']}")
            print(f"   Mean Return: {result['mean_return']:.3f} ± {result['std_return']:.3f}")
            print(f"   Return Range: [{result['min_return']:.3f}, {result['max_return']:.3f}]")
            print(f"   Mean Length: {result['mean_length']:.1f} ± {result['std_length']:.1f}")
        
        return result
    
    def evaluate_multimodal(self, motion_list_path, template_path, 
                            num_episodes_per_motion=5, max_steps=300,
                            num_slaves_per_motion=2, verbose=True):
        """Evaluate multimodal model on all motions"""
        from multimodal_env import load_motion_configs_from_list
        
        # Load motion configs
        configs = load_motion_configs_from_list(motion_list_path, template_path)
        
        if verbose:
            print(f"\n--- Evaluating Multimodal on {len(configs)} motions ---")
        
        all_results = {}
        
        for config in configs:
            if verbose:
                print(f"\n  Motion: {config.name}")
            
            # Reset models_loaded to reload with potentially different dimensions
            # (In practice, all motions should have same skeleton)
            
            result = self.evaluate_single(
                config.metadata_path,
                num_episodes=num_episodes_per_motion,
                max_steps=max_steps,
                num_slaves=num_slaves_per_motion,
                verbose=verbose
            )
            all_results[config.name] = result
        
        # Aggregate
        all_returns = []
        for motion_name, result in all_results.items():
            all_returns.append(result['mean_return'])
        
        aggregate = {
            'per_motion': all_results,
            'overall_mean': np.mean(all_returns),
            'overall_std': np.std(all_returns),
            'best_motion': max(all_results, key=lambda k: all_results[k]['mean_return']),
            'worst_motion': min(all_results, key=lambda k: all_results[k]['mean_return']),
        }
        
        if verbose:
            print(f"\n--- Aggregate Results ---")
            print(f"   Overall Mean: {aggregate['overall_mean']:.3f} ± {aggregate['overall_std']:.3f}")
            print(f"   Best Motion: {aggregate['best_motion']} ({all_results[aggregate['best_motion']]['mean_return']:.3f})")
            print(f"   Worst Motion: {aggregate['worst_motion']} ({all_results[aggregate['worst_motion']]['mean_return']:.3f})")
        
        return aggregate


def find_models():
    """Find available models"""
    models = {
        'single': [],
        'multimodal': []
    }
    
    # Single motion
    for name, sim, muscle in [
        ('best', 'nn/max.pt', 'nn/max_muscle.pt'),
        ('current', 'nn/current.pt', 'nn/current_muscle.pt'),
    ]:
        if os.path.exists(sim) and os.path.exists(muscle):
            models['single'].append({
                'name': name,
                'sim_path': sim,
                'muscle_path': muscle
            })
    
    # Multimodal
    for name, sim, muscle in [
        ('best', 'nn/multimodal_max.pt', 'nn/multimodal_max_muscle.pt'),
        ('current', 'nn/multimodal_current.pt', 'nn/multimodal_current_muscle.pt'),
    ]:
        if os.path.exists(sim) and os.path.exists(muscle):
            models['multimodal'].append({
                'name': name,
                'sim_path': sim,
                'muscle_path': muscle
            })
    
    return models


def evaluate_single_mode(args):
    """Evaluate single-motion models"""
    print("\n" + "=" * 60)
    print("SINGLE-MOTION MODEL EVALUATION")
    print("=" * 60)
    
    models = find_models()
    
    if not models['single']:
        print("❌ No single-motion models found")
        print("   Train with: pixi run train")
        return
    
    evaluator = ModelEvaluator()
    results = {}
    
    for model in models['single']:
        print(f"\n📊 Evaluating: {model['name']}")
        evaluator.load_models(model['sim_path'], model['muscle_path'])
        
        result = evaluator.evaluate_single(
            args.metadata,
            num_episodes=args.episodes,
            max_steps=args.max_steps,
            num_slaves=args.slaves
        )
        results[model['name']] = result
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, result in results.items():
        print(f"  {name}: {result['mean_return']:.3f} ± {result['std_return']:.3f}")


def evaluate_multimodal_mode(args):
    """Evaluate multimodal models"""
    print("\n" + "=" * 60)
    print("MULTI-MODAL MODEL EVALUATION")
    print("=" * 60)
    
    models = find_models()
    
    if not models['multimodal']:
        print("❌ No multimodal models found")
        print("   Train with: pixi run train_multimodal")
        return
    
    evaluator = ModelEvaluator()
    
    for model in models['multimodal']:
        print(f"\n📊 Evaluating: {model['name']}")
        evaluator.load_models(model['sim_path'], model['muscle_path'])
        
        if args.motion:
            # Evaluate on specific motion
            from multimodal_env import load_motion_configs_from_list
            configs = load_motion_configs_from_list(args.motion_list, args.template)
            config = next((c for c in configs if c.name == args.motion), None)
            
            if config:
                result = evaluator.evaluate_single(
                    config.metadata_path,
                    num_episodes=args.episodes,
                    max_steps=args.max_steps,
                    num_slaves=args.slaves
                )
            else:
                print(f"❌ Motion '{args.motion}' not found")
        else:
            # Evaluate on all motions
            result = evaluator.evaluate_multimodal(
                args.motion_list,
                args.template,
                num_episodes_per_motion=args.episodes,
                max_steps=args.max_steps,
                num_slaves_per_motion=args.slaves
            )


def compare_models(args):
    """Compare single vs multimodal models"""
    print("\n" + "=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)
    
    models = find_models()
    
    # First, evaluate single models on the default motion
    print("\n--- Single-Motion Models (on default motion) ---")
    single_results = {}
    
    if models['single']:
        evaluator = ModelEvaluator()
        for model in models['single']:
            evaluator.load_models(model['sim_path'], model['muscle_path'])
            result = evaluator.evaluate_single(
                args.metadata,
                num_episodes=args.episodes,
                max_steps=args.max_steps,
                num_slaves=args.slaves,
                verbose=False
            )
            single_results[f"single_{model['name']}"] = result
            print(f"  {model['name']}: {result['mean_return']:.3f} ± {result['std_return']:.3f}")
    else:
        print("  No single-motion models found")
    
    # Then evaluate multimodal models
    print("\n--- Multi-Modal Models (averaged across motions) ---")
    multi_results = {}
    
    if models['multimodal']:
        evaluator = ModelEvaluator()
        for model in models['multimodal']:
            evaluator.load_models(model['sim_path'], model['muscle_path'])
            result = evaluator.evaluate_multimodal(
                args.motion_list,
                args.template,
                num_episodes_per_motion=args.episodes,
                max_steps=args.max_steps,
                num_slaves_per_motion=args.slaves,
                verbose=False
            )
            multi_results[f"multi_{model['name']}"] = result
            print(f"  {model['name']}: {result['overall_mean']:.3f} ± {result['overall_std']:.3f}")
    else:
        print("  No multimodal models found")
    
    # Final summary
    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)
    
    all_results = []
    for name, result in single_results.items():
        all_results.append((name, result['mean_return'], result['std_return']))
    for name, result in multi_results.items():
        all_results.append((name, result['overall_mean'], result['overall_std']))
    
    all_results.sort(key=lambda x: x[1], reverse=True)
    
    for i, (name, mean, std) in enumerate(all_results):
        marker = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else "  "
        print(f"  {marker} {name}: {mean:.3f} ± {std:.3f}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate MASS Models')
    parser.add_argument('--mode', choices=['single', 'multi', 'compare'],
                        default='compare',
                        help='Evaluation mode')
    parser.add_argument('--metadata', default='data/metadata.txt',
                        help='Single-motion metadata file')
    parser.add_argument('--motion_list', default='data/motion_list.txt',
                        help='Motion list for multimodal')
    parser.add_argument('--template', default='data/metadata.txt',
                        help='Template metadata file')
    parser.add_argument('--motion', type=str, default=None,
                        help='Specific motion to evaluate')
    parser.add_argument('--episodes', type=int, default=5,
                        help='Episodes per evaluation')
    parser.add_argument('--max_steps', type=int, default=300,
                        help='Max steps per episode')
    parser.add_argument('--slaves', type=int, default=4,
                        help='Parallel environments')
    args = parser.parse_args()
    
    print("\n" + "=" * 60)
    print("MASS MODEL EVALUATION")
    print("=" * 60)
    print(f"Mode: {args.mode}")
    print(f"Episodes: {args.episodes}")
    print(f"Max steps: {args.max_steps}")
    
    if args.mode == 'single':
        evaluate_single_mode(args)
    elif args.mode == 'multi':
        evaluate_multimodal_mode(args)
    else:
        compare_models(args)
    
    print("\n✅ Evaluation complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
