#!/usr/bin/env python3
"""
MASS Test Suite - Main Runner
==============================
This script runs all tests in sequence, stopping at the first failure.

Run from project root:
    pixi shell
    python python/run_all_tests.py

Test progression:
    1. test_01_model_inference.py  - Test NN models without environment
    2. test_02_single_env.py       - Test single-motion environment
    3. test_03_multimodal_env.py   - Test multi-modal environment
    4. test_04_visualize.py        - Help with visualization
"""

import os
import sys
import subprocess
import time
from datetime import datetime


def run_test(script_name, description, stop_on_fail=True):
    """Run a test script and return success status"""
    print("\n" + "=" * 70)
    print(f"RUNNING: {description}")
    print(f"Script: {script_name}")
    print("=" * 70 + "\n")
    
    start_time = time.time()
    
    result = subprocess.run(
        [sys.executable, f'python/{script_name}'],
        capture_output=False
    )
    
    elapsed = time.time() - start_time
    
    if result.returncode == 0:
        print(f"\n✅ PASSED: {description} ({elapsed:.1f}s)")
        return True
    else:
        print(f"\n❌ FAILED: {description} (exit code: {result.returncode})")
        return False


def main():
    print("\n" + "=" * 70)
    print("MASS TEST SUITE")
    print("=" * 70)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Working directory: {os.getcwd()}")
    print()
    
    # Check we're in the right directory
    if not os.path.exists('build/pymss.cpython-312-x86_64-linux-gnu.so') and \
       not os.path.exists('build/pymss.so'):
        print("⚠️  Warning: pymss library not found in build/")
        print("   Make sure you've built the project: pixi run build")
        print()
    
    # Define test sequence
    tests = [
        ('test_01_model_inference.py', 'Step 1: Model Inference Test'),
        ('test_02_single_env.py', 'Step 2: Single Environment Test'),
        ('test_03_multimodal_env.py', 'Step 3: Multi-Modal Environment Test'),
    ]
    
    results = []
    all_passed = True
    
    for script_name, description in tests:
        if not os.path.exists(f'python/{script_name}'):
            print(f"\n⚠️  Test script not found: python/{script_name}")
            results.append((description, 'SKIPPED'))
            continue
        
        passed = run_test(script_name, description)
        results.append((description, 'PASSED' if passed else 'FAILED'))
        
        if not passed:
            all_passed = False
            print(f"\n🛑 Stopping test suite due to failure in: {description}")
            break
    
    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    
    for description, status in results:
        symbol = '✅' if status == 'PASSED' else ('❌' if status == 'FAILED' else '⚠️')
        print(f"  {symbol} {status}: {description}")
    
    if all_passed:
        print("\n" + "=" * 70)
        print("🎉 ALL TESTS PASSED!")
        print("=" * 70)
        print("\nNext steps:")
        print("  1. Train a model (if not already done):")
        print("     - Single motion: pixi run train")
        print("     - Multi-modal: pixi run train_multimodal")
        print("  2. Visualize:")
        print("     - python python/test_04_visualize.py --mode single")
        print("     - python python/test_04_visualize.py --mode multi --motion walk")
        return 0
    else:
        print("\n" + "=" * 70)
        print("❌ SOME TESTS FAILED")
        print("=" * 70)
        print("\nDebug steps:")
        print("  1. Check error messages above")
        print("  2. Make sure project is built: pixi run build")
        print("  3. Run individual tests for more details")
        return 1


if __name__ == "__main__":
    sys.exit(main())
