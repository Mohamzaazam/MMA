#!/usr/bin/env python3
"""
Verify BVH Conversion Integrity

Checks the converted BVH files for:
1. File existence and non-empty
2. Valid BVH structure (HIERARCHY, MOTION, ROOT sections)
3. Frame count sanity
4. Corresponding trials.txt metadata

Also copies trials.txt from source if missing.

Usage:
    python verify_conversion.py data/cmu --source /mnt/e/database/cmu/subjects
    python verify_conversion.py data/cmu --fix  # Copy missing trials.txt
"""

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass, asdict


@dataclass
class SubjectVerification:
    """Verification result for a single subject."""
    subject_id: str
    bvh_count: int
    valid_count: int
    invalid_files: List[str]
    has_trials_txt: bool
    total_frames: int


@dataclass
class VerificationReport:
    """Full verification report."""
    output_dir: str
    total_subjects: int
    total_files: int
    valid_files: int
    invalid_files: int
    subjects_with_trials: int
    subjects_missing_trials: List[str]
    total_frames: int
    invalid_details: List[Dict]


def validate_bvh_file(filepath: Path) -> Tuple[bool, str, int]:
    """
    Validate a BVH file.
    
    Checks:
    - File exists and non-empty
    - Has HIERARCHY, MOTION, ROOT sections
    - Frame count declared matches actual motion data lines
    
    Returns: (is_valid, error_message, frame_count)
    """
    try:
        if not filepath.exists():
            return False, "File does not exist", 0
        
        if filepath.stat().st_size == 0:
            return False, "File is empty", 0
        
        with open(filepath, 'r') as f:
            content = f.read()
        
        # Check required sections
        if 'HIERARCHY' not in content:
            return False, "Missing HIERARCHY section", 0
        if 'MOTION' not in content:
            return False, "Missing MOTION section", 0
        if 'ROOT' not in content:
            return False, "Missing ROOT joint", 0
        
        # Split at MOTION to get motion data
        parts = content.split('MOTION')
        if len(parts) != 2:
            return False, "Invalid MOTION section", 0
        
        motion_section = parts[1]
        lines = [l.strip() for l in motion_section.split('\n') if l.strip()]
        
        # Extract declared frame count
        frame_count = 0
        motion_start_idx = 0
        for i, line in enumerate(lines):
            if line.startswith('Frames:'):
                try:
                    frame_count = int(line.split(':')[1].strip())
                except:
                    return False, "Cannot parse frame count", 0
            elif line.startswith('Frame Time:'):
                motion_start_idx = i + 1
                break
        
        if frame_count == 0:
            return False, "Frame count is 0", 0
        
        # Count actual motion data lines
        actual_frames = len(lines) - motion_start_idx
        
        if actual_frames < frame_count:
            return False, f"Truncated: {actual_frames}/{frame_count} frames", actual_frames
        
        return True, "", frame_count
        
    except Exception as e:
        return False, f"Read error: {e}", 0


def verify_subject(subject_dir: Path) -> SubjectVerification:
    """Verify all BVH files for a subject."""
    subject_id = subject_dir.name
    bvh_files = list(subject_dir.glob("*.bvh"))
    
    valid_count = 0
    invalid_files = []
    total_frames = 0
    
    for bvh_file in bvh_files:
        is_valid, error, frames = validate_bvh_file(bvh_file)
        if is_valid:
            valid_count += 1
            total_frames += frames
        else:
            invalid_files.append(f"{bvh_file.name}: {error}")
    
    has_trials = (subject_dir / "trials.txt").exists()
    
    return SubjectVerification(
        subject_id=subject_id,
        bvh_count=len(bvh_files),
        valid_count=valid_count,
        invalid_files=invalid_files,
        has_trials_txt=has_trials,
        total_frames=total_frames,
    )


def copy_trials_metadata(
    output_dir: Path,
    source_dir: Path,
    subjects: List[str],
) -> int:
    """Copy trials.txt from source to output for specified subjects."""
    copied = 0
    
    for subject_id in subjects:
        src_trials = source_dir / subject_id / "trials.txt"
        dst_trials = output_dir / subject_id / "trials.txt"
        
        if src_trials.exists() and not dst_trials.exists():
            dst_trials.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_trials, dst_trials)
            copied += 1
            print(f"  Copied trials.txt for subject {subject_id}")
    
    return copied


def remove_subjects_without_trials(
    output_dir: Path,
    subjects: List[str],
) -> int:
    """Remove subject directories that don't have trials.txt."""
    removed = 0
    
    for subject_id in subjects:
        subject_dir = output_dir / subject_id
        if subject_dir.exists():
            bvh_count = len(list(subject_dir.glob("*.bvh")))
            shutil.rmtree(subject_dir)
            removed += 1
            print(f"  Removed subject {subject_id} ({bvh_count} BVH files)")
    
    return removed


def verify_conversion(
    output_dir: Path,
    source_dir: Path = None,
    fix_metadata: bool = False,
) -> VerificationReport:
    """
    Verify all converted BVH files.
    
    Args:
        output_dir: Directory containing converted BVH files
        source_dir: Original source directory (for metadata copy)
        fix_metadata: If True, copy missing trials.txt files
    """
    print(f"\nVerifying BVH conversion: {output_dir}")
    print("=" * 60)
    
    subjects = sorted([d for d in output_dir.iterdir() if d.is_dir()])
    
    all_results = []
    subjects_missing_trials = []
    invalid_details = []
    
    for subject_dir in subjects:
        result = verify_subject(subject_dir)
        all_results.append(result)
        
        if not result.has_trials_txt:
            subjects_missing_trials.append(result.subject_id)
        
        if result.invalid_files:
            for f in result.invalid_files:
                invalid_details.append({
                    'subject': result.subject_id,
                    'file': f,
                })
    
    # Aggregate stats
    total_files = sum(r.bvh_count for r in all_results)
    valid_files = sum(r.valid_count for r in all_results)
    invalid_files = total_files - valid_files
    total_frames = sum(r.total_frames for r in all_results)
    subjects_with_trials = sum(1 for r in all_results if r.has_trials_txt)
    
    report = VerificationReport(
        output_dir=str(output_dir),
        total_subjects=len(subjects),
        total_files=total_files,
        valid_files=valid_files,
        invalid_files=invalid_files,
        subjects_with_trials=subjects_with_trials,
        subjects_missing_trials=subjects_missing_trials,
        total_frames=total_frames,
        invalid_details=invalid_details,
    )
    
    # Print summary
    print(f"\nSubjects:      {report.total_subjects}")
    print(f"Total files:   {report.total_files}")
    print(f"Valid:         {report.valid_files} ✓")
    print(f"Invalid:       {report.invalid_files} {'✗' if report.invalid_files > 0 else '✓'}")
    print(f"Total frames:  {report.total_frames:,}")
    print(f"\nTrials.txt:    {report.subjects_with_trials}/{report.total_subjects} subjects")
    
    if report.invalid_files > 0:
        print(f"\nInvalid files ({report.invalid_files}):")
        for detail in report.invalid_details[:10]:
            print(f"  {detail['subject']}/{detail['file']}")
        if len(report.invalid_details) > 10:
            print(f"  ... and {len(report.invalid_details) - 10} more")
    
    # Fix metadata if requested
    if fix_metadata and source_dir and subjects_missing_trials:
        print(f"\nCopying missing trials.txt files...")
        copied = copy_trials_metadata(output_dir, source_dir, subjects_missing_trials)
        print(f"Copied {copied} trials.txt files")
    elif subjects_missing_trials:
        print(f"\nSubjects missing trials.txt ({len(subjects_missing_trials)}):")
        print(f"  {subjects_missing_trials[:10]}{'...' if len(subjects_missing_trials) > 10 else ''}")
        if source_dir:
            print(f"  Run with --fix to copy from {source_dir}")
    
    # Save report
    report_path = output_dir / "verification_report.json"
    with open(report_path, 'w') as f:
        json.dump(asdict(report), f, indent=2)
    print(f"\nReport saved to: {report_path}")
    
    return report


def reconvert_invalid_files(
    invalid_details: List[Dict],
    output_dir: Path,
    source_dir: Path,
) -> int:
    """
    Reconvert invalid BVH files.
    
    Args:
        invalid_details: List of {'subject': str, 'file': str} dicts
        output_dir: Directory containing BVH files
        source_dir: Source directory with ASF/AMC files
        
    Returns:
        Number of successfully reconverted files
    """
    import subprocess
    import sys
    
    # Group by subject
    subjects_to_fix = set()
    files_to_fix = []
    
    for detail in invalid_details:
        subject = detail['subject']
        filename = detail['file'].split(':')[0]  # Remove error message
        subjects_to_fix.add(subject)
        files_to_fix.append((subject, filename))
    
    print(f"\nReconverting {len(files_to_fix)} invalid files from {len(subjects_to_fix)} subjects...")
    
    # Delete invalid files first
    for subject, filename in files_to_fix:
        invalid_path = output_dir / subject / filename
        if invalid_path.exists():
            invalid_path.unlink()
            print(f"  Deleted: {subject}/{filename}")
    
    # Reconvert using batch_amc2bvh
    subjects_arg = ','.join(sorted(subjects_to_fix))
    
    cmd = [
        sys.executable,
        'python/amc2bvh/batch_amc2bvh.py',
        str(source_dir.parent if source_dir.name == 'subjects' else source_dir),
        '-o', str(output_dir),
        '--walk-bvh',
        '--subjects', subjects_arg,
        '--force',  # Force reconversion
    ]
    
    print(f"  Running: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if result.returncode == 0:
            print(f"  Reconversion completed successfully")
            return len(files_to_fix)
        else:
            print(f"  Reconversion failed: {result.stderr[:200]}")
            return 0
    except subprocess.TimeoutExpired:
        print(f"  Reconversion timed out")
        return 0
    except Exception as e:
        print(f"  Reconversion error: {e}")
        return 0


def main():
    parser = argparse.ArgumentParser(
        description='Verify BVH conversion integrity',
    )
    parser.add_argument('output_dir', help='Directory containing converted BVH files')
    parser.add_argument('--source', type=str, default=None,
                       help='Source directory for metadata copy and reconversion')
    parser.add_argument('--fix', action='store_true',
                       help='Copy missing trials.txt from source')
    parser.add_argument('--reconvert', action='store_true',
                       help='Reconvert invalid/truncated BVH files')
    parser.add_argument('--remove-no-trials', action='store_true',
                       help='Remove subjects without trials.txt metadata')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    source_dir = Path(args.source) if args.source else None
    
    # Auto-detect source subjects subdirectory
    if source_dir and (source_dir / 'subjects').exists():
        source_dir = source_dir / 'subjects'
    
    report = verify_conversion(output_dir, source_dir, args.fix)
    
    # Reconvert if requested and there are invalid files
    if args.reconvert and report.invalid_files > 0 and source_dir:
        reconverted = reconvert_invalid_files(report.invalid_details, output_dir, source_dir)
        if reconverted > 0:
            print(f"\nRe-verifying after reconversion...")
            report = verify_conversion(output_dir, source_dir, args.fix)
    elif args.reconvert and report.invalid_files > 0 and not source_dir:
        print("\n--reconvert requires --source to be specified")
    
    # Remove subjects without trials.txt if requested
    if getattr(args, 'remove_no_trials', False) and report.subjects_missing_trials:
        print(f"\nRemoving subjects without trials.txt...")
        removed = remove_subjects_without_trials(output_dir, report.subjects_missing_trials)
        print(f"Removed {removed} subjects")
        # Update report
        report = verify_conversion(output_dir, source_dir, False)
    
    if report.invalid_files > 0:
        exit(1)


if __name__ == '__main__':
    main()

