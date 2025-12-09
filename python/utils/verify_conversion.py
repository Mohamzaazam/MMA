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
        
        # Extract frame count
        frame_count = 0
        for line in content.split('\n'):
            line = line.strip()
            if line.startswith('Frames:'):
                try:
                    frame_count = int(line.split(':')[1].strip())
                except:
                    return False, "Cannot parse frame count", 0
                break
        
        if frame_count == 0:
            return False, "Frame count is 0", 0
        
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


def main():
    parser = argparse.ArgumentParser(
        description='Verify BVH conversion integrity',
    )
    parser.add_argument('output_dir', help='Directory containing converted BVH files')
    parser.add_argument('--source', type=str, default=None,
                       help='Source directory for metadata copy')
    parser.add_argument('--fix', action='store_true',
                       help='Copy missing trials.txt from source')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    source_dir = Path(args.source) if args.source else None
    
    # Auto-detect source subjects subdirectory
    if source_dir and (source_dir / 'subjects').exists():
        source_dir = source_dir / 'subjects'
    
    report = verify_conversion(output_dir, source_dir, args.fix)
    
    if report.invalid_files > 0:
        exit(1)


if __name__ == '__main__':
    main()
