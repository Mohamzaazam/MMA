#!/usr/bin/env python3
"""
Batch ASF/AMC to BVH Converter

Converts motion capture data from CMU format (ASF/AMC) to BVH format
using parallel processing with robust error handling.

Usage:
    python batch_amc2bvh.py /mnt/e/database/cmu/subjects -o data/cmu --walk-bvh
    python batch_amc2bvh.py /path/to/subjects -o output --workers 8 --dry-run
"""

import argparse
import json
import logging
import os
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from tqdm import tqdm

from standalone import (
    ASFAMCtoBVH,
    ConversionConfig,
    RotationOrder,
)

# Configure module logger
logger = logging.getLogger(__name__)


@dataclass
class ConversionTask:
    """Represents a single conversion task."""
    subject_id: str
    asf_path: Path
    amc_path: Path
    output_path: Path
    
    @property
    def motion_name(self) -> str:
        return self.amc_path.stem


@dataclass
class ConversionResult:
    """Result of a conversion operation."""
    task: ConversionTask
    success: bool
    error: Optional[str] = None
    duration_seconds: float = 0.0
    output_size_bytes: int = 0
    frame_count: int = 0


@dataclass 
class BatchConversionReport:
    """Summary report of batch conversion."""
    start_time: str
    end_time: str
    duration_seconds: float
    total_tasks: int
    successful: int
    failed: int
    skipped: int
    total_frames: int
    total_output_bytes: int
    failed_tasks: List[Dict] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return asdict(self)


def validate_bvh_file(filepath: Path) -> Tuple[bool, str, int]:
    """
    Validate a BVH file by checking basic structure.
    
    Returns:
        Tuple of (is_valid, error_message, frame_count)
    """
    try:
        if not filepath.exists():
            return False, "File does not exist", 0
        
        if filepath.stat().st_size == 0:
            return False, "File is empty", 0
        
        with open(filepath, 'r') as f:
            content = f.read(4096)  # Read first 4KB
        
        # Check for required BVH sections
        if 'HIERARCHY' not in content:
            return False, "Missing HIERARCHY section", 0
        if 'MOTION' not in content:
            return False, "Missing MOTION section", 0
        if 'ROOT' not in content:
            return False, "Missing ROOT joint", 0
        
        # Try to extract frame count
        frame_count = 0
        for line in content.split('\n'):
            line = line.strip()
            if line.startswith('Frames:'):
                try:
                    frame_count = int(line.split(':')[1].strip())
                except:
                    pass
                break
        
        return True, "", frame_count
        
    except Exception as e:
        return False, f"Cannot read file: {e}", 0


class BatchConverter:
    """
    Batch converter for ASF/AMC to BVH conversion.
    
    Features:
    - Parallel processing with configurable workers
    - Robust error handling
    - Progress tracking
    - Resume support (skip existing files)
    - Verification of output files
    """
    
    def __init__(
        self,
        input_dir: Path,
        output_dir: Path,
        config: ConversionConfig,
        max_workers: int = 4,
        force: bool = False,
        verify: bool = True,
    ):
        """
        Initialize the batch converter.
        
        Args:
            input_dir: Directory containing subjects (with ASF/AMC files)
            output_dir: Directory to write BVH files
            config: ConversionConfig for the ASF/AMC to BVH converter
            max_workers: Maximum parallel conversion workers
            force: Force reconversion even if output exists
            verify: Verify output files after conversion
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.config = config
        self.max_workers = max_workers
        self.force = force
        self.verify = verify
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"BatchConverter initialized: input={input_dir}, output={output_dir}")
    
    def discover_tasks(
        self, 
        subjects: Optional[Set[str]] = None
    ) -> List[ConversionTask]:
        """
        Discover all ASF/AMC pairs in the input directory.
        
        Args:
            subjects: Optional set of subject IDs to filter (e.g., {'01', '02', '86'})
        
        Returns:
            List of ConversionTask objects
        """
        tasks = []
        
        # Check if input_dir contains 'subjects' subdirectory
        subjects_dir = self.input_dir
        if (self.input_dir / 'subjects').exists():
            subjects_dir = self.input_dir / 'subjects'
        
        if not subjects_dir.exists():
            logger.error(f"Input directory does not exist: {subjects_dir}")
            return tasks
        
        # Iterate through subject directories
        for subject_path in sorted(subjects_dir.iterdir()):
            if not subject_path.is_dir():
                continue
            
            subject_id = subject_path.name
            
            # Filter by subjects if specified
            if subjects and subject_id not in subjects:
                continue
            
            # Find ASF file
            asf_files = list(subject_path.glob("*.asf"))
            if not asf_files:
                logger.warning(f"Subject {subject_id}: No ASF file found, skipping")
                continue
            
            asf_path = asf_files[0]  # Use first ASF file
            
            # Find all AMC files
            amc_files = list(subject_path.glob("*.amc"))
            if not amc_files:
                logger.warning(f"Subject {subject_id}: No AMC files found, skipping")
                continue
            
            # Create tasks for each AMC file
            subject_output_dir = self.output_dir / subject_id
            
            for amc_path in sorted(amc_files):
                output_path = subject_output_dir / f"{amc_path.stem}.bvh"
                
                # Skip if output exists and not forcing
                if not self.force and output_path.exists():
                    logger.debug(f"Skipping {amc_path.name}: output exists")
                    continue
                
                tasks.append(ConversionTask(
                    subject_id=subject_id,
                    asf_path=asf_path,
                    amc_path=amc_path,
                    output_path=output_path,
                ))
        
        logger.info(f"Discovered {len(tasks)} conversion tasks")
        return tasks
    
    def convert_single(self, task: ConversionTask) -> ConversionResult:
        """
        Convert a single ASF/AMC pair to BVH.
        
        Args:
            task: ConversionTask to process
            
        Returns:
            ConversionResult with status
        """
        start_time = time.time()
        
        try:
            # Ensure output directory exists
            task.output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Create converter and run conversion
            converter = ASFAMCtoBVH(self.config)
            
            # Suppress print output during batch processing
            import io
            from contextlib import redirect_stdout
            
            with redirect_stdout(io.StringIO()):
                converter.convert(
                    str(task.asf_path),
                    str(task.amc_path),
                    str(task.output_path),
                )
            
            duration = time.time() - start_time
            
            # Verify output if enabled
            output_size = 0
            frame_count = 0
            
            if task.output_path.exists():
                output_size = task.output_path.stat().st_size
                
                if self.verify:
                    is_valid, error, frames = validate_bvh_file(task.output_path)
                    if not is_valid:
                        return ConversionResult(
                            task=task,
                            success=False,
                            error=f"Verification failed: {error}",
                            duration_seconds=duration,
                        )
                    frame_count = frames
            else:
                return ConversionResult(
                    task=task,
                    success=False,
                    error="Output file not created",
                    duration_seconds=duration,
                )
            
            return ConversionResult(
                task=task,
                success=True,
                duration_seconds=duration,
                output_size_bytes=output_size,
                frame_count=frame_count,
            )
            
        except Exception as e:
            duration = time.time() - start_time
            error_msg = f"{type(e).__name__}: {str(e)}"
            logger.error(f"Failed to convert {task.amc_path.name}: {error_msg}")
            
            return ConversionResult(
                task=task,
                success=False,
                error=error_msg,
                duration_seconds=duration,
            )
    
    def convert_all(
        self, 
        tasks: List[ConversionTask],
        progress: bool = True,
    ) -> List[ConversionResult]:
        """
        Convert all tasks in parallel.
        
        Args:
            tasks: List of conversion tasks
            progress: Show progress bar
            
        Returns:
            List of ConversionResult objects
        """
        if not tasks:
            logger.info("No tasks to convert")
            return []
        
        results = []
        
        print(f"\nConverting {len(tasks)} motion files using {self.max_workers} workers...")
        print(f"Output directory: {self.output_dir}")
        print()
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_task = {
                executor.submit(self.convert_single, task): task
                for task in tasks
            }
            
            # Collect results with progress bar
            pbar = tqdm(
                total=len(tasks),
                desc="Converting",
                unit="file",
                ncols=80,
                disable=not progress,
            )
            
            for future in as_completed(future_to_task):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    task = future_to_task[future]
                    logger.error(f"Unexpected error for {task.amc_path.name}: {e}")
                    results.append(ConversionResult(
                        task=task,
                        success=False,
                        error=str(e),
                    ))
                
                pbar.update(1)
            
            pbar.close()
        
        return results
    
    def generate_report(
        self, 
        results: List[ConversionResult],
        start_time: datetime,
        end_time: datetime,
        skipped_count: int = 0,
    ) -> BatchConversionReport:
        """Generate summary report from results."""
        successful = sum(1 for r in results if r.success)
        failed = sum(1 for r in results if not r.success)
        total_frames = sum(r.frame_count for r in results if r.success)
        total_bytes = sum(r.output_size_bytes for r in results if r.success)
        
        failed_tasks = [
            {
                'subject': r.task.subject_id,
                'motion': r.task.motion_name,
                'error': r.error,
            }
            for r in results if not r.success
        ]
        
        return BatchConversionReport(
            start_time=start_time.isoformat(),
            end_time=end_time.isoformat(),
            duration_seconds=(end_time - start_time).total_seconds(),
            total_tasks=len(results) + skipped_count,
            successful=successful,
            failed=failed,
            skipped=skipped_count,
            total_frames=total_frames,
            total_output_bytes=total_bytes,
            failed_tasks=failed_tasks,
        )
    
    def save_report(self, report: BatchConversionReport, filepath: Path):
        """Save report to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(report.to_dict(), f, indent=2)
        logger.info(f"Report saved to {filepath}")
    
    def copy_metadata_files(self, converted_subjects: Set[str]):
        """
        Copy metadata files (trials.txt/trials.json) from source to output directories.
        
        Args:
            converted_subjects: Set of subject IDs that were converted
        """
        # Find the source subjects directory
        subjects_dir = self.input_dir
        if (self.input_dir / 'subjects').exists():
            subjects_dir = self.input_dir / 'subjects'
        
        copied_count = 0
        
        for subject_id in converted_subjects:
            source_dir = subjects_dir / subject_id
            output_dir = self.output_dir / subject_id
            
            if not output_dir.exists():
                continue
            
            # Copy trials.txt and/or trials.json if they exist
            for filename in ['trials.txt', 'trials.json']:
                source_file = source_dir / filename
                if source_file.exists():
                    dest_file = output_dir / filename
                    shutil.copy2(source_file, dest_file)
                    logger.debug(f"Copied metadata: {source_file} -> {dest_file}")
                    copied_count += 1
        
        if copied_count > 0:
            logger.info(f"Copied {copied_count} metadata files")
            print(f"  Copied {copied_count} metadata files")

    
    def run(
        self, 
        subjects: Optional[Set[str]] = None,
        dry_run: bool = False,
    ) -> BatchConversionReport:
        """
        Run the full batch conversion pipeline.
        
        Args:
            subjects: Optional set of subject IDs to filter
            dry_run: Only discover tasks, don't convert
            
        Returns:
            BatchConversionReport with results
        """
        start_time = datetime.now()
        
        print("=" * 60)
        print("ASF/AMC to BVH Batch Converter")
        print("=" * 60)
        print(f"Input:  {self.input_dir}")
        print(f"Output: {self.output_dir}")
        print(f"Format: {self.config.rotation_order.name}")
        print(f"Scale:  {self.config.scale:.4f}")
        print(f"FPS:    {self.config.fps}")
        print()
        
        # Discover tasks
        print("Discovering ASF/AMC file pairs...")
        all_subjects_dir = self.input_dir
        if (self.input_dir / 'subjects').exists():
            all_subjects_dir = self.input_dir / 'subjects'
        
        # Count total AMC files for skip calculation
        total_amc_count = 0
        for subject_path in all_subjects_dir.iterdir():
            if subject_path.is_dir():
                if subjects and subject_path.name not in subjects:
                    continue
                total_amc_count += len(list(subject_path.glob("*.amc")))
        
        tasks = self.discover_tasks(subjects)
        skipped_count = total_amc_count - len(tasks)
        
        print(f"Found {total_amc_count} total motion files")
        print(f"  To convert: {len(tasks)}")
        print(f"  Skipped (already exist): {skipped_count}")
        
        if dry_run:
            print("\n[DRY RUN] No conversions performed.")
            print("\nTasks that would be converted:")
            for task in tasks[:20]:  # Show first 20
                print(f"  {task.subject_id}/{task.motion_name}")
            if len(tasks) > 20:
                print(f"  ... and {len(tasks) - 20} more")
            
            return BatchConversionReport(
                start_time=start_time.isoformat(),
                end_time=datetime.now().isoformat(),
                duration_seconds=0,
                total_tasks=len(tasks),
                successful=0,
                failed=0,
                skipped=skipped_count,
                total_frames=0,
                total_output_bytes=0,
            )
        
        if not tasks:
            print("\nNo files to convert. All files already processed.")
            end_time = datetime.now()
            return BatchConversionReport(
                start_time=start_time.isoformat(),
                end_time=end_time.isoformat(),
                duration_seconds=(end_time - start_time).total_seconds(),
                total_tasks=0,
                successful=0,
                failed=0,
                skipped=skipped_count,
                total_frames=0,
                total_output_bytes=0,
            )
        
        # Run conversions
        results = self.convert_all(tasks)
        
        end_time = datetime.now()
        
        # Generate and print report
        report = self.generate_report(results, start_time, end_time, skipped_count)
        
        print()
        print("=" * 60)
        print("Conversion Summary")
        print("=" * 60)
        print(f"  Duration:   {report.duration_seconds:.1f}s")
        print(f"  Successful: {report.successful}")
        print(f"  Failed:     {report.failed}")
        print(f"  Skipped:    {report.skipped}")
        print(f"  Frames:     {report.total_frames:,}")
        print(f"  Output:     {report.total_output_bytes / (1024*1024):.1f}MB")
        
        # Copy metadata files for converted subjects
        converted_subjects = {r.task.subject_id for r in results if r.success}
        self.copy_metadata_files(converted_subjects)
        
        if report.failed_tasks:
            print(f"\nFailed conversions ({len(report.failed_tasks)}):")
            for ft in report.failed_tasks[:10]:
                print(f"  {ft['subject']}/{ft['motion']}: {ft['error']}")
            if len(report.failed_tasks) > 10:
                print(f"  ... and {len(report.failed_tasks) - 10} more")
        
        # Save report
        report_path = self.output_dir / "conversion_report.json"
        self.save_report(report, report_path)
        print(f"\nReport saved to: {report_path}")
        
        return report


def setup_logging(
    level: int = logging.INFO,
    log_file: Optional[Path] = None,
):
    """Configure logging for the batch converter."""
    handlers = [logging.StreamHandler()]
    
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers,
    )


def main():
    parser = argparse.ArgumentParser(
        description='Batch convert ASF/AMC files to BVH format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  # Convert all subjects with walk-bvh format (recommended)
  %(prog)s /mnt/e/database/cmu -o data/cmu --walk-bvh
  
  # Convert specific subjects only
  %(prog)s /path/to/subjects -o output --subjects 01,02,86
  
  # Dry run to see what would be converted
  %(prog)s /path/to/subjects -o output --dry-run
  
  # Force reconversion of all files
  %(prog)s /path/to/subjects -o output --force
        """
    )
    
    parser.add_argument('input_dir', 
                        help='Input directory containing subjects/ with ASF/AMC files')
    parser.add_argument('-o', '--output', required=True,
                        help='Output directory for BVH files')
    parser.add_argument('-w', '--workers', type=int, default=4,
                        help='Number of parallel workers (default: 4)')
    parser.add_argument('-f', '--fps', type=float, default=120.0,
                        help='Frames per second (default: 120)')
    parser.add_argument('-r', '--rotation', choices=['ZXY', 'ZYX', 'XYZ'], default='ZXY',
                        help='Rotation order (default: ZXY)')
    parser.add_argument('-s', '--scale', type=float, default=1.0,
                        help='Scale factor (default: 1.0, overridden by --walk-bvh)')
    parser.add_argument('--subjects', type=str, default=None,
                        help='Comma-separated list of subject IDs to convert (e.g., 01,02,86)')
    parser.add_argument('--walk-bvh', action='store_true',
                        help='Use walk-bvh format (Character1_* names, joint collapsing, CMU scaling)')
    parser.add_argument('--force', action='store_true',
                        help='Force reconversion even if output exists')
    parser.add_argument('--dry-run', action='store_true',
                        help='Only show what would be converted, do not process')
    parser.add_argument('--no-verify', action='store_true',
                        help='Skip verification of output BVH files')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    log_file = Path(args.output) / "batch_conversion.log" if not args.dry_run else None
    setup_logging(log_level, log_file)
    
    # Build conversion config
    config = ConversionConfig()
    config.fps = args.fps
    config.rotation_order = RotationOrder[args.rotation]
    config.scale = args.scale
    
    if args.walk_bvh:
        config.rotation_order = RotationOrder.ZXY
        config.scale = (1 / 0.45) * 2.54  # Convert CMU units
        config.collapse_joints = ConversionConfig.walk_bvh_collapse_joints()
        config.skip_joints = ConversionConfig.walk_bvh_skip_joints()
        config.end_joints = ConversionConfig.walk_bvh_end_joints()
        config.joint_name_map = ConversionConfig.walk_bvh_joint_map()
    
    # Parse subjects filter
    subjects = None
    if args.subjects:
        subjects = set(s.strip().zfill(2) for s in args.subjects.split(','))
    
    # Create and run converter
    converter = BatchConverter(
        input_dir=Path(args.input_dir),
        output_dir=Path(args.output),
        config=config,
        max_workers=args.workers,
        force=args.force,
        verify=not args.no_verify,
    )
    
    report = converter.run(subjects=subjects, dry_run=args.dry_run)
    
    # Exit with error code if there were failures
    if report.failed > 0:
        sys.exit(1)


if __name__ == '__main__':
    main()
