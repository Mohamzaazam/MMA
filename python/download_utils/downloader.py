#!/usr/bin/env python3
"""
CMU Motion Capture Database Downloader (Enhanced Version)
Downloads ASF and AMC files and organizes them by subject number.

Features:
- Concurrent downloads with configurable parallelism
- Retry mechanism with exponential backoff
- Resumable downloads (skip already downloaded files)
- Progress bars with tqdm
- Proper logging
- Connection pooling with requests
- File integrity validation
- Timeout handling

Data source: http://mocap.cs.cmu.edu/
"""

import hashlib
import logging
import os
import re
import shutil
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional
from urllib.parse import urljoin

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from tqdm import tqdm

# Configure module logger
logger = logging.getLogger(__name__)


@dataclass
class DownloadResult:
    """Result of a download operation."""
    success: bool
    path: Optional[Path] = None
    error: Optional[str] = None
    skipped: bool = False
    bytes_downloaded: int = 0


class DownloadError(Exception):
    """Custom exception for download failures."""
    pass


def validate_asf_file(filepath: Path) -> tuple[bool, str]:
    """
    Validate an ASF (Acclaim Skeleton File) file.
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read(2048)  # Read first 2KB
            
        # ASF files should contain these sections
        required_markers = [':version', ':root', ':bonedata']
        found = sum(1 for marker in required_markers if marker in content.lower())
        
        if found < 2:
            return False, "Missing required ASF sections (:version, :root, :bonedata)"
        
        return True, ""
    except Exception as e:
        return False, f"Cannot read file: {e}"


def validate_amc_file(filepath: Path) -> tuple[bool, str]:
    """
    Validate an AMC (Acclaim Motion Capture) file.
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            lines = []
            for i, line in enumerate(f):
                lines.append(line.strip())
                if i > 50:  # Check first 50 lines
                    break
        
        # AMC files should have frame numbers (integers on their own line)
        # and motion data lines
        has_frame_number = False
        has_motion_data = False
        
        for line in lines:
            if line.isdigit():
                has_frame_number = True
            elif line and not line.startswith('#') and not line.startswith(':'):
                parts = line.split()
                if len(parts) > 1:  # bone_name followed by values
                    has_motion_data = True
        
        if not has_frame_number:
            return False, "No frame numbers found"
        if not has_motion_data:
            return False, "No motion data found"
            
        return True, ""
    except Exception as e:
        return False, f"Cannot read file: {e}"


class CMUMocapDownloader:
    """
    Download and organize CMU Motion Capture data.
    
    Features:
    - Concurrent downloads for faster individual subject downloads
    - Automatic retry with exponential backoff
    - Resume support (skips already downloaded files)
    - Progress tracking with tqdm
    - Connection pooling for efficiency
    
    Example:
        >>> downloader = CMUMocapDownloader(output_dir="./data")
        >>> downloader.download_bulk()  # Download entire dataset
        >>> # or
        >>> downloader.download_subject(86)  # Download specific subject
    """
    
    # Primary bulk download URL for all ASF/AMC files
    BULK_ZIP_URL = "http://mocap.cs.cmu.edu/allasfamc.zip"
    BULK_ZIP_SIZE_APPROX = 500 * 1024 * 1024  # ~500MB expected size
    
    # Alternative: Individual subject download base URL
    BASE_URL = "http://mocap.cs.cmu.edu/subjects"
    
    # Known subject range in CMU dataset
    MIN_SUBJECT = 1
    MAX_SUBJECT = 144
    
    def __init__(
        self,
        output_dir: str = "cmu_mocap_data",
        max_workers: int = 4,
        max_retries: int = 3,
        timeout: int = 30,
        chunk_size: int = 8192,
        verify_ssl: bool = False,
    ):
        """
        Initialize the CMU MoCap downloader.
        
        Args:
            output_dir: Directory to save downloaded files
            max_workers: Maximum concurrent download threads
            max_retries: Number of retry attempts for failed downloads
            timeout: Request timeout in seconds
            chunk_size: Download chunk size in bytes
            verify_ssl: Whether to verify SSL certificates (CMU site has issues)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.max_workers = max_workers
        self.max_retries = max_retries
        self.timeout = timeout
        self.chunk_size = chunk_size
        self.verify_ssl = verify_ssl
        
        # Create session with retry strategy
        self.session = self._create_session()
        
        logger.info(f"Initialized CMUMocapDownloader: output={output_dir}, workers={max_workers}")
    
    def _create_session(self) -> requests.Session:
        """Create a requests session with retry strategy and connection pooling."""
        session = requests.Session()
        
        # Configure retry strategy with exponential backoff
        retry_strategy = Retry(
            total=self.max_retries,
            backoff_factor=1,  # Wait 1, 2, 4 seconds between retries
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET"],
        )
        
        # Mount adapter with connection pooling
        adapter = HTTPAdapter(
            max_retries=retry_strategy,
            pool_connections=self.max_workers,
            pool_maxsize=self.max_workers * 2,
        )
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        # Set default headers
        session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) CMUMocapDownloader/2.0'
        })
        
        return session
    
    def _get_remote_file_size(self, url: str) -> Optional[int]:
        """Get the size of a remote file via HEAD request."""
        try:
            response = self.session.head(
                url, 
                timeout=self.timeout, 
                verify=self.verify_ssl,
                allow_redirects=True
            )
            if response.status_code == 200:
                return int(response.headers.get('content-length', 0))
        except Exception as e:
            logger.debug(f"Could not get file size for {url}: {e}")
        return None
    
    def _download_file(
        self,
        url: str,
        dest: Path,
        expected_size: Optional[int] = None,
        progress_callback: Optional[Callable[[int], None]] = None,
        force: bool = False,
    ) -> DownloadResult:
        """
        Download a file with progress tracking, validation, and resume support.
        
        Args:
            url: URL to download from
            dest: Destination path
            expected_size: Expected file size for validation
            progress_callback: Callback function for progress updates
            force: Force re-download even if file exists
            
        Returns:
            DownloadResult with status information
        """
        # Ensure parent directory exists
        dest.parent.mkdir(parents=True, exist_ok=True)
        
        # Check if file already exists with correct size
        if dest.exists() and not force:
            existing_size = dest.stat().st_size
            if expected_size and existing_size == expected_size:
                logger.debug(f"Skipping {dest.name}: already exists with correct size")
                return DownloadResult(success=True, path=dest, skipped=True)
            
            # If no expected size known, check remote size
            if expected_size is None:
                remote_size = self._get_remote_file_size(url)
                if remote_size and existing_size == remote_size:
                    logger.debug(f"Skipping {dest.name}: already exists with correct size")
                    return DownloadResult(success=True, path=dest, skipped=True)

        temp_path = dest.with_suffix(dest.suffix + '.tmp')
        
        # Determine starting position
        start_byte = 0
        if temp_path.exists() and not force:
            start_byte = temp_path.stat().st_size
            
        # Retry loop for resuming
        max_retries = self.max_retries
        for attempt in range(max_retries + 1):
            try:
                headers = {}
                mode = 'wb'
                if start_byte > 0:
                    headers['Range'] = f"bytes={start_byte}-"
                    mode = 'ab'
                    logger.info(f"Resuming {dest.name} from byte {start_byte}")
                
                response = self.session.get(
                    url,
                    stream=True,
                    timeout=self.timeout,
                    verify=self.verify_ssl,
                    headers=headers
                )
                
                # Handle resizing/completion if server doesn't support range
                if response.status_code == 416: # Range Not Satisfiable
                    # Likely already complete or invalid range
                    remote_size = self._get_remote_file_size(url)
                    if remote_size == start_byte:
                        logger.info(f"Download already complete for {dest.name}")
                        temp_path.rename(dest)
                        return DownloadResult(success=True, path=dest, bytes_downloaded=0)
                    else:
                        # File changed or error, restart
                        logger.warning("Invalid range, restarting download")
                        start_byte = 0
                        mode = 'wb'
                        temp_path.unlink(missing_ok=True)
                        continue # Restart loop
                        
                response.raise_for_status()
                
                total_size = int(response.headers.get('content-length', 0))
                # For range requests, content-length is the remaining size
                if start_byte > 0:
                    total_size += start_byte
                    
                downloaded_this_session = 0
                
                with open(temp_path, mode) as f:
                    for chunk in response.iter_content(chunk_size=self.chunk_size):
                        if chunk:
                            f.write(chunk)
                            chunk_len = len(chunk)
                            downloaded_this_session += chunk_len
                            start_byte += chunk_len
                            if progress_callback:
                                progress_callback(chunk_len)
                
                # If we get here, download (chunk) finished successfully
                # Validate size if possible
                if total_size > 0 and start_byte != total_size:
                     raise DownloadError(f"Incomplete download: got {start_byte}, expected {total_size}")
                
                temp_path.rename(dest)
                return DownloadResult(success=True, path=dest, bytes_downloaded=downloaded_this_session)
                
            except (requests.exceptions.RequestException, DownloadError, Exception) as e:
                logger.warning(f"Download interruption ({attempt+1}/{max_retries}): {e}")
                if attempt == max_retries:
                    logger.error(f"Failed to download {url} after {max_retries} retries")
                    return DownloadResult(success=False, error=str(e))
                
                # Update start_byte for next attempt
                if temp_path.exists():
                     start_byte = temp_path.stat().st_size
                
                import time
                time.sleep(1 * (attempt + 1)) # Backoff

        return DownloadResult(success=False, error="Max retries exceeded")
    
    def download_bulk(
        self, 
        extract: bool = True, 
        force: bool = False,
        keep_zip: bool = False,
    ) -> Path:
        """
        Download the entire ASF/AMC dataset as a single zip file.
        This is the recommended approach - faster and more reliable.
        
        Args:
            extract: If True, extract and organize files after download
            force: Force re-download even if zip already exists
            keep_zip: Keep the zip file after extraction
            
        Returns:
            Path to the downloaded/extracted directory
        """
        zip_path = self.output_dir / "allasfamc.zip"
        
        # Check if already extracted
        subjects_dir = self.output_dir / "subjects"
        if not force and subjects_dir.exists():
            subject_count = len([d for d in subjects_dir.iterdir() if d.is_dir()])
            if subject_count > 100:  # Most of dataset already extracted
                logger.info(f"Dataset already extracted: {subject_count} subjects found")
                print(f"✓ Dataset already extracted: {subject_count} subjects in {subjects_dir}")
                return subjects_dir
        
        print(f"{'=' * 60}")
        print("CMU Motion Capture Dataset - Bulk Download")
        print(f"{'=' * 60}")
        print(f"URL: {self.BULK_ZIP_URL}")
        print(f"Expected size: ~{self.BULK_ZIP_SIZE_APPROX // (1024*1024)}MB")
        print()
        
        # Get actual file size
        remote_size = self._get_remote_file_size(self.BULK_ZIP_URL)
        
        # Create progress bar
        with tqdm(
            total=remote_size or self.BULK_ZIP_SIZE_APPROX,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
            desc="Downloading",
            ncols=80,
        ) as pbar:
            result = self._download_file(
                self.BULK_ZIP_URL,
                zip_path,
                expected_size=remote_size,
                progress_callback=pbar.update,
                force=force,
            )
        
        if not result.success:
            print(f"\n✗ Bulk download failed: {result.error}")
            print("Falling back to individual subject download...")
            logger.warning("Bulk download failed, falling back to individual downloads")
            return self.download_all_subjects()
        
        if result.skipped:
            print(f"✓ Using existing archive: {zip_path}")
        else:
            print(f"✓ Downloaded {result.bytes_downloaded / (1024*1024):.1f}MB to {zip_path}")
        
        if extract:
            extracted_path = self._extract_and_organize(zip_path)
            
            if not keep_zip:
                zip_path.unlink(missing_ok=True)
                logger.info("Removed zip file after extraction")
            
            return extracted_path
        
        return zip_path
    
    def _extract_and_organize(self, zip_path: Path) -> Path:
        """Extract zip and organize files by subject number with progress."""
        print(f"\nExtracting and organizing files...")
        
        organized_dir = self.output_dir / "subjects"
        organized_dir.mkdir(exist_ok=True)
        
        with zipfile.ZipFile(zip_path, 'r') as zf:
            # Get list of ASF/AMC files
            all_files = [f for f in zf.namelist() 
                        if f.lower().endswith(('.asf', '.amc'))]
            
            print(f"Found {len(all_files)} motion capture files")
            
            subjects_found = set()
            
            # Extract with progress bar
            with tqdm(
                total=len(all_files),
                desc="Extracting",
                unit="files",
                ncols=80,
            ) as pbar:
                for file_path in all_files:
                    filename = os.path.basename(file_path)
                    subject_num = self._extract_subject_number(filename)
                    
                    if subject_num:
                        subjects_found.add(subject_num)
                        subject_dir = organized_dir / subject_num
                        subject_dir.mkdir(exist_ok=True)
                        
                        # Extract file directly to subject directory
                        dest_path = subject_dir / filename
                        with zf.open(file_path) as src:
                            with open(dest_path, 'wb') as dst:
                                shutil.copyfileobj(src, dst)
                    
                    pbar.update(1)
        
        print(f"✓ Organized {len(subjects_found)} subjects into {organized_dir}")
        logger.info(f"Extracted {len(all_files)} files into {len(subjects_found)} subject directories")
        
        return organized_dir
    
    def _extract_subject_number(self, filename: str) -> Optional[str]:
        """Extract subject number from filename (e.g., '01.asf' -> '01')."""
        # Pattern: XX.asf or XX_YY.amc
        match = re.match(r'^(\d+)(?:_\d+)?\.(?:asf|amc)$', filename, re.IGNORECASE)
        if match:
            return match.group(1).zfill(2)  # Zero-pad to 2 digits
        return None
    
    def download_subject(
        self, 
        subject_num: int, 
        force: bool = False,
    ) -> DownloadResult:
        """
        Download ASF and AMC files for a specific subject.
        
        Args:
            subject_num: Subject number (e.g., 1, 86, 143)
            force: Force re-download even if files exist
            
        Returns:
            DownloadResult with status and path information
        """
        subject_str = str(subject_num).zfill(2)
        subject_dir = self.output_dir / "subjects" / subject_str
        subject_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Downloading subject {subject_str}")
        
        files_downloaded = 0
        files_skipped = 0
        total_bytes = 0
        errors = []
        
        # Download ASF file
        asf_url = f"{self.BASE_URL}/{subject_str}/{subject_str}.asf"
        asf_path = subject_dir / f"{subject_str}.asf"
        
        result = self._download_file(asf_url, asf_path, force=force)
        if result.success:
            if result.skipped:
                files_skipped += 1
            else:
                files_downloaded += 1
                total_bytes += result.bytes_downloaded
        else:
            errors.append(f"ASF: {result.error}")
            logger.warning(f"Failed to download ASF for subject {subject_str}")
        
        # Download AMC files (try numbered patterns)
        amc_num = 1
        consecutive_failures = 0
        max_failures = 5  # Stop after 5 consecutive 404s
        
        while consecutive_failures < max_failures:
            amc_filename = f"{subject_str}_{str(amc_num).zfill(2)}.amc"
            amc_url = f"{self.BASE_URL}/{subject_str}/{amc_filename}"
            amc_path = subject_dir / amc_filename
            
            result = self._download_file(amc_url, amc_path, force=force)
            
            if result.success:
                if result.skipped:
                    files_skipped += 1
                else:
                    files_downloaded += 1
                    total_bytes += result.bytes_downloaded
                consecutive_failures = 0
            else:
                consecutive_failures += 1
            
            amc_num += 1
        
        success = files_downloaded > 0 or files_skipped > 0
        
        return DownloadResult(
            success=success,
            path=subject_dir,
            bytes_downloaded=total_bytes,
            skipped=(files_downloaded == 0 and files_skipped > 0),
            error="; ".join(errors) if errors else None,
        )
    
    def download_subjects_concurrent(
        self,
        subject_nums: list[int],
        force: bool = False,
        progress: bool = True,
    ) -> dict[int, DownloadResult]:
        """
        Download multiple subjects concurrently.
        
        Args:
            subject_nums: List of subject numbers to download
            force: Force re-download even if files exist
            progress: Show progress bar
            
        Returns:
            Dictionary mapping subject numbers to their download results
        """
        results = {}
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all download tasks
            future_to_subject = {
                executor.submit(self.download_subject, num, force): num
                for num in subject_nums
            }
            
            # Create progress bar
            pbar = tqdm(
                total=len(subject_nums),
                desc="Downloading subjects",
                unit="subject",
                ncols=80,
                disable=not progress,
            )
            
            # Collect results as they complete
            for future in as_completed(future_to_subject):
                subject_num = future_to_subject[future]
                try:
                    results[subject_num] = future.result()
                except Exception as e:
                    logger.error(f"Subject {subject_num} download failed: {e}")
                    results[subject_num] = DownloadResult(
                        success=False, 
                        error=str(e)
                    )
                pbar.update(1)
            
            pbar.close()
        
        return results
    
    def download_all_subjects(
        self,
        start: int = 1,
        end: int = 144,
        force: bool = False,
    ) -> Path:
        """
        Download ASF/AMC files for a range of subjects concurrently.
        
        Args:
            start: Starting subject number
            end: Ending subject number (inclusive)
            force: Force re-download even if files exist
            
        Returns:
            Path to subjects directory
        """
        subjects_dir = self.output_dir / "subjects"
        subjects_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Downloading subjects {start} to {end} ({end - start + 1} subjects)")
        print(f"Using {self.max_workers} concurrent workers")
        print()
        
        subject_nums = list(range(start, end + 1))
        results = self.download_subjects_concurrent(subject_nums, force=force)
        
        # Summarize results
        successful = sum(1 for r in results.values() if r.success)
        skipped = sum(1 for r in results.values() if r.skipped)
        failed = sum(1 for r in results.values() if not r.success)
        total_bytes = sum(r.bytes_downloaded for r in results.values())
        
        print()
        print(f"{'=' * 60}")
        print(f"Download Summary")
        print(f"{'=' * 60}")
        print(f"  Successful: {successful}")
        print(f"  Skipped (already exists): {skipped}")
        print(f"  Failed: {failed}")
        print(f"  Total downloaded: {total_bytes / (1024*1024):.1f}MB")
        
        if failed > 0:
            print(f"\nFailed subjects:")
            for num, result in results.items():
                if not result.success:
                    print(f"  Subject {num}: {result.error}")
        
        return subjects_dir
    
    def list_subjects(self) -> list[dict]:
        """List all downloaded subject directories with file counts."""
        subjects_dir = self.output_dir / "subjects"
        if not subjects_dir.exists():
            return []
        
        subjects = []
        for d in sorted(subjects_dir.iterdir()):
            if d.is_dir() and d.name.isdigit():
                asf_files = list(d.glob("*.asf"))
                amc_files = list(d.glob("*.amc"))
                
                # Calculate total size
                total_size = sum(f.stat().st_size for f in asf_files + amc_files)
                
                subjects.append({
                    'number': d.name,
                    'path': d,
                    'asf_files': len(asf_files),
                    'amc_files': len(amc_files),
                    'total_size_bytes': total_size,
                })
        
        return subjects
    
    def verify_dataset(self, validate_content: bool = True, remove_missing: bool = False) -> dict:
        """
        Verify the integrity of downloaded dataset.
        
        Args:
            validate_content: If True, parse files to verify they're not corrupted
            remove_missing: If True, automatically remove subjects missing ASF files
        
        Returns:
            Dictionary with verification results
        """
        subjects = self.list_subjects()
        
        issues = []
        corrupted_files = []
        subjects_missing_asf = []
        subjects_missing_amc = []
        removed_subjects = []
        
        print("Verifying dataset integrity...")
        
        for s in tqdm(subjects, desc="Checking subjects", unit="subject", ncols=80):
            subject_num = s['number']
            subject_path = s['path']
            
            # Check if ASF exists
            asf_files = list(subject_path.glob("*.asf"))
            if len(asf_files) == 0:
                subjects_missing_asf.append(subject_num)
                issue_msg = f"Subject {subject_num}: Missing ASF file"
                issues.append(issue_msg)
                
                if remove_missing:
                    try:
                        shutil.rmtree(subject_path)
                        removed_subjects.append(subject_num)
                        logger.info(f"Removed subject {subject_num} (missing ASF)")
                        continue # Skip AMC check if removed
                    except Exception as e:
                        logger.error(f"Failed to remove subject {subject_num}: {e}")
                
            elif validate_content:
                # Validate ASF file content
                for asf_file in asf_files:
                    if asf_file.stat().st_size == 0:
                        corrupted_files.append(str(asf_file))
                        issues.append(f"Subject {subject_num}: Empty ASF file {asf_file.name}")
                    else:
                        is_valid, error = validate_asf_file(asf_file)
                        if not is_valid:
                            corrupted_files.append(str(asf_file))
                            issues.append(f"Subject {subject_num}: Corrupted ASF {asf_file.name} - {error}")
            
            # Check if AMC files exist (only if subject wasn't removed)
            amc_files = list(subject_path.glob("*.amc"))
            if len(amc_files) == 0:
                subjects_missing_amc.append(subject_num)
                issues.append(f"Subject {subject_num}: No AMC files")
            elif validate_content:
                # Validate AMC file content
                for amc_file in amc_files:
                    if amc_file.stat().st_size == 0:
                        corrupted_files.append(str(amc_file))
                        issues.append(f"Subject {subject_num}: Empty AMC file {amc_file.name}")
                    else:
                        is_valid, error = validate_amc_file(amc_file)
                        if not is_valid:
                            corrupted_files.append(str(amc_file))
                            issues.append(f"Subject {subject_num}: Corrupted AMC {amc_file.name} - {error}")
        
        # Print summary of subjects missing ASF
        if subjects_missing_asf:
            if remove_missing:
                print(f"\n⚠ Subjects missing ASF skeleton file (REMOVED {len(removed_subjects)}/{len(subjects_missing_asf)}):")
                print(f"  {', '.join(removed_subjects)}")
            else:
                print(f"\n⚠ Subjects missing ASF skeleton file ({len(subjects_missing_asf)}):")
                print(f"  {', '.join(subjects_missing_asf)}")
        
        if corrupted_files:
            print(f"\n⚠ Corrupted files found ({len(corrupted_files)}):")
            for f in corrupted_files[:10]:
                print(f"  - {f}")
            if len(corrupted_files) > 10:
                print(f"  ... and {len(corrupted_files) - 10} more")
        
        return {
            'total_subjects': len(subjects) - len(removed_subjects), # Update total count
            'total_asf': sum(s['asf_files'] for s in subjects),
            'total_amc': sum(s['amc_files'] for s in subjects),
            'total_size_mb': sum(s['total_size_bytes'] for s in subjects) / (1024*1024),
            'subjects_missing_asf': subjects_missing_asf,
            'subjects_missing_amc': subjects_missing_amc,
            'corrupted_files': corrupted_files,
            'removed_subjects': removed_subjects, # Include removed list
            'issues': issues,
            'is_valid': len(issues) == 0,
        }
    
    def cleanup_temp_files(self):
        """Remove any temporary files left from interrupted downloads."""
        temp_files = list(self.output_dir.rglob("*.tmp"))
        for f in temp_files:
            f.unlink()
            logger.info(f"Removed temp file: {f}")
        
        if temp_files:
            print(f"Cleaned up {len(temp_files)} temporary files")
        
        return len(temp_files)
    
    def close(self):
        """Close the session and cleanup resources."""
        self.session.close()
        logger.info("Downloader session closed")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def setup_logging(level: int = logging.INFO, log_file: Optional[str] = None):
    """
    Configure logging for the downloader.
    
    Args:
        level: Logging level (e.g., logging.DEBUG, logging.INFO)
        log_file: Optional file path to write logs to
    """
    handlers = [logging.StreamHandler()]
    
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers,
    )


def main():
    """Main entry point with CLI support."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Download CMU Motion Capture ASF/AMC files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python downloader.py                    # Download entire dataset (bulk)
  python downloader.py -s 86              # Download subject 86 only
  python downloader.py --range 1-50       # Download subjects 1-50
  python downloader.py --individual       # Download all subjects individually
  python downloader.py --verify           # Verify downloaded dataset
  python downloader.py --cleanup          # Remove temporary files
        """,
    )
    parser.add_argument(
        '-o', '--output',
        default='cmu_mocap_data',
        help='Output directory (default: cmu_mocap_data)'
    )
    parser.add_argument(
        '-s', '--subject',
        type=int,
        help='Download specific subject number only'
    )
    parser.add_argument(
        '--range',
        type=str,
        help='Download subject range (e.g., "1-50")'
    )
    parser.add_argument(
        '--bulk',
        action='store_true',
        default=True,
        help='Download bulk archive (default, recommended)'
    )
    parser.add_argument(
        '--individual',
        action='store_true',
        help='Download subjects individually (slower but more control)'
    )
    parser.add_argument(
        '-w', '--workers',
        type=int,
        default=4,
        help='Number of concurrent download workers (default: 4)'
    )
    parser.add_argument(
        '-f', '--force',
        action='store_true',
        help='Force re-download even if files exist'
    )
    parser.add_argument(
        '--verify',
        action='store_true',
        help='Verify downloaded dataset integrity'
    )
    parser.add_argument(
        '--cleanup',
        action='store_true',
        help='Remove temporary files from interrupted downloads'
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    parser.add_argument(
        '--log-file',
        type=str,
        help='Write logs to file'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.WARNING
    setup_logging(level=log_level, log_file=args.log_file)
    
    # Create downloader
    with CMUMocapDownloader(
        output_dir=args.output,
        max_workers=args.workers,
    ) as downloader:
        
        if args.cleanup:
            # Cleanup temporary files
            count = downloader.cleanup_temp_files()
            print(f"Removed {count} temporary files")
            return
        
        if args.verify:
            # Verify dataset
            result = downloader.verify_dataset()
            print(f"\n{'=' * 60}")
            print("Dataset Verification")
            print(f"{'=' * 60}")
            print(f"  Total subjects: {result['total_subjects']}")
            print(f"  Total ASF files: {result['total_asf']}")
            print(f"  Total AMC files: {result['total_amc']}")
            print(f"  Total size: {result['total_size_mb']:.1f}MB")
            print(f"  Valid: {'✓ Yes' if result['is_valid'] else '✗ No'}")
            
            if result['issues']:
                print(f"\nIssues found ({len(result['issues'])}):")
                for issue in result['issues'][:10]:
                    print(f"  - {issue}")
                if len(result['issues']) > 10:
                    print(f"  ... and {len(result['issues']) - 10} more")
            return
        
        if args.subject:
            # Download single subject
            result = downloader.download_subject(args.subject, force=args.force)
            if result.success:
                print(f"\n✓ Subject {args.subject} saved to: {result.path}")
            else:
                print(f"\n✗ Failed to download subject {args.subject}: {result.error}")
            
        elif args.range:
            # Download range of subjects
            start, end = map(int, args.range.split('-'))
            path = downloader.download_all_subjects(start, end, force=args.force)
            print(f"\n✓ Subjects saved to: {path}")
            
        elif args.individual:
            # Download all subjects individually
            path = downloader.download_all_subjects(force=args.force)
            print(f"\n✓ Subjects saved to: {path}")
            
        else:
            # Default: bulk download
            path = downloader.download_bulk(force=args.force)
            print(f"\n✓ Data saved to: {path}")
        
        # Print summary
        subjects = downloader.list_subjects()
        if subjects:
            print(f"\n{'=' * 60}")
            print("Summary")
            print(f"{'=' * 60}")
            print(f"  Subjects: {len(subjects)}")
            total_asf = sum(s['asf_files'] for s in subjects)
            total_amc = sum(s['amc_files'] for s in subjects)
            total_size = sum(s['total_size_bytes'] for s in subjects) / (1024*1024)
            print(f"  ASF files: {total_asf}")
            print(f"  AMC files: {total_amc}")
            print(f"  Total size: {total_size:.1f}MB")


if __name__ == "__main__":
    main()