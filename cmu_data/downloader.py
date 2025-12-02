#!/usr/bin/env python3
"""
CMU Motion Capture Database Downloader
Downloads ASF and AMC files and organizes them by subject number.

Data source: http://mocap.cs.cmu.edu/
"""

import os
import re
import ssl
import zipfile
import urllib.request
from pathlib import Path
from typing import Optional
import shutil


class CMUMocapDownloader:
    """Download and organize CMU Motion Capture data."""
    
    # Primary bulk download URL for all ASF/AMC files
    BULK_ZIP_URL = "http://mocap.cs.cmu.edu:8080/allasfamc.zip"
    
    # Alternative: Individual subject download base URL
    BASE_URL = "http://mocap.cs.cmu.edu/subjects"
    
    def __init__(self, output_dir: str = "cmu_mocap_data"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create SSL context that bypasses certificate verification
        # (CMU site sometimes has certificate issues)
        self.ssl_context = ssl.create_default_context()
        self.ssl_context.check_hostname = False
        self.ssl_context.verify_mode = ssl.CERT_NONE
    
    def download_bulk(self, extract: bool = True) -> Path:
        """
        Download the entire ASF/AMC dataset as a single zip file.
        This is the recommended approach - faster and more reliable.
        
        Args:
            extract: If True, extract and organize files after download
            
        Returns:
            Path to the downloaded/extracted directory
        """
        zip_path = self.output_dir / "allasfamc.zip"
        
        print(f"Downloading CMU MoCap bulk archive...")
        print(f"URL: {self.BULK_ZIP_URL}")
        print(f"This may take a while (~500MB)...")
        
        try:
            self._download_file(self.BULK_ZIP_URL, zip_path)
            print(f"✓ Downloaded to {zip_path}")
        except Exception as e:
            print(f"✗ Bulk download failed: {e}")
            print("Falling back to individual subject download...")
            return self.download_all_subjects()
        
        if extract:
            return self._extract_and_organize(zip_path)
        
        return zip_path
    
    def _download_file(self, url: str, dest: Path, show_progress: bool = True):
        """Download a file with progress indication."""
        # Try with SSL context first, then without
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, context=self.ssl_context) as response:
                total_size = int(response.headers.get('content-length', 0))
                downloaded = 0
                block_size = 8192
                
                with open(dest, 'wb') as f:
                    while True:
                        buffer = response.read(block_size)
                        if not buffer:
                            break
                        downloaded += len(buffer)
                        f.write(buffer)
                        
                        if show_progress and total_size > 0:
                            pct = (downloaded / total_size) * 100
                            print(f"\rProgress: {pct:.1f}% ({downloaded // (1024*1024)}MB)", end="")
                
                if show_progress:
                    print()  # New line after progress
                    
        except urllib.error.URLError:
            # Retry without SSL context for HTTP URLs
            with urllib.request.urlopen(url) as response:
                with open(dest, 'wb') as f:
                    shutil.copyfileobj(response, f)
    
    def _extract_and_organize(self, zip_path: Path) -> Path:
        """Extract zip and organize files by subject number."""
        print(f"Extracting and organizing files...")
        
        # Create temp extraction directory
        temp_dir = self.output_dir / "_temp_extract"
        temp_dir.mkdir(exist_ok=True)
        
        # Extract all files
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(temp_dir)
        
        # Organize by subject number
        organized_dir = self.output_dir / "subjects"
        organized_dir.mkdir(exist_ok=True)
        
        # Find all ASF and AMC files
        asf_files = list(temp_dir.rglob("*.asf"))
        amc_files = list(temp_dir.rglob("*.amc"))
        
        print(f"Found {len(asf_files)} ASF files and {len(amc_files)} AMC files")
        
        # Organize files
        subjects_found = set()
        
        for asf in asf_files:
            subject_num = self._extract_subject_number(asf.name)
            if subject_num:
                subjects_found.add(subject_num)
                subject_dir = organized_dir / subject_num
                subject_dir.mkdir(exist_ok=True)
                shutil.copy2(asf, subject_dir / asf.name)
        
        for amc in amc_files:
            subject_num = self._extract_subject_number(amc.name)
            if subject_num:
                subjects_found.add(subject_num)
                subject_dir = organized_dir / subject_num
                subject_dir.mkdir(exist_ok=True)
                shutil.copy2(amc, subject_dir / amc.name)
        
        # Cleanup
        shutil.rmtree(temp_dir)
        
        print(f"✓ Organized {len(subjects_found)} subjects into {organized_dir}")
        return organized_dir
    
    def _extract_subject_number(self, filename: str) -> Optional[str]:
        """Extract subject number from filename (e.g., '01.asf' -> '01')."""
        # Pattern: XX.asf or XX_YY.amc
        match = re.match(r'^(\d+)(?:_\d+)?\.(?:asf|amc)$', filename)
        if match:
            return match.group(1).zfill(2)  # Zero-pad to 2 digits
        return None
    
    def download_subject(self, subject_num: int) -> Path:
        """
        Download ASF and AMC files for a specific subject.
        
        Args:
            subject_num: Subject number (e.g., 1, 86, 143)
            
        Returns:
            Path to subject directory
        """
        subject_str = str(subject_num).zfill(2)
        subject_dir = self.output_dir / "subjects" / subject_str
        subject_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Downloading subject {subject_str}...")
        
        # Download ASF file
        asf_url = f"{self.BASE_URL}/{subject_str}/{subject_str}.asf"
        asf_path = subject_dir / f"{subject_str}.asf"
        
        try:
            self._download_file(asf_url, asf_path, show_progress=False)
            print(f"  ✓ Downloaded {asf_path.name}")
        except Exception as e:
            print(f"  ✗ Failed to download ASF: {e}")
            return subject_dir
        
        # Download AMC files (try common patterns)
        amc_num = 1
        consecutive_failures = 0
        max_failures = 5  # Stop after 5 consecutive failures
        
        while consecutive_failures < max_failures:
            amc_filename = f"{subject_str}_{str(amc_num).zfill(2)}.amc"
            amc_url = f"{self.BASE_URL}/{subject_str}/{amc_filename}"
            amc_path = subject_dir / amc_filename
            
            try:
                self._download_file(amc_url, amc_path, show_progress=False)
                print(f"  ✓ Downloaded {amc_filename}")
                consecutive_failures = 0
            except:
                consecutive_failures += 1
            
            amc_num += 1
        
        return subject_dir
    
    def download_all_subjects(self, start: int = 1, end: int = 144) -> Path:
        """
        Download ASF/AMC files for a range of subjects.
        
        Args:
            start: Starting subject number
            end: Ending subject number (inclusive)
            
        Returns:
            Path to subjects directory
        """
        subjects_dir = self.output_dir / "subjects"
        subjects_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Downloading subjects {start} to {end}...")
        
        for subject_num in range(start, end + 1):
            try:
                self.download_subject(subject_num)
            except Exception as e:
                print(f"  ⚠ Subject {subject_num} failed: {e}")
        
        return subjects_dir
    
    def list_subjects(self) -> list:
        """List all downloaded subject directories."""
        subjects_dir = self.output_dir / "subjects"
        if not subjects_dir.exists():
            return []
        
        subjects = []
        for d in sorted(subjects_dir.iterdir()):
            if d.is_dir() and d.name.isdigit():
                asf_count = len(list(d.glob("*.asf")))
                amc_count = len(list(d.glob("*.amc")))
                subjects.append({
                    'number': d.name,
                    'path': d,
                    'asf_files': asf_count,
                    'amc_files': amc_count
                })
        
        return subjects


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Download CMU Motion Capture ASF/AMC files"
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
    
    args = parser.parse_args()
    
    downloader = CMUMocapDownloader(output_dir=args.output)
    
    if args.subject:
        # Download single subject
        path = downloader.download_subject(args.subject)
        print(f"\n✓ Subject {args.subject} saved to: {path}")
        
    elif args.range:
        # Download range of subjects
        start, end = map(int, args.range.split('-'))
        path = downloader.download_all_subjects(start, end)
        print(f"\n✓ Subjects saved to: {path}")
        
    elif args.individual:
        # Download all subjects individually
        path = downloader.download_all_subjects()
        print(f"\n✓ Subjects saved to: {path}")
        
    else:
        # Default: bulk download
        path = downloader.download_bulk()
        print(f"\n✓ Data saved to: {path}")
    
    # Print summary
    subjects = downloader.list_subjects()
    if subjects:
        print(f"\nSummary: {len(subjects)} subjects downloaded")
        total_asf = sum(s['asf_files'] for s in subjects)
        total_amc = sum(s['amc_files'] for s in subjects)
        print(f"Total: {total_asf} ASF files, {total_amc} AMC files")


if __name__ == "__main__":
    main()