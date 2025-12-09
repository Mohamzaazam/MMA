#!/usr/bin/env python3
"""
Quick-start example for downloading CMU Motion Capture data.
Run this script directly to download the entire dataset.

Enhanced version with:
- Concurrent downloads
- Progress bars
- Resumable downloads
- Dataset verification
"""

from downloader import CMUMocapDownloader, setup_logging
import logging

# Optional: Enable verbose logging to see detailed progress
# setup_logging(level=logging.DEBUG)

# Initialize downloader with custom settings
downloader = CMUMocapDownloader(
    output_dir="E:/database/cmu/",
    max_workers=4,      # Number of concurrent downloads
    max_retries=3,      # Retry failed downloads up to 3 times
    timeout=120,         # 60 second timeout per request
)

# Display welcome banner
print("=" * 60)
print("CMU Motion Capture Dataset Downloader (Enhanced)")
print("=" * 60)
print()
print("Features:")
print("  • Concurrent downloads for faster individual downloads")
print("  • Automatic retry with exponential backoff")
print("  • Resume support (skips already downloaded files)")
print("  • Progress bars for tracking download status")
print("  • Dataset verification")
print()
print("Output structure:")
print("  cmu_mocap_data/")
print("  └── subjects/")
print("      ├── 01/")
print("      │   ├── 01.asf")
print("      │   ├── 01_01.amc")
print("      │   └── 01_02.amc")
print("      ├── 02/")
print("      │   ├── 02.asf")
print("      │   └── 02_01.amc")
print("      └── ...")
print()

# ============================================================================
# DOWNLOAD OPTIONS - Uncomment ONE of the following:
# ============================================================================

# OPTION 1: Download entire dataset via bulk zip (RECOMMENDED)
# This downloads a single ~500MB zip file - fastest and most reliable
downloader.download_bulk()

# OPTION 2: Download specific subject only
# downloader.download_subject(86)

# OPTION 3: Download a range of subjects (uses concurrent downloads)
# downloader.download_all_subjects(start=1, end=20)

# OPTION 4: Download specific subjects concurrently
# results = downloader.download_subjects_concurrent([1, 5, 10, 86, 143])

# ============================================================================
# POST-DOWNLOAD ACTIONS
# ============================================================================

# Verify dataset integrity and auto-clean invalid subjects
print()
print("=" * 60)
print("Verifying dataset...")
print("=" * 60)
verification = downloader.verify_dataset(validate_content=True, remove_missing=True)
print(f"  Total subjects: {verification['total_subjects']}")
print(f"  Total ASF files: {verification['total_asf']}")
print(f"  Total AMC files: {verification['total_amc']}")
print(f"  Total size: {verification['total_size_mb']:.1f}MB")
print(f"  Corrupted files: {len(verification['corrupted_files'])}")
print(f"  Valid: {'✓ Yes' if verification['is_valid'] else '✗ No'}")

if verification['issues']:
    print(f"\n  Issues found ({len(verification['issues'])}):")
    for issue in verification['issues'][:5]:
        print(f"    - {issue}")
    if len(verification['issues']) > 5:
        print(f"    ... and {len(verification['issues']) - 5} more")

if verification.get('removed_subjects'):
    print(f"\n  ⚠ REMOVED {len(verification['removed_subjects'])} subjects due to missing skeleton (ASF) file:")
    print(f"    {', '.join(verification['removed_subjects'])}")


# Print what was downloaded
subjects = downloader.list_subjects()
print(f"\n{'=' * 60}")
print(f"Download Complete!")
print(f"{'=' * 60}")
print(f"Subjects downloaded: {len(subjects)}")

for s in subjects[:5]:  # Show first 5
    size_kb = s['total_size_bytes'] / 1024
    print(f"  Subject {s['number']}: {s['asf_files']} ASF, {s['amc_files']} AMC files ({size_kb:.0f}KB)")

if len(subjects) > 5:
    print(f"  ... and {len(subjects) - 5} more subjects")

# Cleanup (close the session)
downloader.close()