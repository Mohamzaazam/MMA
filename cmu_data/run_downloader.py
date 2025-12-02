#!/usr/bin/env python3
"""
Quick-start example for downloading CMU Motion Capture data.
Run this script directly to download the entire dataset.
"""

from downloader import CMUMocapDownloader

# Initialize downloader
downloader = CMUMocapDownloader(output_dir="cmu_mocap_data")

# Option 1: Download everything (recommended - single 500MB zip)
print("=" * 60)
print("CMU Motion Capture Dataset Downloader")
print("=" * 60)
print()
print("This will download all ASF/AMC files (~500MB) and organize")
print("them by subject number.")
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

# Uncomment ONE of the following options:

# OPTION 1: Download entire dataset (recommended)
downloader.download_bulk()

# OPTION 2: Download specific subject only
# downloader.download_subject(86)  # Downloads subject 86

# OPTION 3: Download a range of subjects
# downloader.download_all_subjects(start=1, end=20)

# Print what was downloaded
subjects = downloader.list_subjects()
print(f"\n{'=' * 60}")
print(f"Download complete!")
print(f"Subjects: {len(subjects)}")
for s in subjects[:5]:  # Show first 5
    print(f"  Subject {s['number']}: {s['asf_files']} ASF, {s['amc_files']} AMC files")
if len(subjects) > 5:
    print(f"  ... and {len(subjects) - 5} more subjects")