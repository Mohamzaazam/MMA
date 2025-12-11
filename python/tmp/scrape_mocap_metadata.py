#!/usr/bin/env python3
"""
CMU Motion Capture Metadata Scraper

Scrapes motion trial descriptions from the CMU mocap database
and saves them to subject folders for easy reference.

Usage:
    python scrape_mocap_metadata.py /mnt/e/database/cmu
"""

import argparse
import json
import re
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import requests
from bs4 import BeautifulSoup
from urllib3.exceptions import InsecureRequestWarning

# Suppress SSL warnings (CMU site has certificate issues)
warnings.filterwarnings('ignore', category=InsecureRequestWarning)

# CMU Mocap search URL
SEARCH_URL = "https://mocap.cs.cmu.edu/search.php?subjectnumber=%25&motion=%25"


def fetch_mocap_page() -> str:
    """Fetch the CMU mocap search results page."""
    print("Fetching CMU mocap database...")
    response = requests.get(SEARCH_URL, verify=False, timeout=60)
    response.raise_for_status()
    print(f"  Retrieved {len(response.text):,} bytes")
    return response.text


def parse_mocap_data(html: str) -> Dict[str, Dict]:
    """
    Parse the mocap HTML to extract subject and trial information.
    
    Returns:
        Dict mapping subject_id to dict with:
            - description: subject-level description
            - trials: dict mapping trial_id to motion description
    """
    soup = BeautifulSoup(html, 'html.parser')
    subjects = {}
    
    # Find all table rows
    rows = soup.find_all('tr')
    
    current_subject_id = None
    current_subject_desc = None
    
    for row in rows:
        cells = row.find_all('td')
        
        # Check if this is a subject header row (has COLSPAN=3 with subject info)
        colspan_cell = row.find('td', colspan='3')
        if colspan_cell:
            text = colspan_cell.get_text(strip=True)
            # Pattern: "Subject #1 (climb, swing, hang on playground equipment) file index"
            match = re.match(r'Subject #(\d+)\s*\(([^)]+)\)', text)
            if match:
                current_subject_id = match.group(1).zfill(2)
                current_subject_desc = match.group(2).strip()
                subjects[current_subject_id] = {
                    'description': current_subject_desc,
                    'trials': {}
                }
                continue
        
        # Check if this is a trial row (has trial number and description)
        if current_subject_id and len(cells) >= 3:
            # Skip header rows
            first_cell_text = cells[0].get_text(strip=True)
            if first_cell_text == 'Image':
                continue
            
            # Trial rows have: empty cell, trial #, motion description, ...
            try:
                trial_num_text = cells[1].get_text(strip=True)
                if trial_num_text.isdigit():
                    trial_num = int(trial_num_text)
                    trial_id = f"{current_subject_id}_{str(trial_num).zfill(2)}"
                    motion_desc = cells[2].get_text(strip=True)
                    
                    if motion_desc and motion_desc not in ['Motion Description', '']:
                        subjects[current_subject_id]['trials'][trial_id] = motion_desc
            except (IndexError, ValueError):
                continue
    
    return subjects


def save_metadata(subjects: Dict[str, Dict], output_dir: Path, format: str = 'txt'):
    """
    Save metadata to subject folders.
    
    Args:
        subjects: Parsed subject data
        output_dir: Base output directory (e.g., /mnt/e/database/cmu/subjects)
        format: Output format ('txt' or 'json')
    """
    subjects_dir = output_dir / 'subjects' if (output_dir / 'subjects').exists() else output_dir
    
    saved_count = 0
    
    for subject_id, data in subjects.items():
        subject_dir = subjects_dir / subject_id
        
        if not subject_dir.exists():
            print(f"  Skipping subject {subject_id}: directory not found")
            continue
        
        if format == 'json':
            metadata_file = subject_dir / 'trials.json'
            with open(metadata_file, 'w') as f:
                json.dump(data, f, indent=2)
        else:
            # Text format: one line per trial
            metadata_file = subject_dir / 'trials.txt'
            with open(metadata_file, 'w') as f:
                f.write(f"# Subject {subject_id}: {data['description']}\n")
                f.write(f"# Format: trial_id: motion_description\n\n")
                for trial_id, desc in sorted(data['trials'].items()):
                    f.write(f"{trial_id}: {desc}\n")
        
        saved_count += 1
    
    return saved_count


def save_master_catalog(subjects: Dict[str, Dict], output_dir: Path):
    """Save a master catalog of all trials."""
    catalog_file = output_dir / 'motion_catalog.txt'
    
    with open(catalog_file, 'w') as f:
        f.write("# CMU Motion Capture Database - Trial Catalog\n")
        f.write("# Format: trial_id: motion_description\n")
        f.write("# Use this file to search for specific motions\n\n")
        
        total_trials = 0
        for subject_id, data in sorted(subjects.items()):
            f.write(f"\n## Subject {subject_id}: {data['description']}\n")
            for trial_id, desc in sorted(data['trials'].items()):
                f.write(f"{trial_id}: {desc}\n")
                total_trials += 1
    
    print(f"  Saved master catalog: {catalog_file}")
    print(f"  Total trials: {total_trials}")
    
    # Also save as JSON for programmatic access
    json_file = output_dir / 'motion_catalog.json'
    with open(json_file, 'w') as f:
        json.dump(subjects, f, indent=2)
    print(f"  Saved JSON catalog: {json_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Scrape motion trial metadata from CMU mocap database',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  # Scrape and save to CMU data directory
  %(prog)s /mnt/e/database/cmu
  
  # Save as JSON format
  %(prog)s /mnt/e/database/cmu --format json
        """
    )
    
    parser.add_argument('--output_dir', default='/mnt/e/database/cmu',
                        help='Output directory (CMU data root, e.g., /mnt/e/database/cmu)')
    parser.add_argument('--format', choices=['txt', 'json'], default='txt',
                        help='Output format for per-subject files (default: txt)')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    if not output_dir.exists():
        print(f"Error: Output directory does not exist: {output_dir}")
        sys.exit(1)
    
    print("=" * 60)
    print("CMU Motion Capture Metadata Scraper")
    print("=" * 60)
    
    # Fetch and parse
    html = fetch_mocap_page()
    subjects = parse_mocap_data(html)
    
    print(f"\nParsed {len(subjects)} subjects")
    total_trials = sum(len(s['trials']) for s in subjects.values())
    print(f"Total trials: {total_trials}")
    
    # Save master catalog
    print(f"\nSaving master catalog to {output_dir}...")
    save_master_catalog(subjects, output_dir)
    
    # Save per-subject metadata
    print(f"\nSaving per-subject metadata...")
    saved = save_metadata(subjects, output_dir, args.format)
    print(f"  Saved metadata for {saved} subjects")
    
    print("\nDone!")


if __name__ == '__main__':
    main()
