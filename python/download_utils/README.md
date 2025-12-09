# CMU Motion Capture Data Downloader

Download and organize ASF/AMC motion capture files from the [CMU Graphics Lab Motion Capture Database](http://mocap.cs.cmu.edu/).

## Quick Start

```bash
# Download the entire dataset (~500MB)
python download_cmu_mocap.py
```

## Output Structure

After download, data is organized by subject number:

Default path: `E:/database/cmu/`

```
cmu_mocap_data/
└── subjects/
    ├── 01/
    │   ├── 01.asf          # Skeleton definition
    │   ├── 01_01.amc       # Motion 1
    │   ├── 01_02.amc       # Motion 2
    │   └── ...
    ├── 02/
    │   ├── 02.asf
    │   └── 02_01.amc
    ├── 86/
    │   ├── 86.asf
    │   ├── 86_01.amc
    │   └── ...
    └── ...
```

## File Formats

- **ASF (Acclaim Skeleton File)**: Defines the skeleton hierarchy and bone properties
- **AMC (Acclaim Motion Capture)**: Contains frame-by-frame joint rotation data

## Usage Options

### As a Module

```python
from cmu_mocap_downloader import CMUMocapDownloader

downloader = CMUMocapDownloader(output_dir="my_mocap_data")

# Download entire dataset
downloader.download_bulk()

# Or download specific subject
downloader.download_subject(86)

# Or download a range
downloader.download_all_subjects(start=1, end=50)
```

### Command Line

```bash
# Download everything (default)
python cmu_mocap_downloader.py

# Download to custom directory
python cmu_mocap_downloader.py -o /path/to/output

# Download specific subject
python cmu_mocap_downloader.py -s 86

# Download range of subjects
python cmu_mocap_downloader.py --range 1-50

# Download subjects individually (slower but more control)
python cmu_mocap_downloader.py --individual
```

## Working with the Data

### Using with AMCParser (recommended)

```bash
pip install numpy matplotlib
```

```python
# https://github.com/CalciferZh/AMCParser
from amc_parser import parse_asf, parse_amc

asf_path = './cmu_mocap_data/subjects/01/01.asf'
amc_path = './cmu_mocap_data/subjects/01/01_01.amc'

joints = parse_asf(asf_path)
motions = parse_amc(amc_path)

# Access joint data for a specific frame
frame_idx = 180
joints['root'].set_motion(motions[frame_idx])
```

## Dataset Information

- **Source**: CMU Graphics Lab Motion Capture Database
- **Subjects**: ~144 human subjects
- **Motions**: ~2605 motion trials
- **Categories**: Walking, running, dancing, sports, and more
- **Frame Rate**: 120 fps
- **License**: Free for research and commercial use

## Citation

If using this data, please acknowledge:

> The data used in this project was obtained from mocap.cs.cmu.edu.
> The database was created with funding from NSF Grant #0196217.

## Troubleshooting

**SSL Certificate Errors**: The script handles these automatically by disabling SSL verification for the CMU server.

**Slow Download**: The bulk download (~500MB) is faster than downloading subjects individually.

**Missing Subjects**: Not all subject numbers exist (e.g., some numbers are skipped in the original database).