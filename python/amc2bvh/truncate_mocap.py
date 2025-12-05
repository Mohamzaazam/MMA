#!/usr/bin/env python3
import os
import argparse
from pathlib import Path

def truncate_bvh(input_path, output_path, max_frames):
    with open(input_path, 'r') as f:
        lines = f.readlines()

    with open(output_path, 'w') as f:
        motion_section = False
        frames_written = 0
        
        for i, line in enumerate(lines):
            if line.strip().startswith('Frames:'):
                parts = line.split()
                if len(parts) > 1:
                    original_frames = int(parts[1])
                    new_frames = min(original_frames, max_frames)
                    f.write(f"Frames:\t{new_frames}\n")
                else:
                    f.write(line)
            elif line.strip() == 'MOTION':
                motion_section = True
                f.write(line)
            elif motion_section and line.strip().startswith('Frame Time:'):
                f.write(line)
            elif motion_section:
                # This is a data line (or empty line at end)
                if line.strip():
                    if frames_written < max_frames:
                        f.write(line)
                        frames_written += 1
                else:
                    f.write(line)
            else:
                # Header section
                f.write(line)
    
    print(f"Truncated BVH: {input_path} -> {output_path} ({frames_written} frames)")

def truncate_amc(input_path, output_path, max_frames):
    with open(input_path, 'r') as f:
        lines = f.readlines()

    with open(output_path, 'w') as f:
        frames_found = 0
        current_frame_lines = []
        
        # AMC files have a header, then frames starting with a frame number
        # We need to preserve the header
        
        header_done = False
        
        for line in lines:
            stripped = line.strip()
            
            # Check if it's a frame number (integer)
            if stripped.isdigit():
                header_done = True
                frames_found += 1
                
                if frames_found > max_frames:
                    break
                
                f.write(line)
            elif not header_done:
                f.write(line)
            else:
                # Data line for the current frame
                if frames_found <= max_frames:
                    f.write(line)

    print(f"Truncated AMC: {input_path} -> {output_path} ({min(frames_found, max_frames)} frames)")

def main():
    parser = argparse.ArgumentParser(description='Truncate BVH and AMC files to a specific number of frames.')
    parser.add_argument('input', help='Input file or directory')
    parser.add_argument('--frames', '-n', type=int, default=100, help='Number of frames to keep')
    parser.add_argument('--output', '-o', help='Output file or directory (optional). If omitted, adds _truncated suffix.')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite input files')

    args = parser.parse_args()
    
    input_path = Path(args.input)
    
    if input_path.is_file():
        files = [input_path]
        output_dir = input_path.parent if not args.output else Path(args.output).parent
    elif input_path.is_dir():
        files = list(input_path.glob('**/*.bvh')) + list(input_path.glob('**/*.amc'))
        output_dir = input_path if not args.output else Path(args.output)
    else:
        print(f"Error: {input_path} not found.")
        return

    for file_path in files:
        if args.overwrite:
            out_path = file_path
        elif args.output and Path(args.output).suffix:
             # If output is a file path (only valid for single input file)
             if len(files) == 1:
                 out_path = Path(args.output)
             else:
                 print("Error: Output path is a file but multiple input files found.")
                 return
        elif args.output:
            # Output is a directory
            rel_path = file_path.relative_to(input_path) if input_path.is_dir() else file_path.name
            out_path = Path(args.output) / rel_path
            out_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            # Default: add suffix
            out_path = file_path.with_name(f"{file_path.stem}_truncated{file_path.suffix}")

        if file_path.suffix.lower() == '.bvh':
            truncate_bvh(file_path, out_path, args.frames)
        elif file_path.suffix.lower() == '.amc':
            truncate_amc(file_path, out_path, args.frames)

if __name__ == '__main__':
    main()
