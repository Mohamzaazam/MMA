#!/usr/bin/env python3
import os
import argparse
from pathlib import Path


def truncate_bvh(input_path, output_path, start_frame, num_frames):
    with open(input_path, 'r') as f:
        lines = f.readlines()

    with open(output_path, 'w') as f:
        motion_section = False
        frames_written = 0
        skipped_frames = 0
        
        # First pass to find original frame count
        original_frames = 0
        for line in lines:
             if line.strip().startswith('Frames:'):
                 parts = line.split()
                 if len(parts) > 1:
                     original_frames = int(parts[1])
                 break
        
        # Calculate new frame count
        if original_frames > 0:
            available_frames = max(0, original_frames - start_frame)
            if num_frames is None:
                new_total_frames = available_frames
            else:
                new_total_frames = min(available_frames, num_frames)
        else:
             # Fallback if Frames header not found or valid
             new_total_frames = 0 # Or handle differently

        for i, line in enumerate(lines):
            if line.strip().startswith('Frames:'):
                f.write(f"Frames:\t{new_total_frames}\n")
            elif line.strip() == 'MOTION':
                motion_section = True
                f.write(line)
            elif motion_section and line.strip().startswith('Frame Time:'):
                f.write(line)
            elif motion_section:
                # This is a data line (or empty line at end)
                if line.strip():
                    if skipped_frames < start_frame:
                        skipped_frames += 1
                    elif num_frames is None or frames_written < num_frames:
                        f.write(line)
                        frames_written += 1
                else:
                    f.write(line)
            else:
                # Header section
                f.write(line)
    
    print(f"Truncated BVH: {input_path} -> {output_path} (Skipped {skipped_frames}, Kept {frames_written} frames)")

def truncate_amc(input_path, output_path, start_frame, num_frames):
    with open(input_path, 'r') as f:
        lines = f.readlines()

    with open(output_path, 'w') as f:
        frames_skipped = 0
        frames_written = 0
        
        header_done = False
        skip_current_frame = False
        
        for line in lines:
            stripped = line.strip()
            
            # Check if it's a frame number (integer)
            if stripped.isdigit():
                header_done = True
                current_frame_idx = frames_skipped + frames_written # Logical index of found frames
                
                if frames_skipped < start_frame:
                    frames_skipped += 1
                    skip_current_frame = True
                elif num_frames is None or frames_written < num_frames:
                    skip_current_frame = False
                    frames_written += 1
                    # Renumber frame
                    f.write(f"{frames_written}\n")
                else:
                    # Limit reached
                    break
            elif not header_done:
                f.write(line)
            else:
                # Data line for the current frame
                if not skip_current_frame:
                     f.write(line)

    print(f"Truncated AMC: {input_path} -> {output_path} (Skipped {frames_skipped}, Kept {frames_written} frames)")

def main():
    parser = argparse.ArgumentParser(description='Truncate BVH and AMC files to a specific subset of frames.')
    parser.add_argument('input', help='Input file or directory')
    parser.add_argument('--frames', '-n', type=int, default=None, help='Number of frames to keep. If not specified, keeps all frames from start to end.')
    parser.add_argument('--start', '-s', type=int, default=0, help='Number of frames to skip at the beginning.')
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
            truncate_bvh(file_path, out_path, args.start, args.frames)
        elif file_path.suffix.lower() == '.amc':
            truncate_amc(file_path, out_path, args.start, args.frames)

if __name__ == '__main__':
    main()
