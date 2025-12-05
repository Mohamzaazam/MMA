"""
Main converter module for ASF/AMC to BVH conversion.

Provides the main ASFAMCtoBVH converter class and command-line interface
for converting Acclaim skeleton/motion files to BVH format.
"""

import argparse

from .config import ConversionConfig, RotationOrder
from .asf_parser import ASFParser
from .amc_parser import AMCParser
from .bvh_writer import BVHWriter


class ASFAMCtoBVH:
    """Main converter class for ASF/AMC to BVH conversion"""
    
    def __init__(self, config: ConversionConfig = None):
        self.config = config or ConversionConfig()
    
    def convert(self, asf_path: str, amc_path: str, output_path: str) -> str:
        """Convert ASF/AMC files to BVH format."""
        # Parse ASF
        print(f"Parsing skeleton: {asf_path}")
        asf_parser = ASFParser()
        skeleton = asf_parser.parse(asf_path)
        
        # Parse AMC
        print(f"Parsing motion: {amc_path}")
        amc_parser = AMCParser(skeleton)
        motion = amc_parser.parse(amc_path)
        
        # Write BVH
        print(f"Writing BVH: {output_path}")
        writer = BVHWriter(skeleton, motion, self.config)
        writer.write(output_path)
        
        # Summary
        print(f"\nConversion complete!")
        print(f"  Frames: {motion.frame_count}")
        print(f"  Joints: {len(skeleton.joints)}")
        print(f"  FPS: {self.config.fps}")
        print(f"  Scale: {self.config.scale}")
        print(f"  Rotation: {self.config.rotation_order.name}")
        if self.config.collapse_joints:
            print(f"  Collapsed: {', '.join(sorted(self.config.collapse_joints))}")
        if self.config.joint_name_map:
            print(f"  Renamed: Yes (Character1 format)")
        
        return output_path


def main():
    """Command-line interface for ASF/AMC to BVH conversion"""
    parser = argparse.ArgumentParser(
        description='Convert ASF/AMC to BVH',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  # Match C output exactly (default)
  %(prog)s skeleton.asf motion.amc -o output.bvh
  
  # Apply scaling
  %(prog)s skeleton.asf motion.amc -o output.bvh --scale 2.54
  
  # Use walk_.bvh format (collapse joints, rename, scale)
  %(prog)s skeleton.asf motion.amc -o output.bvh --walk-bvh
  
  # Custom options
  %(prog)s skeleton.asf motion.amc -o output.bvh --collapse --rename --scale 2.54
        """
    )
    
    parser.add_argument('asf', help='Input ASF skeleton file')
    parser.add_argument('amc', help='Input AMC motion file')
    parser.add_argument('-o', '--output', default='output.bvh', help='Output BVH file (default: output.bvh)')
    parser.add_argument('-f', '--fps', type=float, default=120.0, help='Frames per second (default: 120)')
    parser.add_argument('-r', '--rotation', choices=['ZYX', 'ZXY', 'XYZ'], default='ZYX',
                       help='Rotation order (default: ZYX)')
    parser.add_argument('-s', '--scale', type=float, default=1.0,
                       help='Scale factor for positions (default: 1.0)')
    parser.add_argument('--collapse', action='store_true',
                       help='Collapse lhipjoint, rhipjoint, thorax, upperneck')
    parser.add_argument('--rename', action='store_true',
                       help='Rename joints to Character1_* format')
    parser.add_argument('--walk-bvh', action='store_true',
                       help='Use walk_.bvh format (enables --collapse, --rename, --scale 2.54)')
    
    args = parser.parse_args()
    
    # Build configuration
    config = ConversionConfig()
    config.fps = args.fps
    config.rotation_order = RotationOrder[args.rotation]
    config.scale = args.scale
    
    # walk-bvh preset
    if args.walk_bvh:
        config.scale = (1 / 0.45) * 2.54
        config.collapse_joints = ConversionConfig.walk_bvh_collapse_joints()
        config.skip_joints = ConversionConfig.walk_bvh_skip_joints()
        config.end_joints = ConversionConfig.walk_bvh_end_joints()
        config.joint_name_map = ConversionConfig.walk_bvh_joint_map()
    else:
        # Individual options
        if args.collapse:
            config.collapse_joints = ConversionConfig.walk_bvh_collapse_joints()
            config.skip_joints = ConversionConfig.walk_bvh_skip_joints()
            config.end_joints = ConversionConfig.walk_bvh_end_joints()
        
        if args.rename:
            config.joint_name_map = ConversionConfig.walk_bvh_joint_map()
    
    # Convert
    converter = ASFAMCtoBVH(config)
    converter.convert(args.asf, args.amc, args.output)


if __name__ == '__main__':
    main()