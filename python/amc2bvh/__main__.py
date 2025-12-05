"""
Entry point for running the package as a module.

Usage:
    python -m asfamc2bvh skeleton.asf motion.amc -o output.bvh
"""

from .main import main

if __name__ == '__main__':
    main()