"""
CLI entry point for segmentation tools.

This will be populated by migrating segmentation_tools.py and multilabel_segmt_tools.py.
"""

import argparse
import sys


def main():
    """Main entry point for imatools-segmentation command."""
    parser = argparse.ArgumentParser(
        description="Segmentation manipulation tools",
        prog="imatools-segmentation"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Placeholder subcommands
    extract_parser = subparsers.add_parser("extract", help="Extract labels")
    mask_parser = subparsers.add_parser("mask", help="Mask segmentations")
    merge_parser = subparsers.add_parser("merge", help="Merge labels")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    print(f"imatools-segmentation {args.command}: Command stub - migration in progress")
    return 0


if __name__ == "__main__":
    sys.exit(main())
