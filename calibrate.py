#!/usr/bin/env python3
"""Generate and parse calibration papers"""
import json
import math
import string
from typing import Dict, List



Charlist = List[str]  # list of characters to use
Charmap = Dict[str, int]  # mapping of character to luminance

ASCII: Charlist = string.digits + string.ascii_letters + string.punctuation + ' '  # doesn't include whitespace

aspect_ratio = 1.0  # width/height of character space

def get_calibration_sheet(charlist: Charlist, wrap: int = 10):
    lines = []
    for i in range(0, len(charlist), wrap):
        lines.append(charlist[i:i+wrap])
    return lines

def build_charmap_from_sheet(img, charlist: Charlist,
                             wrap: int = 10) -> Charmap:
    return {c: ord(c) for c in charlist}


def split_image(img, rows: int = 10, cols: int = 10) -> List[List[int]]:
    """Split an image into a grid of subimages"""
    pass

def calc_luminance(img) -> int:
    """Calculate the average luminance of an image."""
    pass

def reverse_dict(d: Dict) -> Dict:
    """Reverse a dict, with common values grouped into a list.

    >>> reverse_dict({'a': 1, 'b': 1, 'c': 3})
    {1: ['a', 'b'], 3: ['c']}
    """
    rdict: Dict = {}
    for k, v in d.items():
        rdict.setdefault(v, []).append(k)
    return rdict

def sort_dict_by_value(d: Dict) -> Dict:
    return {k: d[k] for k in sorted(d, key=d.get)}

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    main_group = parser.add_mutually_exclusive_group()
    main_group.add_argument('-g', '--generate-calibration', action='store_true',
        help='generate a calibration sheet to print and scan')
    main_group.add_argument('-c', '--calibrate-from-image',
        type=argparse.FileType('rb'),
        help='calculate character weights from an image')
    # TODO: add wrap, charlist to cmdline interface

    args = parser.parse_args()

    if args.generate_calibration:
        [print(l) for l in get_calibration_sheet(ASCII)]
        exit(0)
    if args.calibrate_from_image:
        # build_charmap_from_sheet()
        exit(0)
