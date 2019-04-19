#!/usr/bin/env python3
from typing import Dict, List, Tuple, Iterable

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from asciifier import Charset, Charmap, ASCII, reverse_dict


def make_calibration_sheet(charset: Charset, wrap: int = 10, guidelines: bool = True) -> str:
    """Generate a calibration sheet from a charset."""
    lines = []
    for i in range(0, len(charset), wrap):
        lines.append(charset[i:i+wrap])
    if guidelines:
        h_line = '+'+'-'*wrap+'+'
        lines = ['|'+l for l in lines]
        lines.insert(0, h_line)
        lines.append('+')
    return '\n'.join(lines)


def build_charmap_from_img(cal_img: np.ndarray, charset: Charset = ASCII,
                           wrap: int = 10) -> Charmap:
    """Make a charmap from a corrected image of a calibration sheet."""
    num_lines = math.ceil(len(charset) / wrap)
    char_imgs = split_image(cal_img, cols=wrap, rows=num_lines)
    avg_bs = map(calc_brightness, char_imgs)
    # expand vals to fill 0, 255  TODO: explore other interpolation methods
    scaled_avg_bs = np.interp(avg_bs, [min(avg_bs), max(avg_bs)], [0, 255])
    charmap = make_8bit_charmap(scaled_avg_bs, charset)
    return charmap


def make_8bit_charmap(vals: List[float], charset: Charset = ASCII) -> Charmap:
    """Build a charmap from characters and values."""
    def map_vals_to_charset(vals, charset):
        return {c: int(round(vals[i])) for i, c in enumerate(charset)}
    return reverse_dict(map_vals_to_charset(vals, charset))


def sort_dict_by_value(d: Dict) -> Dict:
    """Sort a dictionary by value."""
    return {k: d[k] for k in sorted(d, key=d.get)}


def calc_brightness(img: np.ndarray) -> float:
    """Calculate the average brightness of an image."""
    return np.mean(img, dtype=float)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    main_group = parser.add_mutually_exclusive_group(required=True)
    main_group.add_argument('-g', '--generate-calibration', action='store_true',
        help='generate a calibration sheet to print and scan')
    # main_group.add_argument('-c', '--calibrate-from-image', metavar='IMAGE',
    #     type=argparse.FileType('rb'),
    #     help='calculate character weights from an image')

    args = parser.parse_args()

    if args.generate_calibration:
        print(make_calibration_sheet(ASCII))
        exit(0)
    # if args.calibrate_from_image:
    #     exit(0)