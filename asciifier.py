#!/usr/bin/env python3
"""Generate and parse calibration papers"""
import json
import math
import string
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


# types
Charlist = List[str]  # list of characters to use
Charmap = Dict[int, List[str]]  # mapping of brightness to characters

ASCII: Charlist = string.digits + string.ascii_letters + string.punctuation + ' '  # doesn't include whitespace

aspect_ratio = 1.0  # width/height of character space


def get_calibration_sheet(charlist: Charlist, wrap: int = 10):
    lines = []
    for i in range(0, len(charlist), wrap):
        lines.append(charlist[i:i+wrap])
    return lines


def convert_image(f, width: int, charmap: Charmap, aspect_ratio: float) -> str:
    # # TODO: support given height
    # if width is None != height is None:
    #     # TODO: allow cropping/defining hard boundaries
    #     raise ValueError("width or height must be defined, but not both.")
    img = Image.open(f)
    ih, iw = img.size
    img_ratio = iw / ih
    new_h: float = (width / img_ratio)  # height based on image proportions
    scaled_h: float = new_h * aspect_ratio  # scaled to match charmap's aspect ratio
    new_size = (width, int(scaled_h))
    resized_img = img.resize(new_size)
    img_a = rgb2gs(np.asarray(resized_img))  # np array of img in grayscale

    return array2ascii(img_a, charmap)



def array2ascii(img: np.ndarray, charmap: Charmap) -> str:
    """Convert an image to ascii text."""
    def find_nearest(img_gs, vals):
        """Match each pixel of a gs image with an array of potential values.

        Expands the image into three dimensions with vals, and for each pixel
        returns the index of the closest val.

        Returns: A two-dimensional array of indices mapping to vals.
        """
        final_shape = (*np.shape(img_gs)[0:2], len(vals))  # 3D array of pixels and potential values
        a = np.ones(final_shape) * np.reshape(np.array(vals), (1, 1, len(vals)))
        diff = np.repeat(img.reshape((*np.shape(img)[0:2], 1)), len(vals), 2) - a
        idx = np.abs(diff).argmin(2)
        return idx
    f = find_nearest(img, list(charmap.keys()))
    lines = [''.join(map(lambda a: list(charmap.values())[a][0], row)) for row in f.astype(int)]
    return '\n'.join(lines) + '\n'


def build_charmap_from_img(cal_img: np.ndarray, charset: Charlist = ASCII,
                           wrap: int = 10) -> Charmap:
    """Make a charmap from a corrected image of a calibration sheet."""
    num_lines = math.ceil(len(charset) / wrap)
    char_imgs = split_image(cal_img, cols=wrap, rows=num_lines)
    avg_bs = map(calc_brightness, char_imgs)
    # expand vals to fill 0, 255  TODO: explore other interpolation methods
    scaled_avg_bs = np.interp(avg_bs, [min(avg_bs), max(avg_bs)], [0, 255])
    charmap = make_8bit_charmap(scaled_avg_bs, charset)
    return charmap


def make_8bit_charmap(vals: List[float], charset: Charlist = ASCII) -> Charmap:
    """Build a charmap from characters and values."""
    def map_vals_to_charset(vals, charset):
        return {c: int(round(vals[i])) for i, c in enumerate(charset)}
    return reverse_dict(map_vals_to_charset(vals, charset))


def split_image(img: np.ndarray, rows: int = 10, cols: int = 10) -> List[np.ndarray]:
    """Split an image into a grid of subimages.

    Returns: a one-dimensional list of images
    """
    h, w, d = np.shape(img)
    avg_h = h / 10
    avg_w = w / 10

    out = []
    h_ind = lambda a: math.floor(a*avg_h)
    w_ind = lambda a: math.floor(a*avg_w)
    # array index is height, width
    for hn, hs in enumerate(range(rows), 1):
        for wn, ws in enumerate(range(cols), 1):
            out.append(img[h_ind(hs):h_ind(hn), w_ind(ws):w_ind(wn), :])
    return out


def calc_brightness(img: np.ndarray) -> float:
    """Calculate the average brightness of an image."""
    return np.mean(img, dtype=float)


def rgb2gs(img: np.ndarray) -> np.ndarray:
    """Convert an RGB image to single-channel grayscale."""
    return np.mean(img[:,:,0:3], axis=2)


def expand_contrast(img: np.ndarray, lower=0, upper=255) -> np.ndarray:
    """Interpolate an image's values to fill a different range."""
    return np.interp(img, [min(img), max(img)], [lower, upper])


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
    """Sort a dictionary by value."""
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
    main_group.add_argument('-i', '--image', type=argparse.FileType('rb'),
        help='Image to convert to ascii')
    # TODO: add wrap, charlist to cmdline interface

    args = parser.parse_args()

    if args.generate_calibration:
        [print(l) for l in get_calibration_sheet(ASCII)]
        exit(0)
    if args.calibrate_from_image:
        exit(0)
    if args.image:
        with open('courier-scaled-charmap.json', 'r') as f:
            weights = json.loads(f.read())
        charmap = reverse_dict(weights)
        print(convert_image(
            args.image,
            width=80,
            charmap=charmap,
            aspect_ratio=6/10))  # 10 CPI and 6 LPI
        exit(0)
