#!/usr/bin/env python3
"""Generate and parse calibration papers"""
# TODO: determine metrics for ability to reproduce image
# TODO: add logging
import json
import math
import os
import string
from typing import Dict, List, Tuple, Iterable

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


# types
Charset = List[str]  # list of characters to use
Charmap = Dict[int, List[str]]  # mapping of brightness to characters

ASCII: Charset = string.digits + string.ascii_letters + string.punctuation + ' '  # doesn't include whitespace

aspect_ratio = 6/10  # width/height of character space, 10 CPI and 6 LPI


def reverse_dict(d: Dict) -> Dict:
    """Reverse a dict, with common values grouped into a list.

    >>> reverse_dict({'a': 1, 'b': 1, 'c': 3})
    {1: ['a', 'b'], 3: ['c']}
    """
    rdict: Dict = {}
    for k, v in d.items():
        rdict.setdefault(v, []).append(k)
    return rdict


def img_to_array(f, width: int, aspect_ratio: float) -> np.ndarray:
    """Convert an image file to a resampled numpy array."""
    img = Image.open(f)
    iw, ih = img.size
    img_ratio = iw / ih
    new_h: float = (width / img_ratio)  # height based on image proportions
    scaled_h: float = new_h * aspect_ratio  # scaled to match charmap's aspect ratio
    new_size = (width, int(scaled_h))
    resized_img = img.resize(new_size)
    img_a = np.asarray(resized_img)
    return img_a


def convert_image(f, width: int, charmap: Charmap, aspect_ratio: float) -> str:
    """Convert an image to ascii.

    Args:
        f: A filepath or file-like object of the image.
        charmap: The charmap to use.
        aspect_ratio: The ratio of width / height of character, used to
            compensate for stretching/compression of the output format.

    Returns: A string of characters from charmap.
    """
    # # TODO: support given height
    # if width is None != height is None:
    #     # TODO: allow cropping/defining hard boundaries
    #     raise ValueError("width or height must be defined, but not both.")
    img_a = img_to_array(f, width, aspect_ratio)
    shape = img_a.shape
    if len(shape) != 2:
        img_a = rgb2gs(img_a)  # np array of img in grayscale
    return array2ascii(expand_contrast(img_a), charmap)


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


def rgb2gs(img: np.ndarray) -> np.ndarray:
    """Convert an RGB image to single-channel grayscale."""
    return np.mean(img[:, :, 0:3], axis=2)


def img_rgb2cmyk(img: np.ndarray) -> np.ndarray:
    if img.shape[2] < 3:
        raise ValueError("Image does not have 3 channels")
    cmyk = np.zeros((*img.shape[:2], 4))  # output array
    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            cmyk[x, y, :] = rgb2cmyk(*img[x, y, :3])
    return 255 - np.clip(cmyk, 0, 255) * 255


def rgb2cmyk(r, g, b) -> Tuple[int, int, int, int]:
    rp, gp, bp = map(lambda a: a/255, (r, g, b))
    k = 1 - max(rp, gp, bp)
    if k == 1:  # true black
        return 0, 0, 0, k
    c, m, y = map((lambda a: (1-a-k) / (1-k)), (rp, gp, bp))
    return c, m, y, k


def expand_contrast(img: np.ndarray, lower=0, upper=255) -> np.ndarray:
    """Interpolate an image's values to fill a different range."""
    return np.interp(img, [np.min(img), np.max(img)], [lower, upper])


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    # main_group = parser.add_mutually_exclusive_group(required=True)
    parser.add_argument('image', help='Image to convert to ascii')
    parser.add_argument('-w', '--width', type=int, default=80,
        help='Width of output text in characters')
    output_group = parser.add_mutually_exclusive_group()
    output_group.add_argument('--rgb', action='store_true',
        help="Write Red, Green, and Blue channels to separate files")
    output_group.add_argument('--cmyk', action='store_true',
        help="Convert to CMYK and write to separate files")
    # TODO: add wrap, charlist to cmdline interface
    parser.add_argument('-c', '--expand-contrast', action='store_true',
        help="Expand image values to fill 0-255 range.")
    parser.add_argument('out', nargs="?", default=".", help='Output location')

    args = parser.parse_args()

    # deal with in and out paths
    # Use infile name if out is a directory, ala `mv`, but replace extension
    # I used plain str for infile in order to get filename
    # TODO: pull out into function, create "Path" objects that implement "with" interface?
    # TODO: support stdout
    infile = args.image
    if not os.path.exists(infile):
        parser.error("image does not exist")
    infile_name, infile_ext = os.path.splitext(os.path.split(infile)[1])

    out = args.out
    if os.path.isdir(out):
        # if a directory is given, use the input filename and .txt
        outpath = out
        outfile_ext = "txt"
        outfile_name = infile_name
        outfile = os.path.join(outpath, outfile_name + '.' + outfile_ext)
    else:
        outfile = out
        outfile_name, outfile_ext = os.path.splitext(outfile)

    # main logic
    # load/make charmap
    with open('courier-scaled-charmap.json', 'r') as f:
        weights = json.loads(f.read())
    charmap = reverse_dict(weights)

    width = args.width
    img_a = img_to_array(infile, width, aspect_ratio)
    shape = img_a.shape
    if args.expand_contrast:
        img_a = expand_contrast(img_a)

    if args.rgb:
        # write rgb channels to 3 files
        if shape[2] < 3:
            raise ValueError("image is not RGB")
        channels = [array2ascii(img_a[:, :, i], charmap) for i in range(3)]

        name, ext = os.path.splitext(outfile)
        for i, c in enumerate(channels):
            with open(name+'_'+{0:'r',1:'g',2:'b'}[i]+ext, 'w') as f:
                f.write(c)
    elif args.cmyk:
        # write cmyk channels to 4 files
        if shape[2] < 3:
            raise ValueError("image is not RGB")
        cmyk = img_rgb2cmyk(img_a)
        channels = [array2ascii(cmyk[:, :, i], charmap) for i in range(4)]

        name, ext = os.path.splitext(outfile)
        for i, c in enumerate(channels):
            with open(name+'_'+{0:'c',1:'m',2:'y',3:'k'}[i]+ext, 'w') as f:
                f.write(c)
    else:
        # write single-pass grayscale output
        if len(shape) != 2:
            img_a = rgb2gs(img_a)  # np array of img in grayscale
        txt = array2ascii(img_a, charmap)
        with open(outfile, 'w') as f:
            f.write(txt)  # 10 CPI and 6 LPI
    exit(0)
