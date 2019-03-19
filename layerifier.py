#!/usr/bin/env python3
"""Overlay multiple files line-by-line."""
import argparse
from typing import List


def combine_lines(lines: List[str]) -> str:
    out = ''
    for l in lines:
        out += l.rstrip('\n') + '\r'
    return out


def get_lines(files: List[object], num_copies: int = 1):
    while(True):
        out = []
        for f in files:
            line = f.readline()
            if line:
                for _ in range(num_copies):
                    out.append(line)
        if out:
            yield out
        else:
            break


parser = argparse.ArgumentParser('layerifier',
    description='Concatenate multiple files line-by-line with "\\r".')
parser.add_argument('-r', '--repeat', type=int, default=1)
parser.add_argument('input', type=argparse.FileType('r'), nargs='+')

args = parser.parse_args()

[print(combine_lines(lines)) for lines in get_lines(args.input, args.repeat)]
