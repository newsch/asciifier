#!/usr/bin/env python3
"""Split text into separate files of fixed width."""
import argparse
import io
import os
from typing import List


parser = argparse.ArgumentParser('split text into pages')
parser.add_argument('-w', '--width', type=int, default=80, help='page width')
parser.add_argument('input', type=argparse.FileType('r'))
parser.add_argument('output', type=str)

args = parser.parse_args()

outs = []
chunk_size = args.width
name, ext = os.path.splitext(args.output)
for ind, l in enumerate(args.input):
    inline = l.rstrip('\n')
    chunk_range = range(0, len(inline), chunk_size)
    num_chunks = len(chunk_range)
    for fnum, i in enumerate(chunk_range):
        if fnum >= len(outs):
            outs.append(open(name+'_'+str(fnum+1)+ext, 'w'))
            if ind > 0:
                outs[fnum].write('\n'*ind)  # fill w/ whitespace for previous lines
        outs[fnum].write(inline[i:i+chunk_size]+'\n')
    if num_chunks < len(outs):  # add blank lines to rightmost files if line does not continue
        for i in range(num_chunks, len(outs)):
            outs[i].write('\n')
[f.close() for f in outs]
