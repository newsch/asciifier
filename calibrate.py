#!/usr/bin/env python3
"""Generate and parse calibration papers"""
import string

ASCII = string.digits + string.ascii_letters + string.punctuation + ' '

def get_calibration_sheet(charset, wrap=10):
    lines = []
    for i in range(0, len(charset), wrap):
        lines.append(charset[i:i+wrap])
    return lines

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--generate-calibration', action='store_true')

    args = parser.parse_args()

    if args.generate_calibration:
        [print(l) for l in get_calibration_sheet(ASCII)]
        exit(0)
