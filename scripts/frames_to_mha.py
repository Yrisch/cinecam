#!/usr/bin/env python3
# cinecam: camera paths through SPH simulations, rendered with shamrock
# Copyright (c) 2026 the cinecam authors
# SPDX-License-Identifier: CECILL-2.1
# cinecam is licensed under the CeCILL 2.1 License, see LICENSE for more information

"""
Stack a sequence of PNG frames into 3D MetaImage (.mha or .mhd) volumes.

Writes two volumes:
  <out>_rgb  : uint8, 3 channels per voxel (ElementNumberOfChannels = 3)
  <out>_grey : uint8, 1 channel (ITU-R BT.601 luma: 0.299 R + 0.587 G + 0.114 B)

With --format mha (default) each volume is a single .mha file (header + data).
With --format mhd each volume is a .mhd text header plus a detached data file
(.raw, or .zraw with --compress).

Volume layout is X = image width, Y = image height, Z = frame index.
Only depends on numpy and Pillow (the MetaImage writer is implemented here).

Usage:
  python frames_to_mha.py frames_dir [-o out_prefix] [--pattern 'frame_*.png'] [--format mha|mhd] [--compress]
"""

import argparse
import glob
import os
import zlib

import numpy as np
from PIL import Image


def load_frames(frames_dir, pattern):
    files = sorted(glob.glob(os.path.join(frames_dir, pattern)))
    if not files:
        raise SystemExit(f"no files matching '{pattern}' in {frames_dir}")

    first = np.asarray(Image.open(files[0]).convert("RGB"))
    vol = np.empty((len(files),) + first.shape, dtype=np.uint8)  # (Z, Y, X, 3)
    vol[0] = first
    for i, f in enumerate(files[1:], start=1):
        img = np.asarray(Image.open(f).convert("RGB"))
        if img.shape != first.shape:
            raise SystemExit(f"{f} has shape {img.shape}, expected {first.shape}")
        vol[i] = img

    print(f"loaded {len(files)} frames of size {first.shape[1]}x{first.shape[0]}")
    return vol


def write_metaimage(path, data, n_channels, compress=False):
    """
    data: uint8 array of shape (Z, Y, X) or (Z, Y, X, C), C-contiguous (X fastest).
    path ending in .mha -> data embedded after the header;
    path ending in .mhd -> data written to a sibling .raw/.zraw file.
    """
    detached = path.endswith(".mhd")
    nz, ny, nx = data.shape[:3]
    raw = np.ascontiguousarray(data).tobytes()
    if compress:
        raw = zlib.compress(raw, level=6)

    header = [
        "ObjectType = Image",
        "NDims = 3",
        "BinaryData = True",
        "BinaryDataByteOrderMSB = False",
        f"CompressedData = {'True' if compress else 'False'}",
    ]
    if compress:
        header.append(f"CompressedDataSize = {len(raw)}")
    header += [
        "TransformMatrix = 1 0 0 0 1 0 0 0 1",
        "Offset = 0 0 0",
        "CenterOfRotation = 0 0 0",
        "AnatomicalOrientation = RAI",
        "ElementSpacing = 1 1 1",
        f"DimSize = {nx} {ny} {nz}",
    ]
    if n_channels > 1:
        header.append(f"ElementNumberOfChannels = {n_channels}")
    if detached:
        data_path = os.path.splitext(path)[0] + (".zraw" if compress else ".raw")
        data_file = os.path.basename(data_path)  # resolved relative to the header
    else:
        data_file = "LOCAL"
    header += [
        "ElementType = MET_UCHAR",
        f"ElementDataFile = {data_file}",  # must be the last header line
    ]

    with open(path, "wb") as fh:
        fh.write(("\n".join(header) + "\n").encode("ascii"))
        if not detached:
            fh.write(raw)
    if detached:
        with open(data_path, "wb") as fh:
            fh.write(raw)
        print(f"wrote {path} + {data_file} ({os.path.getsize(data_path) / 2**20:.1f} MiB)")
    else:
        print(f"wrote {path} ({os.path.getsize(path) / 2**20:.1f} MiB)")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0].strip())
    parser.add_argument("frames_dir", help="directory containing the PNG frames")
    parser.add_argument("-o", "--out", default=None, help="output prefix (default: <frames_dir>/frames)")
    parser.add_argument("--pattern", default="*.png")
    parser.add_argument("--format", choices=["mha", "mhd"], default="mha",
                        help="mha: single file; mhd: header + detached .raw/.zraw")
    parser.add_argument("--compress", action="store_true", help="zlib-compress the voxel data")
    args = parser.parse_args()

    out = args.out or os.path.join(args.frames_dir, "frames")
    vol = load_frames(args.frames_dir, args.pattern)

    write_metaimage(f"{out}_rgb.{args.format}", vol, n_channels=3, compress=args.compress)

    luma = np.array([0.299, 0.587, 0.114], dtype=np.float32)
    grey = np.clip(np.rint(vol.astype(np.float32) @ luma), 0, 255).astype(np.uint8)
    write_metaimage(f"{out}_grey.{args.format}", grey, n_channels=1, compress=args.compress)


if __name__ == "__main__":
    main()
