#!/usr/bin/env python3
"""Paper teaser: one stereo sample in both representations plus a strip of scene types.

Frames are decoded read-only from the released separate-eye HEVC files.
"""
import argparse
import os
import subprocess
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', '/tmp/vr180-teaser-mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERO = ('sample_1CA74D4B-4E63-4E82-81F0-92AC351EE58B', 48)
# (sample_id, frame index, short label); one frame per scene type, no prominent people.
STRIP = [
    ('sample_F78494F1-192C-47DF-A869-69469BA92AED', 0, 'Garden'),
    ('sample_13FAA494-3C03-488A-8204-877B483D91C9', 225, 'Aquarium'),
    ('sample_86B1F49D-D8FA-4D2D-AAFF-26599C5C5AFF', 83, 'Lake'),
    ('sample_C42E3E25-7A12-447D-B666-80139C93DE7F', 0, 'Art gallery'),
    ('sample_8C428012-39C8-442D-971B-F7E2F5C0958C', 225, 'Building'),
    ('sample_B95AEEB9-6752-4627-BA11-A73760A6C175', 0, 'Static display'),
    ('sample_208B2935-FD52-4B78-A4A8-0D6082528E15', 56, 'Mountain trail'),
    ('sample_99D514E4-39D8-42B0-AF97-1030E83C3A8A', 114, 'Waterfall'),
]


def frame(path, index, width):
    """Decode one frame to 8-bit RGB, area-downscaled to the given width."""
    cmd = ['ffmpeg', '-nostdin', '-loglevel', 'error', '-i', str(path),
           '-vf', f'select=eq(n\\,{index}),scale={width}:-2:flags=area', '-frames:v', '1',
           '-f', 'rawvideo', '-pix_fmt', 'rgb24', 'pipe:1']
    probe = subprocess.check_output(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries',
                                     'stream=width,height', '-of', 'csv=p=0', str(path)], text=True)
    w, h = map(int, probe.strip().strip(',').split(',')[:2])
    height = round(h * width / w / 2) * 2
    raw = subprocess.check_output(cmd)
    return np.frombuffer(raw, np.uint8).reshape(height, width, 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-root', type=Path, required=True, help='Local copy of the dataset')
    ap.add_argument('--out', type=Path, default=Path('figures'))
    args = ap.parse_args()
    fig = plt.figure(figsize=(7.0, 3.05), layout='constrained')
    top, bottom = fig.subfigures(2, 1, height_ratios=[1.85, 1])
    axes = top.subplots(1, 4)
    panels = [('h265_fisheye', 'LEFT', 'Native fisheye, left (8160×7200)'),
              ('h265_fisheye', 'RIGHT', 'Native fisheye, right'),
              ('h265_eq', 'LEFT', 'Half-EQ, left (7200×7200)'),
              ('h265_eq', 'RIGHT', 'Half-EQ, right')]
    for ax, (rep, eye, title) in zip(axes, panels):
        ax.imshow(frame(args.data_root / rep / HERO[0] / f'{eye}.mov', HERO[1], 900))
        ax.set_title(title, fontsize=7)
        ax.set_axis_off()
    axes = bottom.subplots(1, len(STRIP))
    for ax, (sid, index, label) in zip(axes, STRIP):
        ax.imshow(frame(args.data_root / 'h265_eq' / sid / 'LEFT.mov', index, 400))
        ax.set_title(label, fontsize=6.5)
        ax.set_axis_off()
    args.out.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out / 'teaser.pdf', dpi=300)
    fig.savefig(args.out / 'teaser.png', dpi=200)
    print(args.out / 'teaser.pdf')


if __name__ == '__main__':
    main()
