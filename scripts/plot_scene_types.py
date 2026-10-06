#!/usr/bin/env python3
"""Scene-type distribution of the release (paper figure), from the dataset manifest."""
import argparse
import collections
import json
import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', '/tmp/vr180-scene-types-mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

TOP = 14


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', type=Path, required=True)
    ap.add_argument('--out', type=Path, default=Path('figures'))
    args = ap.parse_args()
    samples = json.loads(args.manifest.read_text())['samples']
    counts = collections.Counter(s['annotations']['scene_type'] for s in samples)
    video = collections.Counter(s['annotations']['scene_type'] for s in samples if s['media_type'] == 'video')
    ranked = counts.most_common()
    rest = ranked[TOP:]
    labels = [k for k, _ in ranked[:TOP]] + [f'{len(rest)} other types']
    total = [v for _, v in ranked[:TOP]] + [sum(v for _, v in rest)]
    vids = [video[k] for k, _ in ranked[:TOP]] + [sum(video[k] for k, _ in rest)]
    fig, ax = plt.subplots(figsize=(3.4, 2.9), layout='constrained')
    y = range(len(labels))[::-1]
    ax.barh(y, vids, color='#2a6f97', label='video')
    ax.barh(y, [t - v for t, v in zip(total, vids)], left=vids, color='#a9cce3', label='still')
    for yi, t in zip(y, total):
        ax.text(t + 6, yi, str(t), va='center', fontsize=6.5)
    ax.set_yticks(list(y), labels, fontsize=7)
    ax.set_xlabel('Samples', fontsize=7)
    ax.tick_params(axis='x', labelsize=7)
    ax.set_xlim(0, max(total) * 1.12)
    ax.spines[['top', 'right']].set_visible(False)
    ax.legend(fontsize=7, frameon=False, loc='center right')
    args.out.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out / 'scene-types.pdf')
    fig.savefig(args.out / 'scene-types.png', dpi=220)


if __name__ == '__main__':
    main()
