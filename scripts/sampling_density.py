#!/usr/bin/env python3
"""Angular sampling density of the native fisheye and half-equirectangular rasters.

For every direction of the half-EQ domain, the fisheye density follows from the Jacobian of the
factory lookup table (direction -> native pixel); the half-EQ density is analytic. Densities are
linear: sqrt(pixels per steradian), expressed in pixels per degree. Also reports the mean density
in the nine viewports of the coding study and the viewport sampling density itself.
"""
import argparse
import json
import math
import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', '/tmp/vr180-density-mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

MAPS = Path(__file__).resolve().parents[1] / 'geometry-scripts/linux/maps'
FISHEYE = (8160, 7200)
EQ = 7200
VIEWS = [(yaw, pitch) for pitch in [45, 0, -45] for yaw in [-60, 0, 60]]
REGIONS = {'all': list(range(9)), 'center': [4], 'lateral': [0, 2, 3, 5, 6, 8], 'upper': [0, 1, 2], 'lower': [6, 7, 8]}
STEP = 8  # analyse every 8th grid node (512 x 512 directions)


def fisheye_density(table):
    uv = np.load(table, mmap_mode='r')[::STEP, ::STEP].astype(np.float64)
    n = uv.shape[0]
    px = uv[..., 0] * FISHEYE[0]
    py = uv[..., 1] * FISHEYE[1]
    # Grid node centres in degrees: columns are longitude -90..90, rows latitude 90..-90.
    centres = (np.arange(n) * STEP + STEP / 2) / 4096 * 180 - 90
    lon, lat = np.meshgrid(centres, -centres)
    d = math.radians(180 / 4096 * STEP)
    dx_dlon, dx_dlat = np.gradient(px, d, -d, axis=(1, 0))
    dy_dlon, dy_dlat = np.gradient(py, d, -d, axis=(1, 0))
    det = np.abs(dx_dlon * dy_dlat - dx_dlat * dy_dlon)
    per_sr = det / np.cos(np.radians(lat))
    valid = (uv[..., 0] > 0) & (uv[..., 0] < 1) & (uv[..., 1] > 0) & (uv[..., 1] < 1) & (np.abs(lat) < 89.5)
    density = np.where(valid, np.sqrt(per_sr) * math.pi / 180, np.nan)
    return lon, lat, density


def eq_density(lat):
    per_deg = EQ / 180
    return per_deg / np.sqrt(np.cos(np.radians(lat)))


def view_mask(lon, lat, yaw, pitch, fov=60):
    # Directions inside a square rectilinear viewport (x-right, y-up, z-forward; positive pitch up).
    lo, la = np.radians(lon), np.radians(lat)
    v = np.stack([np.cos(la) * np.sin(lo), np.sin(la), np.cos(la) * np.cos(lo)], -1)
    y, p = math.radians(yaw), math.radians(pitch)
    ry = np.array([[math.cos(y), 0, -math.sin(y)], [0, 1, 0], [math.sin(y), 0, math.cos(y)]])
    rp = np.array([[1, 0, 0], [0, math.cos(p), -math.sin(p)], [0, math.sin(p), math.cos(p)]])
    c = v @ ry.T @ rp.T
    t = math.tan(math.radians(fov / 2))
    return (c[..., 2] > 0) & (np.abs(c[..., 0] / c[..., 2]) <= t) & (np.abs(c[..., 1] / c[..., 2]) <= t)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    report = {'units': 'pixels per degree (square root of pixels per square degree of solid angle)',
              'viewport_density_center_px_per_deg': 768 / math.tan(math.radians(30)) * math.pi / 180,
              'calibrations': {}}
    first = None
    for index in sorted(MAPS.glob('*.json')):
        digest = index.stem
        for eye in ['left', 'right']:
            lon, lat, fd = fisheye_density(MAPS / f'{digest}-{eye}.npy')
            ed = eq_density(lat)
            ratio = fd / ed
            row = {'fisheye_center_px_per_deg': float(np.nanmean(fd[np.hypot(lon, lat) < 5])),
                   'fisheye_px_per_deg_by_off_axis_angle': {}, 'regions': {}}
            off = np.degrees(np.arccos(np.clip(np.cos(np.radians(lat)) * np.cos(np.radians(lon)), -1, 1)))
            for a in [0, 15, 30, 45, 60, 75, 85]:
                ring = (np.abs(off - a) < 2.5) & np.isfinite(fd)
                row['fisheye_px_per_deg_by_off_axis_angle'][str(a)] = float(np.nanmean(fd[ring])) if ring.any() else None
            masks = [view_mask(lon, lat, yaw, pitch) for yaw, pitch in VIEWS]
            for name, idx in REGIONS.items():
                m = np.any([masks[i] for i in idx], axis=0) & np.isfinite(fd)
                row['regions'][name] = {'fisheye_px_per_deg': float(np.mean(fd[m])), 'half_eq_px_per_deg': float(np.mean(ed[m])),
                                        'fisheye_to_half_eq_area_ratio': float(np.mean((fd[m] / ed[m]) ** 2))}
            report['calibrations'].setdefault(digest[:12], {})[eye] = row
            if first is None:
                first = (lon, lat, fd, ed, ratio)
    lon, lat, fd, ed, ratio = first
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.3), layout='constrained')
    extent = [-90, 90, -90, 90]
    for ax, data, title in [(axes[0], fd, 'Native fisheye'), (axes[1], ed, 'Half-equirectangular')]:
        im = ax.imshow(data, extent=extent, origin='upper', cmap='viridis', vmin=35, vmax=70)
        ax.set_title(title + ', px/deg', fontsize=8)
    fig.colorbar(im, ax=axes[:2], shrink=.85, extend='max').ax.tick_params(labelsize=7)
    im = axes[2].imshow(ratio ** 2, extent=extent, origin='upper', cmap='coolwarm', vmin=0.5, vmax=1.5)
    axes[2].set_title('Pixels per solid angle,\nfisheye / half-EQ', fontsize=8)
    fig.colorbar(im, ax=axes[2], shrink=.85, extend='both').ax.tick_params(labelsize=7)
    for ax in axes:
        ax.set_xlabel('Longitude (deg)', fontsize=7)
        ax.tick_params(labelsize=7)
        ax.set_xticks([-90, -45, 0, 45, 90])
        ax.set_yticks([-90, -45, 0, 45, 90])
        for yaw, pitch in VIEWS:
            ax.plot(yaw, pitch, '+', color='white' if ax is not axes[2] else 'black', ms=5, mew=1)
    axes[0].set_ylabel('Latitude (deg)', fontsize=7)
    fig.savefig(args.out / 'sampling-density.pdf')
    fig.savefig(args.out / 'sampling-density.png', dpi=200)
    (args.out / 'sampling-density.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'calibrations'}, indent=1))
    for cal, eyes in report['calibrations'].items():
        for eye, row in eyes.items():
            print(cal, eye, 'centre %.1f' % row['fisheye_center_px_per_deg'],
                  {k: round(v, 1) for k, v in row['fisheye_px_per_deg_by_off_axis_angle'].items() if v},
                  {k: (round(v['fisheye_px_per_deg'], 1), round(v['half_eq_px_per_deg'], 1), round(v['fisheye_to_half_eq_area_ratio'], 2)) for k, v in row['regions'].items()})


if __name__ == '__main__':
    main()
