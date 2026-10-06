"""Shared helpers for the coding study: viewport geometry, metrics and rate differences."""
import json
import math
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from scipy.interpolate import PchipInterpolator

HERE = Path(__file__).resolve().parent
BASE = HERE.parents[1]
sys.path.insert(0, str(BASE/'geometry-scripts/linux'))
from geometry import LensMap, canonical, calibration_object  # noqa: E402

DATA = Path(os.environ.get('VR180_DATA', 'VR180'))  # dataset root with h265_fisheye/ and h265_eq/
FF = ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-nostdin']
REGIONS = ['all', 'center', 'lateral', 'upper', 'lower']+[f'view{i}' for i in range(9)]
GRID = [{'yaw_degrees': yaw, 'pitch_degrees': pitch, 'horizontal_fov_degrees': 60, 'vertical_fov_degrees': 60}
        for pitch in [45, 0, -45] for yaw in [-60, 0, 60]]


def save(path, obj):
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(obj, indent=2, allow_nan=False)+'\n')
    temp.replace(path)


def sig(path):
    s = path.stat()
    return {'bytes': s.st_size, 'mtime_ns': s.st_mtime_ns}


def status(folder, stage, **extra):
    save(folder/'status.json', {'stage': stage, 'updated_unix': time.time(), **extra})
    print(json.dumps({'stage': stage, **extra}), flush=True)


def readexact(stream, n):
    chunks = []
    remaining = n
    while remaining:
        b = stream.read(remaining)
        if not b:
            raise EOFError(f'Missing {remaining} of {n} bytes')
        chunks.append(b)
        remaining -= len(b)
    return b''.join(chunks)


def make_maps(lens, forward_valid, width=1536, fish_size=(8160, 7200), eq_size=(7200, 7200)):
    """Sampling maps of the nine viewports into the fisheye and half-EQ rasters, and their common validity."""
    yy, xx = np.indices((width, width), dtype=float)
    x = (2*(xx+.5)/width-1)*math.tan(math.pi/6)
    y = (1-2*(yy+.5)/width)*math.tan(math.pi/6)
    fw, fh = fish_size
    ew, eh = eq_size
    interior = cv2.erode(forward_valid.astype(np.uint8), np.ones((9, 9), np.uint8))
    native, expanded, masks = [], [], []
    for view in GRID:
        yaw = math.radians(view['yaw_degrees'])
        pitch = math.radians(view['pitch_degrees'])
        # x-right/y-up/z-forward. Positive pitch raises the center ray.
        py = y*math.cos(pitch)+math.sin(pitch)
        pz = math.cos(pitch)-y*math.sin(pitch)
        rx = x*math.cos(yaw)+pz*math.sin(yaw)
        rz = -x*math.sin(yaw)+pz*math.cos(yaw)
        length = np.sqrt(rx*rx+py*py+rz*rz)
        u = np.arctan2(rx, rz)/math.pi+.5
        v = .5-np.arcsin(py/length)/math.pi
        uv, valid = lens.sample(u*lens.width-.5, v*lens.height-.5)
        fishxy = (uv*np.array([fw, fh])-.5).astype(np.float32)
        eqxy = (np.stack([u, v], -1)*np.array([ew, eh])-.5).astype(np.float32)
        valid &= (fishxy[..., 0] >= 4) & (fishxy[..., 0] < fw-4) & (fishxy[..., 1] >= 4) & (fishxy[..., 1] < fh-4)
        valid &= (eqxy[..., 0] >= 4) & (eqxy[..., 0] < ew-4) & (eqxy[..., 1] >= 4) & (eqxy[..., 1] < eh-4)
        valid &= cv2.remap(interior, eqxy, None, cv2.INTER_NEAREST) > 0
        valid[:5] = False
        valid[-5:] = False
        valid[:, :5] = False
        valid[:, -5:] = False
        native.append(fishxy)
        expanded.append(eqxy)
        masks.append(valid)
    return np.stack(native), np.stack(expanded), np.stack(masks)


def image_views(rgb, maps):
    return np.stack([cv2.remap(rgb, m, None, cv2.INTER_LANCZOS4) for m in maps])


def luma(rgb):
    return (rgb[..., 0].astype(np.float32)*.2126+rgb[..., 1].astype(np.float32)*.7152+rgb[..., 2].astype(np.float32)*.0722)/65535


def metric(reference, result, mask):
    a = luma(reference)
    b = luma(result)
    d = a-b
    ma = cv2.GaussianBlur(a, (11, 11), 1.5)
    mb = cv2.GaussianBlur(b, (11, 11), 1.5)
    va = cv2.GaussianBlur(a*a, (11, 11), 1.5)-ma*ma
    vb = cv2.GaussianBlur(b*b, (11, 11), 1.5)-mb*mb
    cov = cv2.GaussianBlur(a*b, (11, 11), 1.5)-ma*mb
    sm = ((2*ma*mb+.01**2)*(2*cov+.03**2))/((ma*ma+mb*mb+.01**2)*(va+vb+.03**2))
    return {'n': int(mask.sum()), 'sse': float((d[mask]**2).sum(dtype=np.float64)),
            'ssim_sum': float(sm[mask].sum(dtype=np.float64))}


def combined(rows):
    n = sum(r['n'] for r in rows)
    sse = sum(r['sse'] for r in rows)
    if not n:
        raise ValueError('No shared content to evaluate')
    return {'pixels': n, 'psnr_y_db': -10*math.log10(max(sse/n, 1e-30)),
            'ssim_y': sum(r['ssim_sum'] for r in rows)/n, 'sse_y': sse}


def stats(rows):
    output = {f'view{i}': combined(r) for i, r in enumerate(rows)}
    for name, indices in {'all': list(range(9)), 'center': [4], 'lateral': [0, 2, 3, 5, 6, 8],
                          'upper': [0, 1, 2], 'lower': [6, 7, 8]}.items():
        output[name] = combined([r for i in indices for r in rows[i]])
    return output


def pooled(rows):
    n = sum(r['pixels'] for r in rows)
    sse = sum(r['sse_y'] for r in rows)
    if n <= 0 or sse <= 0:
        raise ValueError('Invalid pooled metric support')
    return {'pixels': n, 'sse_y': sse, 'psnr_y_db': -10*math.log10(sse/n),
            'ssim_y': sum(r['ssim_y']*r['pixels'] for r in rows)/n}


def rate_difference(points, region):
    """BD-rate of fisheye relative to half-EQ (percent): mean PCHIP log-rate difference over the common PSNR range."""
    curves = {}
    for rep in ['fisheye', 'eq']:
        rows = sorted([p for p in points if p['representation'] == rep], key=lambda p: p['regions'][region]['psnr_y_db'])
        q = np.array([p['regions'][region]['psnr_y_db'] for p in rows])
        rate = np.array([p['stereo_mbps'] for p in rows])
        if len(rows) != 4 or not np.isfinite(q).all() or not np.isfinite(rate).all() or (rate <= 0).any():
            return {'status': 'unavailable', 'reason': 'Invalid/missing rate-quality point'}
        if not np.all(np.diff(q) > 0) or not np.all(np.diff(rate) > 0):
            return {'status': 'unavailable', 'reason': 'Duplicate quality or nonmonotonic rate-quality curve; no points removed'}
        curves[rep] = (q, PchipInterpolator(q, np.log(rate), extrapolate=False))
    lo = max(q[0] for q, _ in curves.values())
    hi = min(q[-1] for q, _ in curves.values())
    if hi <= lo:
        return {'status': 'unavailable', 'reason': 'No overlapping quality interval; no extrapolation'}
    value = 100*math.expm1(float((curves['fisheye'][1].integrate(lo, hi)-curves['eq'][1].integrate(lo, hi))/(hi-lo)))
    return {'status': 'available', 'fisheye_vs_eq_percent': value, 'overlap_psnr_db': [float(lo), float(hi)],
            'method': 'Mean log-rate difference with PCHIP over the common viewport PSNR range; negative favours fisheye'}
