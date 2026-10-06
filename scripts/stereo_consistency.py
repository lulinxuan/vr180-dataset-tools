#!/usr/bin/env python3
"""Left/right consistency of every sample, measured on one half-equirectangular frame per eye.

Frame: the middle frame of a video, frame 0 of a still. Both eyes are decoded at 3600 x 3600
(20 px/deg). Inside the central +-45 deg region: relative luma difference, CIELAB colour
difference of the mean colours, and the ratio of mean gradient magnitude. SIFT matches between
the eyes give the vertical offset (median latitude difference of matches within +-20 deg of the
equator, where near objects add little vertical parallax) and the horizontal disparity range.
Read-only on the dataset; results append to a resumable JSONL file.
"""
import argparse
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np

SIZE = 3600
cv2.setNumThreads(4)
PX_PER_DEG = SIZE / 180


def frame(path, index):
    raw = subprocess.check_output(['ffmpeg', '-nostdin', '-loglevel', 'error', '-threads', '4', '-i', str(path), '-vf',
                                   f'select=eq(n\\,{index}),scale={SIZE}:{SIZE}:flags=area', '-frames:v', '1',
                                   '-f', 'rawvideo', '-pix_fmt', 'rgb24', 'pipe:1'])
    return np.frombuffer(raw, np.uint8).reshape(SIZE, SIZE, 3)


def central(img, half_deg=45):
    a = int((90 - half_deg) * PX_PER_DEG)
    b = int((90 + half_deg) * PX_PER_DEG)
    return img[a:b, a:b], a


def photometric(left, right):
    yl = cv2.cvtColor(left, cv2.COLOR_RGB2GRAY).astype(np.float64)
    yr = cv2.cvtColor(right, cv2.COLOR_RGB2GRAY).astype(np.float64)
    valid = (yl > 8) & (yr > 8)
    lab_l = cv2.cvtColor(left, cv2.COLOR_RGB2LAB).reshape(-1, 3)[valid.ravel()].astype(np.float64)
    lab_r = cv2.cvtColor(right, cv2.COLOR_RGB2LAB).reshape(-1, 3)[valid.ravel()].astype(np.float64)
    # OpenCV 8-bit Lab: L scaled by 255/100, a and b offset by 128.
    scale = np.array([100 / 255, 1, 1])
    delta_e = float(np.linalg.norm((lab_l.mean(0) - lab_r.mean(0)) * scale))
    grad = []
    for y in (yl, yr):
        gx = cv2.Sobel(y, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(y, cv2.CV_64F, 0, 1, ksize=3)
        grad.append(np.hypot(gx, gy)[valid].mean())
    return {'luma_difference_percent': float(100 * (yl[valid].mean() - yr[valid].mean()) / ((yl[valid].mean() + yr[valid].mean()) / 2)),
            'mean_colour_delta_e': delta_e,
            'sharpness_log2_ratio': float(np.log2(grad[0] / grad[1]))}


def geometric(left, right, offset):
    sift = cv2.SIFT_create(nfeatures=4000)
    gl = cv2.cvtColor(left, cv2.COLOR_RGB2GRAY)
    gr = cv2.cvtColor(right, cv2.COLOR_RGB2GRAY)
    kl, dl = sift.detectAndCompute(gl, None)
    kr, dr = sift.detectAndCompute(gr, None)
    if dl is None or dr is None or len(kl) < 10 or len(kr) < 10:
        return {'matches': 0}
    pairs = cv2.BFMatcher(cv2.NORM_L2).knnMatch(dl, dr, k=2)
    good = [m for m, n in (p for p in pairs if len(p) == 2) if m.distance < 0.75 * n.distance]
    if len(good) < 10:
        return {'matches': len(good)}
    pl = np.array([kl[m.queryIdx].pt for m in good]) + offset
    pr = np.array([kr[m.trainIdx].pt for m in good]) + offset
    lat_l = 90 - pl[:, 1] / PX_PER_DEG
    dlat = (pl[:, 1] - pr[:, 1]) / PX_PER_DEG  # positive: right-eye feature appears higher
    dlon = (pl[:, 0] - pr[:, 0]) / PX_PER_DEG
    keep = (np.abs(dlat) < 2) & (np.abs(dlon) < 10)
    band = keep & (np.abs(lat_l) < 20)
    out = {'matches': int(keep.sum()), 'equator_matches': int(band.sum())}
    if band.sum() >= 10:
        out['vertical_offset_arcmin'] = float(np.median(dlat[band]) * 60)
        out['vertical_spread_arcmin'] = float(np.median(np.abs(dlat[band] - np.median(dlat[band]))) * 60)
    if keep.sum() >= 10:
        out['horizontal_disparity_deg_p5_p95'] = [float(np.percentile(dlon[keep], 5)), float(np.percentile(dlon[keep], 95))]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-root', type=Path, required=True)
    ap.add_argument('--manifest', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--limit', type=int)
    args = ap.parse_args()
    samples = json.loads(args.manifest.read_text())['samples'][:args.limit]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    done = {json.loads(l)['sample_id'] for l in args.out.read_text().splitlines()} if args.out.exists() else set()
    with args.out.open('a') as stream:
        for s in samples:
            if s['sample_id'] in done:
                continue
            index = s['video']['frame_count'] // 2 if s['media_type'] == 'video' else 0
            row = {'sample_id': s['sample_id'], 'media_type': s['media_type'], 'frame': index,
                   'scene_type': s['annotations']['scene_type'], 'calibration_id': s['calibration_id']}
            try:
                left, offset = central(frame(args.data_root / 'h265_eq' / s['sample_id'] / 'LEFT.mov', index))
                right, _ = central(frame(args.data_root / 'h265_eq' / s['sample_id'] / 'RIGHT.mov', index))
                row.update(photometric(left, right))
                row.update(geometric(left, right, offset))
            except Exception as exc:
                row['error'] = str(exc)
            stream.write(json.dumps(row) + '\n')
            stream.flush()


if __name__ == '__main__':
    main()
