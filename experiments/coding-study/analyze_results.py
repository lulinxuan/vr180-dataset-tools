#!/usr/bin/env python3
"""Tables and figure for the coding study from the study summary.

Reports the summary (median and range of the per-clip BD-rate in each viewport
region) and exploratory analyses: sign counts, a bootstrap interval of the median, a Wilcoxon
signed-rank test, groupings by content label, same-CRF rate ratios and the no-codec control.
The 24 clips are a diversity selection, not a random sample, so intervals and p-values are
descriptive.
"""
import argparse
import json
import math
import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', '/tmp/vr180-analysis-mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
REGIONS = ['all', 'center', 'lateral', 'upper', 'lower']
LABELS = {'all': 'All', 'center': 'Center', 'lateral': 'Lateral', 'upper': 'Upper', 'lower': 'Lower'}


def values(scenes, region):
    return [s['rate_differences'][region]['fisheye_vs_eq_percent'] for s in scenes
            if s['rate_differences'][region]['status'] == 'available']


def bootstrap_median(x, seed=0, n=10000):
    rng = np.random.default_rng(seed)
    x = np.asarray(x)
    medians = np.median(rng.choice(x, size=(n, len(x)), replace=True), axis=1)
    return [float(np.percentile(medians, 2.5)), float(np.percentile(medians, 97.5))]


def describe(x):
    return {'n': len(x), 'median_percent': float(np.median(x)) if x else None,
            'min_percent': float(min(x)) if x else None, 'max_percent': float(max(x)) if x else None}


def bd_rate(points, region, quality):
    """BD-rate of fisheye vs half-EQ (percent) with a given quality function; None if unavailable."""
    from scipy.interpolate import PchipInterpolator
    curves = {}
    for rep in ['fisheye', 'eq']:
        rows = sorted([p for p in points if p['representation'] == rep], key=lambda p: quality(p['regions'][region]))
        q = np.array([quality(p['regions'][region]) for p in rows])
        rate = np.array([p['stereo_mbps'] for p in rows])
        if len(rows) != 4 or not np.isfinite(q).all() or not np.all(np.diff(q) > 0) or not np.all(np.diff(rate) > 0):
            return None
        curves[rep] = (q, PchipInterpolator(q, np.log(rate), extrapolate=False))
    lo = max(q[0] for q, _ in curves.values())
    hi = min(q[-1] for q, _ in curves.values())
    if hi <= lo:
        return None
    return 100 * math.expm1(float((curves['fisheye'][1].integrate(lo, hi) - curves['eq'][1].integrate(lo, hi)) / (hi - lo)))


def ssim_db(region):
    return -10 * math.log10(max(1 - region['ssim_y'], 1e-12))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--summary', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--manifest', type=Path, required=True, help="The dataset's manifest.json")
    ap.add_argument('--allow-partial', action='store_true', help='Pipeline test only; output is marked PARTIAL')
    args = ap.parse_args()
    data = json.loads(args.summary.read_text())
    partial = not data['all_selected_samples_complete']
    if partial and not args.allow_partial:
        ap.error('Summary is incomplete; final analysis requires all 24 selected cases')
    scenes = [s for s in data['scenes'] if s['status'] == 'complete']
    selection = {s['sample_id']: s for s in json.loads((HERE / 'clip-selection.json').read_text())['samples']}
    manifest = {s['sample_id']: s for s in json.loads(args.manifest.read_text())['samples']}

    report = {'partial_pipeline_test': partial, 'complete_cases': len(scenes), 'selected_cases': data['selected_samples'],
              'sign_convention': 'BD-rate of fisheye relative to half-EQ; negative means fisheye needs less rate',
              'summary': {}, 'exploratory': {}}
    for region in REGIONS:
        x = values(scenes, region)
        report['summary'][region] = {**describe(x), 'unavailable': len(scenes) - len(x),
                                            'unavailable_reasons': sorted({s['rate_differences'][region]['reason'] for s in scenes
                                                                           if s['rate_differences'][region]['status'] != 'available'})}
        if not partial and data.get('aggregate'):
            agg = data['aggregate'][region]
            assert agg['available_cases'] == len(x)
            if x:
                assert abs(agg['median_percent'] - np.median(x)) < 1e-9
        ex = {'fisheye_better': sum(v < 0 for v in x), 'eq_better': sum(v > 0 for v in x)}
        if len(x) >= 2:
            ex['bootstrap95_median_percent'] = bootstrap_median(x)
        if len(x) >= 6:
            ex['wilcoxon_p_two_sided'] = float(stats.wilcoxon(x).pvalue)
        report['exploratory'][region] = ex

    groups = {}
    for name, key in [('camera_motion', lambda s: selection[s['sample_id']]['camera_motion']),
                      ('object_motion', lambda s: selection[s['sample_id']]['object_motion']),
                      ('low_light', lambda s: 'present' if selection[s['sample_id']]['low_light_label'] else 'absent'),
                      ('low_texture', lambda s: 'present' if 'Low texture' in manifest[s['sample_id']]['annotations']['challenges'] else 'not present'),
                      ('calibration', lambda s: selection[s['sample_id']]['calibration_id'][:8])]:
        table = {}
        for s in scenes:
            r = s['rate_differences']['all']
            if r['status'] == 'available':
                table.setdefault(str(key(s)), []).append(r['fisheye_vs_eq_percent'])
        groups[name] = {k: describe(v) for k, v in sorted(table.items())}
    report['exploratory']['by_label_all_region'] = groups

    ssim = {}
    for region in REGIONS:
        vals, agree = [], 0
        for s in scenes:
            v = bd_rate(s['points'], region, ssim_db)
            if v is None:
                continue
            vals.append(v)
            p = s['rate_differences'][region]
            if p['status'] == 'available' and np.sign(p['fisheye_vs_eq_percent']) == np.sign(v):
                agree += 1
        ssim[region] = {**describe(vals), 'unavailable': len(scenes) - len(vals), 'fisheye_better': sum(v < 0 for v in vals),
                        'same_sign_as_psnr': agree}
    report['exploratory']['ssim_db_bd_rate'] = ssim

    ratios = {}
    for crf in sorted({p['crf'] for s in scenes for p in s['points']}):
        r = [next(p for p in s['points'] if p['representation'] == 'fisheye' and p['crf'] == crf)['stereo_mbps'] /
             next(p for p in s['points'] if p['representation'] == 'eq' and p['crf'] == crf)['stereo_mbps'] for s in scenes]
        ratios[str(crf)] = describe(r) | {'note': 'fisheye/EQ stereo rate ratio at equal CRF (not equal quality)'}
    report['exploratory']['same_crf_rate_ratio'] = ratios
    if all('no_codec_projection' in s for s in scenes):
        report['exploratory']['no_codec_projection_psnr_db'] = {
            region: describe([s['no_codec_projection'][region]['psnr_y_db'] for s in scenes]) for region in REGIONS}
    psnr = {rep: {str(crf): describe([next(p for p in s['points'] if p['representation'] == rep and p['crf'] == crf)['regions']['all']['psnr_y_db']
                                      for s in scenes]) for crf in sorted({p['crf'] for s in scenes for p in s['points']})}
            for rep in ['fisheye', 'eq']}
    report['exploratory']['psnr_all_by_crf_db'] = psnr

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / 'analysis.json').write_text(json.dumps(report, indent=2) + '\n')

    fig, ax = plt.subplots(figsize=(3.4, 2.3), layout='constrained')
    rng = np.random.default_rng(1)
    for i, region in enumerate(REGIONS):
        x = np.array(values(scenes, region))
        jitter = rng.uniform(-0.18, 0.18, len(x))
        colors = np.where(x < 0, '#0072B2', '#D55E00')
        ax.scatter(np.full(len(x), i) + jitter, x, s=10, c=colors, alpha=.85, linewidths=0)
        if len(x):
            ax.plot([i - .3, i + .3], [np.median(x)] * 2, color='black', lw=1.4)
        missing = len(scenes) - len(x)
        if missing:
            ax.annotate(f'{missing} N/A', (i, 0), xytext=(0, -2), textcoords='offset points', ha='center', va='top', fontsize=6)
    ax.axhline(0, color='grey', lw=.6, ls='--')
    ax.set_xticks(range(len(REGIONS)), [LABELS[r] for r in REGIONS], fontsize=7)
    ax.set_ylabel('BD-rate, fisheye vs. half-EQ (%)', fontsize=7)
    ax.tick_params(labelsize=7)
    ax.grid(axis='y', alpha=.2)
    if partial:
        ax.set_title(f'PARTIAL PIPELINE TEST ({len(scenes)} clips) - NOT RESULTS', fontsize=6.5, color='red')
    fig.savefig(args.out / 'bd-rate-by-region.pdf')
    fig.savefig(args.out / 'bd-rate-by-region.png', dpi=220)
    print(json.dumps({'cases': len(scenes), 'partial': partial,
                      'median_all': report['summary']['all']['median_percent']}))


if __name__ == '__main__':
    main()
