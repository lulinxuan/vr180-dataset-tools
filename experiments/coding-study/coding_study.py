#!/usr/bin/env python3
"""Coding study on the released renders: native fisheye vs half-EQ as delivered.

For each of the 24 selected clips and both eyes, the central 90 frames (one GOP) of the
released fisheye and half-EQ HEVC files are re-encoded with x265 at four CRFs. Each decoded
representation is compared with its own released source in nine common rectilinear
viewports (Y' PSNR, SSIM) on pixels that are valid and non-black in both sources.

  run-all   --output-root OUT [--jobs 6]
  run-one   --output-root OUT --sample S --eye E
  summarize --output-root OUT --output summary.json
"""
import argparse
import fcntl
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

import study_common as p
from study_common import GRID, REGIONS, make_maps, pooled, rate_difference, stats, status

SELECTION = p.HERE/'clip-selection.json'
FRAMES = 90
CRFS = [18, 24, 30, 36]
SIZE = {'fisheye': (8160, 7200), 'eq': (7200, 7200)}
FOLDER = {'fisheye': 'h265_fisheye', 'eq': 'h265_eq'}
X265 = 'pools=8:frame-threads=1:wpp=1:keyint=90:min-keyint=90:scenecut=0:open-gop=0:log-level=info:colorprim=bt709:transfer=2:colormatrix=bt709'
TO_RGB = 'scale=in_color_matrix=bt709:in_range=tv:out_range=pc,format=rgb48le'
CONFIG = {'version': 'released-renders-v1', 'frames': FRAMES, 'interval': 'central 90 frames of the frozen 270-frame interval',
          'crfs': CRFS, 'encoder': 'libx265 4.1 medium, '+X265, 'input': 'released HEVC decoded to yuv420p10le, no conversion',
          'reference': "each representation's own released source", 'viewports': GRID,
          'metric': "Y' BT.709 pooled-MSE PSNR and Gaussian SSIM (11x11, sigma 1.5) in nine 1536x1536 viewports, Lanczos4",
          'mask': 'geometrically valid in both representations and luma > 1/1023 in both sources, eroded 11x11'}


def code_hashes():
    files = [p.HERE/n for n in ['coding_study.py', 'study_common.py']]
    files.append(p.BASE/'geometry-scripts/linux/geometry.py')
    return {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in files}


def spec_for(sample):
    return next(s for s in json.loads(SELECTION.read_text())['samples'] if s['sample_id'] == sample)


def source(spec, rep, eye):
    path = p.DATA/FOLDER[rep]/spec['sample_id']/f'{eye.upper()}.mov'
    probe = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-count_packets',
                                                '-show_streams', '-of', 'json', str(path)]))['streams'][0]
    assert (probe['width'], probe['height']) == SIZE[rep] and probe['pix_fmt'] == 'yuv420p10le'
    assert probe['avg_frame_rate'] == '90/1' and probe['color_range'] == 'tv' and probe['color_space'] == 'bt709'
    assert int(probe['nb_read_packets']) == spec['source_frames']
    return path, {'path': str(path.relative_to(p.DATA)), **p.sig(path), 'bit_rate': int(probe['bit_rate'])}


def rgb_frames(command, width, height):
    process = subprocess.Popen(command, stdout=subprocess.PIPE)
    try:
        for _ in range(FRAMES):
            yield np.frombuffer(p.readexact(process.stdout, width*height*6), '<u2').reshape(height, width, 3)
        if process.stdout.read(1):
            raise ValueError('Extra frames')
        if process.wait(timeout=60):
            raise ValueError('Decoder failed')
    finally:
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=30)


def raw_to_rgb(raw, rep):
    w, h = SIZE[rep]
    return p.FF+['-threads', '2', '-f', 'rawvideo', '-pix_fmt', 'yuv420p10le', '-s', f'{w}x{h}', '-r', '90', '-i', str(raw),
                 '-filter_threads', '2', '-vf', TO_RGB, '-f', 'rawvideo', 'pipe:1']


def stream_to_rgb(movie):
    return p.FF+['-threads', '2', '-i', str(movie), '-map', '0:v:0', '-an', '-filter_threads', '2', '-vf', TO_RGB,
                 '-fps_mode', 'passthrough', '-f', 'rawvideo', 'pipe:1']


def detail(views, masks):
    """Mean absolute luma gradient over the masked viewport pixels."""
    total, count = 0.0, 0
    for v in range(9):
        y = p.luma(views[v])
        g = np.abs(cv2.Sobel(y, cv2.CV_32F, 1, 0, ksize=3))+np.abs(cv2.Sobel(y, cv2.CV_32F, 0, 1, ksize=3))
        total += float(g[masks[v]].sum(dtype=np.float64))
        count += int(masks[v].sum())
    return total/count


def prepare(folder, spec, eye, maps, view_masks):
    cache = folder/'cache'
    cache.mkdir(exist_ok=False)
    first = spec['start_frame']+(spec['frame_count']-FRAMES)//2
    info = {'sample_id': spec['sample_id'], 'eye': eye, 'first_frame': first, 'frames': FRAMES, 'seconds': FRAMES/90,
            'sources': {}, 'source_detail': {}}
    refs = {}
    t = time.monotonic()
    for rep in SIZE:
        path, info['sources'][rep] = source(spec, rep, eye)
        w, h = SIZE[rep]
        raw = cache/f'{rep}-source.yuv'
        subprocess.run(p.FF+['-threads', '4', '-i', str(path), '-map', '0:v:0', '-an', '-vf',
                             f'trim=start_frame={first}:end_frame={first+FRAMES},setpts=PTS-STARTPTS', '-fps_mode', 'passthrough',
                             '-pix_fmt', 'yuv420p10le', '-f', 'rawvideo', str(raw)], check=True)
        if raw.stat().st_size != FRAMES*w*h*3:
            raise ValueError(f'{rep}: unexpected source frame count')
        refs[rep] = np.lib.format.open_memmap(cache/f'{rep}-reference-views.npy', mode='w+', dtype=np.uint16,
                                              shape=(FRAMES, 9, 1536, 1536, 3))
        for frame, rgb in enumerate(rgb_frames(raw_to_rgb(raw, rep), w, h)):
            refs[rep][frame] = p.image_views(rgb, maps[rep])
            if frame % 15 == 0:
                status(folder, 'reference', rep=rep, completed_frames=frame+1, total_frames=FRAMES,
                       elapsed_seconds=time.monotonic()-t)
        refs[rep].flush()
    use = np.lib.format.open_memmap(cache/'metric-masks.npy', mode='w+', dtype=np.bool_, shape=(FRAMES, 9, 1536, 1536))
    for frame in range(FRAMES):
        for v in range(9):
            m = view_masks[v] & (p.luma(refs['fisheye'][frame, v]) > 1/1023) & (p.luma(refs['eq'][frame, v]) > 1/1023)
            use[frame, v] = cv2.erode(m.astype(np.uint8), np.ones((11, 11), np.uint8)) > 0
        if not use[frame].any():
            raise ValueError('No evaluable content')
    use.flush()
    for rep in SIZE:
        info['source_detail'][rep] = float(np.mean([detail(refs[rep][f], use[f]) for f in (0, FRAMES//2, FRAMES-1)]))
    info['mask_fraction'] = float(use.mean())
    info['prepare_seconds'] = time.monotonic()-t
    p.save(folder/'prepared.json', info)
    return info


def encode(folder, info, rep, crf):
    w, h = SIZE[rep]
    output = folder/f'{rep}-crf{crf}.mkv'
    cmd = p.FF+['-f', 'rawvideo', '-pix_fmt', 'yuv420p10le', '-s', f'{w}x{h}', '-r', '90', '-i', str(folder/'cache'/f'{rep}-source.yuv'),
                '-frames:v', str(FRAMES), '-an', '-c:v', 'libx265', '-preset', 'medium', '-crf', str(crf), '-x265-params', X265,
                '-color_range', 'tv', '-colorspace', 'bt709', '-color_primaries', 'bt709', '-n', str(output)]
    t = time.monotonic()
    with (folder/f'{rep}-crf{crf}-encode.stderr').open('w') as log:
        subprocess.run(cmd, stderr=log, check=True)
    probe = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-show_streams', '-show_packets', '-select_streams', 'v:0',
                                                '-of', 'json', str(output)]))
    stream = probe['streams'][0]
    assert (stream['width'], stream['height']) == SIZE[rep] and stream['profile'] == 'Main 10'
    assert len(probe['packets']) == FRAMES
    bits = 8*sum(int(x['size']) for x in probe['packets'])
    return {'rep': rep, 'crf': crf, 'command': cmd, 'wall_seconds': time.monotonic()-t, 'packet_payload_bits': bits,
            'mbps': bits/info['seconds']/1e6, 'bitstream_sha256': hashlib.sha256(output.read_bytes()).hexdigest()}


def evaluate(folder, info, rep, crf, encoded, maps):
    cache = folder/'cache'
    refs = np.load(cache/f'{rep}-reference-views.npy', mmap_mode='r')
    masks = np.load(cache/'metric-masks.npy', mmap_mode='r')
    rows = [[] for _ in range(9)]
    t = time.monotonic()
    w, h = SIZE[rep]
    for frame, rgb in enumerate(rgb_frames(stream_to_rgb(folder/f'{rep}-crf{crf}.mkv'), w, h)):
        views = p.image_views(rgb, maps[rep])
        for v in range(9):
            rows[v].append(p.metric(refs[frame, v], views[v], masks[frame, v]))
        if frame == FRAMES//2 and crf in (18, 36):
            tiles = [cv2.resize(np.concatenate([refs[frame, v], views[v]], axis=1), (640, 320)) for v in range(9)]
            panel = np.concatenate([np.concatenate(tiles[i:i+3], axis=1) for i in (0, 3, 6)], axis=0)
            cv2.imwrite(str(folder/f'{rep}-crf{crf}-middle.jpg'), (panel[:, :, ::-1]/257).astype(np.uint8))
    result = {**encoded, 'sample_id': info['sample_id'], 'eye': info['eye'], 'evaluate_seconds': time.monotonic()-t,
              'quality': stats(rows)}
    p.save(folder/f'{rep}-crf{crf}-result.json', result)
    return result


def run_one(root, sample, eye):
    spec = spec_for(sample)
    folder = root/sample/eye
    folder.mkdir(parents=True, exist_ok=True)
    lock = (folder/'.run.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    identity = {'config': CONFIG, 'spec': spec, 'code_sha256': code_hashes()}
    ident = folder/'identity.json'
    if ident.exists():
        assert json.loads(ident.read_text()) == identity, 'Configuration changed; use a new output root'
    else:
        p.save(ident, identity)
    if (folder/'completed.json').exists():
        return
    cv2.setNumThreads(2)
    cal = p.DATA/'h265_fisheye'/spec['sample_id']/'metadata.json'
    digest = hashlib.sha256(p.canonical(p.calibration_object(cal))).hexdigest()
    lens = p.LensMap(p.BASE/'geometry-scripts/linux/maps'/f'{digest}.json', eye, cal)
    _, fvalid = lens.remap_coordinates('fisheye-to-eq', SIZE['fisheye'], SIZE['eq'])
    fm, em, view_masks = make_maps(lens, fvalid)
    maps = {'fisheye': fm, 'eq': em}
    if (folder/'prepared.json').exists():
        info = json.loads((folder/'prepared.json').read_text())
    else:
        if (folder/'cache').exists():
            raise RuntimeError(f'Incomplete cache in {folder}')
        info = prepare(folder, spec, eye, maps, view_masks)
    for rep in SIZE:
        for crf in CRFS:
            if (folder/f'{rep}-crf{crf}-result.json').exists():
                continue
            status(folder, 'encode', rep=rep, crf=crf)
            evaluate(folder, info, rep, crf, encode(folder, info, rep, crf), maps)
    for rep in SIZE:
        for crf in CRFS:
            r = json.loads((folder/f'{rep}-crf{crf}-result.json').read_text())
            assert hashlib.sha256((folder/f'{rep}-crf{crf}.mkv').read_bytes()).hexdigest() == r['bitstream_sha256']
    removed = []
    for f in sorted((folder/'cache').iterdir()):
        removed.append({'file': f.name, 'bytes': f.stat().st_size})
        f.unlink()
    (folder/'cache').rmdir()
    p.save(folder/'completed.json', {'sample_id': sample, 'eye': eye, 'frames': FRAMES, 'removed_regenerable_files': removed})
    status(folder, 'complete')


def run_all(root, jobs):
    order = [(s['sample_id'], eye) for s in json.loads(SELECTION.read_text())['samples'] for eye in ('left', 'right')]
    (root/'logs').mkdir(parents=True, exist_ok=True)
    running, failed = {}, []
    pending = [k for k in order if not (root/k[0]/k[1]/'completed.json').exists()]
    while pending or running:
        while pending and len(running) < jobs and not failed:
            k = pending.pop(0)
            log = (root/'logs'/f'{k[0]}-{k[1]}.log').open('a')
            running[k] = (subprocess.Popen([sys.executable, str(Path(__file__).resolve()), 'run-one', '--output-root', str(root),
                                            '--sample', k[0], '--eye', k[1]], stdout=log, stderr=subprocess.STDOUT, cwd=p.HERE), log)
        p.save(root/'batch-status.json', {'jobs': jobs, 'running': sorted(map(list, running)), 'pending': len(pending),
                                          'completed': sum((root/k[0]/k[1]/'completed.json').exists() for k in order),
                                          'failed': failed, 'updated_unix': time.time()})
        time.sleep(20)
        for k, (proc, log) in list(running.items()):
            if proc.poll() is not None:
                log.close()
                del running[k]
                if proc.returncode:
                    failed.append([k[0], k[1], proc.returncode])
        if failed and not running:
            break
    p.save(root/'batch-status.json', {'jobs': jobs, 'running': [], 'pending': len(pending), 'failed': failed,
                                      'completed': sum((root/k[0]/k[1]/'completed.json').exists() for k in order),
                                      'updated_unix': time.time()})
    return failed


def summarize(root, output):
    scenes = []
    for spec in json.loads(SELECTION.read_text())['samples']:
        sid = spec['sample_id']
        prepared = {eye: json.loads((root/sid/eye/'prepared.json').read_text()) for eye in ('left', 'right')}
        points = []
        for rep in SIZE:
            for crf in CRFS:
                rows = [json.loads((root/sid/eye/f'{rep}-crf{crf}-result.json').read_text()) for eye in ('left', 'right')]
                points.append({'representation': rep, 'crf': crf,
                               'stereo_mbps': sum(r['packet_payload_bits'] for r in rows)/(FRAMES/90)/1e6,
                               'regions': {region: pooled([r['quality'][region] for r in rows]) for region in REGIONS}})
        scenes.append({'sample_id': sid, 'scene_type_label': spec['scene_type'], 'status': 'complete', 'points': points,
                       'rate_differences': {r: rate_difference(points, r) for r in REGIONS},
                       'source_detail_fisheye_over_eq': float(np.mean([prepared[e]['source_detail']['fisheye']/prepared[e]['source_detail']['eq']
                                                                       for e in prepared])),
                       'source_mbps': {rep: sum(prepared[e]['sources'][rep]['bit_rate'] for e in prepared)/1e6 for rep in SIZE},
                       'first_frame': prepared['left']['first_frame']})
    aggregate = {}
    for region in REGIONS:
        values = [s['rate_differences'][region]['fisheye_vs_eq_percent'] for s in scenes
                  if s['rate_differences'][region]['status'] == 'available']
        aggregate[region] = {'available_cases': len(values), 'unavailable_cases': len(scenes)-len(values),
                             'median_percent': float(np.median(values)) if values else None,
                             'min_percent': min(values) if values else None, 'max_percent': max(values) if values else None}
    report = {'scope': '24 selected clips, both eyes, central 90 frames of the released renders; each representation against its own source',
              'config': CONFIG, 'selected_samples': len(scenes), 'all_selected_samples_complete': True,
              'scenes': scenes, 'aggregate': aggregate}
    output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps({r: aggregate[r] for r in ['all', 'center', 'lateral', 'upper', 'lower']}, indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='command', required=True)
    for name in ('run-one', 'run-all', 'summarize'):
        s = sub.add_parser(name)
        s.add_argument('--output-root', type=Path, required=True)
        if name == 'run-one':
            s.add_argument('--sample', required=True)
            s.add_argument('--eye', choices=['left', 'right'], required=True)
        if name == 'run-all':
            s.add_argument('--jobs', type=int, default=6)
        if name == 'summarize':
            s.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    root = args.output_root.expanduser().resolve()
    if root == p.DATA.resolve() or p.DATA.resolve() in root.parents:
        ap.error('Outputs must be outside the dataset')
    if args.command == 'run-one':
        run_one(root, args.sample, args.eye)
    elif args.command == 'run-all':
        failed = run_all(root, args.jobs)
        if failed:
            raise SystemExit(f'Failed: {failed}')
    else:
        summarize(root, args.output)


if __name__ == '__main__':
    main()
