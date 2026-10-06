#!/usr/bin/env python3
"""Download the precomputed lookup tables (maps/*.npy) from the GitHub release.

The tables (~128 MB each) exceed GitHub's file limit, so they are release assets.
Each download is checked against the size and SHA-256 recorded in maps/<calibration>.json.
"""
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

RELEASE = 'https://github.com/lulinxuan/vr180-dataset-tools/releases/download/maps-v1/'
MAPS = Path(__file__).resolve().parent / 'maps'


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def main():
    for index in sorted(MAPS.glob('*.json')):
        for eye in json.loads(index.read_text())['eyes'].values():
            target = MAPS / eye['file']
            if target.is_file() and target.stat().st_size == eye['bytes'] and sha256(target) == eye['sha256']:
                print('ok', target.name)
                continue
            partial = target.with_suffix('.part')
            print('downloading', target.name, flush=True)
            urllib.request.urlretrieve(RELEASE + eye['file'], partial)
            if partial.stat().st_size != eye['bytes'] or sha256(partial) != eye['sha256']:
                partial.unlink()
                sys.exit(f'{target.name}: size or SHA-256 mismatch')
            partial.rename(target)
            print('ok', target.name)


if __name__ == '__main__':
    main()
