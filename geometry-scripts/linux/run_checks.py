#!/usr/bin/env python3
"""Run numerical and small video tests, recording the actual host platform."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import cv2
import numpy
import scipy

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--report', type=Path, required=True)
args = p.parse_args()
if args.report.exists():
    p.error('Report already exists; choose a new path')
root = Path(__file__).resolve().parent
result = subprocess.run([sys.executable, '-m', 'unittest', '-v', 'test_geometry', 'test_video'],
                        cwd=root, capture_output=True, text=True, timeout=300)
ffmpeg = shutil.which('ffmpeg')
version = subprocess.run([ffmpeg, '-version'], capture_output=True, text=True).stdout.splitlines()[0] if ffmpeg else None
report = {
    'schema': 'vr180.geometry.validation.v1',
    'tested_at_utc': datetime.now(timezone.utc).isoformat(),
    'system': platform.system(), 'machine': platform.machine(),
    'python': platform.python_version(),
    'packages': {'numpy': numpy.__version__, 'scipy': scipy.__version__, 'opencv': cv2.__version__},
    'ffmpeg': version, 'passed': result.returncode == 0,
    'output': result.stdout + result.stderr,
    'scope': 'Numerical lookup/inversion, full-calibration/map integrity, 16-bit image remap, synthetic CFR video count/rate, interpolation option, audio copy, and inverse video. Not physical calibration accuracy or equality to Resolve.',
}
args.report.parent.mkdir(parents=True, exist_ok=True)
args.report.write_text(json.dumps(report, indent=2) + '\n')
print(report['output'])
print(f"Host: {report['system']} {report['machine']}; passed: {report['passed']}")
raise SystemExit(result.returncode)
