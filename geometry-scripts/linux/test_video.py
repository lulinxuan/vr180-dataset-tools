"""Small CPU-only integration tests; no dataset movies or Apple SDK needed."""
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parent


def run(command):
    return subprocess.run(command, check=True, capture_output=True, text=True, timeout=90)


class VideoTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ffmpeg = shutil.which('ffmpeg')
        cls.ffprobe = shutil.which('ffprobe')
        if not cls.ffmpeg or not cls.ffprobe:
            raise RuntimeError('Install FFmpeg and ffprobe to run the video tests')
        cls.temp = tempfile.TemporaryDirectory()
        cls.base = Path(cls.temp.name)
        cls.source = cls.base/'source.mkv'
        cls.cal = next((ROOT/'calibrations').glob('*.json'))
        run([cls.ffmpeg, '-v', 'error', '-nostdin', '-n',
             '-f', 'lavfi', '-i', 'testsrc2=size=136x120:rate=30:duration=0.2',
             '-f', 'lavfi', '-i', 'sine=frequency=440:sample_rate=48000:duration=0.2',
             '-c:v', 'ffv1', '-pix_fmt', 'yuv420p10le', '-colorspace', 'bt709',
             '-color_range', 'tv', '-c:a', 'pcm_s16le', str(cls.source)])

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def convert(self, source, output, direction, *extra):
        return run([sys.executable, str(ROOT/'convert.py'), '--calibration', str(self.cal),
                    '--eye', 'left', '--direction', direction, '--input', str(source),
                    '--output', str(output), *extra])

    def streams(self, path):
        # count_frames forces decoding; this checks actual frames, not just headers.
        return json.loads(run([self.ffprobe, '-v', 'error', '-count_frames',
                               '-show_streams', '-of', 'json', str(path)]).stdout)['streams']

    def test_limited_video_exits_and_preserves_count(self):
        out = self.base/'limited.mkv'
        result = self.convert(self.source, out, 'fisheye-to-eq', '--allow-resized-native',
                              '--output-size', '64x64', '--frames', '3')
        self.assertEqual(json.loads(result.stdout)['frames'], 3)
        streams = self.streams(out)
        self.assertEqual(len(streams), 1)
        self.assertEqual(int(streams[0]['nb_read_frames']), 3)
        self.assertEqual(streams[0]['avg_frame_rate'], '30/1')

    def test_interpolation_is_applied_and_recorded(self):
        digests = set()
        for mode in ('linear', 'lanczos4'):
            out = self.base/f'interp-{mode}.mkv'
            result = json.loads(self.convert(self.source, out, 'fisheye-to-eq', '--allow-resized-native',
                                             '--output-size', '64x64', '--frames', '1',
                                             '--interpolation', mode).stdout)
            self.assertEqual(result['interpolation'], mode)
            self.assertEqual(json.loads(out.with_name(out.name+'.json').read_text())['interpolation'], mode)
            digests.add(result['output_sha256'])
        self.assertEqual(len(digests), 2)

    def test_whole_video_audio_and_reverse(self):
        eq = self.base/'whole.mkv'
        self.convert(self.source, eq, 'fisheye-to-eq', '--allow-resized-native',
                     '--output-size', '64x64')
        native = self.base/'inverse.mkv'
        self.convert(eq, native, 'eq-to-fisheye', '--output-size', '136x120')
        for path in (eq, native):
            streams = self.streams(path)
            video = next(s for s in streams if s['codec_type'] == 'video')
            audio = next(s for s in streams if s['codec_type'] == 'audio')
            self.assertEqual(int(video['nb_read_frames']), 6)
            self.assertEqual(video['avg_frame_rate'], '30/1')
            self.assertEqual(video['pix_fmt'], 'gbrp16le')
            self.assertEqual(audio['codec_name'], 'pcm_s16le')
            # Compare decoded audio bytes; remapping must not change the audio.
            command = [self.ffmpeg, '-v', 'error', '-i', str(path), '-map', '0:a:0',
                       '-f', 'hash', '-hash', 'sha256', '-']
            source_command = command.copy()
            source_command[4] = str(self.source)
            self.assertEqual(run(command).stdout, run(source_command).stdout)


if __name__ == '__main__':
    unittest.main(verbosity=2)
