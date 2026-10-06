# Fisheye ↔ half-equirectangular converter

Converts one eye of a VR180 sample between the native fisheye raster (8160 × 7200) and half-equirectangular (180° × 180°, 7200 × 7200 by default), using the factory lens calibration stored in the sample's `metadata.json`. It needs Python, NumPy, SciPy, OpenCV and FFmpeg, and runs on Linux and macOS.

The lookup tables for the dataset's two calibrations were generated with Apple's immersive-media framework and are included; a map is selected by the SHA-256 of the sample's complete `calibration.ilpd`. A camera with a different calibration needs its own table.

## Install

Python 3.10+ and FFmpeg 5.1+ are required.

```sh
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
python fetch_maps.py                       # four lookup tables, about 512 MB
python run_checks.py --report checks.json  # self-test
```

## Convert

```sh
# Native fisheye -> half-EQ, 16-bit RGB FFV1 output
python convert.py --calibration /data/h265_fisheye/<sample_id>/metadata.json \
  --eye left --direction fisheye-to-eq \
  --input /data/h265_fisheye/<sample_id>/LEFT.mov --output LEFT-eq.mkv

# Half-EQ -> native fisheye
python convert.py --calibration /data/h265_fisheye/<sample_id>/metadata.json \
  --eye left --direction eq-to-fisheye --input LEFT-eq.mkv --output LEFT-fisheye.mkv
```

- `--eye right` for the right eye. PNG, TIFF and JPEG images are accepted; image output is PNG or TIFF at the input bit depth.
- `--codec libx265 --crf 18` or `--codec hevc_nvenc --bitrate 400M` write HEVC instead of FFV1.
- `--interpolation lanczos4` resamples with Lanczos4 instead of the default bilinear; it keeps more fine detail (see Accuracy) but is slower.
- `--frames N` converts the first N frames only; `--output-size WxH` changes the output raster.
- Each run writes the output, a `.valid.png` mask of covered pixels and a `.json` record. Existing files are never overwritten.
- Audio is copied without re-encoding when the output container supports it.

## Conventions

- Half-EQ covers one eye, 180° horizontally and vertically.
- Lookup tables are 4096 × 4096 float32 maps of normalized source coordinates, sampled bilinearly; the reverse direction inverts the same table numerically. Pixel centres follow `source_pixel = uv * [width, height] - 0.5`.
- Calibration pixel coordinates always refer to the native fisheye raster.
- Inputs are BT.709 limited-range YUV or full-range RGB. Only the matrix and range are converted; no tone or colour transform is applied.

## Accuracy

Round trips on three scenes (both calibrations, both eyes, 16-bit RGB, no lossy coding):

| Domain | Interpolation | RGB PSNR (dB) | Luma SSIM |
| --- | --- | --- | --- |
| Native fisheye | Bilinear | 42.8–53.1 | 0.9906–0.9978 |
| Native fisheye | Lanczos4 | 53.9–64.9 | 0.9994–0.9999 |
| Half-EQ | Bilinear | 45.3–54.8 | 0.9945–0.9983 |
| Half-EQ | Lanczos4 | 57.3–66.3 | 0.9997–0.9999 |

- Bilinear interpolation (the default) softens fine detail. The Lanczos4 rows correspond to `--interpolation lanczos4`; for one scene the converter's round-trip output was checked bit for bit against the measured one.
- 5.4–5.8% of the non-black native pixels, mainly at the lateral edges, lie outside the half-EQ domain and cannot be restored by a round trip.
- The converter's half-EQ differs in viewing orientation from the dataset's half-EQ files rendered in DaVinci Resolve (about 45 px at 7200²). A spherical rotation fitted per calibration and eye reduces the median feature residual to 1.05–2.01 px; the converter does not apply it.
- The calibration is the manufacturer's and has not been measured independently.

## Tests

`validation-linux.json` (9 functional checks) and `validation-media-linux.json` (16 native-resolution conversions with the default bilinear interpolation: both calibrations, a still and a video each, both eyes and directions) record passing runs on Linux x86-64, with the package versions they used.
