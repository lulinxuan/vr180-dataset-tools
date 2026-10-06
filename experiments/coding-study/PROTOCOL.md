# Coding study: native fisheye vs. half-equirectangular

As delivered, does the native fisheye or the half-equirectangular (half-EQ) representation reach a given viewport quality with less bitrate, and where in the field of view do the two differ? The study re-encodes the dataset's released renders of both representations.

## Clips

`clip-selection.json` lists 24 video clips, fixed before any clip was encoded. Eligible clips are 90 fps videos of at least 3 s, excluding three clips used to develop the pipeline and their overlap components. One clip is chosen per overlap component of `split-groups.json`, with at most two clips per recording date, three per scene type and fourteen per calibration. Clips are ranked deterministically for diversity of scene type, date, calibration, camera motion and low light, with a SHA-256 tie break; the first three slots go to clips labelled with camera translation. The set covers 22 scene types, 19 recording dates and both calibrations (14 and 10 clips). It is a diversity sample, not a random sample of the dataset.

From each clip and eye, the study uses the 90 frames (1 s, one GOP) at the centre of the 270-frame interval given in the selection file.

## Sources

The released `h265_fisheye` (8160 × 7200) and `h265_eq` (7200 × 7200) files, HEVC Main 10, 4:2:0, BT.709 limited range, rendered in DaVinci Resolve at about 570 and 510 Mb/s per eye. The 90 frames are decoded to 10-bit 4:2:0 and passed to the encoder without conversion.

## Encoding

x265 4.1 through FFmpeg: preset medium, Main 10, CRF 18, 24, 30 and 36, closed GOP of 90 frames (`keyint=90:min-keyint=90:scenecut=0:open-gop=0`), `pools=8:frame-threads=1`. Every stream is fully decoded.

## Evaluation

- Each decoded stream and its own source are rendered into nine rectilinear viewports of 1536 × 1536 pixels and 60° × 60° (yaw −60°, 0°, 60° × pitch +45°, 0°, −45°) with Lanczos4 interpolation, using the dataset's lookup table for the fisheye.
- Only pixels that are valid in both representations and non-black (luma > 1/1023) in both sources are scored.
- Y′ PSNR from pooled squared error (primary) and Gaussian SSIM (11 × 11, σ = 1.5) per viewport and per region: all nine views; centre; lateral (yaw ±60°, six views); upper (pitch +45°); lower (pitch −45°).
- Stereo bitrate: the sum of both eyes' video packet sizes per second.
- Rate difference at equal quality (BD-rate): mean difference of PCHIP-interpolated log rate over the PSNR range covered by both curves; negative values favour fisheye.
- Source detail: mean absolute luma gradient of each source in the scored viewport pixels, to check that neither render is markedly softer.

Each representation is compared with its own render, so detail lost when Resolve projects the camera image to half-EQ is not counted.

## Running

1. Set the environment variable `VR180_DATA` to the dataset root (the folder containing `h265_fisheye/` and `h265_eq/`); the lookup tables must be fetched with `geometry-scripts/linux/fetch_maps.py`.
2. `python coding_study.py run-all --output-root OUT --jobs 6` (each eye needs about 60 GB of scratch space and 25 GB of memory per concurrent encoder).
3. `python coding_study.py summarize --output-root OUT --output summary.json`
4. `python analyze_results.py --summary summary.json --manifest $VR180_DATA/manifest.json --out analysis`
5. `python plot_results.py --summary summary.json --out figures`

`study_common.py` provides the viewport geometry, metrics and rate-difference functions; each run records the SHA-256 of the code it used. The results are in [`results/`](results/).
