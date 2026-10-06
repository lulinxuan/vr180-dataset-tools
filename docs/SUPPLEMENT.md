# Supplementary details

Technical details for the paper that are not covered by the dataset documentation (<https://huggingface.co/datasets/lulinxuan/VR180>, `docs/`), the converter README or the coding-study protocol.

## Clip extraction

Clips were chosen in a desktop tool by selecting a contiguous interval during stereo playback. The tool passes the zero-based start frame and frame count to the Blackmagic RAW SDK `CreateJobTrim` operation, reopens the output and checks frame count, raster, frame rate, lens data and immersive attributes before writing the sample. Trimming copies the compressed frames and adds no RGB encoding step. Stills are copies of original one-frame camera clips. Capture parameters in the metadata are read from the first frame of the left eye.

## Geometry evaluation

- **Round trips.** Three scenes (an airshow, a street and a zoo), frame 0 of both eyes, covering both calibrations. Decoded HEVC frames are processed as 16-bit RGB without further lossy coding: 8160 × 7200 → 7200² → 8160 × 7200 for native round trips, and 7200² → 8160 × 7200 → 7200² for half-EQ round trips. Bilinear is the converter's default; the Lanczos4 results correspond to its `--interpolation lanczos4` option.
- **Metrics.** RGB PSNR and SSIM on BT.709 luma in [0, 1] with an 11 × 11 Gaussian window (σ = 1.5), population covariance, K1 = 0.01 and K2 = 0.03. Pixels whose interpolation support or SSIM window touches invalid coordinates are excluded, as are reference pixels with luma ≤ 1/1023.
- **Coverage.** 5.37–5.84% of the non-black native pixels lie outside the reverse coordinate domain, mainly in lateral edge bands; a further 0.28–0.29 percentage points are excluded by the metric margins.
- **Orientation relative to the Resolve renders.** On a street scene, matched features between the converter's half-EQ and the dataset's half-EQ were displaced by 45.55 and 44.32 px (two eyes, 7200² scale). A spherical rotation fitted on one scene and applied unchanged to another scene with the same calibration and eye left median feature residuals of 1.05–2.01 px over eight directed tests (features measured at 1800² and rescaled); applied to six later frames of two clips, the medians were 0.73–1.57 px.
- **Linux.** Python 3.12.14, NumPy 2.5.3, SciPy 1.18.1, OpenCV 5.0.0 and FFmpeg 9.0.2 on x86-64: 9 functional checks (lookup and inversion, map integrity, 16-bit remapping, video frame counts and rates, interpolation option, audio) and 16 native-resolution conversions (one still and one video per calibration, both eyes, both directions; three frames per video) passed.

## Representation colour check

For every sample, frame 0 of the left eye of both HEVC representations was decoded at 816 px width, and the mean HSV saturation and the standard deviation of luma over non-black pixels were compared. The fisheye/half-EQ ratios are 0.85–1.41 (median 1.02) for saturation and 0.75–1.17 (median 1.00) for luma spread.
