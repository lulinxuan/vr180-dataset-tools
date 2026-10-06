# VR180 dataset tools

Code for the stereo VR180 dataset at <https://huggingface.co/datasets/lulinxuan/VR180>: 1,211 samples recorded with two Blackmagic URSA Cine Immersive cameras, each as camera-native Blackmagic RAW, native fisheye HEVC (8160 × 7200 per eye) and half-equirectangular HEVC (7200 × 7200 per eye), with factory lens calibration and scene annotations.

| Directory | Content |
| --- | --- |
| [`geometry-scripts/linux/`](geometry-scripts/linux/) | Fisheye ↔ half-equirectangular converter using the factory calibration |
| [`experiments/coding-study/`](experiments/coding-study/) | Fisheye vs. half-equirectangular coding study ([protocol](experiments/coding-study/PROTOCOL.md)) |
| [`scripts/`](scripts/) | Paper figures and dataset analyses: overview figure, scene-type distribution, angular sampling density of the two representations, left/right consistency of every sample |
| [`docs/SUPPLEMENT.md`](docs/SUPPLEMENT.md) | Supplementary technical details for the paper |

## Converter

```sh
cd geometry-scripts/linux
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
python fetch_maps.py
python convert.py --calibration <dataset>/h265_fisheye/<sample_id>/metadata.json \
  --eye left --direction fisheye-to-eq \
  --input <dataset>/h265_fisheye/<sample_id>/LEFT.mov --output LEFT-eq.mkv
```

See [`geometry-scripts/linux/README.md`](geometry-scripts/linux/README.md) for options, conventions and measured accuracy.

## Coding study

Re-encodes the released fisheye and half-equirectangular renders of 24 clips and compares them in common viewports; see [`experiments/coding-study/PROTOCOL.md`](experiments/coding-study/PROTOCOL.md). Results, including per-clip rate–quality curves, are in [`experiments/coding-study/results/`](experiments/coding-study/results/).

## License

The code is released under the [MIT License](LICENSE). The calibration files in `geometry-scripts/linux/calibrations/` and the lookup tables of the `maps-v1` release are derived from the dataset and share its [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) license.

## Citation

Paper: [A Camera-Native Stereo VR180 Dataset](https://arxiv.org/abs/2610.10607) (arXiv:2610.10607).

```bibtex
@misc{lu2026vr180,
  author        = {Lu, Linxuan},
  title         = {A Camera-Native Stereo {VR180} Dataset},
  year          = {2026},
  eprint        = {2610.10607},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV},
  url           = {https://arxiv.org/abs/2610.10607}
}
```
