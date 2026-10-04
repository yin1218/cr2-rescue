# Contributing

Thanks for helping! Bug reports, test cases and code are all welcome.

## Reporting a problem

Open an [issue](https://github.com/yin1218/cr2-rescue/issues/new/choose) with:

- the output of `cr2-rescue scan <your folder>` (and of `recover`, if it ran),
- `report.csv` from the output folder,
- the camera model and the recovery tool you used.

**Please do not attach personal photos.** If a specific file is needed to reproduce a problem, say so in the issue and
we can find a way that keeps your pictures private (often the first ~1 MB of a CR2 — the header and thumbnail — is
enough, or a synthetic case can be built with `cr2rescue.synth`).

## Development

```bash
git clone https://github.com/yin1218/cr2-rescue && cd cr2-rescue
pip install -e '.[dev]'
pytest
```

The tests build a damaged memory card from synthetic photos (`src/cr2rescue/synth.py`) and recover it end to end.
When you fix a recovery problem, try to add the failure mode to `synth.make_card` or a focused test, so it stays fixed.

Code layout:

| module | what it does |
|---|---|
| `scan.py` | find CR2 headers in files, group copies of the same photo, guess the cluster size |
| `tiff.py` | parse the CR2/TIFF structure (preview, thumbnail, small image, EXIF tags) |
| `jpeg.py` | baseline JPEG decoder that tracks bit positions and per-MCU state (Numba) |
| `reference.py` | thumbnail / small-image references, alignment, colour fit |
| `assemble.py` | rebuild one preview cluster by cluster: verify, continue from copies, re-sync, fill |
| `gsearch.py` | global search for clusters that continue a photo |
| `recover.py` | the end-to-end pipeline and output folders |
| `export.py` | EXIF and file dates |
| `cli.py` | command line |

Style: plain Python, NumPy, short functions with a docstring that says *why*. Keep new dependencies to a minimum.

## Ideas

- CR3 support (ISO BMFF container, JPEG preview in a `PRVW` box)
- Nikon NEF / Sony ARW (also carry full-size JPEG previews)
- A simple GUI
