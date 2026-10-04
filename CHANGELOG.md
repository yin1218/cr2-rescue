# Changelog

## 0.1.0 — 2026-10-05

First public release.

- `cr2-rescue scan` and `cr2-rescue recover`.
- Groups every copy of a photo found in recovered files or a raw card image.
- Per-cluster verification of the embedded full-size JPEG against the thumbnail and the 16-bit small image.
- Lossless repair from other copies and from a global search over every cluster of every input file.
- Re-synchronisation after missing clusters; fill from the small image where data is gone.
- Final whole-run check that drops data from a look-alike photo (e.g. a burst shot) that passed the per-cluster test.
- Picture area of the small image located per camera model; colour fit robust to damaged rows.
- EXIF (capture time, camera, lens, exposure) and file dates; full metadata with ExifTool when installed.
- Synthetic damaged-card generator and end-to-end tests.
