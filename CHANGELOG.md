# Changelog

## 0.1.0 — 2026-10-05

First public release.

- `cr2-rescue scan` and `cr2-rescue recover`.
- Groups every copy of a photo found in recovered files or a raw card image.
- Per-cluster verification of the embedded full-size JPEG against the thumbnail and the 16-bit small image.
- Lossless repair from other copies and from a global search over every cluster of every input file.
- Re-synchronisation after missing clusters; fill from the small image where data is gone.
- Final whole-run check that drops data from a look-alike photo (e.g. a burst shot) that passed the per-cluster test.
- Picture area of the small image located per camera model, using only small images that match their own thumbnail (on a damaged card many are overwritten even when the preview is intact); colour fit robust to damaged rows.
- Learns from the other photos of the same camera: a JPEG header overwritten in every copy is borrowed from them (identical bytes for the same settings), the small image's place is inferred when its IFD is lost, and a photo without a thumbnail borrows the colour style of the nearest photo in time, refitted to its own decoded picture.
- A small image that shows another picture than the one that decodes is not used; only the straight run from the header is kept, as `unverified`.
- Re-sync in smooth areas (sky, walls), where neighbouring rows match almost equally well: the number of missing bytes picks the row.
- A damaged small image no longer spoils an intact preview: a preview that decodes in one piece to its EOI marker and matches the thumbnail stays intact.
- Overwritten rows of the small image are detected without a reference and mended; thumbnail colours are transferred region by region.
- EXIF (capture time, camera, lens, exposure) and file dates; full metadata with ExifTool when installed.
- Synthetic damaged-card generator and end-to-end tests.
