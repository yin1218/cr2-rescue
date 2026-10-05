# Changelog

## 0.2.0 — 2026-10-05

- `cr2-rescue sharpen`: gives the blurry, filled-in part of partial photos detail again.
  - **neighbour**: real detail from a photo of the same camera taken up to 3 minutes apart. SIFT matches on the
    decoded part; the transform (similarity, affine or homography) is the one that best predicts the matches held
    out next to the hole, since it has to extrapolate into it. Colours fitted, optical flow for what moved (only
    where the coarse pictures disagree), and the detail used only where the neighbour's coarse picture agrees
    with the fill and the flow did not bend it out of shape.
  - **upscale**: an x4 AI super-resolution model run with `realesrgan-ncnn-vulkan` (default `realesrgan-x4plus`)
    on the box around the hole, colours pinned to the fill.
  - By default both: the neighbour where it is trusted, the upscaler for the rest. Decoded pixels are never
    changed. Writes `sharpened/` with the same file names and EXIF, and `sharpen.json`.
  - A stopped run goes on where it was.
  - Optional dependency: `pip install 'cr2-rescue[sharpen]'` (OpenCV).
- `recover` writes `masks/<photo>.png` for partial photos (white = decoded data) and a `mask` column in the report.
- Progress shows up as it goes when the output is sent to a file.

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
