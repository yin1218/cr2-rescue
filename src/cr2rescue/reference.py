"""Low-resolution references of the picture, used to judge where JPEG data belongs and to fill gaps.

A CR2 holds two small, independently stored versions of the same frame:

* the camera thumbnail (JPEG, 160x120 on most bodies, the 3:2 frame letterboxed inside), and
* an uncompressed 16-bit linear RGB image (e.g. 660x441 on the EOS M2, including a masked sensor border).

Both survive surprisingly often when the big preview is broken, because they live in other disk clusters.
"""
from __future__ import annotations

import io

import numpy as np
from PIL import Image

from .jpeg import ycc


def open_thumb(thumb_bytes):
    try:
        im = Image.open(io.BytesIO(bytes(thumb_bytes)))
        im.load()
        return im.convert('RGB')
    except Exception:
        return None


def thumb_reference(thumb: Image.Image, aspect: float, width=None):
    """Thumbnail -> picture-only reference with exact geometry, `width` pixels wide (default: native).

    The frame sits letterboxed in the thumbnail and the first/last content rows are blended with the black
    bars, so those rows are replaced by their neighbours and the true (fractional) content box is resampled.
    """
    t = np.asarray(thumb, np.float32)
    H, W = t.shape[:2]
    ch = W / aspect
    width = width or W
    size = (width, int(round(width / aspect)))
    if ch > H - 1:  # not letterboxed
        return thumb.resize(size, Image.BICUBIC)
    top = (H - ch) / 2
    r0, r1 = int(np.ceil(top)), int(np.floor(top + ch))
    a = np.concatenate([t[r0:r0 + 1], t[r0:r1], t[r1 - 1:r1]])
    y0 = top - (r0 - 1)
    im = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
    return im.resize(size, Image.BICUBIC, box=(0, y0, W, y0 + ch))


def small_array(buf, info):
    """Raw bytes of the uncompressed image -> (h, w, 3) float32, zero-padded if truncated."""
    n = info['width'] * info['height'] * info['spp'] * 2
    b = bytes(buf[:n]).ljust(n, b'\0')
    a = np.frombuffer(b, '<u2').reshape(info['height'], info['width'], info['spp'])[..., :3]
    return a.astype(np.float32)


def _lum(a, black):
    return np.clip(a.mean(axis=2) - black, 1, None) ** (1 / 2.2)


def _norm(x):
    x = x - x.mean()
    return x / (np.linalg.norm(x) + 1e-9)


def black_level(a):
    """The masked border reads the black level; the 0.2th percentile is a robust stand-in."""
    return float(np.percentile(a.mean(axis=2), 0.2))


class _Target:
    """One photo used for alignment: luminance of its small image, and what the box must match."""

    def __init__(self, a, thumb_ref, fine=None):
        self.G = Image.fromarray(_lum(a, black_level(a)).astype(np.float32))
        self.t = self._prep(thumb_ref)
        self.f = self._prep(fine) if fine is not None else None

    @staticmethod
    def _prep(im):
        T = np.asarray(im.convert('L'), np.float32)
        return (T.shape[1], T.shape[0]), _norm(T[1:-1, 1:-1].ravel())

    def score(self, box, fine):
        (tw, th), T = self.f if fine and self.f else self.t
        x0, y0, cw, ch = box
        c = self.G.resize((tw, th), Image.BILINEAR, box=(x0, y0, x0 + cw, y0 + ch))
        return float(_norm(np.asarray(c)[1:-1, 1:-1].ravel()) @ T)


def align_small(items, aspect):
    """Find the box of the small image that shows exactly the preview frame (it includes a masked border).

    items: [(small array, thumbnail reference, fine target or None)] of photos from one camera model -- the
    box is the same for all of them, so their scores are summed. The fine target is an intact preview reduced
    to about the small image's size; it pins the box down to one pixel, which the thumbnail alone cannot.
    Brute-force search over box width and position, then hill-climbing. Returns ((x0, y0, w, h), score)."""
    if not isinstance(items, list):
        raise TypeError('items must be a list of (array, thumb_ref[, fine])')
    T = [_Target(*it) for it in items]
    h, w = items[0][0].shape[:2]
    fine = any(t.f for t in T)

    def score(x0, y0, cw, fine=False):
        return sum(t.score((x0, y0, cw, cw / aspect), fine) for t in T) / len(T)

    coarse = []
    lo = max(int(w * 0.9), 8)
    for cw in range(w, lo - 1, -2):
        ch = cw / aspect
        if ch > h:
            continue
        for y0 in range(0, int(h - ch) + 1, 2):
            for x0 in range(0, w - cw + 1, 2):
                coarse.append((score(x0, y0, cw), (x0, y0, cw)))
    if not coarse:
        return None, -1
    coarse.sort(reverse=True)
    best = (-2, None)
    seen = {}
    for _, start in coarse[:8]:  # hill-climb on the 1-pixel grid from the best coarse boxes
        top = (seen.setdefault(start, score(*start, fine=fine)), start)
        for _ in range(16):
            x0, y0, cw = top[1]
            nxt = top
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for dw in (-2, -1, 0, 1, 2):  # both edges can move independently
                        x, y, c = x0 + dx, y0 + dy, cw + dw
                        if (dx, dy, dw) == (0, 0, 0) or x < 0 or y < 0 or x + c > w or y + c / aspect > h:
                            continue
                        if (x, y, c) not in seen:
                            seen[(x, y, c)] = score(x, y, c, fine=fine)
                        if seen[(x, y, c)] > nxt[0]:
                            nxt = (seen[(x, y, c)], (x, y, c))
            if nxt is top:
                break
            top = nxt
        if top[0] > best[0]:
            best = top
    x0, y0, cw = best[1]
    return (x0, y0, cw, int(round(cw / aspect))), best[0]


def preview_reduced(jpeg_bytes, scale=8):
    """Decode a JPEG at 1/scale size (fast DCT scaling). None if it does not decode."""
    try:
        im = Image.open(io.BytesIO(bytes(jpeg_bytes)))
        im.draft('RGB', (im.width // scale, im.height // scale))
        im.load()
        return im.convert('RGB')
    except Exception:
        return None


def bands_ok(ref_rgb, thumb_ref, bands=16, thr=0.6):
    """Per horizontal band: does the reference correlate with the camera thumbnail? (detects overwritten
    parts of the small image)"""
    t = np.asarray(thumb_ref.convert('L'), np.float32)
    g = np.asarray(ref_rgb.convert('L').resize((t.shape[1], t.shape[0]), Image.BOX), np.float32)
    edges = np.linspace(0, t.shape[0], bands + 1).astype(int)
    out = []
    for i in range(bands):
        x, y = g[edges[i]:edges[i + 1]].ravel(), t[edges[i]:edges[i + 1]].ravel()
        if y.std() < 3:
            out.append(bool(abs(x.mean() - y.mean()) < 25 and x.std() < 8))
            continue
        out.append(bool(x.std() > 0 and np.corrcoef(x, y)[0, 1] > thr))
    return out


def _robust_fit(X, Y, rows, bands=8):
    """Least squares X @ M ~ Y that ignores overwritten rows of the small image (least median of squares
    over fits to single row bands, then trimmed refits)."""
    r_of = lambda M: np.abs(X @ M - Y).sum(1)
    cands = [np.linalg.lstsq(X, Y, rcond=None)[0]]
    per_row = len(X) // rows
    for b in np.array_split(np.arange(rows), bands):
        sl = slice(b[0] * per_row, (b[-1] + 1) * per_row)
        cands.append(np.linalg.lstsq(X[sl], Y[sl], rcond=None)[0])
    M = min(cands, key=lambda M: np.median(r_of(M)))
    for _ in range(3):
        r = r_of(M)
        keep = r <= max(3 * np.median(r), 0.02)
        M = np.linalg.lstsq(X[keep], Y[keep], rcond=None)[0]
    return M


def render_small(a, box, thumb_ref=None):
    """Linear camera RGB -> display RGB. With a thumbnail, colour matrix + tone curve are fitted to it (the
    thumbnail carries the camera's picture style); otherwise a plain gamma with grey-world balance."""
    x0, y0, cw, ch = box
    black = black_level(a)
    lin = np.clip(a[y0:y0 + ch, x0:x0 + cw] - black, 0, None)
    flat = np.c_[lin.reshape(-1, 3), np.ones(ch * cw)]
    if thumb_ref is None:
        g = lin.reshape(-1, 3)
        gain = g.mean() / np.maximum(g.mean(0), 1e-6)
        v = g * gain
        v = (v / max(np.percentile(v, 99.5), 1e-6)).clip(0, 1) ** (1 / 2.2)
        return Image.fromarray((v * 255 + 0.5).astype(np.uint8).reshape(ch, cw, 3))
    t = thumb_ref
    small = np.stack([np.asarray(Image.fromarray(lin[..., c]).resize(t.size, Image.BOX)) for c in range(3)], -1)
    X = np.c_[small.reshape(-1, 3), np.ones(small.shape[0] * small.shape[1])]
    Yg = np.asarray(t, np.float32).reshape(-1, 3) / 255
    Yl = Yg ** 2.2
    M = _robust_fit(X, Yl, small.shape[0])
    pred_small = np.clip(X @ M, 0, None) ** (1 / 2.2)
    pred = np.clip(flat @ M, 0, None) ** (1 / 2.2)
    r = np.abs(X @ M - Yl).sum(1)
    keep = r <= max(3 * np.median(r), 0.02)
    pred_small, Yg = pred_small[keep], Yg[keep]
    out = np.empty_like(pred)
    for c in range(3):  # picture-style tone curve: binned median of thumbnail value per predicted value
        edges = np.quantile(pred_small[:, c], np.linspace(0, 1, 33))
        xs, ys = [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (pred_small[:, c] >= lo) & (pred_small[:, c] <= hi)
            if m.sum() > 5:
                xs.append(np.median(pred_small[m, c]))
                ys.append(np.median(Yg[m, c]))
        keep = np.r_[True, np.diff(xs) > 1e-4] if xs else []
        xs, ys = list(np.asarray(xs)[keep]), list(np.asarray(ys)[keep])
        if len(xs) < 2:
            out[:, c] = pred[:, c]
            continue
        ys = np.maximum.accumulate(ys)
        xs = [0.0] + xs + [max(1.0, xs[-1] * 1.2)]
        ys = [0.0] + list(ys) + [1.0]
        out[:, c] = np.interp(pred[:, c], xs, ys)
    return Image.fromarray((np.clip(out, 0, 1) * 255 + 0.5).astype(np.uint8).reshape(ch, cw, 3))


def mcu_reference(ref_rgb: Image.Image, jp):
    """Reference image covering the preview frame -> per-MCU mean (Y, Cb, Cr), shape (3, nmcu).

    MCU columns/rows past the picture edge (padding) take the edge values."""
    W, H = ref_rgb.size
    sx, sy = W / jp.W, H / jp.H
    bw, bh = jp.mcux * jp.mcu_w * sx, jp.mcuy * jp.mcu_h * sy  # MCU grid extent in reference pixels
    a = np.asarray(ref_rgb.convert('RGB'))
    pad_x, pad_y = int(np.ceil(bw - W)) + 1, int(np.ceil(bh - H)) + 1
    a = np.pad(a, ((0, pad_y), (0, pad_x), (0, 0)), mode='edge')
    im = Image.fromarray(a)
    method = Image.BOX if W >= jp.mcux * 2 else Image.BICUBIC
    small = im.resize((jp.mcux * 2, jp.mcuy * 2), method, box=(0, 0, bw, bh))
    c = ycc(small)
    return c.reshape(3, jp.mcuy, 2, jp.mcux, 2).mean((2, 4)).reshape(3, -1)
