"""Synthetic Canon-style CR2 files and a damaged "memory card", for tests and the demo.

No real photos are needed: a procedural landscape is stored the way a Canon body stores it (full-size 4:2:2
JPEG preview, letterboxed 160x120 thumbnail, small 16-bit linear RGB image with a masked border, RAW
blob), files are laid out in clusters on a fake card, damaged in the ways seen in practice, and "recovered"
the way carving tools do it.
"""
from __future__ import annotations

import io
import os
import struct

import numpy as np
from PIL import Image, ImageDraw, ImageFilter


# ----------------------------------------------------------------------------------------------- picture
def _noise(rng, w, h, octaves=6, base=4):
    out = np.zeros((h, w), np.float32)
    amp, tot = 1.0, 0.0
    for o in range(octaves):
        n = base * 2 ** o
        g = rng.random((max(2, n * h // w), n)).astype(np.float32)
        out += amp * np.asarray(Image.fromarray(g).resize((w, h), Image.BICUBIC))
        tot += amp
        amp *= 0.55
    return out / tot


def scene(w=1536, h=1024, seed=0):
    """A procedural landscape: sky, sun, clouds, mountain ranges, forest, lake with reflections."""
    rng = np.random.default_rng(seed)
    y = np.linspace(0, 1, h)[:, None]
    hue = rng.uniform(-0.15, 0.15)
    top = np.array([0.18 + hue, 0.38, 0.75 - hue])
    hor = np.array([0.95, 0.78 + hue / 2, 0.62])
    img = (top * (1 - y[..., None] ** 0.7) + hor * y[..., None] ** 0.7) * np.ones((h, w, 1))
    sx, sy = rng.uniform(0.2, 0.8) * w, rng.uniform(0.15, 0.35) * h
    yy, xx = np.mgrid[0:h, 0:w]
    r = np.hypot(xx - sx, yy - sy) / w
    img += (np.exp(-r * 18) * 0.6)[..., None] * np.array([1.0, 0.85, 0.5])
    img[r < 0.03] = [1.0, 0.97, 0.85]
    cl = np.clip((_noise(rng, w, h, 7, 3) - 0.5) * 3.0, 0, 1) * (1 - y) ** 1.5
    img = img * (1 - cl[..., None] * 0.7) + cl[..., None] * 0.7
    horizon = int(h * rng.uniform(0.58, 0.66))
    for k, (shade, height, rough) in enumerate(((0.55, 0.30, 0.9), (0.38, 0.22, 1.2), (0.22, 0.14, 1.6))):
        ridge = _noise(rng, w, 1, 8, 2)[0]
        ridge = horizon - (ridge * rough + 0.2) * height * h
        col = np.array([0.25, 0.32, 0.45]) * shade / 0.55 + 0.25 * (2 - k) / 2 * hor
        m = yy > ridge[None, :]
        tex = _noise(rng, w, h, 6, 8)[..., None]
        img = np.where(m[..., None] & (yy < horizon)[..., None], col * (0.75 + 0.5 * tex), img)
    trees = Image.new('L', (w, h), 0)
    d = ImageDraw.Draw(trees)
    for _ in range(int(w / 9)):
        x = rng.uniform(0, w)
        th = rng.uniform(0.03, 0.09) * h
        tw = th * rng.uniform(0.25, 0.4)
        d.polygon([(x, horizon - th), (x - tw, horizon + 2), (x + tw, horizon + 2)], fill=255)
    tm = np.asarray(trees, np.float32)[..., None] / 255
    img = img * (1 - tm) + tm * np.array([0.06, 0.16, 0.09]) * (0.8 + 0.4 * _noise(rng, w, h, 5, 16)[..., None])
    refl = img[horizon - 1::-1][:h - horizon]
    ripple = _noise(rng, w, h - horizon, 6, 24)
    shift = ((ripple - 0.5) * 12).astype(int)
    rows = np.clip(np.arange(h - horizon)[:, None] + shift, 0, h - horizon - 1)
    refl = refl[rows, np.arange(w)[None, :]]
    img[horizon:] = refl * 0.65 + np.array([0.05, 0.12, 0.16])
    img += rng.normal(0, 0.012, img.shape)
    im = Image.fromarray((np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8))
    return im.filter(ImageFilter.UnsharpMask(1.2, 60, 2))


# ----------------------------------------------------------------------------------------------- CR2 writer
def _thumbnail(img):
    """160x120 thumbnail with the frame letterboxed, edge rows blended with the bars (as Canon does)."""
    big = Image.new('RGB', (1600, 1200))
    ch = 1600 * img.height / img.width
    content = img.resize((1600, int(round(ch))), Image.BICUBIC)
    big.paste(content, (0, int(round((1200 - ch) / 2))))
    b = io.BytesIO()
    big.resize((160, 120), Image.BOX).save(b, 'JPEG', quality=90)
    return b.getvalue()


def _small(img, border=(3, 5, 2, 3), seed=0):
    """Small 16-bit linear RGB image (1/8 scale) with a masked border reading the black level."""
    rng = np.random.default_rng(seed)
    top, left, bottom, right = border
    w, h = img.width // 8, img.height // 8
    a = np.asarray(img.resize((w, h), Image.BOX), np.float32) / 255
    lin = np.where(a <= 0.04045, a / 12.92, ((a + 0.055) / 1.055) ** 2.4)
    raw = 2048 + lin * np.array([5200, 11000, 7400]) + rng.normal(0, 6, lin.shape)
    H, W = h + top + bottom, w + left + right
    full = 2048 + rng.normal(0, 6, (H, W, 3))
    full[top:top + h, left:left + w] = raw
    return np.clip(full, 0, 65535).astype('<u2'), (W, H)


class _Ifd:
    def __init__(self):
        self.e = []

    def add(self, tag, typ, vals):
        self.e.append((tag, typ, vals))


def _pack_ifd(ifd, at, nxt):
    """Serialise an IFD placed at offset `at`; out-of-line values follow the entry table."""
    ents = sorted(ifd.e)
    n = len(ents)
    extra_at = at + 2 + n * 12 + 4
    head, extra = [struct.pack('<H', n)], b''
    for tag, typ, vals in ents:
        if typ == 2:
            data = vals.encode() + b'\0'
            cnt = len(data)
        elif typ == 3:
            data, cnt = struct.pack('<%dH' % len(vals), *vals), len(vals)
        elif typ == 4:
            data, cnt = struct.pack('<%dI' % len(vals), *vals), len(vals)
        elif typ == 5:
            data, cnt = b''.join(struct.pack('<II', *v) for v in vals), len(vals)
        else:
            raise ValueError(typ)
        if len(data) <= 4:
            field = data.ljust(4, b'\0')
        else:
            field = struct.pack('<I', extra_at + len(extra))
            extra += data + b'\0' * (len(data) & 1)
        head.append(struct.pack('<HHI', tag, typ, cnt) + field)
    head.append(struct.pack('<I', nxt))
    return b''.join(head) + extra


def make_cr2(img, when='2024:05:01 10:00:00', subsec='00', model='Canon EOS Synthetic', seed=0, quality=90):
    """Build a CR2-like file around `img` (the camera-processed picture). Returns (bytes, preview_jpeg)."""
    rng = np.random.default_rng(seed + 1000)
    b = io.BytesIO()
    img.save(b, 'JPEG', quality=quality, subsampling=1)
    prev = b.getvalue()
    thumb = _thumbnail(img)
    small, (sw, sh) = _small(img, seed=seed)
    small = small.tobytes()
    raw = rng.integers(0, 256, len(prev) * 3 // 2, dtype=np.uint8).tobytes()

    to = 0x2000
    po = (to + len(thumb) + 0xFFF) // 0x1000 * 0x1000
    so = po + len(prev)
    ro = so + len(small)
    ifd0, exif, ifd1, ifd2, ifd3 = _Ifd(), _Ifd(), _Ifd(), _Ifd(), _Ifd()
    at0 = 0x10
    # sizes are fixed by the entry lists, so lay out in two passes
    ifd0.add(0x100, 4, [img.width]); ifd0.add(0x101, 4, [img.height]); ifd0.add(0x103, 3, [6])
    ifd0.add(0x10F, 2, 'Canon'); ifd0.add(0x110, 2, model); ifd0.add(0x111, 4, [po]); ifd0.add(0x112, 3, [1])
    ifd0.add(0x117, 4, [len(prev)]); ifd0.add(0x8769, 4, [0])
    exif.add(0x829A, 5, [(1, 250)]); exif.add(0x829D, 5, [(56, 10)]); exif.add(0x8827, 3, [200])
    exif.add(0x9003, 2, when); exif.add(0x9291, 2, subsec); exif.add(0x920A, 5, [(35, 1)])
    exif.add(0xA434, 2, 'EF-M22mm f/2 STM')
    ifd1.add(0x201, 4, [to]); ifd1.add(0x202, 4, [len(thumb)])
    ifd2.add(0x100, 4, [sw]); ifd2.add(0x101, 4, [sh]); ifd2.add(0x102, 3, [16, 16, 16]); ifd2.add(0x103, 3, [1])
    ifd2.add(0x111, 4, [so]); ifd2.add(0x115, 3, [3]); ifd2.add(0x117, 4, [len(small)])
    ifd3.add(0x103, 3, [6]); ifd3.add(0x111, 4, [ro]); ifd3.add(0x117, 4, [len(raw)])
    ifd3.add(0xC640, 3, [1, 1000, 1000])
    sizes = [len(_pack_ifd(x, 0, 0)) for x in (ifd0, exif, ifd1, ifd2, ifd3)]
    offs = [at0]
    for s in sizes[:-1]:
        offs.append((offs[-1] + s + 3) // 4 * 4)
    a0, ae, a1, a2, a3 = offs
    ifd0.e = [(t, ty, [ae] if t == 0x8769 else v) for t, ty, v in ifd0.e]
    blob = bytearray(to)
    blob[:16] = b'II*\x00' + struct.pack('<I', at0) + b'CR\x02\x00' + struct.pack('<I', a3)
    for ifd, at, nxt in ((ifd0, a0, a1), (exif, ae, 0), (ifd1, a1, a2), (ifd2, a2, a3), (ifd3, a3, 0)):
        p = _pack_ifd(ifd, at, nxt)
        blob[at:at + len(p)] = p
    assert len(blob) == to
    out = bytes(blob) + thumb + b'\0' * (po - to - len(thumb)) + prev + small + raw
    return out, prev


# ----------------------------------------------------------------------------------------------- the card
def _put(disk, cl, C, data):
    disk[cl * C:cl * C + len(data)] = data


def make_card(folder, cluster=8192, size=(1536, 1024), seed=0, image=None):
    """Write a damaged set of 'recovered' files into `folder`. Returns {name: expected outcome, ...}.

    Card layout (each letter = one photo, files start at cluster boundaries):

      A | B(first part) | C | B(rest) | D | E | F | G
    * A: intact
    * B: fragmented -- its second half lies after C, so a carved copy holds C's data there
    * C: two clusters of its preview were overwritten later (no other copy exists)
    * D: damaged on the card, but a second copy (another tool's output) is damaged elsewhere
    * E: the carved file stops in the middle of the preview
    * F: the whole preview was overwritten; thumbnail and small image survive
    * G: intact
    Files are carved from each header to the next header, like most carving tools do.
    image: also write the raw card (all clusters, before carving) to this path.
    """
    os.makedirs(folder, exist_ok=True)
    C = cluster
    names = 'ABCDEFG'
    files, previews = {}, {}
    for k, n in enumerate(names):
        img = scene(size[0], size[1], seed=seed * 100 + k)
        files[n], previews[n] = make_cr2(img, when=f'2024:05:01 10:00:{10 + k:02d}', subsec=f'{11 * k:02d}',
                                         seed=seed * 100 + k)
    ncl = {n: (len(d) + C - 1) // C for n, d in files.items()}
    from .tiff import parse_cr2
    lay = {n: parse_cr2(files[n]) for n in names}
    po = {n: lay[n].preview[0] for n in names}
    pl = {n: lay[n].preview[1] for n in names}

    # B is split in the middle of its preview
    split_b = (po['B'] + pl['B'] // 2) // C
    order = [('A', 0, ncl['A']), ('B', 0, split_b), ('C', 0, ncl['C']), ('B', split_b, ncl['B']),
             ('D', 0, ncl['D']), ('E', 0, ncl['E']), ('F', 0, ncl['F']), ('G', 0, ncl['G'])]
    total = sum(b - a for _, a, b in order)
    disk = bytearray(total * C)
    where = {}
    cl = 0
    for n, a, b in order:
        for k in range(a, b):
            where[(n, k)] = cl
            _put(disk, cl, C, files[n][k * C:(k + 1) * C])
            cl += 1
    rng = np.random.default_rng(seed + 7)

    def wipe(n, k, junk=None):
        c = where[(n, k)]
        disk[c * C:(c + 1) * C] = junk if junk is not None else rng.integers(0, 256, C, dtype=np.uint8).tobytes()

    # C: two clusters of the preview overwritten with another photo's preview data
    c0 = (po['C'] + pl['C'] * 2 // 3) // C
    wipe('C', c0, files['G'][po['G'] + 5 * C:po['G'] + 6 * C])
    wipe('C', c0 + 1, files['G'][po['G'] + 6 * C:po['G'] + 7 * C])
    # D: one cluster overwritten on the card; the second copy has a different cluster zeroed
    d1 = (po['D'] + pl['D'] // 3) // C
    d2 = (po['D'] + pl['D'] * 3 // 4) // C
    wipe('D', d1)
    # F: every preview cluster overwritten
    for k in range(po['F'] // C + 1, (po['F'] + pl['F']) // C):
        wipe('F', k)

    if image:
        with open(image, 'wb') as f:
            f.write(rng.integers(0, 256, 5 * C, dtype=np.uint8).tobytes() + bytes(disk))  # file-system area first
    # carve: header -> next header
    heads = sorted(where[(n, 0)] for n in names)
    out = {}
    for i, h in enumerate(heads):
        end = heads[i + 1] if i + 1 < len(heads) else total
        data = bytes(disk[h * C:end * C])
        n = next(x for x in names if where[(x, 0)] == h)
        if n == 'E':
            data = data[:po['E'] + pl['E'] // 2]  # carving stopped early
        fname = f'f{h * C // 512:07d}.cr2'
        with open(os.path.join(folder, fname), 'wb') as f:
            f.write(data)
        out[n] = fname
    # a second copy of D from "another recovery tool": correct where the card copy was damaged,
    # but with a different cluster missing (zeros), stored inside a bigger file at a cluster boundary
    d = bytearray(files['D'])
    d[d2 * C:(d2 + 1) * C] = b'\0' * C
    other = os.path.join(folder, 'other-tool')
    os.makedirs(other, exist_ok=True)
    with open(os.path.join(other, 'recup_dir.1.bin.cr2'), 'wb') as f:
        f.write(rng.integers(0, 256, 3 * C, dtype=np.uint8).tobytes() + bytes(d))
    expected = dict(A='intact', B='repaired', C='partial', D='repaired', E='partial', F='preview-only', G='intact')
    return dict(files=out, expected=expected, previews=previews, cluster=C,
                times={n: f'2024:05:01 10:00:{10 + k:02d}.{11 * k:02d}' for k, n in enumerate(names)})
