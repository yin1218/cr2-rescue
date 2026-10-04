"""Minimal, damage-tolerant TIFF / Canon CR2 container parsing.

Only what recovery needs: where the embedded JPEG preview, the camera thumbnail, the small uncompressed
RGB image and the RAW data live, plus a handful of EXIF tags. Truncated or partly overwritten files are
expected, so every read is bounds-checked and a missing value is ``None`` instead of an exception.
"""
from __future__ import annotations

import re
import struct
from dataclasses import dataclass, field

CR2_MAGIC = re.compile(rb'II\*\x00\x10\x00\x00\x00CR\x02\x00')

_SIZE = {1: 1, 2: 1, 3: 2, 4: 4, 5: 8, 6: 1, 7: 1, 8: 2, 9: 4, 10: 8, 11: 4, 12: 8, 13: 4}
_FMT = {3: 'H', 4: 'I', 8: 'h', 9: 'i', 11: 'f', 12: 'd', 13: 'I'}

# tags
WIDTH, HEIGHT, BPS, COMPRESSION = 0x100, 0x101, 0x102, 0x103
MAKE, MODEL, ORIENTATION, STRIP_OFFSETS, SPP, STRIP_BYTES = 0x10F, 0x110, 0x112, 0x111, 0x115, 0x117
JPEG_OFFSET, JPEG_LENGTH, EXIF_IFD = 0x201, 0x202, 0x8769
CR2_SLICE = 0xC640
EXPOSURE, FNUMBER, ISO, DTO, SUBSEC, FOCAL, LENS = 0x829A, 0x829D, 0x8827, 0x9003, 0x9291, 0x920A, 0xA434


class IFD(dict):
    """tag -> value. Integers/floats come back as tuples, ASCII as str, rationals as (num, den) tuples."""

    def one(self, tag, default=None):
        v = self.get(tag)
        if v is None:
            return default
        if isinstance(v, (tuple, list)):
            return v[0] if v else default
        return v


def read_ifd(d, off, endian='<'):
    """Parse the IFD at ``off``. Returns (IFD, next_ifd_offset) or (None, 0) if it is out of range/garbage."""
    if off <= 0 or off + 2 > len(d):
        return None, 0
    n = struct.unpack_from(endian + 'H', d, off)[0]
    if n == 0 or n > 1000 or off + 2 + n * 12 > len(d):
        return None, 0
    ifd = IFD()
    for i in range(n):
        p = off + 2 + i * 12
        tag, typ, cnt = struct.unpack_from(endian + 'HHI', d, p)
        if typ not in _SIZE or cnt > 1 << 24:
            continue
        size = _SIZE[typ] * cnt
        if size <= 4:
            raw = d[p + 8:p + 8 + size]
        else:
            at = struct.unpack_from(endian + 'I', d, p + 8)[0]
            if at + size > len(d):
                ifd[tag] = None  # value lives in a part of the file we do not have
                continue
            raw = d[at:at + size]
        ifd[tag] = _decode(raw, typ, cnt, endian)
    p = off + 2 + n * 12
    nxt = struct.unpack_from(endian + 'I', d, p)[0] if p + 4 <= len(d) else 0
    return ifd, nxt


def _decode(raw, typ, cnt, endian):
    raw = bytes(raw)
    if typ == 2:
        return raw.split(b'\0', 1)[0].decode('latin-1').strip()
    if typ in (1, 6, 7):
        return raw
    if typ in (5, 10):
        f = endian + ('I' if typ == 5 else 'i') * (2 * cnt)
        v = struct.unpack(f, raw)
        return tuple((v[2 * k], v[2 * k + 1]) for k in range(cnt))
    return struct.unpack(endian + _FMT[typ] * cnt, raw)


@dataclass
class Layout:
    """Byte ranges inside one CR2 (offsets relative to the start of the TIFF header)."""
    preview: tuple | None = None       # (offset, length) of the full-size JPEG preview
    thumb: tuple | None = None         # (offset, length) of the small JPEG thumbnail (e.g. 160x120)
    small: dict | None = None          # uncompressed 16-bit RGB image: offset, length, width, height
    raw: tuple | None = None           # (offset, length) of the lossless-JPEG RAW data
    tags: dict = field(default_factory=dict)

    @property
    def capture_time(self):
        t = self.tags.get('DateTimeOriginal')
        if not t:
            return None
        s = self.tags.get('SubSecTimeOriginal')
        return f'{t}.{s}' if s else t

    @property
    def end(self):
        """Smallest file size that holds every part we know about."""
        e = 0
        for r in (self.preview, self.thumb, self.raw):
            if r:
                e = max(e, r[0] + r[1])
        if self.small:
            e = max(e, self.small['offset'] + self.small['length'])
        return e


def _strip(ifd):
    off, cnt = ifd.get(STRIP_OFFSETS), ifd.get(STRIP_BYTES)
    if not off or not cnt:
        return None
    return int(off[0]), int(sum(cnt))  # Canon writes one strip; several are assumed contiguous


def parse_cr2(d) -> Layout:
    """Parse a CR2 that starts at d[0]. Raises ValueError if this is not a CR2 header."""
    if not CR2_MAGIC.match(bytes(d[:12])):
        raise ValueError('not a CR2 header')
    lay = Layout()
    off = struct.unpack_from('<I', d, 4)[0]
    ifds = []
    seen = set()
    while off and off not in seen and len(ifds) < 8:
        seen.add(off)
        ifd, off = read_ifd(d, off)
        if ifd is None:
            break
        ifds.append(ifd)
    if not ifds:
        raise ValueError('IFD0 unreadable')
    ifd0 = ifds[0]
    lay.preview = _strip(ifd0)
    for ifd in ifds[1:]:
        if ifd.get(JPEG_OFFSET) and ifd.get(JPEG_LENGTH):
            lay.thumb = (int(ifd.one(JPEG_OFFSET)), int(ifd.one(JPEG_LENGTH)))
        elif ifd.one(COMPRESSION) == 1 and ifd.get(STRIP_OFFSETS) and ifd.one(BPS) == 16:
            o, n = _strip(ifd)
            lay.small = dict(offset=o, length=n, width=int(ifd.one(WIDTH)), height=int(ifd.one(HEIGHT)),
                             spp=int(ifd.one(SPP, 3)))
        elif ifd.get(STRIP_OFFSETS) and (ifd.get(CR2_SLICE) is not None or ifd.one(COMPRESSION) == 6):
            lay.raw = _strip(ifd)
    t = lay.tags
    for k, tag in (('Make', MAKE), ('Model', MODEL)):
        if isinstance(ifd0.get(tag), str):
            t[k] = ifd0[tag]
    if ifd0.one(ORIENTATION):
        t['Orientation'] = int(ifd0.one(ORIENTATION))
    if ifd0.one(EXIF_IFD):
        ex, _ = read_ifd(d, int(ifd0.one(EXIF_IFD)))
        if ex:
            for k, tag in (('DateTimeOriginal', DTO), ('SubSecTimeOriginal', SUBSEC), ('LensModel', LENS)):
                if isinstance(ex.get(tag), str) and ex[tag]:
                    t[k] = ex[tag]
            for k, tag in (('ExposureTime', EXPOSURE), ('FNumber', FNUMBER), ('FocalLength', FOCAL)):
                v = ex.one(tag)
                if isinstance(v, tuple) and v[1]:
                    t[k] = v
            if ex.one(ISO):
                t['ISO'] = int(ex.one(ISO))
    return lay


def find_headers(d, start=0, end=None):
    """Offsets of every CR2 header in a byte buffer (recovered files often hold several photos)."""
    return [m.start() for m in CR2_MAGIC.finditer(d, start, len(d) if end is None else end)]
