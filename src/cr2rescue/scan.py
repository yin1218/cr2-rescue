"""Find every CR2 photo in a set of files and group the copies of each photo.

Recovery tools (PhotoRec, Disk Drill, ...) often write the same photo several times, or several photos into
one file. Every CR2 header found is a *copy*; copies with the same camera model and capture time (to 1/100 s)
are the same photo, and each may be damaged in a different place.
"""
from __future__ import annotations

import os
import re
from collections import Counter, OrderedDict
from dataclasses import dataclass, field
from math import gcd

from .gsearch import open_map
from .tiff import Layout, find_headers, parse_cr2

DEFAULT_CLUSTER = 32768
HEADER_READ = 1 << 20


class Store:
    """Per-process cache of read-only file maps (bounded, so hundreds of files do not exhaust handles)."""

    def __init__(self, max_open=64):
        self.maps = OrderedDict()
        self.max_open = max_open
        self.sizes = {}

    def map(self, path):
        m = self.maps.get(path)
        if m is None:
            m = open_map(path)
            self.maps[path] = m
            if len(self.maps) > self.max_open:
                _, old = self.maps.popitem(last=False)
                if hasattr(old, 'close'):
                    old.close()
        else:
            self.maps.move_to_end(path)
        return m

    def read(self, path, a, b):
        if b <= 0:
            return b''
        return bytes(self.map(path)[max(0, a):b])

    def size(self, path):
        if path not in self.sizes:
            self.sizes[path] = os.path.getsize(path)
        return self.sizes[path]

    def getter(self, path, shift):
        """get(x0, x1) -> bytes [shift + x0, shift + x1) of the file (photo-relative view)."""
        def get(x0, x1):
            if shift + x0 < 0:
                return b''
            return self.read(path, shift + x0, shift + x1)
        return get


STORE = Store()


@dataclass
class Copy:
    path: str
    offset: int
    layout: Layout

    @property
    def label(self):
        return f'{os.path.basename(self.path)}@{self.offset}'


@dataclass
class Photo:
    key: str
    copies: list = field(default_factory=list)
    name: str = ''

    @property
    def layout(self):
        return self.copies[0].layout

    @property
    def capture_time(self):
        return self.layout.capture_time


CR2_EXT = re.compile(r'\.(cr2)$', re.I)


def collect(paths, all_files=False, recursive=True):
    """Input paths (files or folders) -> sorted list of files to search."""
    out = []
    for p in paths:
        if os.path.isdir(p):
            walk = os.walk(p) if recursive else [(p, [], os.listdir(p))]
            for root, dirs, files in walk:
                dirs[:] = [d for d in dirs if not d.startswith('.')]
                for f in files:
                    if f.startswith('.'):
                        continue
                    if all_files or CR2_EXT.search(f):
                        out.append(os.path.join(root, f))
        elif os.path.isfile(p):
            out.append(p)
        else:
            raise FileNotFoundError(p)
    return sorted(set(os.path.abspath(p) for p in out))


def headers_in(path, store=STORE):
    m = store.map(path)
    if not len(m):
        return []
    return find_headers(m)


def guess_cluster(offsets_per_file, sizes=(), default=DEFAULT_CLUSTER):
    """Guess the card's cluster (allocation unit) size.

    Files on FAT/exFAT start at cluster boundaries, so distances between photos found in one recovered file
    are multiples of the cluster size; so are the sizes of files carved from one header to the next.
    Returns the largest power of two the evidence supports, capped at the default (assuming a smaller cluster
    than the real one is safe, only slower; a larger one is not)."""
    g, n = 0, 0
    for offs in offsets_per_file:
        for a, b in zip(offs, offs[1:]):
            g = gcd(g, b - a)
            n += 1
    if n >= 2 and g:
        return int(min(max(g & -g, 512), default))  # largest power of two dividing every distance
    sizes = [s for s in sizes if s > 0]
    if len(sizes) >= 3:
        c = default
        while c >= 512:
            if sum(s % c == 0 for s in sizes) >= 0.6 * len(sizes):
                return c
            c //= 2
    if n == 1 and g:
        return int(min(max(g & -g, 512), default))
    return default


def file_phase(offsets, cluster):
    """Byte offset of the first cluster boundary in a file (where the copies' headers sit)."""
    if not offsets:
        return 0
    return Counter(o % cluster for o in offsets).most_common(1)[0][0]


def scan(paths, cluster=None, store=STORE, log=print):
    """Returns (photos, phases {path: phase}, cluster, problems)."""
    found = {}
    problems = []
    for i, path in enumerate(paths):
        try:
            found[path] = headers_in(path, store)
        except (OSError, ValueError) as e:
            problems.append((path, f'unreadable: {e}'))
            found[path] = []
    if cluster is None:
        cluster = guess_cluster([v for v in found.values() if len(v) >= 2],
                                [store.size(p) for p, v in found.items() if v])
    phases = {p: file_phase(v, cluster) for p, v in found.items()}
    groups = {}
    for path, offs in found.items():
        m = store.map(path)
        for o in offs:
            try:
                lay = parse_cr2(bytes(m[o:o + HEADER_READ]))
            except (ValueError, IndexError) as e:
                problems.append((path, f'header at {o} unreadable: {e}'))
                continue
            if not lay.preview:
                problems.append((path, f'header at {o} has no preview'))
                continue
            t = lay.capture_time
            key = f"{lay.tags.get('Model', '?')}|{t}" if t else f'{path}@{o}'
            groups.setdefault(key, Photo(key)).copies.append(Copy(path, o, lay))
    photos = sorted(groups.values(), key=lambda p: (p.capture_time or '~', p.key))
    _name(photos)
    return photos, phases, cluster, problems


def _name(photos):
    """Output name: the original file name when one copy is a whole file (IMG_1234.CR2 -> IMG_1234),
    otherwise the capture time."""
    used = set()
    for p in photos:
        p.copies.sort(key=lambda c: (c.offset != 0, ' ' in os.path.basename(c.path), c.path, c.offset))
        c = p.copies[0]
        if c.offset == 0:
            stem = os.path.splitext(os.path.basename(c.path))[0]
        elif p.capture_time:
            stem = p.capture_time.replace(':', '-', 2).replace(' ', '_').replace(':', '-').replace('.', '_')
        else:
            stem = f'{os.path.splitext(os.path.basename(c.path))[0]}_{c.offset}'
        base, k = stem, 2
        while stem.lower() in used:
            stem = f'{base}_{k}'
            k += 1
        used.add(stem.lower())
        p.name = stem
