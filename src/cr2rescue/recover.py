"""End-to-end recovery: scan -> rebuild every photo -> global search for missing clusters -> export."""
from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import time
from dataclasses import dataclass, field
from multiprocessing import get_context

import numpy as np
from PIL import Image

from . import export
from .assemble import Assembler
from .gsearch import pick_continuation, scan_file
from .jpeg import Jpeg, markers_ok
from .reference import (align_small, bands_ok, open_thumb, preview_reduced, render_small, small_array,
                        small_fit, small_score, thumb_reference)
from .scan import STORE, collect, scan

GOOD_SMALL = 0.9  # an intact small image matches its own thumbnail at least this well (overwritten: ~0)
FOLDERS = ('intact', 'repaired', 'partial', 'preview-only', 'unverified')


@dataclass
class Options:
    cluster: int = 0                 # 0 = auto
    jobs: int = 0                    # 0 = all CPUs
    global_search: bool = True
    passes: int = 3
    exiftool: str = 'auto'           # auto | yes | no
    all_files: bool = False
    keep_work: bool = False
    quality: int = 95
    min_coverage: float = 0.1        # partial results below this go to preview-only (if a reference exists)


@dataclass
class Result:
    key: str
    name: str
    category: str = 'failed'
    coverage: float = 0.0
    reference: str = ''
    copies: int = 0
    primary: str = ''
    sources: list = field(default_factory=list)
    file: str = ''                   # work file holding the output
    note: str = ''
    owns: list = field(default_factory=list)  # fingerprints of the clusters this photo is made of


def fingerprint(b):
    """Identity of a cluster's content (its first 4 KB), stable across processes."""
    return hashlib.blake2b(bytes(b[:4096]), digest_size=10).digest()


def _owns(buf, po, cluster):
    """Fingerprints of every whole cluster of a finished preview (buf = preview bytes at photo offset po)."""
    x0 = -(-po // cluster) * cluster
    return [fingerprint(buf[x - po:x - po + 4096]) for x in range(x0, po + len(buf) - 4096, cluster)]


# ----------------------------------------------------------------------------------------------- reference
def _thumb(photo):
    for c in photo.copies:
        if c.layout.thumb:
            o, n = c.layout.thumb
            b = STORE.read(c.path, c.offset + o, c.offset + o + n)
            if len(b) == n:
                im = open_thumb(b)
                if im is not None:
                    return im
    return None


def _small_of(c):
    s = c.layout.small
    if not s:
        return None
    a0 = c.offset + s['offset']
    buf = STORE.read(c.path, a0, a0 + s['length'])
    return small_array(buf, s) if len(buf) == s['length'] else None


def _align_key(photo, aspect):
    s = photo.layout.small
    return (photo.layout.tags.get('Model'), s['width'], s['height'], round(aspect, 3)) if s else None


def align_models(photos, log=print, per_model=6, max_tries=60):
    """Where the picture sits inside the small image depends only on the camera model: find it once per model,
    from a few photos whose small image and thumbnail (and ideally whole preview) are intact.

    On a damaged card many small images are overwritten even when the preview is fine, so every candidate is
    first fitted to its own thumbnail and used only if it matches (`small_fit`). A partly damaged one can still
    match at a wrong box; whatever disagrees with the joint box is dropped and the box found again."""
    groups = {}
    for p in photos:
        c = p.copies[0]
        if not (c.layout.small and c.layout.preview):
            continue
        po, pl = c.layout.preview
        try:
            jp = Jpeg(STORE.read(c.path, c.offset + po, c.offset + po + min(pl, 65536)) + b'\0' * 16)
        except ValueError:
            continue
        groups.setdefault(_align_key(p, jp.W / jp.H), []).append((p, jp.W / jp.H))
    boxes = {}
    for k, members in groups.items():
        good, tried = [], 0
        for p, aspect in members:
            if tried >= max_tries or len(good) >= 2 * per_model or (
                    len(good) >= per_model and any(g[2] is not None for g in good)):
                break
            thumb = _thumb(p)
            if thumb is None:
                continue
            tnat = thumb_reference(thumb, aspect)
            for c in p.copies:
                a = _small_of(c)
                if a is None or a.shape[:2] != (k[2], k[1]):
                    continue
                tried += 1
                sc, own = small_fit(a, tnat, aspect)
                if sc < GOOD_SMALL:
                    continue
                fine = None
                if c.layout.preview:
                    po, pl = c.layout.preview
                    prev = STORE.read(c.path, c.offset + po, c.offset + po + pl)
                    if len(prev) == pl and markers_ok(prev):
                        fine = preview_reduced(prev)
                        if fine is not None and not all(bands_ok(fine, tnat)):
                            fine = None
                good.append((a, tnat, fine, own))
                break
        items = sorted(good, key=lambda it: it[2] is None)[:per_model]  # a verified preview pins the box best
        box = None
        for _ in range(3):
            if not items:
                break
            box, sc = align_small([it[:3] for it in items], k[3], starts=[it[3] for it in items])
            fit = [it for it in items if box and small_score(it[0], it[1], box) >= GOOD_SMALL]
            if len(fit) == len(items):
                break
            items, box = fit, None
        if box is not None:
            boxes[k] = box
            fine_n = sum(it[2] is not None for it in items)
            checked = f', {fine_n} preview-checked' if fine_n else ''
            log(f'  {k[0]}: picture area of the {k[1]}x{k[2]} image = {box} (match {sc:.3f}, {len(items)} photo(s)'
                f'{checked}; {len(good)} of {tried} small images intact)')
        else:
            log(f'  {k[0]}: no intact small image among {tried} tried -- using thumbnails')
    return boxes


def reference_for(photo, W, H, boxes=None):
    """Best low-res reference of the full frame: the small RGB image when it is intact, else the thumbnail.
    Returns (image, kind) with kind in {'small', 'thumb', 'small-unverified', None}."""
    aspect = W / H
    thumb = _thumb(photo)
    tnat = thumb_reference(thumb, aspect) if thumb else None
    for c in photo.copies:
        a = _small_of(c)
        if a is None:
            continue
        box = (boxes or {}).get(_align_key(photo, aspect))
        if tnat is None:
            if box is None:
                h, w = a.shape[:2]
                cw = min(w, int(h * aspect))
                ch = int(round(cw / aspect))
                box = ((w - cw) // 2, (h - ch) // 2, cw, ch)
            return render_small(a, box), 'small-unverified'
        if box is None:
            box, _ = align_small([(a, tnat, None)], aspect)
            if box is None:
                continue
        img = render_small(a, box, tnat)
        if all(bands_ok(img, tnat)):
            return img, 'small'
    if thumb is not None:
        return thumb_reference(thumb, aspect, width=max(W // 8, tnat.width)), 'thumb'
    return None, None


# ----------------------------------------------------------------------------------------------- one photo
_S = {}


def _init(settings):
    _S.update(settings)


def _sources(photo, primary, extras):
    out = [(primary.label, STORE.getter(primary.path, primary.offset))]
    for c in photo.copies:
        if c is not primary:
            out.append((c.label + ' copy', STORE.getter(c.path, c.offset)))
    for path, shift in extras:
        out.append((f'{os.path.basename(path)}@{shift} found', STORE.getter(path, shift)))
    return out


def process(job):
    """Rebuild one photo. job = (photo, extras, pass_no). Returns (Result, open holes)."""
    photo, extras, pno = job
    cluster, work = _S['cluster'], _S['work']
    r = Result(photo.key, photo.name, copies=len(photo.copies))
    try:
        return _process(photo, extras, pno, cluster, work, r)
    except Exception as e:  # one bad photo must not stop the rest
        r.note = f'error: {type(e).__name__}: {e}'[:300]
        return r, []


def _process(photo, extras, pno, cluster, work, r):
    # usable copies = JPEG header readable
    cands = []
    for c in photo.copies:
        po, pl = c.layout.preview
        head = STORE.read(c.path, c.offset + po, c.offset + po + min(pl, 65536))
        try:
            jp = Jpeg(head + b'\0' * 16)
        except ValueError:
            continue
        cands.append((c, jp))
    if not cands:
        thumb = _thumb(photo)
        if thumb is not None:
            r.category, r.reference = 'preview-only', 'thumb'
            r.file = _save(work, photo, pno, thumb.resize((thumb.width * 4, thumb.height * 4), Image.LANCZOS))
        r.note = 'JPEG header of the preview is damaged in every copy'
        return r, []
    W, H = cands[0][1].W, cands[0][1].H
    ref, kind = reference_for(photo, W, H, _S.get('boxes'))
    r.reference = kind or 'none'
    blur = (5, 3) if kind == 'thumb' else None

    if ref is None:  # nothing to check against: keep a copy that decodes to the end, as-is
        for c, jp in cands:
            po, pl = c.layout.preview
            b = STORE.read(c.path, c.offset + po, c.offset + po + pl)
            if len(b) == pl and markers_ok(b):
                r.category, r.coverage, r.primary = 'unverified', 1.0, c.label
                r.file = _save(work, photo, pno, b)
                r.note = 'no thumbnail to verify against'
                return r, []
        r.note = 'no reference and no complete copy'
        return r, []

    # primary copy = the one with the longest good start
    scored = []
    for c, jp in cands:
        po, pl = c.layout.preview
        try:
            a = Assembler([(c.label, STORE.getter(c.path, c.offset))], po, pl, ref, cluster, blur).run(first_only=True)
        except ValueError:
            continue
        scored.append((a.coverage, c.offset == 0, c))
    if not scored:
        r.note = 'preview unreadable'
        return r, []
    scored.sort(key=lambda t: (t[0], t[1]), reverse=True)
    primary = scored[0][2]
    po, pl = primary.layout.preview
    a = Assembler(_sources(photo, primary, extras), po, pl, ref, cluster, blur).run()
    r.primary = primary.label
    r.coverage = a.coverage
    used = sorted(set(int(k) for k in np.unique(a.how) if k >= 0))
    r.sources = [a.sources[k][0] for k in used]

    clean = len(a.log) == 1 and a.log[0][0] == 'ok' and a.log[0][4] is None
    if clean and a.have.all():
        r.category = 'intact'
        b = STORE.read(primary.path, primary.offset + po, primary.offset + po + pl)
        r.file = _save(work, photo, pno, b)
        r.owns = _owns(b, po, cluster)
        return r, []
    lossless = a.jpeg_bytes()
    if lossless is not None:
        r.category = 'repaired'
        r.file = _save(work, photo, pno, lossless)
        r.owns = _owns(bytes(a.buf), po, cluster)
        return r, []
    holes = []
    for h in a.holes:
        h['key'] = photo.key
        holes.append(h)
    if a.coverage > 0 and a.coverage >= _S.get('min_coverage', 0):
        r.category = 'partial'
        r.file = _save(work, photo, pno, a.image(ref))
    else:
        r.category = 'preview-only'
        r.file = _save(work, photo, pno, ref.resize((W, H), Image.LANCZOS))
    return r, holes


def _save(work, photo, pno, what):
    path = os.path.join(work, f'{photo.name}.p{pno}.jpg')
    tags = photo.layout.tags
    if isinstance(what, (bytes, bytearray)):
        export.write_jpeg_bytes(path, bytes(what), tags)
    else:
        export.write_image(path, what, tags, quality=_S.get('quality', 95))
    return path


# ----------------------------------------------------------------------------------------------- global
_H = []


def _init_search(settings, holes):
    _S.update(settings)
    _H[:] = holes


def _search_one(args):
    path, phase, cluster, skip = args
    return path, scan_file(path, phase, cluster, _H, skip=skip)


def _better(new, old):
    rank = {'intact': 4, 'repaired': 3, 'unverified': 2, 'partial': 1, 'preview-only': 0, 'failed': -1}
    if old is None:
        return True
    if new.category == old.category:
        return new.coverage > old.coverage + 1e-6
    return rank[new.category] > rank[old.category]


def recover(paths, out_dir, opt: Options = None, log=print):
    opt = opt or Options()
    t0 = time.time()
    files = collect(paths, all_files=opt.all_files)
    log(f'Searching {len(files)} file(s) for CR2 photos ...')
    photos, phases, cluster, problems = scan(files, cluster=opt.cluster or None, log=log)
    log(f'Found {sum(len(p.copies) for p in photos)} copies of {len(photos)} photo(s); cluster size {cluster} bytes')
    for path, why in problems[:20]:
        log(f'  ! {os.path.basename(path)}: {why}')
    os.makedirs(out_dir, exist_ok=True)
    work = os.path.join(out_dir, '.work')
    os.makedirs(work, exist_ok=True)
    jobs = opt.jobs or os.cpu_count() or 1
    log('Locating the picture area of the small images ...')
    boxes = align_models(photos, log)
    settings = dict(cluster=cluster, work=work, quality=opt.quality, min_coverage=opt.min_coverage, boxes=boxes)
    ctx = get_context('spawn')

    best = {}
    extras = {p.key: [] for p in photos}
    by_key = {p.key: p for p in photos}
    searched = set()
    todo = list(photos)
    for pno in range(opt.passes + 1):
        if not todo:
            break
        log(f'Pass {pno + 1}: rebuilding {len(todo)} photo(s) ...')
        holes = []
        with ctx.Pool(jobs, initializer=_init, initargs=(settings,)) as pool:
            jobs_in = [(p, extras[p.key], pno) for p in todo]
            for k, (res, hs) in enumerate(pool.imap_unordered(process, jobs_in)):
                old = best.get(res.key)
                if _better(res, old):
                    if old and old.file and os.path.exists(old.file):
                        os.remove(old.file)
                    best[res.key] = res
                elif res.file and os.path.exists(res.file):
                    os.remove(res.file)
                holes += [h for h in hs if (h['key'], h['c'], h['j']) not in searched]
                if (k + 1) % 25 == 0 or k + 1 == len(jobs_in):
                    log(f'  {k + 1}/{len(jobs_in)}  ({time.time() - t0:.0f}s)')
        _summary(best, log)
        if not opt.global_search or pno == opt.passes or not holes:
            break
        log(f'Global search: {len(holes)} hole(s) x every {cluster // 1024} KB cluster of {len(files)} file(s) ...')
        skip = {}
        for res in best.values():
            if res.category == 'intact':
                p = by_key[res.key]
                c = next(c for c in p.copies if c.label == res.primary)
                po, pl = c.layout.preview
                skip.setdefault(c.path, []).append((c.offset + po, c.offset + po + pl))
        cands = {}
        args = [(f, phases[f], cluster, skip.get(f, [])) for f in files]
        with ctx.Pool(jobs, initializer=_init_search, initargs=(settings, holes)) as pool:
            for path, found in pool.imap_unordered(_search_one, args):
                for i, lst in found.items():
                    for o, e, n, lv in lst:
                        cands.setdefault(i, []).append((e, path, o, n, lv))
        owner = {fp: res.key for res in best.values() for fp in res.owns}
        todo = []
        for i, h in enumerate(holes):
            searched.add((h['key'], h['c'], h['j']))
            # data that is part of another, finished photo cannot be this photo's continuation
            cs = [x for x in cands.get(i, []) if owner.get(fingerprint(STORE.read(x[1], x[2], x[2] + 4096)),
                                                           h['key']) == h['key']]
            keep = pick_continuation(h, sorted(cs), lambda f, o, n: STORE.read(f, o, o + n))
            new = [(f, o - h['c']) for e, f, o, n in keep if (f, o - h['c']) not in extras[h['key']]]
            if new:
                extras[h['key']] += new
                if by_key[h['key']] not in todo:
                    todo.append(by_key[h['key']])
        log(f'  continuation found for {len(todo)} photo(s)')

    report = _export(photos, best, out_dir, opt, log)
    if not opt.keep_work:
        shutil.rmtree(work, ignore_errors=True)
    log(f'Done in {time.time() - t0:.0f}s -> {out_dir}')
    return report


def _summary(best, log):
    n = {}
    for r in best.values():
        n[r.category] = n.get(r.category, 0) + 1
    log('  ' + ', '.join(f'{k}: {n[k]}' for k in FOLDERS + ('failed',) if n.get(k)))


def _export(photos, best, out_dir, opt, log):
    rows = []
    heads = []
    hdir = os.path.join(out_dir, '.work', 'heads')
    use_exiftool = opt.exiftool == 'yes' or (opt.exiftool == 'auto' and export.exiftool_available())
    for p in photos:
        r = best.get(p.key) or Result(p.key, p.name, note='not processed')
        dest = ''
        if r.file and os.path.exists(r.file):
            if r.category == 'partial':
                fname = f'{p.name} ({max(1, int(r.coverage * 100))}%).jpg'
            elif r.category == 'preview-only':
                fname = f'{p.name} (preview only).jpg'
            else:
                fname = f'{p.name}.jpg'
            folder = os.path.join(out_dir, r.category)
            os.makedirs(folder, exist_ok=True)
            dest = os.path.join(folder, fname)
            st = os.stat(r.file)
            shutil.move(r.file, dest)
            os.utime(dest, (st.st_atime, st.st_mtime))
            if use_exiftool:
                c = p.copies[0]
                n = min(c.layout.preview[0], 1 << 20)
                os.makedirs(hdir, exist_ok=True)
                head = os.path.join(hdir, f'{len(heads):05d}.cr2')
                with open(head, 'wb') as f:
                    f.write(STORE.read(c.path, c.offset, c.offset + n))
                heads.append((dest, head))
        rows.append(dict(name=p.name, category=r.category, coverage=round(r.coverage, 4), reference=r.reference,
                         capture_time=p.capture_time or '', model=p.layout.tags.get('Model', ''),
                         copies=len(p.copies), primary=r.primary, sources=' | '.join(r.sources),
                         output=os.path.relpath(dest, out_dir) if dest else '', note=r.note))
    if heads:
        log('Copying full metadata with exiftool ...')
        export.copy_all_metadata(heads, log)
    with open(os.path.join(out_dir, 'report.csv'), 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ['name'])
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(out_dir, 'report.json'), 'w', encoding='utf-8') as f:
        json.dump(rows, f, ensure_ascii=False, indent=1)
    n = {}
    for row in rows:
        n[row['category']] = n.get(row['category'], 0) + 1
    log('Result: ' + ', '.join(f'{n[k]} {k}' for k in FOLDERS + ('failed',) if n.get(k)))
    return rows


# ----------------------------------------------------------------------------------------------- quick scan
def _quick(photo):
    """Fast health check: how much of the best copy decodes correctly before the first bad cluster."""
    best = 0.0
    thumb = _thumb(photo)
    for c in photo.copies:
        po, pl = c.layout.preview
        try:
            jp = Jpeg(STORE.read(c.path, c.offset + po, c.offset + po + min(pl, 65536)) + b'\0' * 16)
        except ValueError:
            continue
        if thumb is None:
            b = STORE.read(c.path, c.offset + po, c.offset + po + pl)
            return photo.key, (1.0 if len(b) == pl and markers_ok(b) else None)
        ref = thumb_reference(thumb, jp.W / jp.H, width=jp.W // 8)
        try:
            a = Assembler([(c.label, STORE.getter(c.path, c.offset))], po, pl, ref, _S['cluster'],
                          (5, 3)).run(first_only=True)
        except ValueError:
            continue
        best = max(best, a.coverage)
    return photo.key, best


def quick_scan(paths, cluster=0, jobs=0, all_files=False, log=print):
    files = collect(paths, all_files=all_files)
    photos, phases, cluster, problems = scan(files, cluster=cluster or None, log=log)
    ctx = get_context('spawn')
    out = {}
    with ctx.Pool(jobs or os.cpu_count() or 1, initializer=_init, initargs=(dict(cluster=cluster),)) as pool:
        for key, cov in pool.imap_unordered(_quick, photos):
            out[key] = cov
    return files, photos, cluster, problems, out
