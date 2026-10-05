"""Give the filled-in part of partial photos its detail back (`cr2-rescue sharpen`).

Where the data of a partial photo is gone, `recover` fills in the picture from the camera's thumbnail (about 1/32
of the detail) or its small image (1/8): right colours and shapes, but blurry. Two ways to do better:

* neighbour -- a photo taken seconds before or after usually shows the same scene with all its detail. It is
  aligned (SIFT + homography, optical flow for what moved), its colours fitted, and its fine detail added to the
  fill wherever its coarse picture agrees with the fill. The detail is real, but only where the neighbour saw
  the same thing (not on people or branches that moved).
* upscale -- an AI super-resolution model (Real-ESRGAN family, run with realesrgan-ncnn-vulkan) redraws the fill
  from its own low-res content. Works everywhere, but the detail is invented.

By default both are used: the neighbour where it is trusted, the upscaler for the rest. Decoded data is never
changed. Needs OpenCV: pip install 'cr2-rescue[sharpen]'."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import time
from datetime import datetime
from multiprocessing import get_context

import numpy as np
from PIL import Image

NEIGHBOUR_SECONDS = 180   # look for a neighbour at most this far apart in time
NEIGHBOUR_TRIES = 4       # ... among the nearest this many photos
MIN_INLIERS = 30          # SIFT matches agreeing on one homography (same scene, seconds apart: 50-400)
FILL_SCALE = {'thumb': 32}  # how much coarser than the photo the fill is; the small image: 8
UPSCALER = 'realesrgan-ncnn-vulkan'
MODEL = 'realesrgan-x4plus'
SEAM = 2                  # px inside the hole over which the new detail fades in

_cv2 = None


def _need_cv2():
    global _cv2
    if _cv2 is None:
        try:
            import cv2
        except ImportError:
            raise ImportError("cr2-rescue sharpen needs OpenCV: pip install 'cr2-rescue[sharpen]'") from None
        _cv2 = cv2
    return _cv2


# ----------------------------------------------------------------------------------------------- shared
def _blur(x, sigma):
    """Gaussian blur (OpenCV's: an order of magnitude faster than scipy's on a whole photo)."""
    cv2 = _need_cv2()
    return cv2.GaussianBlur(np.ascontiguousarray(x, np.float32), (0, 0), sigma, borderType=cv2.BORDER_REFLECT)


def _edt(m):
    """Distance from each True pixel to the nearest False one."""
    cv2 = _need_cv2()
    return cv2.distanceTransform(m.astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)


def low(x, f, sigma=None):
    """What survives in a fill that is f times coarser than the photo: the picture at 1/f, back up, softened."""
    cv2 = _need_cv2()
    h, w = x.shape[:2]
    sigma = f * 3 / 8 if sigma is None else sigma
    small = cv2.resize(x, (max(1, w // f), max(1, h // f)), interpolation=cv2.INTER_AREA)
    k = max(1, int(sigma // 2))  # the blur is done k times smaller (the result is smooth at that scale)
    mid = cv2.resize(small, (max(1, w // k), max(1, h // k)), interpolation=cv2.INTER_CUBIC)
    return cv2.resize(_blur(mid, sigma / k), (w, h), interpolation=cv2.INTER_LINEAR)


def seam(hole):
    """0 on the decoded data, rising to 1 within SEAM px inside the hole."""
    return np.clip(_edt(hole) / SEAM, 0, 1).astype(np.float32)


def compose(T, hole, A=None, trust=None, C=None):
    """The fill T with the neighbour's version A where trusted and the upscaled C everywhere else in the hole."""
    e = seam(hole)
    k = e * trust if A is not None else np.zeros_like(e)
    out = T + k[..., None] * (A - T) if A is not None else T.copy()
    if C is not None:
        out += ((1 - k) * e)[..., None] * (C - T)
    return np.clip(out, 0, 255)


# ----------------------------------------------------------------------------------------------- neighbour
def _fit(kind, src, dst):
    """3x3 transform of the given kind taking src points to dst points (least squares, inliers only)."""
    cv2 = _need_cv2()
    if kind == 'homography':
        M, _ = cv2.findHomography(src, dst, 0)
        return M
    est = cv2.estimateAffinePartial2D if kind == 'similarity' else cv2.estimateAffine2D
    M, _ = est(src, dst, method=cv2.LMEDS)
    return None if M is None else np.vstack([M, [0, 0, 1]])


def _project(M, p):
    q = np.c_[p, np.ones(len(p))] @ M.T
    return q[:, :2] / q[:, 2:]


def align(T, N, real, scale=0.5):
    """Transform taking N onto T, from SIFT matches on the decoded part of T. All matches lie on one side of the
    hole, so the model has to extrapolate into it: the matches nearest the hole are held out and the model that
    predicts them best (similarity, affine or homography; simpler wins ties) is used.
    Returns (3x3 matrix, inliers) or (None, inliers)."""
    cv2 = _need_cv2()
    g = lambda a: cv2.cvtColor(cv2.resize(a, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
                               .clip(0, 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    dist = _edt(real)
    m = cv2.resize((dist > 48).astype(np.uint8) * 255, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    sift = cv2.SIFT_create(nfeatures=20000)
    kt, dt = sift.detectAndCompute(g(T), m)
    kn, dn = sift.detectAndCompute(g(N), None)
    if dt is None or dn is None or len(kt) < 8 or len(kn) < 8:
        return None, 0
    good = [p[0] for p in cv2.BFMatcher().knnMatch(dt, dn, k=2) if len(p) == 2 and p[0].distance < 0.75 * p[1].distance]
    if len(good) < 8:
        return None, len(good)
    pt = np.float32([kt[a.queryIdx].pt for a in good]) / scale
    pn = np.float32([kn[a.trainIdx].pt for a in good]) / scale
    H, inl = cv2.findHomography(pn, pt, cv2.USAC_MAGSAC, 4.0)
    if H is None:
        return None, 0
    inl = inl.ravel().astype(bool)
    n = int(inl.sum())
    if n < MIN_INLIERS:
        return None, n
    pt, pn = pt[inl], pn[inl]
    h, w = real.shape
    d = dist[np.clip(pt[:, 1].astype(int), 0, h - 1), np.clip(pt[:, 0].astype(int), 0, w - 1)]
    near = d <= np.percentile(d, 30)
    best, err = None, np.inf
    for kind in ('similarity', 'affine', 'homography'):
        M = _fit(kind, pn[~near], pt[~near])
        if M is None:
            continue
        e = float(np.median(np.linalg.norm(_project(M, pn[near]) - pt[near], axis=1)))
        if e < 0.9 * err:
            best, err = kind, e
    M = _fit(best, pn, pt) if best else None
    if M is None or not 0.5 < abs(np.linalg.det(M[:2, :2])) < 2:
        return None, n
    return M, n


def flow_warp(LT, LN, f=8):
    """Optical flow moving LN onto LT, used only where the two disagree (things that moved): the fill's own
    geometry is coarse, so where they already agree the homography is the better guide. Returns (move, strain):
    strain = how much the flow bends the picture; content that changed shape (a hand, a person) gets forced into
    the fill's coarse shape and melts, so such places are not trusted (rigid scenery < 0.25, melted 0.6-1.7)."""
    cv2 = _need_cv2()
    h, w = LT.shape[:2]
    sm = lambda a: cv2.resize(a, (w // f, h // f), interpolation=cv2.INTER_AREA)
    g = lambda a: cv2.cvtColor(sm(a).clip(0, 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    fl = dis.calc(g(LT), g(LN), None)  # LT(p) ~ LN(p + fl(p))
    d = np.abs(sm(LT) - sm(LN)).mean(2)
    fl = fl * _blur(np.clip((d - 6) / 8, 0, 1), 4)[..., None]
    gu, gv = np.gradient(_blur(fl[..., 0], 1)), np.gradient(_blur(fl[..., 1], 1))
    strain = np.sqrt(gu[0] ** 2 + gu[1] ** 2 + gv[0] ** 2 + gv[1] ** 2)
    strain = cv2.resize(_blur(strain, 2), (w, h), interpolation=cv2.INTER_LINEAR)
    fl = cv2.resize(fl, (w, h), interpolation=cv2.INTER_CUBIC) * f
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    mx, my = (gx + fl[..., 0]).astype(np.float32), (gy + fl[..., 1]).astype(np.float32)
    return (lambda a: cv2.remap(a, mx, my, cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT)), strain


def colour_fit(src, dst, where):
    """Per-channel gain + offset taking src to dst over `where`."""
    out = np.empty_like(src)
    for c in range(3):
        a, b = src[..., c][where], dst[..., c][where]
        k, o = np.linalg.lstsq(np.c_[a, np.ones_like(a)], b, rcond=None)[0]
        out[..., c] = src[..., c] * k + o
    return out


def from_neighbour(T, real, N, f=32, n_real=None, H=None):
    """The fill T redrawn with the detail of neighbour N. Returns (A, trust, inliers) or None if N does not align.
    n_real: N's own decoded-data mask if N is partial too (its fill is not used)."""
    cv2 = _need_cv2()
    T, N = np.asarray(T, np.float32), np.asarray(N, np.float32)
    h, w = real.shape
    inl = 0
    if H is None:
        H, inl = align(T, N, real)
        if H is None:
            return None
    Nw = cv2.warpPerspective(N, H, (w, h), flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT)
    ok = np.ones((h, w), np.float32) if n_real is None else n_real.astype(np.float32)
    cover = cv2.warpPerspective(ok, H, (w, h), flags=cv2.INTER_LINEAR)
    near = real & (_edt(real) < 400) & (cover > 0.99)
    if near.sum() < 1000:
        return None
    Nw = colour_fit(Nw, T, near)
    LT = low(T, f)
    move, strain = flow_warp(LT, low(Nw, f))
    Nw, cover = move(Nw), move(cover) > 0.99
    LN = low(Nw, f)
    A = Nw + (LT - LN)  # the fill's coarse picture, the neighbour's fine detail
    d = np.abs(LT - LN).mean(2)
    trust = np.clip((14 - d) / (14 - 5), 0, 1) * np.clip((0.8 - strain) / (0.8 - 0.35), 0, 1) * cover
    return A, _blur(trust.astype(np.float32), 6), inl


# ----------------------------------------------------------------------------------------------- upscaler
def find_upscaler(path=None):
    """Path of the realesrgan-ncnn-vulkan executable (given, or on PATH), or None."""
    if path:
        return path if os.path.isfile(path) else shutil.which(path)
    return shutil.which(UPSCALER)


def run_upscaler(src, dst, upscaler, model):
    """x4 super-resolution of image file src into dst."""
    cmd = [upscaler, '-i', src, '-o', dst, '-n', model, '-s', '4']
    models = os.path.join(os.path.dirname(os.path.abspath(upscaler)), 'models')
    if os.path.isdir(models):
        cmd += ['-m', models]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode or not os.path.exists(dst):
        raise RuntimeError(f'{os.path.basename(upscaler)} failed: {(r.stderr or r.stdout).strip()[-300:]}')


def upscale(T, hole, f=32, upscaler=UPSCALER, model=MODEL):
    """The fill T redrawn by the upscaler: down to a size that still holds all its detail (1/16 for a 1/32
    fill: two x4 passes; 1/4 for a 1/8 fill: one pass) and back up. Only the box around the hole goes through
    the model; its colours stay those of the fill."""
    T = np.asarray(T, np.float32)
    passes = 2 if f >= 16 else 1
    s = 4 ** passes
    h, w = hole.shape
    ys, xs = np.nonzero(hole)
    pad = 16 * s  # context around the hole for the model
    y0, x0 = max(ys.min() - pad, 0) // s * s, max(xs.min() - pad, 0) // s * s
    y1, x1 = -(-min(ys.max() + 1 + pad, h) // s) * s, -(-min(xs.max() + 1 + pad, w) // s) * s
    box = np.pad(T, ((0, max(0, y1 - h)), (0, max(0, x1 - w)), (0, 0)), mode='edge')[y0:y1, x0:x1]
    small = box.reshape((y1 - y0) // s, s, (x1 - x0) // s, s, 3).mean((1, 3))
    with tempfile.TemporaryDirectory() as tmp:
        cur = os.path.join(tmp, '0.png')
        Image.fromarray(small.round().clip(0, 255).astype(np.uint8)).save(cur)
        for i in range(passes):
            nxt = os.path.join(tmp, f'{i + 1}.png')
            run_upscaler(cur, nxt, upscaler, model)
            cur = nxt
        big = np.asarray(Image.open(cur).convert('RGB'), np.float32)
    if big.shape[:2] != box.shape[:2]:
        raise RuntimeError(f'model {model} is not a x4 model')
    C = T.copy()
    C[y0:y1, x0:x1] = big[:min(y1, h) - y0, :min(x1, w) - x0]
    return C + (low(T, f) - low(C, f))


# ----------------------------------------------------------------------------------------------- driver
def _when(row):
    try:
        t = row['capture_time']
        return datetime.strptime(t[:19], '%Y:%m:%d %H:%M:%S').timestamp() + float('0.' + t[20:] if t[20:] else 0)
    except (KeyError, ValueError):
        return None


def neighbours(row, rows):
    """Photos of the same camera taken closest in time, nearest first: complete ones, then partial ones."""
    t = _when(row)
    if t is None:
        return []
    cands = []
    for r in rows:
        if r is row or not r.get('output') or r.get('model') != row.get('model'):
            continue
        if r['category'] not in ('intact', 'repaired') and not (r['category'] == 'partial' and r.get('mask')):
            continue
        u = _when(r)
        if u is not None and abs(u - t) <= NEIGHBOUR_SECONDS:
            cands.append((r['category'] == 'partial', abs(u - t), r['name'], r))
    return [c[3] for c in sorted(cands)[:NEIGHBOUR_TRIES]]


def _rgb(path):
    return np.asarray(Image.open(path).convert('RGB'), np.float32)


def _mask(path):
    return np.asarray(Image.open(path).convert('L')) > 127


_J = {}


def _init(settings):
    _J.update(settings)


def sharpen_one(job):
    """job = (row, candidate neighbour rows). Returns a report row."""
    row, cands = job
    src, base = _J['src'], _J['out']
    out = dict(name=row['name'], output='', method='', neighbour='', neighbour_share=0.0, note='')
    try:
        T = _rgb(os.path.join(src, row['output']))
        real = _mask(os.path.join(src, row['mask']))
        hole = ~real
        if not hole.any():
            out['note'] = 'nothing filled in'
            return out
        f = FILL_SCALE.get(row.get('reference'), 8)
        A = trust = C = None
        if _J['neighbour']:
            best = None
            for r in cands:
                N = _rgb(os.path.join(src, r['output']))
                if N.shape != T.shape:
                    continue
                H, n = align(T, N, real)
                if H is not None and (best is None or n > best[0]):
                    best = (n, r, N, H)
            if best:
                n, r, N, H = best
                nr = _mask(os.path.join(src, r['mask'])) if r['category'] == 'partial' else None
                got = from_neighbour(T, real, N, f, nr, H)
                if got:
                    A, trust, _ = got
                    out['neighbour'] = r['name']
                    out['neighbour_share'] = round(float((trust * seam(hole)).sum() / hole.sum()), 3)
            if A is None:
                out['note'] = 'no neighbouring photo of the same scene'
        if _J['upscaler']:
            C = upscale(T, hole, f, _J['upscaler'], _J['model'])
        if A is None and C is None:
            return out
        out['method'] = '+'.join(m for m, x in (('neighbour', A), ('upscale', C)) if x is not None)
        pic = Image.fromarray(compose(T, hole, A, trust, C).astype(np.uint8))
        dest = os.path.join(base, os.path.basename(row['output']))
        with Image.open(os.path.join(src, row['output'])) as im:
            extra = {k: im.info[k] for k in ('exif', 'icc_profile') if im.info.get(k)}
        pic.save(dest, quality=_J['quality'], subsampling=1, **extra)
        st = os.stat(os.path.join(src, row['output']))
        os.utime(dest, (st.st_atime, st.st_mtime))
        out['output'] = os.path.relpath(dest, base)
    except Exception as e:  # one bad photo must not stop the rest
        out['note'] = f'error: {type(e).__name__}: {e}'[:300]
    return out


def sharpen(rescued, out_dir=None, method='both', upscaler=None, model=MODEL, jobs=2, quality=95, names=None,
            log=print):
    """Sharpen every partial photo of a `cr2-rescue recover` output folder. Writes <out_dir>/<same file name>
    (default <rescued>/sharpened) and sharpen.json. Returns the report rows."""
    _need_cv2()
    t0 = time.time()
    with open(os.path.join(rescued, 'report.json'), encoding='utf-8') as f:
        rows = json.load(f)
    todo = [r for r in rows if r['category'] == 'partial' and r.get('output') and r.get('mask')
            and (names is None or r['name'] in names)]
    if any(r['category'] == 'partial' and not r.get('mask') for r in rows):
        log('Some partial photos have no mask (recovered with cr2-rescue < 0.2): run recover again to sharpen them.')
    up = None
    if method in ('both', 'upscale'):
        up = find_upscaler(upscaler)
        if up is None:
            msg = (f'{UPSCALER} not found (get it from https://github.com/xinntao/Real-ESRGAN/releases and put it on '
                   'PATH, or pass --upscaler PATH)')
            if method == 'upscale':
                raise SystemExit(msg)
            log(msg + '; using the neighbouring photos only.')
    out_dir = out_dir or os.path.join(rescued, 'sharpened')
    os.makedirs(out_dir, exist_ok=True)
    report = []
    try:  # a run that was stopped goes on where it was
        with open(os.path.join(out_dir, 'sharpen.json'), encoding='utf-8') as f:
            report = [r for r in json.load(f) if r['output'] and os.path.exists(os.path.join(out_dir, r['output']))]
    except (OSError, ValueError, KeyError, TypeError):
        pass
    if report:
        done_before = {r['name'] for r in report}
        todo = [r for r in todo if r['name'] not in done_before]
        log(f'{len(done_before)} photo(s) already done in {out_dir}, going on with the rest '
            '(another -o to do them again)')
    log(f'Sharpening {len(todo)} partial photo(s): ' + ' + '.join(
        m for m, on in (('neighbouring photos', method in ('both', 'neighbour')), (f'{model} upscaling', up)) if on))
    settings = dict(src=rescued, out=out_dir, neighbour=method in ('both', 'neighbour'), upscaler=up, model=model,
                    quality=quality)
    jobs_in = [(r, neighbours(r, rows)) for r in todo]
    if jobs <= 1:
        _init(settings)
        results = map(sharpen_one, jobs_in)
        pool = None
    else:
        pool = get_context('spawn').Pool(jobs, initializer=_init, initargs=(settings,))
        results = pool.imap_unordered(sharpen_one, jobs_in)
    def write():
        with open(os.path.join(out_dir, 'sharpen.json'), 'w', encoding='utf-8') as f:
            json.dump(sorted(report, key=lambda r: r['name']), f, ensure_ascii=False, indent=1)

    try:
        for k, res in enumerate(results):
            report.append(res)
            write()  # as it goes, so a long run that is stopped still says what is done
            what = res['method'] or 'skipped'
            if res['neighbour']:
                what += f' ({int(res["neighbour_share"] * 100)}% from {res["neighbour"]})'
            log(f'  {k + 1}/{len(jobs_in)} {res["name"]}: {what}{"; " + res["note"] if res["note"] else ""}')
    finally:
        if pool:
            pool.close()
            pool.join()
    report.sort(key=lambda r: r['name'])
    write()
    done = sum(1 for r in report if r['output'])
    log(f'Done in {time.time() - t0:.0f}s: {done} sharpened -> {out_dir}')
    return report
