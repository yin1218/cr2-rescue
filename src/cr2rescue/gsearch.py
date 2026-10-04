"""Global cluster search.

When no copy of a photo has the right data after a hole, the missing cluster may still be *somewhere* on
the card: inside another recovered file, in the slack after another photo, or in a carved fragment. For every
open hole we take the saved decoder state (bit position + DC predictors) and decode every cluster of every
input file from it. The real continuation is the one that

* decodes cleanly for a long stretch,
* matches the low-res reference of the missing part,
* keeps the picture level of the data just before the hole (the DC chain is unbroken), and
* is unique (after merging byte-identical copies of the same cluster).
"""
from __future__ import annotations

import mmap
import os

import numpy as np
from numba import njit


@njit(cache=True)
def _mcu(buf, nbits, pos, maxcode, mincode, valptr, huffval, blk_dc, blk_ac, dcd):
    """Decode one MCU (DC differences into dcd). Returns the new bit position, -1 on invalid data or -2
    when out of bits."""
    for b in range(len(blk_dc)):
        t = blk_dc[b]
        code = 0
        s = -1
        for ln in range(1, 17):
            if pos >= nbits:
                return -2
            code = (code << 1) | ((buf[pos >> 3] >> (7 - (pos & 7))) & 1)
            pos += 1
            if maxcode[t, ln] >= code:
                s = huffval[t, valptr[t, ln] + code - mincode[t, ln]]
                break
        if s < 0 or s > 11:
            return -1
        v = 0
        for _ in range(s):
            if pos >= nbits:
                return -2
            v = (v << 1) | ((buf[pos >> 3] >> (7 - (pos & 7))) & 1)
            pos += 1
        if s > 0 and v < (1 << (s - 1)):
            v -= (1 << s) - 1
        dcd[b] = v
        t = blk_ac[b]
        k = 1
        while k < 64:
            code = 0
            rs = -1
            for ln in range(1, 17):
                if pos >= nbits:
                    return -2
                code = (code << 1) | ((buf[pos >> 3] >> (7 - (pos & 7))) & 1)
                pos += 1
                if maxcode[t, ln] >= code:
                    rs = huffval[t, valptr[t, ln] + code - mincode[t, ln]]
                    break
            if rs < 0:
                return -1
            r = rs >> 4
            s = rs & 15
            if s == 0:
                if r == 15:
                    k += 16
                    continue
                break
            k += r
            if k > 63:
                return -1
            pos += s
            k += 1
    return pos


@njit(cache=True)
def _scan(U, cstarts, take, lead, bit_in, nmax, maxcode, mincode, valptr, huffval, blk_dc, blk_ac, blk_comp,
          ny, p0, p1, p2, ref, qy, qc, thr, min_mcus, out, oute, outk, outb):
    nl = len(lead)
    nb = len(blk_dc)
    tmp = np.empty(nl + take, np.uint8)
    tmp[:nl] = lead
    dcd = np.zeros(nb, np.int64)
    sd = np.zeros((3, nmax), np.float64)
    nref = ref.shape[1]
    cnt = 0
    for ci in range(len(cstarts)):
        a = cstarts[ci]
        m = min(take, len(U) - a)
        if m < take // 2:
            continue
        tmp[nl:nl + m] = U[a:a + m]
        nbits = (nl + m) * 8
        pos = bit_in
        p = np.array([p0, p1, p2], np.int64)
        k = 0
        es = 0.0
        ng = 0
        gy = 0.0
        gb = 0.0
        gr = 0.0
        ok = True
        while k < nmax and k < nref:
            r = _mcu(tmp, nbits, pos, maxcode, mincode, valptr, huffval, blk_dc, blk_ac, dcd)
            if r == -2:
                break
            if r == -1:
                ok = False
                break
            pos = r
            ysum = 0.0
            for b in range(nb):
                c = blk_comp[b]
                p[c] += dcd[b]
                if c == 0:
                    ysum += p[0]
            # signed differences, averaged over groups of 4 MCUs: detail a soft reference lacks cancels out
            sd[0, k] = 128 + ysum / ny * qy / 8 - ref[0, k]
            sd[1, k] = 128 + p[1] * qc / 8 - ref[1, k]
            sd[2, k] = 128 + p[2] * qc / 8 - ref[2, k]
            gy += sd[0, k]
            gb += sd[1, k]
            gr += sd[2, k]
            k += 1
            if k % 4 == 0:
                es += (abs(gy) + 0.5 * (abs(gb) + abs(gr))) / 4
                ng += 1
                gy = 0.0
                gb = 0.0
                gr = 0.0
                if (ng >= 3 and es / ng > 2.5 * thr) or (ng >= 24 and es / ng > 1.5 * thr):
                    ok = False
                    break
        if ok and k >= min_mcus and ng > 0 and es / ng < thr:
            out[cnt] = ci
            oute[cnt] = es / ng
            outk[cnt] = k
            for c in range(3):
                outb[cnt, c] = np.median(sd[c, :k])
            cnt += 1
            if cnt >= len(out):
                break
    return cnt


def open_map(path):
    """Read-only memory map (works for multi-GB disk images); empty files give b''."""
    if os.path.getsize(path) == 0:
        return b''
    with open(path, 'rb') as f:
        return mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)


def scan_file(path, phase, cluster, holes, skip=(), chunk=256 << 20, nmax=8000, min_mcus=96):
    """Try every cluster of one file as the continuation of every hole.
    skip: [(a, b)] byte ranges to ignore (data that already belongs to an intact photo).
    Returns {hole_index: [(file_offset, err, n_mcus, (level Y, Cb, Cr))]}."""
    d = open_map(path)
    size = len(d)
    found = {}
    out = np.zeros(256, np.int64)
    oute = np.zeros(256, np.float64)
    outk = np.zeros(256, np.int64)
    outb = np.zeros((256, 3), np.float64)
    chunk = max(cluster, chunk // cluster * cluster)
    for a0 in range(phase, size, chunk):
        a1 = min(size, a0 + chunk + cluster)  # one cluster of overlap for the last candidate
        e = np.frombuffer(d[a0:a1], np.uint8)
        keep = np.ones(len(e), bool)
        keep[1:] = ~((e[:-1] == 0xFF) & (e[1:] == 0x00))
        U = e[keep]
        uidx = np.cumsum(keep) - 1
        starts = np.arange(a0, min(a0 + chunk, size), cluster)
        if skip:
            sk = np.array(sorted(skip), np.int64)
            i = np.searchsorted(sk[:, 0], starts, side='right') - 1
            starts = starts[~((i >= 0) & (starts < sk[np.maximum(i, 0), 1]))]
        if len(starts) == 0:
            continue
        cst = uidx[starts - a0].astype(np.int64)
        for hi, h in enumerate(holes):
            T = h['tables']
            cnt = _scan(U, cst, cluster, h['lead_u'], h['bit_in'], nmax, T['maxcode'], T['mincode'], T['valptr'],
                        T['huffval'], T['blk_dc'], T['blk_ac'], T['blk_comp'], h['ny'], h['pred'][0], h['pred'][1],
                        h['pred'][2], h['ref'], float(h['qy']), float(h['qc']), float(h['thr']), min_mcus, out,
                        oute, outk, outb)
            for q in range(cnt):
                found.setdefault(hi, []).append((int(starts[out[q]]), float(oute[q]), int(outk[q]),
                                                 tuple(map(float, outb[q]))))
    if isinstance(d, mmap.mmap):
        d.close()
    return found


def pick_continuation(hole, cands, read, max_level=1.1, ratio=1.5):
    """Choose the one true continuation among candidates [(err, path, offset, n, level)], or [] if unsure.

    read(path, offset, n) -> bytes, used to merge byte-identical copies of the same cluster."""
    ok = []
    for e, f, o, n, lv in cands:
        if e >= 0.6 * hole['thr']:
            continue
        if hole['bias'] is not None and max(abs(a - b) for a, b in zip(lv, hole['bias'])) > max_level:
            continue
        ok.append((e, f, o, n))
    groups = {}
    for e, f, o, n in ok:
        groups.setdefault(bytes(read(f, o, 4096)), []).append((e, f, o, n))
    best = sorted((min(g), g) for g in groups.values())
    if not best:
        return []
    if len(best) > 1 and best[1][0][0] < ratio * best[0][0][0]:
        return []  # ambiguous
    return best[0][1]
