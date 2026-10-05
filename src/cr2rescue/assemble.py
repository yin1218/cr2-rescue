"""Rebuild a damaged JPEG preview cluster by cluster.

Memory cards store files in fixed-size clusters (e.g. 32 KB). After a crash or a bad recovery, some
clusters of a photo hold the wrong data: another photo's bytes, zeros, or nothing at all. The JPEG decoder
cannot tell, so a normal viewer shows grey or garbage from the first bad cluster on.

The assembler walks the preview one cluster at a time and checks every decoded cluster against a low-res
reference of the same picture (DC values = the 1/8-scale image). At the first wrong cluster it tries, in
order:

1. **exact continuation** -- the same cluster position in another copy of the photo, or any cluster the
   global search found elsewhere on the card. The decoder state (bit position + DC predictors) is kept, so
   a match is lossless;
2. **resync** -- skip the hole, find the bit offset where valid data starts again, locate which MCU it
   belongs to by correlation with the reference, and fix the DC offset. The gap is filled from the
   reference later.

At the end every run that did not come straight from the primary copy is judged over its whole length
(`check_segments`), which catches look-alike data that passed cluster by cluster.
"""
from __future__ import annotations

import re
import warnings

import numpy as np
from PIL import Image, ImageFilter

from .jpeg import Jpeg, _decode, unstuff, ycc, rgb_of
from .reference import mcu_reference

SKIP = 4  # MCUs dropped after a resync point while the decoder settles
MAX_MAD, MIN_CORR = 10.0, 0.8  # a run from another source must match the reference this well (check_segments)
_MARKER = re.compile(rb'\xff[^\x00\xff]')


def plausible(chunk):
    """Can these bytes be baseline JPEG entropy-coded data? Such data never contains a marker (FF followed
    by anything but 00; restart markers are not used) and is never mostly zeros. Random data, RAW sensor
    data, headers and wiped clusters fail this test at once; only another JPEG's entropy data passes."""
    if not chunk:
        return True
    if chunk.count(0) > len(chunk) // 4:
        return False
    return _MARKER.search(chunk) is None


def sliding_ncc(seq, ref):
    """Normalised cross-correlation of seq (len L) against every window of ref."""
    from scipy.signal import fftconvolve
    L = len(seq)
    s = (seq - seq.mean()) / (seq.std() + 1e-6)
    num = fftconvolve(ref, s[::-1], mode='valid') / L
    c1 = np.concatenate([[0], np.cumsum(ref)])
    c2 = np.concatenate([[0], np.cumsum(ref * ref)])
    mu = (c1[L:] - c1[:-L]) / L
    var = (c2[L:] - c2[:-L]) / L - mu * mu
    out = num / np.sqrt(np.maximum(var, 4.0))  # flat windows (std < 2) cannot be matched
    out[var < 4.0] = 0
    return np.clip(out, -1, 1)


class Assembler:
    def __init__(self, sources, po, pl, ref_rgb, cluster=32768, blur=None, log=None, own_header=True):
        """sources: list of (name, get) where get(x0, x1) returns photo-relative bytes [x0, x1) (may be short).
        sources[0] is the primary copy. po/pl: preview offset/length relative to the CR2 header.
        blur: (rows, cols) box for the error when the reference is soft (the camera thumbnail).
        own_header: False when the JPEG header was copied from another photo, so the data right after it is
        not vouched for by it."""
        self.sources, self.po, self.pl, self.cluster = sources, po, pl, cluster
        self.own_header = own_header
        self.buf = bytearray(bytes(sources[0][1](po, po + pl)).ljust(pl, b'\0'))
        self.orig = bytes(self.buf)
        self.reparse()
        jp = self.jp
        self.ent0 = po + jp.sos_end  # photo-relative offset of the first entropy-coded byte
        self.ref = mcu_reference(ref_rgb, jp)
        self.coefs = np.zeros((jp.nmcu, jp.nblk, 64), np.int16)
        self.dc = np.zeros((jp.nmcu, jp.nblk), np.int64)
        self.have = np.zeros(jp.nmcu, bool)
        self.how = np.full(jp.nmcu, -1, np.int16)  # source index of every MCU (-1 = missing)
        self.log = []
        self.holes = []  # decoder state at every hole no source could fill (input of the global search)
        self.segs = []  # (first MCU, count, source) of every placed run of MCUs
        self.thr = 18.0
        self.blur = blur
        self.say = log or (lambda *a: None)

    # ---------- photo-relative byte offset  <->  bit in the current unstuffed entropy stream
    def reparse(self):
        self.jp = Jpeg(bytes(self.buf))
        e = np.frombuffer(bytes(self.buf[self.jp.sos_end:]), np.uint8)
        st = np.zeros(len(e), bool)
        st[1:] = (e[:-1] == 0xFF) & (e[1:] == 0x00)
        self._stuffed_at = np.flatnonzero(~st)

    def x_of_bit(self, bit):
        sa = self._stuffed_at
        return self.ent0 + int(sa[min(bit // 8, len(sa) - 1)])

    def bit_of_x(self, x):
        return self.jp.bit_of(x - self.ent0)

    def ent_end(self):
        return self.po + self.pl

    @property
    def coverage(self):
        return float(self.have.mean())

    @property
    def lossless(self):
        """True when every MCU came from an unbroken chain of exact data (no resync gap)."""
        return bool(self.have.all()) and not any(l[0] in ('resync', 'drop') for l in self.log)

    # ---------- scoring
    def err(self, m0, mm):
        r = self.ref[:, m0:m0 + mm.shape[1]]
        n = r.shape[1]
        d = mm[:, :n] - r
        if self.blur and n:
            d = self._blur_span(m0, d)
        return np.abs(d[0]) + 0.5 * (np.abs(d[1]) + np.abs(d[2]))

    def _blur_span(self, m0, d):
        """Box-average the signed difference over neighbouring MCUs of the same span (2-D, raster layout).
        Detail the soft reference lacks averages out; wrong data leaves a coarse error."""
        from scipy.ndimage import uniform_filter
        W = self.jp.mcux
        n = d.shape[1]
        r0 = m0 // W
        rows = (m0 + n - 1) // W - r0 + 1
        idx = np.arange(m0, m0 + n) - r0 * W
        grid = np.zeros((3, rows * W))
        mask = np.zeros(rows * W)
        grid[:, idx] = d
        mask[idx] = 1
        grid = grid.reshape(3, rows, W)
        mask = mask.reshape(rows, W)
        m = uniform_filter(mask, self.blur, mode='constant')
        out = np.empty_like(d)
        for c in range(3):
            g = uniform_filter(grid[c], self.blur, mode='constant')
            out[c] = (g / np.maximum(m, 1e-6)).reshape(-1)[idx]
        return out

    def cluster_ok(self, e):
        """Verdict for the MCUs that start inside one cluster."""
        if len(e) == 0:
            return True
        if len(e) < 8:
            return bool(np.median(e) < self.thr)
        sm = np.convolve(e, np.ones(32) / 32, mode='valid') if len(e) >= 32 else e
        return bool(np.median(e) < self.thr and (e > 2.5 * self.thr).mean() < 0.2 and sm.max() < 4 * self.thr)

    def first_bad_cluster(self, st, n, e, x_of_bit, x_end_ok):
        """Walk the decoded MCUs cluster by cluster. Returns (keep, c): MCUs to keep and the start of the
        first bad cluster (None if everything decoded is good and the decode reached x_end_ok)."""
        C = self.cluster
        if n == 0:
            return 0, self.x_of_bit(int(st[0])) // C * C
        xs = x_of_bit(st[:n + 1])
        cl = xs[:n] // C
        i = 0
        while i < n:
            k = i + int(np.searchsorted(cl[i:], cl[i], side='right'))
            c = int(cl[i]) * C
            if not self.cluster_ok(e[i:k]) or not self.bytes_ok(c):
                return self._keep_before(st, n, c), c
            i = k
        if xs[n] >= x_end_ok:
            return n, None
        c = int(xs[n]) // C * C  # decode error inside this cluster
        inside = cl == c // C
        if inside.sum() >= 64 and self.cluster_ok(e[inside]):
            return n, c + C  # the failing MCU straddles into the next (damaged) cluster
        return self._keep_before(st, n, c), c

    def _keep_before(self, st, n, c):
        """Number of decoded MCUs that end before photo offset c. The MCU that straddles c is dropped: its
        tail lies in the bad cluster, and restarting from its first bit lets a replacement cluster be tried."""
        return int(np.searchsorted(st[1:n + 1], self.bit_of_x(c), side='right'))

    def bytes_ok(self, c):
        """Byte-level sanity check of the entropy-coded part of the cluster starting at photo offset c."""
        a = max(c, self.ent0) - self.po
        b = min(c + self.cluster, self.ent_end() - 2) - self.po
        return a >= b or plausible(bytes(self.buf[a:b]))

    # ---------- main loop
    def run(self, max_steps=200, first_only=False, check=True):
        """first_only: stop after the first decode (a quick health check of one copy).
        check: finish with check_segments (leave it out to look at what was found before judging it)."""
        jp = self.jp
        j, b, pred = 0, 0, (0, 0, 0)
        src = 0
        first = True
        tried = {}
        for _ in range(max_steps):
            if j >= jp.nmcu:
                break
            co, st, n, errf = self.jp.decode(b, jp.nmcu - j)
            dcabs = jp.dc_chain(co[:n], pred)
            e = self.err(j, jp.mcu_means(dcabs))
            if first and n > 600:
                self.thr = float(np.clip(3 * np.median(e[:600]), 18.0, 30.0))
                first = False
            sa = self._stuffed_at
            x_of_bit = lambda bits: self.ent0 + sa[np.minimum(np.asarray(bits) // 8, len(sa) - 1)]
            done_x = self.ent_end() if n < jp.nmcu - j else 0
            keep, c = self.first_bad_cluster(st, n, e, x_of_bit, done_x)
            self._place(j, co, dcabs, keep, src)
            self.log.append(('ok', self.sources[src][0], j, keep, c))
            if keep > 0:
                pred = jp.pred_of(dcabs[keep - 1], jp.ny)
            j += keep
            b = int(st[keep])
            if first_only or c is None or c >= self.ent_end() or j >= jp.nmcu:
                break
            visits = tried.get((c, j), 0)
            tried[(c, j)] = visits + 1
            if visits >= 2:
                break
            # 1) exact continuation from another source (first visit only: a second visit means the data
            #    switched in failed right away, so go straight to resync)
            hit = self._continue(j, b, pred, c) if visits == 0 else None
            if hit is not None:
                src = hit
                continue
            if visits == 0:
                self._record_hole(j, b, pred, c)
            # 2) resync after the hole
            k = min(600, keep)
            rate = (self.x_of_bit(int(st[keep])) - self.x_of_bit(int(st[keep - k]))) / k if k >= 50 else None
            self._at = (self.x_of_bit(b), rate)
            first_pass = sorted({0, src})
            r = self._resync(j, c, first_pass) or self._resync(
                j, c, [k for k in range(len(self.sources)) if k not in first_pass], max_e=8)
            if r is None:
                self.log.append(('stop', j, c))
                break
            src, j, b, pred = r
        if check and not first_only:
            self.check_segments()
        return self

    # ---------- final check
    def segment_fit(self, m0, n):
        """(correlation, mean abs difference) of the picture level of MCUs [m0, m0+n) against the reference,
        both box-averaged to the reference's own resolution."""
        from scipy.ndimage import uniform_filter
        W = self.jp.mcux
        r0 = m0 // W
        rows = (m0 + n - 1) // W - r0 + 1
        idx = np.arange(m0, m0 + n) - r0 * W
        mask = np.zeros(rows * W)
        mask[idx] = 1
        mask = mask.reshape(rows, W)
        den = uniform_filter(mask, self.blur or (3, 3), mode='constant')
        out = []
        for v in (self.ref[0, m0:m0 + n], self.jp.mcu_means(self.dc[m0:m0 + n])[0]):
            g = np.zeros(rows * W)
            g[idx] = v
            g = uniform_filter(g.reshape(rows, W), self.blur or (3, 3), mode='constant')
            out.append((g / np.maximum(den, 1e-6)).reshape(-1)[idx])
        r, d = out
        if r.std() < 3:  # a flat reference says nothing about structure
            return 1.0, float(np.abs(r - d).mean())
        c = float(np.corrcoef(r, d)[0, 1]) if d.std() > 0 else 0.0
        return c, float(np.abs(r - d).mean())

    def check_segments(self):
        """Drop runs that came from another source but do not show this picture.

        Each cluster was checked on its own, and in smooth areas (a wall, a dark background) another photo's
        data can pass that test: the error stays small although the content is wrong. Judged over its whole
        length, a foreign run gives itself away by a low correlation with the reference or a shifted level.
        Dropped MCUs are filled from the reference like any other gap."""
        for m0, n, src in self.segs:
            if src == 0 and m0 == 0 and self.own_header:
                continue  # the start of the primary copy: identified by its own header
            c, mad = self.segment_fit(m0, n)
            bad = mad > MAX_MAD or (n >= 1000 and c < MIN_CORR)
            if bad:
                self.have[m0:m0 + n] = False
                self.how[m0:m0 + n] = -1
                self.log.append(('drop', self.sources[src][0], m0, n, round(c, 3), round(mad, 1)))

    def _place(self, m0, co, dcabs, n, src):
        n = min(n, self.jp.nmcu - m0)
        if n <= 0:
            return
        self.coefs[m0:m0 + n] = co[:n]
        self.dc[m0:m0 + n] = dcabs[:n]
        self.have[m0:m0 + n] = True
        self.how[m0:m0 + n] = src
        self.segs.append((m0, n, src))

    def _span_ok(self, j, b, pred, x_from, x_to, piece_bytes):
        """Decode from MCU j (bit b) through photo bytes [x_from, x_to) = piece_bytes and judge them."""
        jp = self.jp
        xb = self.x_of_bit(b)
        lead = bytes(self.buf[xb - self.po: x_from - self.po])
        piece = lead + bytes(piece_bytes)
        bit_in = b - self.bit_of_x(xb)
        co, st, n, errf = jp.decode_piece(piece, bit_in, min(6000, jp.nmcu - j))
        _, sa = unstuff(piece)
        lim = len(piece) - 64  # the last MCU may run past the piece
        ends = sa[np.minimum(st[1:n + 1] // 8, len(sa) - 1)]
        n_in = int(np.searchsorted(ends, min(lim, x_to - xb), side='left'))
        if errf == 1 and sa[min(int(st[n]) // 8, len(sa) - 1)] < lim:
            return False, 0  # invalid code inside the candidate: not this stream
        short = x_to - x_from < self.cluster  # last, partial cluster of the preview
        if n_in < (8 if short else 48):
            return False, 0
        dcabs = jp.dc_chain(co[:n_in], pred)
        e = self.err(j, jp.mcu_means(dcabs))
        if len(e) < n_in:
            return False, 0
        return self.cluster_ok(e), n_in

    def _record_hole(self, j, b, pred, c):
        jp = self.jp
        xb = self.x_of_bit(b)
        lead = bytes(self.buf[xb - self.po:c - self.po])
        lead_u = unstuff(lead)[0] if lead else np.zeros(0, np.uint8)
        # picture level relative to the reference just before the hole: a true continuation keeps it (the DC
        # predictor chain is unbroken), a foreign stream lands at a random offset
        k = np.flatnonzero(self.have[:j])[-3000:]
        bias = (np.median(jp.mcu_means(self.dc[k]) - self.ref[:, k], axis=1).tolist() if len(k) >= 64 else None)
        ref = np.ascontiguousarray(self.ref[:, j:j + 8000], np.float64)
        self.holes.append(dict(j=j, c=c, pred=pred, bias=bias, lead_u=lead_u, bit_in=b - self.bit_of_x(xb), ref=ref,
                               qy=jp.qy, qc=jp.qc, thr=self.thr, ny=jp.ny, tables=jp.tables()))

    def _continue(self, j, b, pred, c):
        C = self.cluster
        for k, (name, get) in enumerate(self.sources):
            chunk = bytes(get(c, min(c + C, self.ent_end())))
            if len(chunk) < min(C, self.ent_end() - c) // 2:
                continue
            if chunk == bytes(self.buf[c - self.po:c - self.po + len(chunk)]):
                continue  # the same bytes that just failed
            if not plausible(chunk[:max(0, self.ent_end() - 2 - c)]):
                continue
            ok, n_in = self._span_ok(j, b, pred, c, c + len(chunk), chunk)
            if ok:
                tail = bytes(get(c, self.ent_end()))
                self.buf[c - self.po:] = tail.ljust(self.ent_end() - c, b'\0')
                self.reparse()
                self.log.append(('switch', name, c, n_in))
                return k
        return None

    def _resync(self, j, c, which, kmax=2048, probe=300, long=3000, max_e=None):
        C = self.cluster
        stop = self.ent_end() if max_e is None else min(self.ent_end(), c + max_e * C)
        for e in range(c + C, stop, C):
            for k in which:
                name, get = self.sources[k]
                tail = bytes(get(e, self.ent_end()))
                if len(tail) < 4096:
                    continue
                r = self._resync_at(j, tail, kmax, probe, long, e)
                if r is not None:
                    m0, bit_rel, pred, peak, margin = r
                    self.buf[c - self.po:e - self.po] = b'\0' * (e - c)  # hole
                    self.buf[e - self.po:] = tail.ljust(self.ent_end() - e, b'\0')
                    self.reparse()
                    b = self.bit_of_x(e) + bit_rel
                    self.log.append(('resync', name, e, m0, round(peak, 3), round(margin, 3)))
                    return k, m0, b, pred
        return None

    def _resync_at(self, j, tail, kmax, probe, long, e=None):
        jp = self.jp
        buf, _ = unstuff(tail[:400000])
        T = (jp.maxcode, jp.mincode, jp.valptr, jp.huffval, jp.blk_dc, jp.blk_ac)
        groups = {}
        for k in range(min(kmax, len(buf) * 8)):  # bit offsets that decode cleanly, grouped by where they sync
            co, st, n, err = _decode(buf, len(buf) * 8, k, probe, *T)
            if n < probe:
                continue
            key = int(st[60])
            if key not in groups:
                groups[key] = k
        cands = []
        for key, k in groups.items():
            co, st, n, err = _decode(buf, len(buf) * 8, k, probe, *T)
            mm = jp.mcu_means(jp.dc_chain(co))
            cands.append((np.abs(np.diff(mm)).mean(), k))
        cands.sort()
        best = None
        for rough, k in cands[:4]:  # the true stream is the smoothest
            co, st, n, err = _decode(buf, len(buf) * 8, k, long, *T)
            if n - SKIP < 200:
                continue
            co, st = co[SKIP:n], st[SKIP:n + 1]
            dc = jp.dc_chain(co)
            mm = jp.mcu_means(dc)
            L = min(len(co), 1500)
            if jp.nmcu - L + 1 <= j:
                continue
            score = np.zeros(jp.nmcu - L + 1)
            tot = 0.0
            for ci, w in ((0, 1.0), (1, 0.5), (2, 0.5)):
                if ci:  # a nearly flat colour channel only adds noise to the match
                    w *= min(1.0, float(mm[ci, :L].std()) / 4.0)
                score += w * sliding_ncc(mm[ci, :L], self.ref[ci])
                tot += w
            score /= tot
            score[:j] = -1
            m0 = int(np.argmax(score))
            s2 = score.copy()
            s2[max(0, m0 - 2 * jp.mcux):m0 + 2 * jp.mcux] = -1
            peak, margin = float(score[m0]), float(score[m0] - s2.max())
            if best is None or peak > best[0]:
                best = (peak, margin, m0, co, st, dc, mm, L)
        if best is None:
            return None
        peak, margin, m0, co, st, dc, mm, L = best
        if peak < 0.8 or margin < 0.08 and not self._where_expected(j, m0, e, co, st):
            return None
        r = self.ref[:, m0:m0 + L]
        off = np.round([np.median(r[0] - mm[0, :L]) * 8 / jp.qy, np.median(r[1] - mm[1, :L]) * 8 / jp.qc,
                        np.median(r[2] - mm[2, :L]) * 8 / jp.qc]).astype(np.int64)
        # verify the placed segment really matches (catches wrong-row placements in smooth pictures)
        dcabs = dc + np.array([off[0]] * jp.ny + [off[1], off[2]])
        e = self.err(m0, jp.mcu_means(dcabs[:L]))
        if np.median(e) > self.thr:
            return None
        return m0, int(st[0]), (int(off[0]), int(off[1]), int(off[2])), peak, margin

    def _where_expected(self, j, m0, e, co, st):
        """Does the byte count put the resynced data near MCU m0? In smooth areas (sky, a wall) the rows just
        above and below match the reference almost as well, so the score alone cannot pick the row; the
        number of bytes missing before e, at the bytes per MCU around the hole, can."""
        if e is None or getattr(self, '_at', None) is None:
            return False
        x_j, rate = self._at
        after = (int(st[-1]) - int(st[0])) / 8 / len(co)
        rate = after if rate is None else (rate + after) / 2
        pred = j + (e - x_j + int(st[0]) / 8) / rate
        return abs(m0 - pred) <= max(1.5 * self.jp.mcux, 0.05 * (pred - j))

    def whole(self):
        """Does the primary copy decode in one piece to the very end: every MCU, the last one finishing right
        before the EOI marker? (a gap or foreign data inside the stream shifts where the decode ends)"""
        co, st, n, errf = self.jp.decode(0, self.jp.nmcu)
        if n < self.jp.nmcu:
            return False
        eoi = bytes(self.buf).rfind(b'\xff\xd9', max(0, len(self.buf) - 64))
        return eoi >= 0 and 0 <= self.po + eoi - self.x_of_bit(int(st[n])) <= 1

    def run_straight(self):
        """Without a usable reference: keep what decodes straight on from the header of the primary copy, up to
        the cluster where decoding breaks (and any implausible clusters right before it)."""
        jp = self.jp
        co, st, n, errf = jp.decode(0, jp.nmcu)
        keep = n
        if n < jp.nmcu:
            sa = self._stuffed_at
            c = int(self.ent0 + sa[min(int(st[n]) // 8, len(sa) - 1)]) // self.cluster * self.cluster
            while c - self.cluster > self.ent0 and not self.bytes_ok(c - self.cluster):
                c -= self.cluster
            keep = self._keep_before(st, n, c)
        self._place(0, co, jp.dc_chain(co[:keep], (0, 0, 0)), keep, 0)
        self.log.append(('straight', self.sources[0][0], 0, keep, None))
        return self

    def picture(self, min_share=0.02):
        """What has been decoded so far as a low-res float RGB image: one pixel per square of whole MCUs (two
        MCU rows when an MCU is twice as wide as tall), NaN where nothing was decoded. Returns (image, (fx, fy))
        with the share of the frame's width and height it covers, or None when too little was decoded."""
        jp = self.jp
        if self.have.mean() < min_share:
            return None
        k = max(1, round(jp.mcu_w / jp.mcu_h))
        cols, rows = jp.W // jp.mcu_w, jp.H // (jp.mcu_h * k)
        if not cols or not rows:
            return None
        c = np.where(self.have, jp.mcu_means(self.dc), np.nan).reshape(3, jp.mcuy, jp.mcux)
        c = c[:, :rows * k, :cols].reshape(3, rows, k, cols).mean(2)
        Y, Cb, Cr = c[0], c[1] - 128, c[2] - 128
        rgb = np.stack([Y + 1.402 * Cr, Y - 0.344136 * Cb - 0.714136 * Cr, Y + 1.772 * Cb], -1)
        return np.clip(rgb, 0, 255), (cols * jp.mcu_w / jp.W, rows * k * jp.mcu_h / jp.H)

    # ---------- output
    def jpeg_bytes(self):
        """The rebuilt original JPEG stream when the repair is lossless, else None."""
        if not self.lossless:
            return None
        jp = Jpeg(bytes(self.buf))
        co, st, n, err = jp.decode(0, jp.nmcu)
        if n != jp.nmcu or not np.array_equal(co[:, :, 1:], self.coefs[:, :, 1:]) or \
                not np.array_equal(jp.dc_chain(co), self.dc):
            return None
        out = bytes(self.buf)
        if out.endswith(b'\xff\xd9'):
            return out
        after = np.flatnonzero(jp.unstuffed_before >= (int(st[n]) + 7) // 8)
        end = jp.sos_end + (int(after[0]) if len(after) else len(jp.unstuffed_before))
        return out[:end] + b'\xff\xd9'

    def image(self, fill_rgb, match=True, soften=0.6):
        """Decoded MCUs over the up-scaled low-res fill. The fill is softened a little and its Y/Cb/Cr are
        matched to the decoded picture with a robust linear fit over the MCUs both have."""
        jp = self.jp
        img = jp.render(self.coefs, self.dc)
        mask = self.have.reshape(jp.mcuy, jp.mcux).repeat(jp.mcu_h, 0).repeat(jp.mcu_w, 1)[:jp.H, :jp.W]
        if mask.all() or fill_rgb is None:
            return Image.fromarray(img)
        f = fill_rgb.filter(ImageFilter.GaussianBlur(soften)) if soften else fill_rgb
        if match and self.have.sum() > 2000:
            f = self._match(f)
        fill = np.asarray(f.resize((jp.W, jp.H), Image.LANCZOS))
        return Image.fromarray(np.where(mask[..., None], img, fill))

    def _match(self, f):
        h = self.have
        R = mcu_reference(f, self.jp)[:, h]
        M = self.jp.mcu_means(self.dc[h])
        c = ycc(f)
        for k in range(3):
            x, y = R[k], M[k]
            keep = np.ones(len(x), bool)
            for _ in range(3):
                with warnings.catch_warnings():  # a flat reference makes the fit ill-conditioned; the clip below copes
                    warnings.simplefilter('ignore')
                    a, b = np.polyfit(x[keep], y[keep], 1)
                r = y - (a * x + b)
                keep = np.abs(r) < 3 * 1.4826 * np.median(np.abs(r[keep])) + 1
            a = float(np.clip(a, 0.7, 1.4))
            b = float(np.median(y[keep] - a * x[keep]))
            c[k] = a * c[k] + b
        return Image.fromarray(rgb_of(c))
