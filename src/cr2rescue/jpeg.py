"""Baseline JPEG entropy decoder that can start at ANY bit position.

A normal decoder gives up at the first damaged byte. To splice a broken stream back together we need to
decode from an arbitrary bit, keep the decoder state (bit position + DC predictors) across a hole, and look
at per-MCU DC values (the 1/8-scale picture) to judge whether data is in the right place.

Supports 3-component baseline JPEG with Y sampled 1x1, 2x1, 1x2 or 2x2 (4:4:4, 4:2:2, 4:4:0, 4:2:0) and
chroma 1x1, no restart markers -- which covers the full-size previews Canon embeds in CR2 files.
"""
from __future__ import annotations

import struct

import numpy as np
from numba import njit

ZIGZAG = np.array([0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5, 12, 19, 26, 33, 40, 48, 41, 34, 27,
                   20, 13, 6, 7, 14, 21, 28, 35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51, 58,
                   59, 52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63])


class Unsupported(ValueError):
    pass


class Jpeg:
    """Header tables + the unstuffed entropy-coded data of one JPEG."""

    def __init__(self, j: bytes):
        j = bytes(j)
        if j[:2] != b'\xff\xd8':
            raise Unsupported('no SOI marker')
        self.q, huff, i = {}, {}, 2
        self.restart = 0
        sof = None
        while True:
            if i + 4 > len(j) or j[i] != 0xFF:
                raise Unsupported('broken JPEG header')
            m = j[i + 1]
            L = struct.unpack('>H', j[i + 2:i + 4])[0]
            seg = j[i + 4:i + 2 + L]
            if m == 0xDB:
                k = 0
                while k < len(seg):
                    if seg[k] >> 4:
                        raise Unsupported('16-bit quantisation tables')
                    self.q[seg[k] & 15] = np.frombuffer(seg[k + 1:k + 65], np.uint8).astype(np.float32)
                    k += 65
            elif m in (0xC0, 0xC1):
                sof = seg
            elif m in (0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF):
                raise Unsupported('only baseline Huffman JPEG is supported')
            elif m == 0xC4:
                k = 0
                while k < len(seg):
                    tc, th = seg[k] >> 4, seg[k] & 15
                    bits = list(seg[k + 1:k + 17])
                    n = sum(bits)
                    huff[(tc, th)] = (bits, list(seg[k + 17:k + 17 + n]))
                    k += 17 + n
            elif m == 0xDD:
                self.restart = struct.unpack('>H', seg[:2])[0]
            elif m == 0xDA:
                self.scan = [(seg[1 + 2 * k], seg[2 + 2 * k] >> 4, seg[2 + 2 * k] & 15) for k in range(seg[0])]
                self.sos_end = i + 2 + L
                break
            i += 2 + L
        if sof is None:
            raise Unsupported('no baseline frame header')
        self.H, self.W = struct.unpack('>HH', sof[1:5])
        self.comps = [(sof[6 + 3 * k], sof[7 + 3 * k] >> 4, sof[7 + 3 * k] & 15, sof[8 + 3 * k])
                      for k in range(sof[5])]
        if self.restart:
            raise Unsupported('restart markers are not supported')
        if len(self.comps) != 3 or len(self.scan) != 3:
            raise Unsupported('only 3-component, single-scan JPEG is supported')
        (_, hy, vy, _), (_, hb, vb, _), (_, hr, vr, _) = self.comps
        if (hb, vb, hr, vr) != (1, 1, 1, 1) or hy not in (1, 2) or vy not in (1, 2):
            raise Unsupported(f'unsupported sampling {[(c[1], c[2]) for c in self.comps]}')
        self.hy, self.vy = hy, vy
        self.ny = hy * vy  # luma blocks per MCU
        self.maxcode = np.full((4, 18), -1, np.int32)
        self.mincode = np.zeros((4, 18), np.int32)
        self.valptr = np.zeros((4, 18), np.int32)
        self.huffval = np.zeros((4, 256), np.uint8)
        for (tc, th), (bits, vals) in huff.items():
            if th > 1:
                raise Unsupported('Huffman table id > 1')
            t = tc * 2 + th
            code, p = 0, 0
            for ln in range(1, 17):
                if bits[ln - 1]:
                    self.valptr[t, ln] = p
                    self.mincode[t, ln] = code
                    code += bits[ln - 1]
                    p += bits[ln - 1]
                    self.maxcode[t, ln] = code - 1
                code <<= 1
            self.huffval[t, :len(vals)] = vals
        dc = {c[0]: c[1] for c in self.scan}
        ac = {c[0]: c[2] for c in self.scan}
        ids = [self.comps[0][0]] * self.ny + [self.comps[1][0], self.comps[2][0]]
        self.blk_comp = np.array([0] * self.ny + [1, 2], np.int32)
        self.blk_dc = np.array([dc[x] for x in ids], np.int32)
        self.blk_ac = np.array([2 + ac[x] for x in ids], np.int32)
        self.blk_q = [self.comps[0][3]] * self.ny + [self.comps[1][3], self.comps[2][3]]
        self.nblk = len(ids)
        self.qy = float(self.q[self.blk_q[0]][0])
        self.qc = float(self.q[self.blk_q[-1]][0])
        self.mcu_w, self.mcu_h = 8 * hy, 8 * vy
        self.mcux = (self.W + self.mcu_w - 1) // self.mcu_w
        self.mcuy = (self.H + self.mcu_h - 1) // self.mcu_h
        self.nmcu = self.mcux * self.mcuy
        ent = np.frombuffer(j[self.sos_end:], np.uint8)
        stuff = np.zeros(len(ent), bool)
        if len(ent) > 1:
            stuff[1:] = (ent[:-1] == 0xFF) & (ent[1:] == 0x00)
        self.buf = ent[~stuff].copy()
        self.unstuffed_before = np.cumsum(~stuff) - (~stuff)

    def tables(self):
        """Everything the numba kernels need (picklable)."""
        return dict(maxcode=self.maxcode, mincode=self.mincode, valptr=self.valptr, huffval=self.huffval,
                    blk_dc=self.blk_dc, blk_ac=self.blk_ac, blk_comp=self.blk_comp)

    def bit_of(self, stuffed_index):
        """Bit position in the unstuffed stream of a byte offset relative to the start of entropy data."""
        return int(self.unstuffed_before[min(stuffed_index, len(self.unstuffed_before) - 1)]) * 8

    def decode(self, start_bit, max_mcus):
        return _decode(self.buf, len(self.buf) * 8, start_bit, max_mcus, self.maxcode, self.mincode,
                       self.valptr, self.huffval, self.blk_dc, self.blk_ac)

    def decode_piece(self, piece, start_bit, n):
        """Decode n MCUs from an arbitrary stuffed byte string (no header), starting at start_bit."""
        buf, _ = unstuff(piece)
        return _decode(buf, len(buf) * 8, start_bit, n, self.maxcode, self.mincode, self.valptr, self.huffval,
                       self.blk_dc, self.blk_ac)

    # --- DC helpers
    def dc_chain(self, coefs, pred=(0, 0, 0)):
        """Absolute DC per block from the coded DC differences (all luma blocks share one predictor)."""
        d = coefs[:, :, 0].astype(np.int64)
        ny = self.ny
        y = (np.cumsum(d[:, :ny].reshape(-1)) + pred[0]).reshape(-1, ny)
        cb = np.cumsum(d[:, ny]) + pred[1]
        cr = np.cumsum(d[:, ny + 1]) + pred[2]
        return np.concatenate([y, cb[:, None], cr[:, None]], 1)

    def mcu_means(self, dc):
        """Per-MCU mean (Y, Cb, Cr) from absolute DC values, shape (3, n)."""
        ny = self.ny
        return np.stack([128 + dc[:, :ny].mean(1) * self.qy / 8, 128 + dc[:, ny] * self.qc / 8,
                         128 + dc[:, ny + 1] * self.qc / 8])

    @staticmethod
    def pred_of(dcabs_row, ny):
        return int(dcabs_row[ny - 1]), int(dcabs_row[ny]), int(dcabs_row[ny + 1])

    # --- pixels
    def render(self, coefs, dc_abs, rows_per_chunk=32):
        """RGB uint8 image from zigzag coefficients of all MCUs (raster order) and absolute DC values."""
        mx, my = self.mcux, self.mcuy
        out = np.zeros((my * self.mcu_h, mx * self.mcu_w, 3), np.uint8)
        q = np.stack([self.q[t] for t in self.blk_q])  # (nblk, 64) zigzag order
        for r0 in range(0, my, rows_per_chunk):
            r1 = min(my, r0 + rows_per_chunk)
            a, b = r0 * mx, r1 * mx
            c = coefs[a:b].astype(np.float32)
            c[:, :, 0] = dc_abs[a:b]
            c *= q[None]
            nat = np.zeros_like(c)
            nat[:, :, ZIGZAG] = c
            pix = np.einsum('ki,nbkl,lj->nbij', IDCT, nat.reshape(-1, self.nblk, 8, 8), IDCT) + 128
            rows = r1 - r0
            pix = pix.reshape(rows, mx, self.nblk, 8, 8)
            yb = pix[:, :, :self.ny].reshape(rows, mx, self.vy, self.hy, 8, 8)
            Y = yb.transpose(0, 2, 4, 1, 3, 5).reshape(rows * self.mcu_h, mx * self.mcu_w)
            ch = []
            for k in (self.ny, self.ny + 1):
                p = pix[:, :, k].transpose(0, 2, 1, 3).reshape(rows * 8, mx * 8)
                ch.append(p.repeat(self.vy, 0).repeat(self.hy, 1))
            Cb, Cr = ch[0] - 128, ch[1] - 128
            rgb = np.stack([Y + 1.402 * Cr, Y - 0.344136 * Cb - 0.714136 * Cr, Y + 1.772 * Cb], -1)
            out[r0 * self.mcu_h:r1 * self.mcu_h] = np.clip(rgb + 0.5, 0, 255).astype(np.uint8)
        return out[:self.H, :self.W]


def unstuff(b):
    """Remove JPEG byte stuffing (FF 00 -> FF). Returns (bytes array, stuffed index of every kept byte)."""
    e = np.frombuffer(bytes(b), np.uint8)
    stuff = np.zeros(len(e), bool)
    if len(e) > 1:
        stuff[1:] = (e[:-1] == 0xFF) & (e[1:] == 0x00)
    return e[~stuff].copy(), np.flatnonzero(~stuff)


@njit(cache=True)
def _decode(buf, nbits, pos, max_mcus, maxcode, mincode, valptr, huffval, blk_dc, blk_ac):
    """Decode up to max_mcus MCUs from bit pos. Returns (coefs, MCU start bits, n decoded, err)
    with err 0 = done, 1 = invalid code, 2 = ran out of data."""
    nb = len(blk_dc)
    coefs = np.zeros((max_mcus, nb, 64), np.int16)
    starts = np.zeros(max_mcus + 1, np.int64)
    n = 0
    while n < max_mcus:
        starts[n] = pos
        for b in range(nb):
            t = blk_dc[b]
            code = 0
            s = -1
            for ln in range(1, 17):
                if pos >= nbits:
                    return coefs, starts, n, 2
                code = (code << 1) | ((buf[pos >> 3] >> (7 - (pos & 7))) & 1)
                pos += 1
                if maxcode[t, ln] >= code:
                    s = huffval[t, valptr[t, ln] + code - mincode[t, ln]]
                    break
            if s < 0 or s > 11:
                return coefs, starts, n, 1
            v = 0
            for _ in range(s):
                if pos >= nbits:
                    return coefs, starts, n, 2
                v = (v << 1) | ((buf[pos >> 3] >> (7 - (pos & 7))) & 1)
                pos += 1
            if s > 0 and v < (1 << (s - 1)):
                v -= (1 << s) - 1
            coefs[n, b, 0] = v
            t = blk_ac[b]
            k = 1
            while k < 64:
                code = 0
                rs = -1
                for ln in range(1, 17):
                    if pos >= nbits:
                        return coefs, starts, n, 2
                    code = (code << 1) | ((buf[pos >> 3] >> (7 - (pos & 7))) & 1)
                    pos += 1
                    if maxcode[t, ln] >= code:
                        rs = huffval[t, valptr[t, ln] + code - mincode[t, ln]]
                        break
                if rs < 0:
                    return coefs, starts, n, 1
                r = rs >> 4
                s = rs & 15
                if s == 0:
                    if r == 15:
                        k += 16
                        continue
                    break
                k += r
                if k > 63:
                    return coefs, starts, n, 1
                v = 0
                for _ in range(s):
                    if pos >= nbits:
                        return coefs, starts, n, 2
                    v = (v << 1) | ((buf[pos >> 3] >> (7 - (pos & 7))) & 1)
                    pos += 1
                if v < (1 << (s - 1)):
                    v -= (1 << s) - 1
                coefs[n, b, k] = v
                k += 1
        n += 1
    starts[n] = pos
    return coefs, starts, n, 0


def _idct_matrix():
    c = np.zeros((8, 8))
    for k in range(8):
        for n in range(8):
            c[k, n] = (np.sqrt(1 / 8) if k == 0 else np.sqrt(2 / 8)) * np.cos(np.pi * (2 * n + 1) * k / 16)
    return c.astype(np.float32)


IDCT = _idct_matrix()


def ycc(rgb):
    """RGB (PIL image or array) -> stacked (Y, Cb, Cr) float arrays, JFIF full range."""
    a = np.asarray(rgb, np.float32)
    R, G, B = a[..., 0], a[..., 1], a[..., 2]
    return np.stack([0.299 * R + 0.587 * G + 0.114 * B,
                     128 - 0.168736 * R - 0.331264 * G + 0.5 * B,
                     128 + 0.5 * R - 0.418688 * G - 0.081312 * B])


def rgb_of(c):
    Y, Cb, Cr = c[0], c[1] - 128, c[2] - 128
    rgb = np.stack([Y + 1.402 * Cr, Y - 0.344136 * Cb - 0.714136 * Cr, Y + 1.772 * Cb], -1)
    return np.clip(rgb + 0.5, 0, 255).astype(np.uint8)


def markers_ok(j):
    """Entropy-coded data may only contain FF00 or RSTn; anything else means foreign data was spliced in."""
    import re
    j = bytes(j)
    try:
        jp_end = Jpeg(j).sos_end
    except ValueError:
        return False
    body = j[jp_end:]
    if not body.endswith(b'\xff\xd9'):
        return False
    return re.search(rb'\xff[^\x00\xd0-\xd7\xff]', body[:-2]) is None
