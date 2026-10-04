import io

import numpy as np
import pytest
from PIL import Image

from cr2rescue import synth
from cr2rescue.reference import (align_small, bands_ok, open_thumb, preview_reduced, render_small, small_array,
                                 thumb_reference)
from cr2rescue.tiff import parse_cr2

BORDER_BOX = (5, 3, 192, 128)  # synth._small border (top 3, left 5) around a 1536/8 x 1024/8 picture


@pytest.fixture(scope='module')
def shots():
    out = []
    for seed in range(3):
        im = synth.scene(1536, 1024, seed=10 + seed)
        data, prev = synth.make_cr2(im, seed=seed)
        lay = parse_cr2(data)
        s = lay.small
        a = small_array(data[s['offset']:s['offset'] + s['length']], s)
        to, tl = lay.thumb
        thumb = open_thumb(data[to:to + tl])
        out.append(dict(im=im, a=a, thumb=thumb, tref=thumb_reference(thumb, 1.5), prev=prev))
    return out


def test_thumbnail_reference_removes_letterbox(shots):
    t = shots[0]
    assert t['thumb'].size == (160, 120)
    ref = t['tref']
    assert ref.size == (160, 107)
    rows = np.asarray(ref.convert('L'), np.float32).mean(1)
    assert rows[0] > 30 and rows[-1] > 30  # no black bar rows left
    want = np.asarray(t['im'].resize(ref.size, Image.BOX).convert('L'), np.float32)
    assert np.corrcoef(want.ravel(), np.asarray(ref.convert('L'), np.float32).ravel())[0, 1] > 0.97


def test_align_with_preview_is_exact(shots):
    items = [(s['a'], s['tref'], preview_reduced(s['prev'])) for s in shots]
    box, score = align_small(items, 1.5)
    assert box == BORDER_BOX and score > 0.97


def test_align_with_thumbnails_only_is_close(shots):
    # a 160-pixel thumbnail cannot pin a 192-pixel box to one pixel; the preview target above does that
    box, score = align_small([(s['a'], s['tref'], None) for s in shots], 1.5)
    assert all(abs(p - q) <= 2 for p, q in zip(box, BORDER_BOX)) and score > 0.95


def test_rendered_small_image_matches_picture(shots):
    s = shots[0]
    img = render_small(s['a'], BORDER_BOX, s['tref'])
    assert img.size == (192, 128)
    want = np.asarray(s['im'].resize(img.size, Image.BOX), np.float32)
    assert np.abs(np.asarray(img, np.float32) - want).mean() < 6
    assert all(bands_ok(img, s['tref']))


def test_bands_ok_flags_overwritten_rows(shots):
    s = shots[0]
    a = s['a'].copy()
    a[60:80] = np.random.default_rng(0).integers(0, 65535, a[60:80].shape)
    bands = bands_ok(render_small(a, BORDER_BOX, s['tref']), s['tref'])
    assert not all(bands) and sum(bands) >= 10


def test_preview_reduced(shots):
    im = preview_reduced(shots[0]['prev'])
    assert im.size == (192, 128)
    assert preview_reduced(b'not a jpeg') is None
