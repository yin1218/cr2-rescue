import io

import numpy as np
import pytest
from PIL import Image

from cr2rescue import synth
from cr2rescue.reference import (align_small, bands_ok, match_colours, mend_rows, open_thumb, preview_reduced,
                                 render_small, small_array, small_rows_ok, small_style, thumb_reference)
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


def _close(img, im):
    want = np.asarray(im.resize(img.size, Image.BOX), np.float32)
    return np.abs(np.asarray(img, np.float32) - want).mean()


def test_overwritten_rows_are_found_and_mended(shots):
    s = shots[0]
    x0, y0, cw, ch = BORDER_BOX
    a = s['a'].copy()
    noise = np.random.default_rng(1).integers(0, 65535, a.shape).astype(a.dtype)
    a[y0 + 40:y0 + 52] = noise[y0 + 40:y0 + 52]
    a[y0 + 90, x0 + 100:] = noise[y0 + 90, x0 + 100:]  # foreign data starting in the middle of a row
    assert small_rows_ok(s['a'], BORDER_BOX).all()
    ok = small_rows_ok(a, BORDER_BOX)
    assert np.flatnonzero(~ok).tolist() == list(range(40, 52)) + [90]
    mended = render_small(mend_rows(a, BORDER_BOX, ok), BORDER_BOX, s['tref'])
    assert _close(mended, s['im']) < 7
    assert all(bands_ok(mended, s['tref']))


def test_colour_style_carries_over_to_another_photo(shots):
    """Without a thumbnail, the style fitted on a neighbouring photo draws the small image in the camera's
    colours; a plain grey-world balance does not."""
    style = small_style(shots[0]['a'], BORDER_BOX, shots[0]['tref'])
    s = shots[1]
    styled = render_small(s['a'], BORDER_BOX, style=style)
    plain = render_small(s['a'], BORDER_BOX)
    assert _close(styled, s['im']) < 3 < 10 < _close(plain, s['im'])


def test_style_fits_a_partial_target(shots):
    """The target may leave pixels out (NaN), e.g. the part of the preview that decoded."""
    s = shots[0]
    target = np.asarray(s['im'].resize((96, 64), Image.BOX), np.float32)
    target[40:] = np.nan
    style = small_style(s['a'], BORDER_BOX, target)
    assert _close(render_small(s['a'], BORDER_BOX, style=style), s['im']) < 6


def test_match_colours_takes_the_thumbnail_colours(shots):
    s = shots[0]
    off = Image.fromarray(np.clip(np.asarray(render_small(s['a'], BORDER_BOX, s['tref']), np.float32)
                                  * [1.0, 0.85, 1.1] + [0, 10, -15], 0, 255).astype(np.uint8))
    fixed = match_colours(off, s['tref'])
    assert fixed.size == off.size
    assert _close(fixed, s['im']) < 0.5 * _close(off, s['im'])
    t = np.asarray(s['tref'], np.float32).copy()
    t[:, :60] = np.nan  # unknown parts are left to their neighbours
    assert np.isfinite(np.asarray(match_colours(off, t), np.float32)).all()
