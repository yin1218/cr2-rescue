import io

import numpy as np
import pytest
from PIL import Image

from cr2rescue import synth
from cr2rescue.assemble import Assembler
from cr2rescue.jpeg import Jpeg


def _jpeg(im):
    b = io.BytesIO()
    im.save(b, 'JPEG', quality=90, subsampling=1)
    return b.getvalue()


@pytest.fixture(scope='module')
def picture():
    pic = synth.scene(768, 512, seed=5)
    return pic, _jpeg(pic)


@pytest.fixture
def assembled(picture):
    pic, j = picture
    ref = pic.resize((pic.width // 8, pic.height // 8), Image.BOX)
    a = Assembler([('a', lambda x0, x1: j[x0:x1]), ('b', lambda x0, x1: b'')], 0, len(j), ref, cluster=4096).run()
    return pic, a


def _dc_of(im):
    jp = Jpeg(_jpeg(im))
    co, st, n, err = jp.decode(0, jp.nmcu)
    return jp.dc_chain(co)


def _splice(a, dc, m0, n):
    a.dc[m0:m0 + n] = dc[m0:m0 + n]
    a.segs.append((m0, n, 1))
    a.check_segments()
    return [l for l in a.log if l[0] == 'drop']


def test_clean_run_is_kept(assembled):
    pic, a = assembled
    assert a.have.all() and not any(l[0] == 'drop' for l in a.log)


def test_own_data_from_another_source_is_kept(assembled):
    pic, a = assembled
    m0, n = a.jp.mcux * 10, a.jp.mcux * 12
    assert _splice(a, a.dc.copy(), m0, n) == []
    assert a.have.all()


@pytest.mark.parametrize('foreign', ['shifted', 'other'])
def test_foreign_run_is_dropped(assembled, foreign):
    """Another photo's data (a near-identical burst shot, or a different scene) that slipped past the
    per-cluster check must not end up in the picture."""
    pic, a = assembled
    if foreign == 'shifted':
        other = Image.fromarray(np.roll(np.asarray(pic), 160, axis=1))
    else:
        other = synth.scene(768, 512, seed=9)
    m0, n = a.jp.mcux * 8, a.jp.mcux * 24
    drops = _splice(a, _dc_of(other), m0, n)
    assert len(drops) == 1 and drops[0][2:4] == (m0, n)
    assert not a.have[m0:m0 + n].any() and a.have[:m0].all() and a.have[m0 + n:].all()
    assert not a.lossless


def _grey(pic):
    return Image.new('RGB', (pic.width // 8, pic.height // 8), (128, 128, 128))


def test_picture_shows_what_decoded(picture):
    pic, j = picture
    half = len(j) // 2
    ref = pic.resize((pic.width // 8, pic.height // 8), Image.BOX)
    a = Assembler([('a', lambda x0, x1: j[x0:min(x1, half)])], 0, len(j), ref, cluster=4096).run()
    img, (fx, fy) = a.picture()
    assert img.shape == (32, 48, 3) and (fx, fy) == (1.0, 1.0)  # 16x8 MCUs, two MCU rows per pixel
    known = np.isfinite(img).all(2)
    assert 0.3 < known.mean() < 0.6 and known[0].all() and not known[-1].any()
    want = np.asarray(pic.resize((48, 32), Image.BOX), np.float32)
    assert np.abs(img[known] - want[known]).mean() < 4
    head = Jpeg(j).sos_end
    empty = Assembler([('a', lambda x0, x1: j[x0:min(x1, head)])], 0, len(j), ref, cluster=4096)
    assert empty.picture() is None


def test_run_straight_keeps_an_intact_stream(picture):
    pic, j = picture
    a = Assembler([('a', lambda x0, x1: j[x0:x1])], 0, len(j), _grey(pic), cluster=4096).run_straight()
    assert a.have.all() and a.jpeg_bytes() == j


def test_run_straight_stops_where_the_data_breaks(picture):
    """Without a reference nothing can be checked: keep only what decodes straight on from the header."""
    pic, j = picture
    jp = Jpeg(j)
    co, st, n, err = jp.decode(0, jp.nmcu)
    c = int(len(j) * 0.6) // 4096 * 4096
    bad = j[:c] + np.random.default_rng(3).integers(0, 256, 4096, dtype=np.uint8).tobytes() + j[c + 4096:]
    a = Assembler([('a', lambda x0, x1: bad[x0:x1])], 0, len(bad), _grey(pic), cluster=4096).run_straight()
    kept = np.flatnonzero(a.have)
    assert a.log[-1][0] == 'straight' and 0.3 < a.coverage < 0.75
    assert np.array_equal(kept, np.arange(len(kept)))  # one run from the start
    assert np.array_equal(a.coefs[kept], co[kept])  # all of it the photo's own data


def _sky(w=768, h=512, seed=0):
    """A smooth sky: each row of MCUs looks almost like the ones above and below."""
    rng = np.random.default_rng(seed)
    y = np.linspace(0, 1, h)[:, None, None]
    base = np.array([70, 130, 220]) * (1 - y) + np.array([190, 210, 235]) * y
    clouds = np.asarray(Image.fromarray(rng.integers(0, 255, (h // 128, w // 24), np.uint8)).resize((w, h), Image.BICUBIC),
                        np.float32)[..., None] / 255
    img = base * (1 - 0.2 * clouds) + 255 * 0.2 * clouds + rng.normal(0, 2, (h, w, 3))
    return Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))


def test_resync_in_a_smooth_picture():
    """With a sharp reference the rows next to the right one score nearly as high; the number of missing bytes
    picks the row, and every MCU after the hole lands in its own place."""
    pic = _sky()
    j = _jpeg(pic)
    C = 2048
    hole = len(j) // 3 // C * C
    bad = j[:hole] + bytes(C) + j[hole + C:]
    ref = pic.resize((pic.width // 8, pic.height // 8), Image.BOX)
    a = Assembler([('a', lambda x0, x1: bad[x0:x1])], 0, len(bad), ref, cluster=C, blur=(3, 3)).run()
    assert [l[0] for l in a.log] == ['ok', 'resync', 'ok'] and a.log[1][5] < 0.08  # the score alone was not enough
    assert a.coverage > 0.9
    jp = Jpeg(j)
    co, st, n, err = jp.decode(0, jp.nmcu)
    assert np.array_equal(a.coefs[a.have], co[a.have])


def test_whole_needs_the_stream_to_end_at_eoi(picture):
    pic, j = picture
    ref = pic.resize((pic.width // 8, pic.height // 8), Image.BOX)
    whole = lambda b: Assembler([('a', lambda x0, x1: b[x0:x1])], 0, len(b), ref, cluster=4096).whole()
    assert whole(j)
    mid = len(j) // 2
    assert not whole(j[:mid] + bytes(4096) + j[mid + 4096:])
    assert not whole(j[:-2] + b'\x55' * 300 + j[-2:])  # every MCU decodes, but the stream ends too early
