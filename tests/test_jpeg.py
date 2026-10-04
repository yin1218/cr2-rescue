import io

import numpy as np
import pytest
from PIL import Image

from cr2rescue import synth
from cr2rescue.jpeg import Jpeg, Unsupported, markers_ok


@pytest.fixture(scope='module')
def picture():
    return synth.scene(400, 264, seed=3)


def _jpeg(im, **kw):
    b = io.BytesIO()
    im.save(b, 'JPEG', **kw)
    return b.getvalue()


@pytest.mark.parametrize('subsampling,sampling', [(0, (1, 1)), (1, (2, 1)), (2, (2, 2))])
def test_decoder_matches_pil(picture, subsampling, sampling):
    j = _jpeg(picture, quality=90, subsampling=subsampling)
    jp = Jpeg(j)
    assert (jp.hy, jp.vy) == sampling
    coefs, starts, n, err = jp.decode(0, jp.nmcu)
    assert err == 0 and n == jp.nmcu
    out = jp.render(coefs[:n], jp.dc_chain(coefs[:n])).astype(int)
    ref = np.asarray(Image.open(io.BytesIO(j)).convert('RGB')).astype(int)
    assert out.shape == ref.shape
    d = np.abs(out - ref)
    assert d.mean() < 1.0          # chroma upsampling differs slightly from libjpeg's
    assert np.percentile(d, 99) <= 4


def test_mcu_means_follow_picture(picture):
    jp = Jpeg(_jpeg(picture, quality=90, subsampling=1))
    coefs, _, n, _ = jp.decode(0, jp.nmcu)
    mm = jp.mcu_means(jp.dc_chain(coefs[:n]))
    assert mm.shape == (3, jp.nmcu)
    y = np.asarray(picture.convert('L'), np.float32)
    first = y[:8, :16].mean()
    assert abs(mm[0, 0] - first) < 4


def test_progressive_is_rejected(picture):
    with pytest.raises(Unsupported):
        Jpeg(_jpeg(picture, progressive=True))


def test_restart_markers_are_rejected(picture):
    j = _jpeg(picture, quality=80)
    i = j.index(b'\xff\xda')
    with pytest.raises(Unsupported):
        Jpeg(j[:i] + b'\xff\xdd\x00\x04\x00\x10' + j[i:])


def test_markers_ok(picture):
    j = _jpeg(picture, quality=80)
    assert markers_ok(j)
    assert not markers_ok(j[:-2])                               # no end marker
    k = len(j) // 2
    assert not markers_ok(j[:k] + b'\xff\xd8' + j[k + 2:])      # a stray marker in the data
