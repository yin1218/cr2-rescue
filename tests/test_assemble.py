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
