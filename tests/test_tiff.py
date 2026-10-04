import pytest

from cr2rescue import synth
from cr2rescue.tiff import find_headers, parse_cr2


@pytest.fixture(scope='module')
def cr2():
    im = synth.scene(512, 344, seed=4)
    data, preview = synth.make_cr2(im, when='2021:06:15 13:45:23', subsec='73', model='Canon EOS Test')
    return data, preview


def test_layout(cr2):
    data, preview = cr2
    lay = parse_cr2(data)
    po, pl = lay.preview
    assert data[po:po + pl] == preview
    to, tl = lay.thumb
    assert data[to:to + 2] == b'\xff\xd8'
    assert lay.small['width'] == 64 + 8 and lay.small['height'] == 43 + 5 and lay.small['spp'] == 3
    assert lay.raw is not None


def test_tags(cr2):
    lay = parse_cr2(cr2[0])
    t = lay.tags
    assert t['Make'] == 'Canon' and t['Model'] == 'Canon EOS Test'
    assert t['DateTimeOriginal'] == '2021:06:15 13:45:23'
    assert lay.capture_time == '2021:06:15 13:45:23.73'
    assert t['ISO'] == 200 and t['FNumber'] == (56, 10)


def test_find_headers(cr2):
    data = cr2[0]
    blob = b'\0' * 1000 + data + b'junk' * 100 + data
    assert find_headers(blob) == [1000, 1000 + len(data) + 400]


def test_truncated_header_does_not_crash(cr2):
    data = cr2[0]
    for n in (16, 40, 200, 1000):
        try:
            parse_cr2(data[:n])
        except ValueError:
            pass
