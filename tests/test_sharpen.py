import json

import numpy as np
import pytest
from PIL import Image
from scipy.ndimage import gaussian_filter

cv2 = pytest.importorskip('cv2')

from cr2rescue import sharpen as S
from cr2rescue.cli import main

W, H = 768, 512


def _scene(seed, w=W + 160, h=H + 120):
    """Detail at every scale (the kind a thumbnail loses) in three colours."""
    rng = np.random.default_rng(seed)
    x = sum(gaussian_filter(rng.normal(0, 1, (h, w, 3)), (s, s, 0)) * s for s in (1, 2, 4, 16, 48))
    x = (x - x.mean()) / x.std()
    return np.clip(128 + 45 * x, 0, 255).astype(np.float32)


def _fill(x, f=32):
    """What recover puts where the data is gone: the picture at 1/f, back up."""
    small = cv2.resize(x, (x.shape[1] // f, x.shape[0] // f), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (x.shape[1], x.shape[0]), interpolation=cv2.INTER_CUBIC)


@pytest.fixture(scope='module')
def damaged():
    scene = _scene(3)
    truth = scene[60:60 + H, 80:80 + W]
    # the neighbour: same scene, camera moved a little (shift + 1 degree)
    M = cv2.getRotationMatrix2D((W / 2 + 80, H / 2 + 60), 1.0, 1.0)
    M[:, 2] += (-23, -11)
    N = cv2.warpAffine(scene, M, (scene.shape[1], scene.shape[0]), flags=cv2.INTER_CUBIC)[60:60 + H, 80:80 + W]
    hole = np.zeros((H, W), bool)
    hole[int(H * 0.6):] = True
    T = np.where(hole[..., None], _fill(truth), truth)
    return truth, T, hole, N


def _rmse(a, b, where):
    return float(np.sqrt(((a - b)[where] ** 2).mean()))


def test_neighbour_brings_back_the_detail(damaged):
    truth, T, hole, N = damaged
    A, trust, inliers = S.from_neighbour(T, ~hole, N, f=32)
    assert inliers >= S.MIN_INLIERS
    out = S.compose(T, hole, A, trust)
    inner = hole & (S.seam(hole) == 1) & (np.arange(W) > 40)[None] & (np.arange(W) < W - 40)[None]
    before, after = _rmse(T, truth, inner), _rmse(out, truth, inner)
    assert after < 0.4 * before, (before, after)
    assert np.array_equal(out[~hole], T[~hole])  # decoded data is never changed


def test_any_pixel_type_is_accepted(damaged):
    truth, T, hole, N = damaged
    assert S.from_neighbour(T.astype(np.float64), ~hole, N.round().astype(np.uint8), f=32) is not None


def test_neighbour_of_another_scene_is_not_used(damaged):
    truth, T, hole, _ = damaged
    other = _scene(4)[:H, :W]
    assert S.from_neighbour(T, ~hole, other, f=32) is None


def _fake_upscaler(src, dst, upscaler, model):
    im = Image.open(src)
    im.resize((im.width * 4, im.height * 4), Image.LANCZOS).save(dst)


@pytest.mark.parametrize('f', [32, 8])
def test_upscale_keeps_size_colours_and_decoded_data(monkeypatch, f):
    monkeypatch.setattr(S, 'run_upscaler', _fake_upscaler)
    h, w = 330, 500  # not multiples of 16; the hole touches the bottom and right edges
    truth = _scene(5, w, h)
    hole = np.zeros((h, w), bool)
    hole[200:, 150:] = True
    T = np.where(hole[..., None], _fill(truth, f), truth)
    C = S.upscale(T, hole, f, 'x', 'm')
    assert C.shape == T.shape
    assert np.abs(S.low(C, f) - S.low(T, f))[hole].mean() < 3  # colours of the fill kept
    out = S.compose(T, hole, C=C)
    assert np.array_equal(out[~hole], T[~hole])


def test_sharpen_command(recovered, tmp_path, monkeypatch):
    monkeypatch.setattr(S, 'run_upscaler', _fake_upscaler)
    exe = tmp_path / 'realesrgan-ncnn-vulkan'
    exe.write_text('')
    out = tmp_path / 'sharp'
    assert main(['sharpen', str(recovered['out']), '-o', str(out), '--upscaler', str(exe), '-j', '1']) == 0
    report = {r['name']: r for r in json.loads((out / 'sharpen.json').read_text())}
    partial = [r for r in recovered['rows'] if r['category'] == 'partial']
    assert partial and set(report) == {r['name'] for r in partial}
    for r in partial:
        got = report[r['name']]
        assert 'upscale' in got['method'] and got['output'], got
        a, b = Image.open(recovered['out'] / r['output']), Image.open(out / got['output'])
        assert a.size == b.size
        assert a.getexif().get_ifd(0x8769).get(0x9003) == b.getexif().get_ifd(0x8769).get(0x9003)


def test_a_stopped_run_goes_on_where_it_was(recovered, tmp_path, monkeypatch):
    monkeypatch.setattr(S, 'run_upscaler', _fake_upscaler)
    out = tmp_path / 'sharp'
    first = S.sharpen(str(recovered['out']), str(out), upscaler=__file__, jobs=1, log=lambda *a: None)
    done = [r for r in first if r['output']]
    assert len(done) >= 2
    (out / done[0]['output']).unlink()  # as if the run had stopped before this one
    seen, real_one = [], S.sharpen_one
    monkeypatch.setattr(S, 'sharpen_one', lambda job: seen.append(job[0]['name']) or real_one(job))
    again = S.sharpen(str(recovered['out']), str(out), upscaler=__file__, jobs=1, log=lambda *a: None)
    assert seen == [done[0]['name']]
    assert {r['name'] for r in again} == {r['name'] for r in first} and all(r['output'] for r in again)


def test_without_an_upscaler_only_neighbours_are_used(recovered, tmp_path, monkeypatch):
    monkeypatch.setenv('PATH', str(tmp_path))
    rows = S.sharpen(str(recovered['out']), str(tmp_path / 'sharp'), jobs=1, log=lambda *a: None)
    assert rows and all('upscale' not in r['method'] for r in rows)
