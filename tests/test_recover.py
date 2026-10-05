import io
import struct
import os

import numpy as np
import pytest
from PIL import Image

from cr2rescue import synth
from cr2rescue.recover import Options, recover


def _pixels(path):
    return np.asarray(Image.open(path).convert('RGB'), np.float32)


def _orig(card, letter):
    return np.asarray(Image.open(io.BytesIO(card['previews'][letter])).convert('RGB'), np.float32)


def test_categories(card, recovered):
    got = {k: v['category'] for k, v in recovered['by_photo'].items()}
    assert got == card['expected']


def test_intact_and_repaired_are_lossless(card, recovered):
    for letter, row in recovered['by_photo'].items():
        if row['category'] in ('intact', 'repaired'):
            out = _pixels(recovered['out'] / row['output'])
            assert np.array_equal(out, _orig(card, letter)), letter


def test_repairs_use_the_right_sources(recovered):
    b, d = recovered['by_photo']['B'], recovered['by_photo']['D']
    assert 'found' in b['sources']          # global search found B's second half after photo C
    assert 'copy' in d['sources']           # D was completed from its second copy


def test_partial_results_are_close(card, recovered):
    for letter in ('C', 'E'):
        row = recovered['by_photo'][letter]
        assert 0.3 < row['coverage'] < 1
        out, ref = _pixels(recovered['out'] / row['output']), _orig(card, letter)
        psnr = 10 * np.log10(255 ** 2 / ((out - ref) ** 2).mean())
        assert psnr > 22, (letter, psnr)
    assert recovered['by_photo']['C']['coverage'] > 0.85   # resynced after the overwritten clusters


def test_partial_results_have_a_mask(recovered):
    for row in recovered['rows']:
        if row['category'] != 'partial':
            assert not row['mask']
            continue
        m = Image.open(recovered['out'] / row['mask'])
        assert m.mode == '1' and m.size == Image.open(recovered['out'] / row['output']).size
        assert abs(np.asarray(m).mean() - row['coverage']) < 0.02


def test_preview_only_keeps_picture(card, recovered):
    row = recovered['by_photo']['F']
    out, ref = _pixels(recovered['out'] / row['output']), _orig(card, 'F')
    assert out.shape == ref.shape
    assert np.corrcoef(out.mean(2).ravel(), ref.mean(2).ravel())[0, 1] > 0.9


def test_metadata(card, recovered):
    for letter, row in recovered['by_photo'].items():
        p = recovered['out'] / row['output']
        ex = Image.open(p).getexif()
        sub = ex.get_ifd(0x8769)
        when, sub_s = card['times'][letter].split('.')
        assert ex[0x0110] == 'Canon EOS Synthetic'
        assert sub[0x9003] == when and sub[0x9291] == sub_s
        assert abs(os.path.getmtime(p) - __import__('time').mktime(
            __import__('time').strptime(when, '%Y:%m:%d %H:%M:%S'))) < 2


def test_reports(recovered):
    out = recovered['out']
    assert (out / 'report.csv').exists() and (out / 'report.json').exists()
    assert not (out / '.work').exists()


def test_raw_card_image(tmp_path):
    """A raw image of the card (e.g. from dd/ddrescue) holds every cluster, so B's continuation is found and E
    is not truncated; D has a single copy there, so its overwritten cluster stays filled in."""
    card = synth.make_card(str(tmp_path / 'carved'), cluster=4096, size=(768, 512), seed=2,
                           image=str(tmp_path / 'card.img'))
    rows = recover([str(tmp_path / 'card.img')], str(tmp_path / 'out'), Options(jobs=2, exiftool='no', all_files=True),
                   log=lambda *a: None)
    got = {r['capture_time']: r['category'] for r in rows}
    want = dict(A='intact', B='repaired', C='partial', D='partial', E='intact', F='preview-only', G='intact')
    assert {card['times'][n]: c for n, c in want.items()} == got


def _card_with_small_images(folder, damage):
    """One CR2 per entry of `damage`: None = intact small image, 'other' = overwritten by unrelated picture data
    (another scene upside down: synthetic scenes all look alike), (dx, dy) = its own small image shifted (fits its
    thumbnail well, but at a wrong box)."""
    from cr2rescue.tiff import parse_cr2
    for k, how in enumerate(damage):
        im = synth.scene(1536, 1024, seed=60 + k)
        data, _ = synth.make_cr2(im, when=f'2024:05:02 10:00:{k:02d}', seed=k)
        s = parse_cr2(data).small
        if how is not None:
            a, _ = synth._small(synth.scene(1536, 1024, seed=90 + k).rotate(180) if how == 'other' else im, seed=k)
            if how != 'other':
                a = np.roll(a, how[::-1], axis=(0, 1))
            data = data[:s['offset']] + a.tobytes() + data[s['offset'] + s['length']:]
        with open(os.path.join(folder, f'IMG_{k:04d}.CR2'), 'wb') as f:
            f.write(data)


def test_alignment_ignores_damaged_small_images(tmp_path):
    from cr2rescue.recover import align_models
    from cr2rescue.scan import collect, scan
    # on a damaged card the small image is often overwritten while the preview is fine
    _card_with_small_images(str(tmp_path), ['other', 'other', (14, 7), None, 'other', None, None, 'other'])
    photos = scan(collect([str(tmp_path)]), log=lambda *a: None)[0]
    lines = []
    boxes = align_models(photos, log=lines.append, per_model=4)
    assert list(boxes.values()) == [(5, 3, 192, 128)], lines
    assert '3 photo(s)' in lines[0] and '4 of 7 small images intact' in lines[0], lines


def test_alignment_without_intact_small_images(tmp_path):
    from cr2rescue.recover import align_models
    from cr2rescue.scan import collect, scan
    _card_with_small_images(str(tmp_path), ['other', 'other', 'other'])
    lines = []
    assert align_models(scan(collect([str(tmp_path)]), log=lambda *a: None)[0], log=lines.append) == {}
    assert 'no intact small image' in lines[0]


def _camera_folder(folder, damage, size=(768, 512), cluster=4096):
    """Photos taken one second apart with one camera; per entry of `damage` a set of what is overwritten:
    'header' = the preview's JPEG header, 'thumb' = the thumbnail, 'ifd' = the IFD of the small image (and the
    RAW's after it), 'small' = the small image, 'other' = the small image shows another scene, 'foreign' = two
    clusters of the preview hold another photo's preview data. Returns the preview of each photo."""
    from cr2rescue.jpeg import Jpeg
    from cr2rescue.tiff import parse_cr2, read_ifd
    rng = np.random.default_rng(5)
    shots = [synth.make_cr2(synth.scene(*size, seed=70 + k), when=f'2024:05:03 10:00:{k:02d}', seed=k)
             for k in range(len(damage))]
    previews = []
    for k, how in enumerate(damage):
        data, prev = shots[k]
        d = bytearray(data)
        lay = parse_cr2(data)
        po, pl = lay.preview
        if 'header' in how:
            d[po:po + Jpeg(prev).sos_end] = rng.integers(1, 255, Jpeg(prev).sos_end, dtype=np.uint8).tobytes()
        if 'thumb' in how:
            to, tl = lay.thumb
            d[to:to + tl] = rng.integers(0, 256, tl, dtype=np.uint8).tobytes()
        s = lay.small
        if 'small' in how:
            d[s['offset']:s['offset'] + s['length']] = rng.integers(0, 256, s['length'], dtype=np.uint8).tobytes()
        if 'other' in how:
            a, _ = synth._small(synth.scene(*size, seed=99).rotate(180), seed=k)
            d[s['offset']:s['offset'] + s['length']] = a.tobytes()
        if 'foreign' in how:
            other_po = parse_cr2(shots[k - 1][0]).preview[0]
            c = (po + pl // 2) // cluster * cluster
            d[c:c + 2 * cluster] = shots[k - 1][0][other_po + c - po:other_po + c - po + 2 * cluster]
        if 'ifd' in how:
            at = struct.unpack_from('<I', d, 4)[0]
            for _ in range(2):  # IFD0 -> IFD1 (thumbnail) -> IFD2 (small image)
                at = read_ifd(bytes(d), at)[1]
            d[at:at + 2] = b'\0\0'
        with open(os.path.join(folder, f'IMG_{k:04d}.CR2'), 'wb') as f:
            f.write(bytes(d))
        previews.append(prev)
    return previews


@pytest.fixture(scope='module')
def camera(tmp_path_factory):
    folder = tmp_path_factory.mktemp('camera')
    damage = [(), (), (), ('header',), ('thumb', 'ifd', 'foreign'), ('thumb', 'other'), (),
              ('header', 'thumb', 'small')]
    previews = _camera_folder(str(folder), damage)
    lines = []
    rows = recover([str(folder)], str(folder / 'out'), Options(jobs=2, exiftool='no', cluster=4096),
                   log=lambda *a: lines.append(' '.join(map(str, a))))
    by = {r['name']: r for r in rows}
    return dict(out=folder / 'out', rows=[by[f'IMG_{k:04d}'] for k in range(len(damage))], previews=previews,
                log='\n'.join(lines))


def test_learns_what_the_camera_shares(camera):
    log = camera['log']
    assert 'same preview header in 6 of 6 photos' in log  # the two with an overwritten header do not count
    assert 'small image right after the preview in all 5' in log  # nor does the one without its IFD
    assert 'small image placed from the camera model for 1 copy' in log
    assert '3 photo(s) without thumbnail, colour style borrowed from a neighbour for 3' in log


def test_header_copied_from_the_same_camera(camera):
    r = camera['rows'][3]
    assert r['category'] == 'repaired' and 'JPEG header copied' in r['note']
    want = np.asarray(Image.open(io.BytesIO(camera['previews'][3])).convert('RGB'), np.float32)
    assert np.array_equal(_pixels(camera['out'] / r['output']), want)


def test_small_image_without_thumbnail_still_catches_foreign_data(camera):
    """No thumbnail and no IFD for the small image: it is found from the camera model, drawn in a neighbour's
    colours, refitted to what decoded, and still rejects the other photo's clusters."""
    r = camera['rows'][4]
    assert r['reference'] == 'small-style+fit' and r['category'] == 'partial' and 0.6 < r['coverage'] < 0.97
    want = np.asarray(Image.open(io.BytesIO(camera['previews'][4])).convert('RGB'), np.float32)
    got = _pixels(camera['out'] / r['output'])
    assert 10 * np.log10(255 ** 2 / ((got - want) ** 2).mean()) > 22


def test_small_image_of_another_picture_is_not_trusted(camera):
    r = camera['rows'][5]
    assert r['category'] == 'unverified' and r['reference'] == 'none' and 'another picture' in r['note']
    want = np.asarray(Image.open(io.BytesIO(camera['previews'][5])).convert('RGB'), np.float32)
    assert np.array_equal(_pixels(camera['out'] / r['output']), want)


def test_damaged_small_image_does_not_spoil_an_intact_preview(tmp_path, monkeypatch):
    """The small image is stored after the preview and can be the damaged one (its lower part from a burst shot
    that the coarse thumbnail cannot tell apart): a preview that decodes in one piece to its end and matches the
    thumbnail stays intact."""
    import tempfile
    from cr2rescue import recover as R
    from cr2rescue.scan import collect, scan
    previews = _camera_folder(str(tmp_path), [(), ()])
    real = R.reference_for

    def burst_tail(*args, **kw):
        ref, kind, small = real(*args, **kw)
        other = synth.scene(768, 512, seed=99).resize(ref.size)
        ref.paste(other.crop((0, ref.height * 3 // 5, ref.width, ref.height)), (0, ref.height * 3 // 5))
        return ref, kind, small

    monkeypatch.setattr(R, 'reference_for', burst_tail)
    photos, _, cluster, _ = scan(collect([str(tmp_path)]), cluster=4096, log=lambda *a: None)
    R._init(dict(cluster=cluster, work=tempfile.mkdtemp(dir=tmp_path), quality=90,
                 boxes=R.align_models(photos, log=lambda *a: None)))
    r, _ = R.process((next(p for p in photos if p.name == 'IMG_0001'), [], 1))
    assert r.category == 'intact' and r.reference == 'thumb' and 'small image does not match' in r.note
    want = np.asarray(Image.open(io.BytesIO(previews[1])).convert('RGB'), np.float32)
    assert np.array_equal(_pixels(r.file), want)


def test_nothing_left_to_check_against(camera):
    r = camera['rows'][7]
    assert r['category'] == 'failed' and 'nothing to check the picture against' in r['note']
    assert [r['category'] for r in camera['rows'][:3]] == ['intact'] * 3
