import io
import os

import numpy as np
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
