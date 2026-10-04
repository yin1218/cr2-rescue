import pytest

from cr2rescue import synth
from cr2rescue.recover import Options, recover

SIZE = (1024, 672)
CLUSTER = 4096


@pytest.fixture(scope='session')
def card(tmp_path_factory):
    """A damaged, synthetic memory card (see synth.make_card for the scenarios)."""
    folder = tmp_path_factory.mktemp('card')
    info = synth.make_card(str(folder), cluster=CLUSTER, size=SIZE, seed=1)
    info['folder'] = str(folder)
    return info


@pytest.fixture(scope='session')
def recovered(card, tmp_path_factory):
    out = tmp_path_factory.mktemp('out')
    rows = recover([card['folder']], str(out), Options(jobs=2, exiftool='no'), log=lambda *a: None)
    by_photo = {}
    for letter, fname in card['files'].items():
        stem = fname.rsplit('.', 1)[0]
        by_photo[letter] = next(r for r in rows if r['name'] == stem)
    return dict(out=out, rows=rows, by_photo=by_photo)
