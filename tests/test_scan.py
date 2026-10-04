from cr2rescue.assemble import plausible
from cr2rescue.scan import guess_cluster


def test_cluster_from_header_distances():
    assert guess_cluster([[0, 3 * 32768, 10 * 32768], [22016, 22016 + 7 * 32768]]) == 32768
    assert guess_cluster([[0, 5 * 8192, 9 * 8192]]) == 8192


def test_cluster_from_file_sizes():
    sizes = [104 * 8192, 18 * 8192, 191 * 8192, 105 * 8192, 152778, 879887]
    assert guess_cluster([], sizes) == 8192


def test_cluster_never_above_default():
    assert guess_cluster([[0, 1 << 20, 2 << 20]]) == 32768
    assert guess_cluster([]) == 32768


def test_plausible_entropy_data():
    import numpy as np
    import io
    from cr2rescue import synth
    b = io.BytesIO()
    synth.scene(256, 160, seed=1).save(b, 'JPEG', quality=90)
    j = b.getvalue()
    ent = j[j.index(b'\xff\xda') + 20:-2]
    assert plausible(ent[:4096])
    assert not plausible(b'\0' * 4096)
    assert not plausible(np.random.default_rng(0).integers(0, 256, 4096, dtype=np.uint8).tobytes())  # random data
    assert not plausible(j[:4096])                # header markers
