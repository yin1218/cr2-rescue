"""Build the before/after pictures used in the README from a synthetic, damaged memory card.

    python examples/make_demo.py            # writes docs/images/demo.jpg and docs/images/social-preview.png

No real photos are involved: synth.py draws the scenes and damages the card the way real recoveries go wrong.
"""
import io
import os
import sys
import tempfile

from PIL import Image, ImageDraw, ImageFile, ImageFont

from cr2rescue import synth
from cr2rescue.recover import Options, recover
from cr2rescue.tiff import parse_cr2

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, '..', 'docs', 'images')
CASES = [('B', 'Fragmented file', '檔案碎片化'), ('C', 'Clusters overwritten', '部分資料被覆蓋'),
         ('E', 'Truncated file', '檔案被截斷'), ('F', 'Preview destroyed', '大圖全毀')]


def font(size):
    for name in ('Arial Unicode.ttf', 'NotoSansCJK-Regular.ttc', 'PingFang.ttc', 'DejaVuSans.ttf'):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    try:
        return ImageFont.load_default(size)
    except TypeError:
        return ImageFont.load_default()


def what_a_viewer_shows(path):
    """Decode the damaged preview the way an ordinary viewer does (keeps going past errors)."""
    data = open(path, 'rb').read()
    po, pl = parse_cr2(data).preview
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    try:
        im = Image.open(io.BytesIO(data[po:po + pl]))
        im.load()
        return im.convert('RGB')
    except Exception:
        return Image.new('RGB', (1536, 1024), (128, 128, 128))
    finally:
        ImageFile.LOAD_TRUNCATED_IMAGES = False


def main():
    os.makedirs(OUT, exist_ok=True)
    tmp = tempfile.mkdtemp()
    card = synth.make_card(os.path.join(tmp, 'card'), cluster=8192, size=(1536, 1024), seed=3)
    rows = recover([os.path.join(tmp, 'card')], os.path.join(tmp, 'out'), Options(exiftool='no'), log=print)
    by_name = {r['name']: r for r in rows}
    tw, th = 480, 320
    pad, head, label = 24, 70, 34
    W = pad + 2 * (tw + pad)
    H = head + len(CASES) * (th + label + pad) + pad
    sheet = Image.new('RGB', (W, H), (250, 250, 250))
    d = ImageDraw.Draw(sheet)
    f_big, f_small = font(30), font(20)
    d.text((pad + tw // 2, 26), 'Before / 修復前', fill=(170, 30, 30), font=f_big, anchor='mm')
    d.text((2 * pad + tw + tw // 2, 26), 'After cr2-rescue / 修復後', fill=(20, 120, 50), font=f_big, anchor='mm')
    pairs = []
    for i, (letter, en, zh) in enumerate(CASES):
        fname = card['files'][letter]
        row = by_name[fname.rsplit('.', 1)[0]]
        before = what_a_viewer_shows(os.path.join(tmp, 'card', fname))
        after = Image.open(os.path.join(tmp, 'out', row['output'])).convert('RGB')
        pairs.append((before, after))
        y = head + i * (th + label + pad)
        d.text((pad, y + 4), f'{en} / {zh}', fill=(40, 40, 40), font=f_small)
        cat = row['category'] + (f" ({row['coverage'] * 100:.0f}%)" if row['category'] == 'partial' else '')
        d.text((W - pad, y + 4), f'→ {cat}', fill=(90, 90, 90), font=f_small, anchor='ra')
        sheet.paste(before.resize((tw, th), Image.LANCZOS), (pad, y + label))
        sheet.paste(after.resize((tw, th), Image.LANCZOS), (2 * pad + tw, y + label))
    sheet.save(os.path.join(OUT, 'demo.jpg'), quality=88, optimize=True)

    # 1280x640 social preview (GitHub repository settings -> Social preview)
    sp = Image.new('RGB', (1280, 640), (17, 24, 39))
    d = ImageDraw.Draw(sp)
    before, after = pairs[0]
    sp.paste(before.resize((420, 280), Image.LANCZOS), (40, 300))
    sp.paste(after.resize((420, 280), Image.LANCZOS), (480, 300))
    d.text((250, 600), 'before', fill=(248, 113, 113), font=font(24), anchor='mm')
    d.text((690, 600), 'after', fill=(74, 222, 128), font=font(24), anchor='mm')
    d.text((40, 50), 'cr2-rescue', fill=(255, 255, 255), font=font(84))
    d.text((40, 160), 'Recover corrupted Canon CR2 RAW photos', fill=(209, 213, 219), font=font(36))
    d.text((40, 215), '救回損毀、無法開啟的 Canon CR2 照片', fill=(209, 213, 219), font=font(36))
    for k, line in enumerate(['fragmented', 'truncated', 'overwritten', 'grey / half-grey', 'after PhotoRec',
                              'full-size JPEG', 'EXIF kept']):
        d.text((940, 300 + k * 40), '• ' + line, fill=(156, 163, 175), font=font(26))
    sp.save(os.path.join(OUT, 'social-preview.png'), optimize=True)
    print('wrote', os.path.normpath(OUT))


if __name__ == '__main__':
    sys.exit(main())
