"""Write recovered pictures with the photo's metadata (capture time, camera, lens, exposure, orientation)."""
from __future__ import annotations

import os
import shutil
import struct
import subprocess
import time

from PIL import Image, TiffImagePlugin


def exif_for(tags):
    """Minimal EXIF built from the tags parsed out of the CR2 header."""
    ex = Image.Exif()
    for k, tag in (('Make', 0x010F), ('Model', 0x0110)):
        if tags.get(k):
            ex[tag] = tags[k]
    if tags.get('Orientation'):
        ex[0x0112] = int(tags['Orientation'])
    if tags.get('DateTimeOriginal'):
        ex[0x0132] = tags['DateTimeOriginal']
    ex[0x0131] = 'cr2-rescue'
    sub = {}
    if tags.get('DateTimeOriginal'):
        sub[0x9003] = tags['DateTimeOriginal']
        sub[0x9004] = tags['DateTimeOriginal']
    if tags.get('SubSecTimeOriginal'):
        sub[0x9291] = tags['SubSecTimeOriginal']
    for k, tag in (('ExposureTime', 0x829A), ('FNumber', 0x829D), ('FocalLength', 0x920A)):
        v = tags.get(k)
        if v:
            sub[tag] = TiffImagePlugin.IFDRational(int(v[0]), int(v[1]))
    if tags.get('ISO'):
        sub[0x8827] = int(tags['ISO'])
    if tags.get('LensModel'):
        sub[0xA434] = tags['LensModel']
    if sub:
        ex[0x8769] = sub
    return ex


def insert_exif(jpeg, ex):
    """Add an APP1/EXIF segment to camera JPEG bytes without re-encoding the picture."""
    data = ex.tobytes()
    if not data.startswith(b'Exif\x00\x00'):
        data = b'Exif\x00\x00' + data
    if len(data) + 2 > 0xFFFF or jpeg[:2] != b'\xff\xd8':
        return jpeg
    return jpeg[:2] + b'\xff\xe1' + struct.pack('>H', len(data) + 2) + data + jpeg[2:]


def capture_timestamp(tags):
    t = tags.get('DateTimeOriginal')
    if not t:
        return None
    try:
        return time.mktime(time.strptime(t, '%Y:%m:%d %H:%M:%S'))
    except ValueError:
        return None


def write_jpeg_bytes(path, jpeg, tags):
    with open(path, 'wb') as f:
        f.write(insert_exif(jpeg, exif_for(tags)))
    touch(path, tags)


def write_image(path, im, tags, quality=95):
    im.save(path, quality=quality, subsampling=1, exif=exif_for(tags))
    touch(path, tags)


def touch(path, tags):
    ts = capture_timestamp(tags)
    if ts:
        os.utime(path, (ts, ts))


def exiftool_available():
    return shutil.which('exiftool') is not None


def copy_all_metadata(pairs, log=print):
    """Copy every tag (incl. maker notes) with exiftool: pairs = [(jpeg_path, cr2_header_path)]."""
    if not pairs or not exiftool_available():
        return False
    lines = []
    for jpg, head in pairs:
        lines += ['-tagsFromFile', head, '-all:all', '-FileModifyDate<DateTimeOriginal', jpg, '-execute']
    argfile = pairs[0][1] + '.args'
    with open(argfile, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines[:-1]) + '\n')
    r = subprocess.run(['exiftool', '-charset', 'filename=utf8', '-@', argfile, '-common_args', '-q', '-q', '-m',
                        '-overwrite_original'], capture_output=True, text=True)
    os.remove(argfile)
    if r.returncode:
        log(f'exiftool: {r.stderr.strip()[:300]}')
    return r.returncode == 0
