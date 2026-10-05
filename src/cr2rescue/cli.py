"""Command line interface: `cr2-rescue scan`, `cr2-rescue recover` and `cr2-rescue sharpen`."""
from __future__ import annotations

import argparse
import os
import sys

from . import __version__

EPILOG = """examples:
  cr2-rescue scan ./recovered
  cr2-rescue recover ./recovered -o ./rescued
  cr2-rescue recover card.img --all-files -o ./rescued      (raw image of the memory card)
  cr2-rescue sharpen ./rescued                              (detail for the blurry filled-in parts)

docs: https://github.com/yin1218/cr2-rescue"""


def _size(s):
    s = s.strip().lower()
    if s == 'auto':
        return 0
    mult = 1024 if s.endswith('k') else 1024 * 1024 if s.endswith('m') else 1
    n = int(s.rstrip('km')) * mult
    if n <= 0 or n & (n - 1):
        raise argparse.ArgumentTypeError('cluster size must be a power of two, e.g. 32k')
    return n


def build_parser():
    p = argparse.ArgumentParser(
        prog='cr2-rescue',
        description='Recover the full-size photo from corrupted, truncated or fragmented Canon CR2 files.',
        epilog=EPILOG, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--version', action='version', version=f'cr2-rescue {__version__}')
    sub = p.add_subparsers(dest='cmd', required=True)

    def common(q):
        q.add_argument('paths', nargs='+', help='CR2 files, folders, or a disk image (with --all-files)')
        q.add_argument('--cluster', type=_size, default=0, metavar='SIZE',
                       help='file-system cluster size, e.g. 32k (default: auto, falls back to 32k)')
        q.add_argument('--all-files', action='store_true',
                       help='search every file, not just *.cr2 (recovery-tool output, disk images)')
        q.add_argument('-j', '--jobs', type=int, default=0, help='parallel processes (default: all CPUs)')

    s = sub.add_parser('scan', help='list the photos found and how damaged they look (read-only, fast)')
    common(s)

    r = sub.add_parser('recover', help='rebuild every photo and write JPEGs')
    common(r)
    r.add_argument('-o', '--out', required=True, help='output folder (created if missing)')
    r.add_argument('--no-global-search', dest='global_search', action='store_false',
                   help='do not search other files for missing clusters (faster)')
    r.add_argument('--passes', type=int, default=3, help='global search rounds (default: 3)')
    r.add_argument('--exiftool', choices=('auto', 'yes', 'no'), default='auto',
                   help='copy all metadata incl. maker notes with exiftool if installed (default: auto)')
    r.add_argument('--quality', type=int, default=95, help='JPEG quality for re-encoded pictures (default: 95)')
    r.add_argument('--keep-work', action='store_true', help='keep the temporary .work folder')

    h = sub.add_parser('sharpen', help="give the blurry filled-in part of partial photos detail again "
                                       "(pip install 'cr2-rescue[sharpen]')")
    h.add_argument('rescued', help='output folder of cr2-rescue recover')
    h.add_argument('-o', '--out', help='output folder (default: RESCUED/sharpened)')
    h.add_argument('--method', choices=('both', 'neighbour', 'upscale'), default='both',
                   help='neighbour = real detail from a photo taken seconds apart; upscale = AI upscaler '
                        '(realesrgan-ncnn-vulkan); both = neighbour where it fits, upscaler elsewhere (default)')
    h.add_argument('--upscaler', metavar='PATH', help='realesrgan-ncnn-vulkan executable (default: found on PATH)')
    h.add_argument('--model', default='realesrgan-x4plus', help='x4 upscaling model (default: realesrgan-x4plus)')
    h.add_argument('-j', '--jobs', type=int, default=2, help='photos at a time (default: 2; ~2 GB of memory each)')
    h.add_argument('--quality', type=int, default=95, help='JPEG quality (default: 95)')
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(line_buffering=True)  # progress shows up even when the output goes to a file
    if args.cmd == 'scan':
        from .recover import quick_scan
        files, photos, cluster, problems, health = quick_scan(args.paths, args.cluster, args.jobs, args.all_files)
        print(f'{len(files)} file(s), {sum(len(p.copies) for p in photos)} CR2 copies, {len(photos)} photo(s), '
              f'cluster {cluster} bytes\n')
        print(f'{"photo":32} {"captured":22} {"copies":>6}  status')
        n_ok = 0
        for p in photos:
            h = health.get(p.key)
            if h is None:
                st = 'preview unreadable'
            elif h >= 1.0:
                st = 'intact'
                n_ok += 1
            else:
                st = f'damaged ({int(h * 100)}% decodes correctly)'
            print(f'{p.name[:32]:32} {(p.capture_time or "-")[:22]:22} {len(p.copies):>6}  {st}')
        for path, why in problems:
            print(f'! {os.path.basename(path)}: {why}', file=sys.stderr)
        print(f'\n{n_ok} intact, {len(photos) - n_ok} need repair.  Next: cr2-rescue recover {" ".join(args.paths)} -o OUT')
        return 0
    if args.cmd == 'sharpen':
        from .sharpen import sharpen
        try:
            rows = sharpen(args.rescued, args.out, args.method, args.upscaler, args.model, args.jobs, args.quality)
        except ImportError as e:
            print(e, file=sys.stderr)
            return 2
        return 0 if any(r['output'] for r in rows) or not rows else 1
    from .recover import Options, recover
    opt = Options(cluster=args.cluster, jobs=args.jobs, global_search=args.global_search, passes=args.passes,
                  exiftool=args.exiftool, all_files=args.all_files, keep_work=args.keep_work, quality=args.quality)
    rows = recover(args.paths, args.out, opt)
    return 0 if rows else 1


if __name__ == '__main__':
    raise SystemExit(main())
