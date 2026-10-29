"""Campaign-wide non-destructive inventory of recorded CPI background frames.

Usage: python python/proc/inventoryBackgrounds.py /mnt/CPI-raw/CLACE2013
"""
import argparse
import csv
from pathlib import Path
from recordedBackgrounds import scan_recorded_backgrounds


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('directory', type=Path)
    p.add_argument('--csv', type=Path, help='optional CSV output path')
    args = p.parse_args()
    paths = sorted(args.directory.rglob('*.roi'))
    rows = []
    for path in paths:
        # This is an inventory, so it need not decode or store image pixels
        data = path.read_bytes()
        _, s = scan_recorded_backgrounds(data)
        row = {'file': str(path), 'frames': s['frames'],
               'background_flags': s['background_flags'],
               'decoded_backgrounds': s['decoded'],
               'invalid_backgrounds': s['invalid_backgrounds'],
               'versions': str(s['versions']), 'image_types': str(s['types']),
               'BGRate': str(s['bg_rates'])}
        rows.append(row)
        print(f"{path.name}: frames={s['frames']} flags={s['background_flags']} "
              f"decoded={s['decoded']} invalid={s['invalid_backgrounds']} "
              f"types={s['types']} BGRate={s['bg_rates']}", flush=True)
    print(f"TOTAL: files={len(rows)}, decoded backgrounds="
          f"{sum(r['decoded_backgrounds'] for r in rows)}")
    if args.csv:
        with args.csv.open('w', newline='') as f:
            wr = csv.DictWriter(f, fieldnames=rows[0].keys() if rows else
                ['file','frames','background_flags','decoded_backgrounds',
                 'invalid_backgrounds','versions','image_types','BGRate'])
            wr.writeheader()
            wr.writerows(rows)


if __name__ == '__main__':
    main()
