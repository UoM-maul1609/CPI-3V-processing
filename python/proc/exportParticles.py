#!/usr/bin/env python3
"""Export recorded CPI ROI images and a re-importable particle catalogue.

Reads existing imageStats MAT files. Does not recalculate statistics, classify
particles, change originals or connect to a database. Filenames use source ROI
indices rather than filtered export order, so lowering thresholds fills gaps.
"""
from __future__ import annotations
import argparse
import csv
from datetime import datetime, timedelta
import json
import math
import os
from pathlib import Path
import re
import tempfile

import numpy as np
from PIL import Image, ImageDraw
from scipy.io import loadmat

_FIELDS = ('source_id','project','flight','date','timestamp','source_mat','source_roi','roi_index','particle_index','size_um','focus','eligible','export_status','png_relative_path','processing_options','error')

def _single(value):
    while isinstance(value, np.ndarray) and value.size == 1:
        value = value.flat[0]
    return value

def _field(struct, name):
    struct = _single(struct)
    if isinstance(struct, np.void) and struct.dtype.names and name in struct.dtype.names:
        return _single(struct[name])
    if isinstance(struct, np.ndarray) and struct.dtype.names and name in struct.dtype.names:
        return _single(struct[name])
    raise KeyError(name)

def _at(struct, name, index):
    val = _field(struct, name)
    return np.asarray(val).reshape(-1)[index]

def _dt_from_matlab(num):
    num = float(num)
    if not math.isfinite(num) or num <= 366:
        return None
    return datetime.fromordinal(int(num)) + timedelta(days=num % 1) - timedelta(days=366)

def _safe(text):
    return re.sub(r'[^A-Za-z0-9_-]+','_',str(text)).strip('_') or 'unknown'

def _flight_rows(csvpath, project):
    if not csvpath:
        return []
    with open(csvpath, newline='', encoding='utf-8-sig') as handle:
        reader=csv.DictReader(handle)
        required={'date','flight'}
        if not required.issubset(set(reader.fieldnames or [])):
            raise ValueError('Flight CSV must contain date and flight columns')
        out=[]
        for row in reader:
            if row.get('project','').strip() and row['project'].strip()!=project:
                continue
            day=(row.get('date') or '').strip()
            try:
                start=datetime.fromisoformat((row.get('start_datetime') or '').strip()) if (row.get('start_datetime') or '').strip() else None
                end=datetime.fromisoformat((row.get('end_datetime') or '').strip()) if (row.get('end_datetime') or '').strip() else None
                if start is None and row.get('start_time','').strip():
                    start=datetime.fromisoformat(day+'T'+row['start_time'].strip())
                if end is None and row.get('end_time','').strip():
                    end=datetime.fromisoformat(day+'T'+row['end_time'].strip())
                if start and end and end <= start:
                    end += timedelta(days=1)
                if not start and not end:
                    datetime.strptime(day,'%Y-%m-%d')
            except ValueError as exc:
                raise ValueError(f'Invalid flight catalogue date/time: {row}') from exc
            out.append(dict(date=day,flight=(row.get('flight') or '').strip(),start=start,end=end,
                            source_key=(row.get('source_key') or '').strip()))
    return out

def _choose_flight(ts, matstem, rows):
    # Explicit source-file match is safest; otherwise require unique time range.
    # Day-only matching is allowed only when exactly one flight exists that day.
    if ts is None:
        return ''
    candidates=[r for r in rows if r['source_key'] and
                (r['source_key'] == matstem or r['source_key'] == matstem+'.roi')]
    if not candidates:
        candidates=[r for r in rows if r['start'] and r['end'] and r['start'] <= ts < r['end']]
    if not candidates:
        candidates=[r for r in rows if not r['end'] and r['date']==ts.date().isoformat()
                    and (r['start'] is None or r['start'] <= ts)]
    labels={r['flight'] for r in candidates if r['flight']}
    return labels.pop() if len(labels)==1 else ''

def _boundaries(dat, i):
    try:
        foc=_field(dat,'foc')
        entries=np.asarray(foc).reshape(-1)
        value=_field(entries[i], 'boundaries')
        points=np.asarray(value, dtype=float).reshape(-1,2)
        return points[np.isfinite(points).all(axis=1)]
    except (KeyError,IndexError,TypeError,ValueError):
        return np.empty((0,2))

def _png_bytes_array(roi):
    arr=np.asarray(_field(roi,'IM'))
    if arr.ndim != 2:
        raise ValueError('ROI pixels are not a 2-D image')
    return np.asarray(np.clip(arr,0,255), dtype=np.uint8)

def _encode_image(arr, outline):
    # Original pixel values are kept intact except for an OPTIONAL overlay.
    img=Image.fromarray(arr, 'L')
    if outline is not None and len(outline)>2:
        img=img.convert('RGB')
        draw=ImageDraw.Draw(img)
        # CPI boundaries use coordinates (row,column), matching numpy image indexing.
        xy=[(float(p[1]),float(p[0])) for p in outline]
        draw.line(xy+[xy[0]],fill=(255,0,0),width=1)
    return img

def _write_png(img, target):
    target.parent.mkdir(parents=True,exist_ok=True)
    fd, tmp=tempfile.mkstemp(prefix='.partial_',suffix='.png',dir=target.parent)
    os.close(fd)
    try:
        img.save(tmp,format='PNG',compress_level=6)
        os.replace(tmp,target)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)

def export(source, output, project=None, flight_csv=None, min_size_um=0., min_focus=None,
           skip_existing=True, overwrite=False):
    source=Path(source).resolve()
    if not source.is_dir():
        raise NotADirectoryError(source)
    root=Path(output).resolve()
    project=project or source.name
    rows=_flight_rows(flight_csv,project)
    project_dir=root/_safe(project)
    mat_files=sorted(p for p in source.glob('*.mat') if p.name not in {'full_backgrounds.mat','timeseries.mat','timeseries_class.mat'})
    if not mat_files:
        raise FileNotFoundError(f'No per-ROI MAT files found in {source}')
    catalogue=[]
    errors=[]
    written=existing=filtered=0
    for mat in mat_files:
        try:
            data=loadmat(mat,variable_names=['ROI_N','dat'])
            rois=np.asarray(_field(data['ROI_N'],'IMAGE')).reshape(-1)
            times=np.asarray(_field(data['ROI_N'],'Time')).reshape(-1)
            dat=data['dat']
            sizes=np.asarray(_field(dat,'len')).reshape(-1)
            focus=np.asarray(_field(_field(dat,'foc'),'focus')).reshape(-1)
            if not (len(rois)==len(times)==len(sizes)==len(focus)):
                raise ValueError(f'inconsistent ROI/stat counts: {len(rois)}, {len(times)}, {len(sizes)}, {len(focus)}')
            try:
                opts=_field(dat,'proc_options')
                opts=str(_single(opts))
            except (KeyError,TypeError):
                opts=''
        except Exception as exc:
            errors.append({'file':mat.name,'error':str(exc)})
            print(f'WARNING: skipping {mat.name}: {exc}',flush=True)
            continue
        # The existing CPI statistics are one detection per recorded ROI.
        for i in range(len(rois)):
            source_id=f'{_safe(project)}_{_safe(mat.stem)}_roi{i+1:07d}_p01'
            dt=_dt_from_matlab(times[i]); date=dt.date().isoformat() if dt else 'undated'
            flight=_choose_flight(dt,mat.stem,rows)
            folder=project_dir / (_safe(date)+'_'+_safe(flight) if flight else _safe(date))
            png=folder/(source_id+'.png')
            try:
                size=float(sizes[i]); foc=float(focus[i])
            except (TypeError,ValueError):
                size=foc=float('nan')
            eligible=(math.isfinite(size) and size>=min_size_um and
                      (min_focus is None or (math.isfinite(foc) and foc>=min_focus)))
            status='filtered' if not eligible else 'pending'
            error=''
            if eligible:
                if png.exists() and skip_existing and not overwrite:
                    status='exists';existing+=1
                else:
                    try:
                        # MAT currently holds only one particle contour per ROI. No overlay
                        # is added unless the source explicitly stores multiple detections.
                        arr=_png_bytes_array(rois[i]); img=_encode_image(arr,None)
                        _write_png(img,png)
                        status='exported';written+=1
                    except Exception as exc:
                        status='error';error=str(exc)
            else:
                filtered+=1
            catalogue.append(dict(source_id=source_id,project=project,flight=flight,date=date,
                 timestamp=dt.isoformat(timespec='milliseconds') if dt else '',source_mat=mat.name,
                 source_roi=mat.stem+'.roi',roi_index=i+1,particle_index=1,
                 size_um=size if math.isfinite(size) else '',focus=foc if math.isfinite(foc) else '',
                 eligible=int(eligible),export_status=status,
                 png_relative_path=png.relative_to(root).as_posix() if status in ('exported','exists') else '',
                 processing_options=opts,error=error))
        print(f'{mat.name}: {len(rois)} ROIs',flush=True)
    project_dir.mkdir(parents=True,exist_ok=True)
    catalog_path=project_dir/'particle_catalogue.csv'
    fd,tmp=tempfile.mkstemp(prefix='.catalogue_',suffix='.csv',dir=project_dir)
    os.close(fd)
    try:
        with open(tmp,'w',newline='',encoding='utf-8') as f:
            writer=csv.DictWriter(f,fieldnames=_FIELDS)
            writer.writeheader();writer.writerows(catalogue)
        os.replace(tmp,catalog_path)
    finally:
        if os.path.exists(tmp):os.unlink(tmp)
    error_file=project_dir/'export_errors.json'
    error_file.write_text(json.dumps(errors,indent=2)+'\n',encoding='utf-8')
    print(f'Export complete: {written} written, {existing} already present, {filtered} filtered, '
          f'{len(errors)} MAT files unreadable; catalogue={catalog_path}',flush=True)
    return dict(written=written,existing=existing,filtered=filtered,errors=errors,catalogue=str(catalog_path))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('path',help='Directory of previously processed ROI MAT files')
    p.add_argument('--png-dir',required=True,help='PNG output root')
    p.add_argument('--project',help='Campaign name (default: input directory name)')
    p.add_argument('--flight-csv',help='Flight mapping CSV with date,flight and optional time windows')
    p.add_argument('--min-size-um',type=float,default=0.,help='Minimum CPI measured maximum length in µm')
    p.add_argument('--min-focus',type=float,help='Minimum existing CPI focus value (higher is better)')
    p.add_argument('--overwrite',action='store_true',help='Regenerate existing images')
    args=p.parse_args()
    if args.min_size_um < 0 or (args.min_focus is not None and not math.isfinite(args.min_focus)):
        p.error('Invalid size/focus threshold')
    export(args.path,args.png_dir,args.project,args.flight_csv,args.min_size_um,args.min_focus,overwrite=args.overwrite)

if __name__=='__main__':main()
