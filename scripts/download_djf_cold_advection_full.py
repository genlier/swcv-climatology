"""Checkpointed, three-worker temperature-only DJF retrieval with boundary pads."""
import argparse
import calendar
import hashlib
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import cdsapi
import numpy as np
import pandas as pd
import xarray as xr

from download_djf_cold_advection_pilot import canonical, AREA

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'data/raw/era5_djf_temperature_full'
MANIFEST=ROOT/'data/manifests/era5_djf_temperature_full_manifest.json'
NC_LOCK=threading.Lock()


def requests_for_winters(first=1980,last=2023):
    result={}
    for winter in range(first,last+1):
        for year,month in [(winter-1,11),(winter-1,12),(winter,1),(winter,2),(winter,3)]:
            days=[29,30] if month==11 else [1,2] if month==3 else list(range(1,calendar.monthrange(year,month)[1]+1))
            result[year,month]=days
    return result


def expected_times(year,month,days):
    return pd.DatetimeIndex([pd.Timestamp(year,month,day,hour) for day in days for hour in [0,6,12,18]]).as_unit('ns')


def validate(path,year,month,days):
    if not path.exists() or path.stat().st_size==0:
        return False
    try:
        with NC_LOCK, xr.open_dataset(path) as raw:
            ds=canonical(raw)
            return ('t' in ds and pd.DatetimeIndex(ds.time.values).equals(expected_times(year,month,days))
                    and ds.sizes['lat']==57 and ds.sizes['lon']==61
                    and np.array_equal(ds.lat.values,np.arange(24,38.01,.25))
                    and np.array_equal(ds.lon.values,np.arange(100,115.01,.25))
                    and bool(np.isfinite(ds.t).all()))
    except Exception:
        return False


def retrieve(year,month,days):
    target=OUT/f'era5_t700_{year}{month:02d}.nc'
    pilot=ROOT/f'data/raw/era5_djf_temperature_pilot/era5_t700_{year}{month:02d}.nc'
    request={'product_type':['reanalysis'],'variable':['temperature'],'pressure_level':['700'],
             'year':[str(year)],'month':[f'{month:02d}'],'day':[f'{d:02d}' for d in days],
             'time':['00:00','06:00','12:00','18:00'],'area':AREA,'grid':[.25,.25],
             'data_format':'netcdf','download_format':'unarchived'}
    if validate(pilot,year,month,days):
        target=pilot
    elif target.exists() and not validate(target,year,month,days):
        raise RuntimeError(f'Existing raw file invalid; preserved for inspection: {target}')
    elif not validate(target,year,month,days):
        partial=target.with_suffix('.nc.part')
        if validate(partial,year,month,days):
            partial.replace(target)
        else:
            if partial.exists():
                partial.rename(partial.with_name(partial.name+f'.invalid_{time.time_ns()}'))
            print(f'request {year}-{month:02d}: {len(days)} dates',flush=True)
            for attempt in range(1,4):
                try:
                    cdsapi.Client().retrieve('reanalysis-era5-pressure-levels',request,str(partial))
                    if not validate(partial,year,month,days):
                        raise RuntimeError(f'Invalid downloaded payload: {partial}')
                    partial.replace(target)
                    break
                except Exception:
                    if validate(partial,year,month,days):
                        partial.replace(target)
                        break
                    if attempt==3:
                        raise
                    if partial.exists():
                        partial.rename(partial.with_name(partial.name+f'.invalid_{time.time_ns()}'))
                    time.sleep(10*attempt)
    entry={'year':year,'month':month,'days':days,'request':request,
           'path':target.relative_to(ROOT).as_posix(),'bytes':target.stat().st_size,
           'sha256':hashlib.sha256(target.read_bytes()).hexdigest()}
    print(f'verified {year}-{month:02d}',flush=True)
    return entry


def write_manifest(entries,failures,requests,status,first,last):
    value={'dataset':'reanalysis-era5-pressure-levels','updated_at_utc':datetime.now(timezone.utc).isoformat(),
           'status':status,'winters':[first,last],'expected_files':len(requests),
           'expected_dates':sum(map(len,requests.values())),
           'files':sorted(entries,key=lambda e:(e['year'],e['month'])),
           'failures':failures,'scope':'full DJF plus two-day boundary pads; temperature only'}
    temp=MANIFEST.with_suffix('.json.tmp')
    temp.write_text(json.dumps(value,indent=2),encoding='utf-8')
    temp.replace(MANIFEST)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--first',type=int,default=1980)
    parser.add_argument('--last',type=int,default=2023)
    parser.add_argument('--workers',type=int,default=3)
    args=parser.parse_args()
    if not 1980<=args.first<=args.last<=2023 or not 1<=args.workers<=3:
        parser.error('Winter range 1980-2023 and workers 1-3 required')
    OUT.mkdir(parents=True,exist_ok=True)
    requests=requests_for_winters(args.first,args.last)
    entries=[]; failures=[]
    write_manifest(entries,failures,requests,'running',args.first,args.last)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures={pool.submit(retrieve,y,m,d):(y,m) for (y,m),d in requests.items()}
        for future in as_completed(futures):
            year,month=futures[future]
            try:
                entries.append(future.result())
            except Exception as exc:
                failures.append({'year':year,'month':month,'error':str(exc)})
                print(f'failed {year}-{month:02d}: {exc}',flush=True)
            write_manifest(entries,failures,requests,'running',args.first,args.last)
    write_manifest(entries,failures,requests,'failed' if failures else 'complete',args.first,args.last)
    if failures:
        raise SystemExit(2)


if __name__=='__main__':
    main()
