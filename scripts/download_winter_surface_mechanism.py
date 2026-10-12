"""Resumable, modest-concurrency continuous ERA5 near-surface retrieval."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
from datetime import datetime,timezone
import argparse
import calendar
import hashlib
import json
import os
import subprocess
import sys
import threading
import zipfile

import cdsapi
import numpy as np
import pandas as pd
import xarray as xr

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'data/raw/era5_winter_surface_mechanism'
MANIFEST=ROOT/'data/manifests/winter_surface_mechanism_manifest.json'
VARIABLES={'t2m','d2m','skt','blh','u10','v10'}
API_VARIABLES=['2m_temperature','2m_dewpoint_temperature','skin_temperature',
               'boundary_layer_height','10m_u_component_of_wind','10m_v_component_of_wind']
AREA=[34,101,27,109]
# NetCDF/HDF5 builds need not be thread-safe. Network retrieval stays parallel;
# dataset inspection/normalization is serialized for durable restart safety.
NETCDF_LOCK=threading.RLock()


def months(year):
    return [11,12] if year==1979 else [1,2] if year==2023 else [1,2,11,12]


def expected(year):
    return pd.DatetimeIndex(np.concatenate([pd.date_range(f'{year}-{m:02d}-01',
                            periods=calendar.monthrange(year,m)[1]*4,freq='6h').values for m in months(year)]))


def canonical(ds):
    return ds.rename({a:b for a,b in [('valid_time','time'),('latitude','lat'),('longitude','lon')] if a in ds.coords}).sortby('time')


def validate(path,year):
    if not path.exists():return False
    try:
        with NETCDF_LOCK, xr.open_dataset(path) as raw:
            ds=canonical(raw)
            if not VARIABLES.issubset(ds.data_vars):return False
            if not pd.DatetimeIndex(ds.time.values).equals(expected(year)):return False
            if ds.sizes['lat']!=29 or ds.sizes['lon']!=33:return False
            if not np.allclose(np.sort(ds.lat.values),np.arange(27,34.01,.25)):return False
            if not np.allclose(np.sort(ds.lon.values),np.arange(101,109.01,.25)):return False
            return all(np.isfinite(ds[v].values).all() for v in VARIABLES)
    except Exception:return False


def sha(path):
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):digest.update(chunk)
    return digest.hexdigest()


def normalize_download(partial):
    # CDS may return an archive even with unarchived requested; never extract paths.
    if not zipfile.is_zipfile(partial):return
    datasets=[]
    with zipfile.ZipFile(partial) as archive:
        members=[n for n in archive.namelist() if n.endswith('.nc')]
        if not members:raise RuntimeError('CDS archive has no NetCDF payload')
        for number,name in enumerate(members):
            member=partial.with_name(partial.name+f'.member{number}.nc')
            member.write_bytes(archive.read(name))
            with NETCDF_LOCK, xr.open_dataset(member) as ds:datasets.append(ds.load())
    normalized=partial.with_name(partial.name+'.normalized.nc')
    with NETCDF_LOCK:
        merged=xr.merge(datasets,compat='no_conflicts')
        merged.to_netcdf(normalized)
    normalized.replace(partial)


def download(year):
    target=OUT/f'era5_winter_surface_{year}.nc'
    partial=target.with_suffix('.nc.part')
    if not validate(target,year):
        if not validate(partial,year):
            print(f'request surface {year}',flush=True)
            client=cdsapi.Client(timeout=120,retry_max=5,quiet=False)
            client.retrieve('reanalysis-era5-single-levels',
                            {'product_type':['reanalysis'],'variable':API_VARIABLES,'year':[str(year)],
                             'month':[f'{m:02d}' for m in months(year)],'day':[f'{d:02d}' for d in range(1,32)],
                             'time':['00:00','06:00','12:00','18:00'],'area':AREA,'grid':[.25,.25],
                             'data_format':'netcdf','download_format':'unarchived'},str(partial))
            normalize_download(partial)
            if not validate(partial,year):raise RuntimeError(f'Invalid surface file year {year}')
        partial.replace(target)
    print(f'verified surface {year}: {target.stat().st_size} bytes',flush=True)
    return {'year':year,'months':months(year),'path':target.relative_to(ROOT).as_posix(),
            'bytes':target.stat().st_size,'sha256':sha(target)}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--workers',type=int,default=2,choices=[1,2])
    parser.add_argument('--years',type=int,nargs='+')
    args=parser.parse_args()
    years=args.years or list(range(1979,2024))
    if any(y<1979 or y>2023 for y in years):parser.error('Years must be 1979–2023')
    OUT.mkdir(parents=True,exist_ok=True)
    state={'dataset':'reanalysis-era5-single-levels','scope':'winter surface mechanism six-hourly',
           'status':'running','pid':os.getpid(),'requested_years':years,'workers':args.workers,
           'started_utc':datetime.now(timezone.utc).isoformat(),'files':[],'failures':[]}
    def save():
        state['updated_utc']=datetime.now(timezone.utc).isoformat()
        temporary=MANIFEST.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(state,indent=2),encoding='utf-8')
        temporary.replace(MANIFEST)
    save()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        pending={pool.submit(download,y):y for y in years}
        for future in as_completed(pending):
            try:state['files'].append(future.result())
            except Exception as error:
                state['failures'].append({'year':pending[future],'error':str(error)})
                print(f'failed surface {pending[future]}: {type(error).__name__}: {error}',flush=True)
            state['files'].sort(key=lambda x:x['year'])
            save()
    state['status']='complete' if not state['failures'] and len(state['files'])==45 else 'partial'
    save()
    if state['status']=='complete':
        state['processing']='running';save()
        for script in ['extract_winter_surface_mechanism.py','analyze_winter_surface_mechanism.py','check_winter_surface_circulation_controls.py']:
            state['processing_step']=script;save()
            result=subprocess.run([sys.executable,str(ROOT/'scripts'/script)],cwd=ROOT)
            if result.returncode:
                state['processing']='failed';save();raise SystemExit(result.returncode)
        state['processing']='complete';save()
    if state['failures']:raise SystemExit(1)


if __name__=='__main__':main()
