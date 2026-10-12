"""Isolated, resumable post-review ERA5 wind/temperature profile download."""
from pathlib import Path
import argparse
import json
from datetime import datetime, timezone
from concurrent.futures import ProcessPoolExecutor, as_completed
import cdsapi
import numpy as np
import pandas as pd
import xarray as xr
from download_era5_model_level_sigma_full import selection, canonical, sha256, AREA, GRID

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'data/raw/era5_vertical_revision_djf'
MANIFEST=ROOT/'data/manifests/era5_vertical_revision_djf_manifest.json'
LEVELS=list(range(100,126))
DATASET='reanalysis-era5-complete'

def make_request(dates):
    return {'date':'/'.join(dates),'levelist':'/'.join(map(str,LEVELS)),
        'levtype':'ml','param':'130/131/132','stream':'oper','time':'00/to/18/by/6',
        'type':'an','area':'/'.join(map(str,AREA)),'grid':'/'.join(map(str,GRID)),
        'format':'netcdf'}

def validate(path, dates):
    if not path.exists(): return False
    try:
        with xr.open_dataset(path) as raw:
            ds=canonical(raw)
            assert {'u','v','t'}.issubset(ds.data_vars)
            assert set(np.asarray(ds.level).astype(int))==set(LEVELS)
            times=pd.DatetimeIndex(ds.time.values)
            assert times.is_unique and len(times)==4*len(dates)
            assert set(times.strftime('%Y-%m-%d'))==set(dates)
            assert set(times.hour)=={0,6,12,18}
            assert float(ds.lat.max())>=AREA[0] and float(ds.lat.min())<=AREA[2]
            assert float(ds.lon.min())<=AREA[1] and float(ds.lon.max())>=AREA[3]
            for name in ('u','v','t'):
                assert bool(np.isfinite(ds[name]).all()), f'non-finite {name}'
            assert bool((abs(ds.u)<250).all() & (abs(ds.v)<250).all()), 'Invalid wind range or encoded missing sentinel'
            assert bool(((ds.t>150)&(ds.t<350)).all()), 'Invalid temperature range or encoded missing sentinel'
        return True
    except Exception as exc:
        print(f'validation failed {path.name}: {exc}',flush=True)
        return False

def retrieve_year(year,dates,max_dates_per_request):
    """Retrieve and validate one calendar year in an isolated worker process."""
    target=OUT/f'era5_vertical_revision_dates_{year}.nc'
    partial=target.with_suffix('.nc.part')
    date_chunks=[dates[i:i+max_dates_per_request]
        for i in range(0,len(dates),max_dates_per_request)]
    requests=[make_request(chunk_dates) for chunk_dates in date_chunks]
    if not validate(target,dates):
        if validate(partial,dates):
            partial.replace(target)
        else:
            partial.unlink(missing_ok=True)
            chunk_dir=OUT/'_chunks'/str(year)
            chunk_dir.mkdir(parents=True,exist_ok=True)
            chunk_paths=[]
            for index,(chunk_dates,request) in enumerate(zip(date_chunks,requests),start=1):
                chunk_path=chunk_dir/f'chunk_{index:02d}.nc'
                chunk_partial=chunk_path.with_suffix('.nc.part')
                if not validate(chunk_path,chunk_dates):
                    if validate(chunk_partial,chunk_dates):
                        chunk_partial.replace(chunk_path)
                    else:
                        chunk_partial.unlink(missing_ok=True)
                        print(f'request {year} chunk {index}/{len(date_chunks)}: '
                            f'{len(chunk_dates)} dates, 26 levels, u/v/t',flush=True)
                        cdsapi.Client().retrieve(DATASET,request,str(chunk_partial))
                        if not validate(chunk_partial,chunk_dates):
                            raise RuntimeError(f'{year} chunk {index} payload failed QC')
                        chunk_partial.replace(chunk_path)
                chunk_paths.append(chunk_path)
            if len(chunk_paths)==1:
                chunk_paths[0].replace(partial)
            else:
                datasets=[]
                for chunk_path in chunk_paths:
                    with xr.open_dataset(chunk_path) as raw:
                        datasets.append(canonical(raw).load())
                combined=xr.concat(datasets,dim='time').sortby('time')
                combined.to_netcdf(partial)
                combined.close()
                for dataset in datasets:
                    dataset.close()
            if not validate(partial,dates):
                raise RuntimeError(f'{year} assembled payload failed QC')
            partial.replace(target)
    if not validate(target,dates):
        raise RuntimeError(f'{year} stored payload failed QC')
    result={'year':year,'dates':dates,'bytes':target.stat().st_size,
        'sha256':sha256(target),'path':target.relative_to(ROOT).as_posix(),
        'requests':requests,'max_dates_per_request':max_dates_per_request}
    print(f'verified {year}: {target.stat().st_size} bytes',flush=True)
    return result

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--year',type=int,help='Pilot calendar year; omit for all frozen dates')
    parser.add_argument('--workers',type=int,choices=[1,2],default=1,
        help='Independent year-level worker processes; maximum 2 for CDS stability')
    parser.add_argument('--max-dates-per-request',type=int,default=20,
        help='Split large calendar-year requests to stay below CDS cost limits')
    parser.add_argument('--plan-only',action='store_true')
    args=parser.parse_args()
    if args.max_dates_per_request<1:
        parser.error('--max-dates-per-request must be positive')
    points,years=selection()
    if args.year is not None:
        if args.year not in years: raise ValueError('Year outside frozen selection')
        years={args.year:years[args.year]}
    plan={'dataset':DATASET,'levels':LEVELS,'variables':['u','v','t'],
        'scope':'existing frozen 899-event winter cohort; calendar dates only',
        'planned_years':sorted(years),'planned_dates':sum(map(len,years.values())),
        'event_cohort':int(points.event_id.nunique()),'area_nwse':AREA,'grid_degrees':GRID,
        'status':'planned','workers':args.workers,'files':[],
        'max_dates_per_request':args.max_dates_per_request,
        'purpose':'Vertical coordinate/shear/stability diagnostics; not a complete boundary-layer heat budget'}
    if args.plan_only:
        print(json.dumps(plan,indent=2)); return
    OUT.mkdir(parents=True,exist_ok=True)
    if MANIFEST.exists():
        previous=json.loads(MANIFEST.read_text())
        plan['files']=previous.get('files',[])
    def checkpoint():
        plan['updated_at_utc']=datetime.now(timezone.utc).isoformat()
        temporary=MANIFEST.with_suffix('.json.part')
        temporary.write_text(json.dumps(plan,indent=2)+'\n')
        temporary.replace(MANIFEST)
    checkpoint()
    plan['status']='downloading';plan['active_years']=[];plan['failures']=[];checkpoint()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        work=sorted(years.items(),key=lambda item:(
            (OUT/f'era5_vertical_revision_dates_{item[0]}.nc').exists(),item[0]))
        futures={pool.submit(retrieve_year,year,dates,args.max_dates_per_request):year
            for year,dates in work}
        order={year:index for index,(year,_) in enumerate(work)}
        plan['active_years']=[year for year,_ in work[:args.workers]];checkpoint()
        pending=set(futures.values())
        for future in as_completed(futures):
            year=futures[future];pending.remove(year)
            try:
                result=future.result()
                plan['files']=[x for x in plan['files'] if x['year']!=year]
                plan['files'].append(result);plan['files'].sort(key=lambda x:x['year'])
            except Exception as exc:
                plan['failures'].append({'year':year,'error':f'{type(exc).__name__}: {exc}'})
                print(f'failed {year}: {exc}',flush=True)
            plan['active_years']=sorted(pending,key=order.get)[:args.workers];checkpoint()
    plan['status']='selected_scope_complete' if not plan['failures'] else 'incomplete'
    plan.pop('active_years',None);checkpoint()
    if plan['failures']: raise SystemExit(2)

if __name__=='__main__':main()
