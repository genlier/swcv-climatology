"""QC-gated, resumable three-member ua extraction; preserves original wind outputs."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
import json
import numpy as np
import pandas as pd
import xarray as xr
from extract_cmip6_wnpac_wind_ncss import download, area_mean
from download_era5_model_level_sigma_full import sha256
ROOT=Path(__file__).resolve().parents[1]
FILES=ROOT/'data/manifests/cmip6_multimember_ua_revision_files.csv'
PARTS=ROOT/'data/processed/cmip6_multimember_ua_revision_parts'
SUBSETS=ROOT/'data/raw/cmip6_multimember_ua_revision_subsets'
STATUS=ROOT/'data/processed/cmip6_multimember_ua_revision_extraction.json'
PERIODS=[(1951,1980),(1985,2014)]
YEARS=[y for first,last in PERIODS for y in range(first,last+1)]
METRICS=['north_u850_ms','south_u850_ms','wnpac_u850_ms']

def valid_part(path):
    if not path.exists(): return False
    try:
        data=pd.read_csv(path)
        return data.year.tolist()==YEARS and bool(np.isfinite(data[METRICS]).all().all())
    except Exception: return False

def extract(keys,group):
    model,experiment,member=keys
    output=PARTS/f'{model}_{experiment}_{member}.csv'
    if valid_part(output): return {'model':model,'experiment':experiment,'member':member,'path':output.relative_to(ROOT).as_posix(),'sha256':sha256(output)}
    pieces=[];provenance=[]
    for row in group.sort_values('filename').itertuples():
        for first,last in PERIODS:
            start=max(first,int(row.file_start_year));end=min(last,int(row.file_end_year))
            if start>end: continue
            for begin in range(start,end+1,10):
                finish=min(begin+9,end)
                target=SUBSETS/model/experiment/member/f'{Path(row.filename).stem}__{begin}_{finish}.nc'
                target.parent.mkdir(parents=True,exist_ok=True)
                query={'var':'ua','time_start':f'{begin}-01-01T00:00:00Z','time_end':f'{finish}-12-30T23:59:59Z',
                    'accept':'netcdf4','addLatLon':'true','north':32.5,'south':5,'west':90,'east':140,'vertCoord':85000}
                if not target.exists():
                    partial=target.with_suffix('.nc.part');download(row.opendap_url.replace('/dodsC/','/ncss/grid/'),query,partial);partial.replace(target)
                coder=xr.coders.CFDatetimeCoder(use_cftime=True)
                with xr.open_dataset(target,decode_times=coder) as ds:
                    for attr,value in [('source_id',model),('experiment_id',experiment),('variant_label',member)]:
                        if ds.attrs.get(attr) not in (None,value): raise RuntimeError(f'Identity mismatch {attr}')
                    field=ds.ua
                    units=field.attrs.get('units','').replace(' ','').replace('**','').replace('^','')
                    if units not in ['ms-1','m/s']: raise RuntimeError(f'Unsupported ua units: {units}')
                    plev=next((x for x in ['plev','lev'] if x in field.coords),None)
                    if plev is None or not np.isclose(np.asarray(field[plev]),85000).any(): raise RuntimeError('Exact 850 hPa missing')
                    field=field.sel({plev:85000})
                    field=field.sel(time=field.time.dt.month.isin([6,7,8]))
                    north=area_mean(field,22.5,32.5,110,140).load();south=area_mean(field,5,15,90,130).load()
                    pieces.append(pd.DataFrame({'year':north.time.dt.year.values,'month':north.time.dt.month.values,
                        'north_u850_ms':north.values,'south_u850_ms':south.values,'wnpac_u850_ms':(north-south).values}))
                provenance.append({'path':target.relative_to(ROOT).as_posix(),'sha256':sha256(target),'request':query,'opendap_url':row.opendap_url})
                print(f'{keys}: verified {begin}-{finish}',flush=True)
    monthly=pd.concat(pieces,ignore_index=True).sort_values(['year','month'])
    expected={(year,month) for year in YEARS for month in [6,7,8]}
    if monthly.duplicated(['year','month']).any() or set(zip(monthly.year,monthly.month))!=expected: raise RuntimeError('Duplicate or incomplete JJA months')
    if not np.isfinite(monthly[METRICS]).all().all(): raise RuntimeError('Non-finite regional indices')
    annual=monthly.groupby('year')[METRICS].mean().reset_index()
    annual.insert(0,'member',member);annual.insert(0,'experiment',experiment);annual.insert(0,'model',model)
    annual.to_csv(output,index=False)
    output.with_suffix('.provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    assert valid_part(output)
    return {'model':model,'experiment':experiment,'member':member,'path':output.relative_to(ROOT).as_posix(),'sha256':sha256(output)}

def main():
    resolved=json.loads((ROOT/'data/processed/cmip6_multimember_ua_revision_resolution.json').read_text())
    if resolved['status']!='resolved' or resolved['completed_groups']!=48: raise RuntimeError('Complete 48-group resolution required')
    files=pd.read_csv(FILES); groups=list(files.groupby(['model','experiment','member']))
    assert len(groups)==48
    PARTS.mkdir(parents=True,exist_ok=True)
    status={'status':'extracting','expected_groups':48,'files':[],'failures':[]}
    def checkpoint():
        temporary=STATUS.with_suffix('.json.part');temporary.write_text(json.dumps(status,indent=2)+'\n');temporary.replace(STATUS)
    checkpoint()
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures={pool.submit(extract,keys,group):keys for keys,group in groups}
        for future in as_completed(futures):
            try: status['files'].append(future.result())
            except Exception as exc: status['failures'].append({'group':list(futures[future]),'error':f'{type(exc).__name__}: {exc}'});print(f'FAILED {futures[future]}: {exc}',flush=True)
            checkpoint()
    status['status']='complete' if not status['failures'] else 'incomplete';checkpoint()
    if status['failures']: raise SystemExit(2)

if __name__=='__main__':main()
