"""Fixed one-degree risk cells and continuous colocated surface/dynamic proxies."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import xarray as xr
from analyze_winter_genesis_regimes import target_times
from extract_winter_local_convergence import spherical_terms
from download_winter_surface_mechanism import canonical,sha

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'data/processed'
LAT=np.arange(26.75,33.26,.25)
LON=np.arange(101.75,109.26,.25)


def cells(lat,lon):
    lat=np.asarray(lat);lon=np.asarray(lon)
    valid=(lat>=27)&(lat<=33)&(lon>=102)&(lon<=109)
    iy=np.clip(np.floor(lat-27).astype(int),0,5)
    ix=np.clip(np.floor(lon-102).astype(int),0,6)
    return np.where(valid,iy*7+ix,-1)


def load(path,times):
    with xr.open_dataset(path) as raw:
        ds=raw.rename({a:b for a,b in [('valid_time','time'),('latitude','lat'),
             ('longitude','lon'),('pressure_level','level')] if a in raw.coords})
        return ds.sortby('lat').sortby('lon').sel(time=times,lat=LAT,lon=LON).load()


def stencil(pressure,level):
    s=pressure>=(level+25)*100
    return s[:,1:-1,1:-1]&s[:,:-2,1:-1]&s[:,2:,1:-1]&s[:,1:-1,:-2]&s[:,1:-1,2:]


def main():
    target=target_times();times=target.union(target-pd.Timedelta(hours=24)).sort_values()
    sy=times.year+(times.month==12).astype(int)
    early=times[(sy<=2001)&times.month.isin([12,1,2])]
    shape=(len(LAT)-2,len(LON)-2)
    totals={lev:np.zeros(shape) for lev in [700,850]}
    for year in sorted(set(early.year)):
        wanted=early[early.year==year]
        sp=load(ROOT/f'data/raw/era5_surface_pressure_6hourly/era5_surface_pressure_{year}.nc',wanted).sp.values
        for lev in totals:totals[lev]+=stencil(sp,lev).sum(axis=0)
    support={lev:totals[lev]/len(early)>=.99 for lev in totals}
    lat,lon=np.meshgrid(LAT[1:-1],LON[1:-1],indexing='ij')
    nodecell=cells(lat,lon)
    # The outside upper boundary is a genesis allocation rule; nodes at 33/109
    # are excluded from averaging so disjoint cell supports contain 16 nodes.
    nodecell[(lat>=33)|(lon>=109)]=-1
    weights=np.cos(np.deg2rad(lat))
    cellnodes={str(lev):[int((support[lev]&(nodecell==c)).sum()) for c in range(42)] for lev in support}
    def average(field,static=None):
        values=np.full((len(field),42),np.nan);coverage=np.zeros_like(values)
        for cell in range(42):
            mask=nodecell==cell
            if static is not None:mask=mask&static
            if mask.sum()<(4 if static is not None else 1):continue
            f=field[:,mask];w=weights[mask]
            denom=(np.isfinite(f)*w).sum(axis=1)
            coverage[:,cell]=denom/w.sum()
            values[:,cell]=np.nansum(f*w,axis=1)/np.where(denom>0,denom,np.nan)
            values[coverage[:,cell]<.8,cell]=np.nan
        return values,coverage
    surface_manifest=json.loads((ROOT/'data/manifests/winter_surface_mechanism_manifest.json').read_text())
    if surface_manifest.get('processing')!='complete':raise RuntimeError('Surface processing not complete')
    surfacefiles={entry['year']:entry for entry in surface_manifest['files']}
    pieces=[];provenance=[]
    for year in sorted(set(times.year)):
        wanted=times[times.year==year]
        spath=ROOT/f'data/raw/era5_surface_pressure_6hourly/era5_surface_pressure_{year}.nc'
        sp=load(spath,wanted).sp.values
        result={}
        for lev in [700,850]:
            path=ROOT/f'data/raw/era5_{lev}hpa_6hourly/era5_{lev}hpa_{year}.nc'
            ds=load(path,wanted).sel(level=lev)
            u,v=ds.u.values.astype(float),ds.v.values.astype(float)
            unsafe=sp<(lev+25)*100
            u[unsafe]=np.nan;v[unsafe]=np.nan
            for name,field in zip(['convergence','stretching'],spherical_terms(u,v,LAT,LON)):
                values,cov=average(field[:,1:-1,1:-1],support[lev])
                result[f'{name}_{lev}']=(('time','cell'),values.astype('float32'))
                result[f'{name}_{lev}_coverage']=(('time','cell'),cov.astype('float32'))
            stat=path.stat();provenance.append({'path':path.relative_to(ROOT).as_posix(),'bytes':stat.st_size,'mtime_ns':stat.st_mtime_ns})
        entry=surfacefiles[year];path=ROOT/entry['path']
        if sha(path)!=entry['sha256']:raise RuntimeError('Surface checksum mismatch')
        with xr.open_dataset(path) as raw:
            surface=canonical(raw).sortby('lat').sortby('lon').sel(time=wanted).load()
        conv,_=spherical_terms(surface.u10.values,surface.v10.values,surface.lat.values,surface.lon.values)
        near={'t2m':surface.t2m,'skin_air':surface.skt-surface.t2m,
              'dewpoint_depression':surface.t2m-surface.d2m,'log_blh':np.log1p(surface.blh),
              'convergence10':xr.DataArray(conv,dims=['time','lat','lon'],coords={a:surface[a] for a in ['time','lat','lon']})}
        for name,field in near.items():
            f=field.sel(lat=LAT[1:-1],lon=LON[1:-1]).values
            values,_=average(f)
            result[name]=(('time','cell'),values.astype('float32'))
        pieces.append(xr.Dataset(result,coords={'time':wanted,'cell':np.arange(42)}))
        print(f'spatial mechanism {year}',flush=True)
    cube=xr.concat(pieces,dim='time').sortby('time')
    if not pd.DatetimeIndex(cube.time.values).equals(times):raise RuntimeError('Spatial time grid mismatch')
    cube['cell_lat']=('cell',27.5+np.arange(42)//7)
    cube['cell_lon']=('cell',102.5+np.arange(42)%7)
    cube.attrs.update(scope='Fixed 1-degree environmental cells; pressure-level stencil safety; observational associations only')
    cube.to_netcdf(OUT/'winter_spatial_mechanism.nc',encoding={name:{'zlib':True,'complevel':4} for name in cube.data_vars if 'time' in cube[name].dims})
    meta=pd.read_csv(OUT/'events_terrain_qc_intensity.csv.gz')
    genesis=pd.to_datetime(meta.genesis_time,utc=True).dt.tz_localize(None)
    season=genesis.dt.year+genesis.dt.month.eq(12).astype(int)
    selected=meta.track_cluster.eq(1)&genesis.dt.month.isin([12,1,2])&season.between(1980,2023)
    events=meta[selected].copy();events['time']=genesis[selected].dt.floor('6h')
    events['cell']=cells(events.genesis_lat,events.genesis_lon)
    counts=events[events.cell.ge(0)].groupby(['time','cell']).size().unstack(fill_value=0).reindex(index=target,columns=np.arange(42),fill_value=0).fillna(0).astype(int)
    if counts.to_numpy().sum()!=int(events.cell.ge(0).sum()):raise RuntimeError('Spatial event count conservation failed')
    xr.Dataset({'events':(('time','cell'),counts.to_numpy().astype('int16'))},coords={'time':target,'cell':np.arange(42)}).to_netcdf(OUT/'winter_spatial_genesis_counts.nc')
    events[['event_id','time','cell','genesis_lat','genesis_lon']].to_csv(OUT/'winter_spatial_genesis_allocation.csv',index=False)
    audit={'scope':'Fixed spatial risk set and colocated environmental proxies','n_times':len(times),'n_cells':42,
        'events_total':len(events),'events_allocated':int(events.cell.ge(0).sum()),
        'excluded_events':events[events.cell.lt(0)].event_id.tolist(),'early_safe_nodes':cellnodes,
        'raw_pressure_provenance':provenance,'surface_source_manifest':'data/manifests/winter_surface_mechanism_manifest.json',
        'limits':['Regional-cell means are not exact centre measurements','Cell grid not an orographic basin polygon',
                  'Pressure-level missingness is terrain and weather dependent','Raw pressure files size/mtime audited, not newly full-hashed']}
    (OUT/'winter_spatial_mechanism_extraction.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in audit.items() if k not in ['raw_pressure_provenance','early_safe_nodes']},indent=2))


if __name__=='__main__':main()
