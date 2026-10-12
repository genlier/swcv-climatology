"""Continuous terrain-masked basin convergence and stretching diagnostics."""
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
from analyze_winter_genesis_regimes import canonical,target_times

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'data/processed'
R=6371000.
OMEGA=7.292115e-5


def spherical_terms(u,v,lat,lon):
    phi=np.deg2rad(lat)
    lam=np.deg2rad(lon)
    cosine=np.cos(phi)[None,:,None]
    tangent=np.tan(phi)[None,:,None]
    divergence=(np.gradient(u,lam,axis=2)/(R*cosine)
                +np.gradient(v,phi,axis=1)/R-v*tangent/R)
    vorticity=(np.gradient(v,lam,axis=2)/(R*cosine)
               -np.gradient(u,phi,axis=1)/R+u*tangent/R)
    convergence=-divergence
    stretching=(vorticity+2*OMEGA*np.sin(phi)[None,:,None])*convergence
    return convergence,stretching


def main():
    target=target_times()
    needed=target.union(target-pd.Timedelta(hours=24)).sort_values()
    pieces=[]
    for year in sorted(set(needed.year)):
        times=needed[needed.year==year]
        lat=np.arange(27.75,32.26,.25)
        lon=np.arange(103.75,108.26,.25)
        with xr.open_dataset(ROOT/f'data/raw/era5_surface_pressure_6hourly/era5_surface_pressure_{year}.nc') as raw:
            sp=canonical(raw).sp.sel(time=times,lat=lat,lon=lon).values
        result={'time':times}
        for level in [700,850]:
            path=ROOT/f'data/raw/era5_{level}hpa_6hourly/era5_{level}hpa_{year}.nc'
            with xr.open_dataset(path) as raw:
                ds=raw.rename({a:b for a,b in [('valid_time','time'),('latitude','lat'),('longitude','lon'),('pressure_level','level')] if a in raw.coords})
                ds=ds.sortby('lat').sortby('lon').sel(time=times,lat=lat,lon=lon,level=level)
                u,v=ds.u.values.astype(float),ds.v.values.astype(float)
            safe=sp>=(level+25)*100
            u[~safe]=np.nan
            v[~safe]=np.nan
            conv,stretch=spherical_terms(u,v,lat,lon)
            weight=np.cos(np.deg2rad(lat[1:-1]))[None,:,None]*np.ones((1,len(lat)-2,len(lon)-2))
            for name,field in [('convergence',conv),('stretching',stretch)]:
                f=field[:,1:-1,1:-1]
                valid=np.isfinite(f)
                denom=(valid*weight).sum(axis=(1,2))
                coverage=denom/weight.sum()
                mean=np.nansum(f*weight,axis=(1,2))/np.where(denom>0,denom,np.nan)
                # Strict 80% regional valid-area gate, reported separately.
                mean[coverage<.8]=np.nan
                result[f'{name}_{level}']=mean
                result[f'{name}_{level}_coverage']=coverage
        pieces.append(pd.DataFrame(result))
        print(f'basin convergence/stretching {year}',flush=True)
    table=pd.concat(pieces,ignore_index=True).sort_values('time')
    if not pd.DatetimeIndex(table.time).equals(needed):
        raise RuntimeError('Continuous diagnostic timestamps mismatch')
    table.to_csv(OUT/'winter_local_convergence_series.csv.gz',index=False)
    print(table.describe().to_string(),flush=True)


if __name__=='__main__':main()
