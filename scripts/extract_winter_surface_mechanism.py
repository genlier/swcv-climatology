"""Extract predeclared continuous basin/west-flank surface diagnostics."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import xarray as xr
from download_winter_surface_mechanism import canonical,validate,sha
from extract_winter_local_convergence import spherical_terms

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'data/processed'


def regional(field,lat0=28,lat1=32,lon0=104,lon1=108):
    selected=field.sel(lat=slice(lat0,lat1),lon=slice(lon0,lon1))
    return selected.weighted(np.cos(np.deg2rad(selected.lat))).mean(['lat','lon'])


def main():
    manifest=json.loads((ROOT/'data/manifests/winter_surface_mechanism_manifest.json').read_text())
    if manifest['status']!='complete' or len(manifest['files'])!=45:
        raise RuntimeError('Expected complete 45-year download')
    pieces=[]
    for entry in manifest['files']:
        path=ROOT/entry['path']
        if not validate(path,entry['year']) or sha(path)!=entry['sha256']:
            raise RuntimeError(f'Surface source validation failed: {path.name}')
        with xr.open_dataset(path) as raw:ds=canonical(raw).sortby('lat').sortby('lon').load()
        core_t=regional(ds.t2m)
        west_t=regional(ds.t2m,lon0=102,lon1=103.75)
        conv,_=spherical_terms(ds.u10.values,ds.v10.values,ds.lat.values,ds.lon.values)
        conv=xr.DataArray(conv,dims=['time','lat','lon'],coords={c:ds[c] for c in ['time','lat','lon']})
        part=pd.DataFrame({'time':ds.time.values,'basin_t2m_k':core_t.values,
                           'west_minus_basin_t2m_k':(west_t-core_t).values,
                           'basin_skin_minus_air_k':regional(ds.skt-ds.t2m).values,
                           'basin_blh_m':regional(ds.blh).values,
                           'basin_dewpoint_depression_k':regional(ds.t2m-ds.d2m).values,
                           'basin_convergence10_s':regional(conv).values})
        part['basin_log_blh']=np.log1p(part.basin_blh_m)
        pieces.append(part)
        print(f'extract near-surface regional series {entry["year"]}',flush=True)
    table=pd.concat(pieces).sort_values('time')
    if table.time.duplicated().any() or not np.isfinite(table.drop(columns='time')).all().all():
        raise RuntimeError('Nonfinite or duplicated surface regional data')
    table.to_csv(OUT/'winter_surface_mechanism_series.csv.gz',index=False)
    (OUT/'winter_surface_mechanism_coverage.json').write_text(json.dumps({
        'years':[1979,2023],'n_times':len(table),'columns':list(table.columns),
        'spatial_definition':'basin 28–32N 104–108E; west flank 28–32N 102–103.75E',
        'limits':'Surface temperature differences are not pressure-gradient or heat-flux estimates; BLH is model diagnostic.'},indent=2),encoding='utf-8')


if __name__=='__main__':main()
