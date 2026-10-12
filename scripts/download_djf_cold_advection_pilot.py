"""Resumable all-date 700-hPa temperature retrieval for DJF 2011."""
import calendar
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import cdsapi
import pandas as pd
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'data/raw/era5_djf_temperature_pilot'
MANIFEST = ROOT / 'data/manifests/era5_djf_temperature_pilot_manifest.json'
MONTHS = [(2010, 12), (2011, 1), (2011, 2)]
AREA = [38, 100, 24, 115]


def canonical(ds):
    mapping = {old: new for old, new in [('valid_time','time'),('pressure_level','level'),('latitude','lat'),('longitude','lon')] if old in ds.coords}
    ds = ds.rename(mapping)
    if 'level' in ds.dims:
        ds = ds.sel(level=700, drop=True)
    return ds.sortby('lat').sortby('lon')


def validate(path, year, month):
    if not path.exists():
        return False
    try:
        with xr.open_dataset(path) as raw:
            ds = canonical(raw)
            expected = pd.date_range(f'{year}-{month:02d}-01', periods=calendar.monthrange(year,month)[1]*4, freq='6h')
            return ('t' in ds and pd.DatetimeIndex(ds.time.values).equals(expected)
                    and float(ds.lat.min()) == AREA[2] and float(ds.lat.max()) == AREA[0]
                    and float(ds.lon.min()) == AREA[1] and float(ds.lon.max()) == AREA[3]
                    and ds.sizes['lat']==57 and ds.sizes['lon']==61
                    and not bool(ds.t.isnull().any()))
    except Exception:
        return False


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    entries=[]
    for year, month in MONTHS:
        target=OUT/f'era5_t700_{year}{month:02d}.nc'
        partial=target.with_suffix('.nc.part')
        request={'product_type':['reanalysis'],'variable':['temperature'],'pressure_level':['700'],
                 'year':[str(year)],'month':[f'{month:02d}'],
                 'day':[f'{d:02d}' for d in range(1,calendar.monthrange(year,month)[1]+1)],
                 'time':['00:00','06:00','12:00','18:00'],'area':AREA,'grid':[.25,.25],
                 'data_format':'netcdf','download_format':'unarchived'}
        if not validate(target, year, month):
            if validate(partial, year, month):
                partial.replace(target)
            else:
                print(f'request temperature {year}-{month:02d}',flush=True)
                cdsapi.Client().retrieve('reanalysis-era5-pressure-levels',request,str(partial))
                if not validate(partial, year, month):
                    raise RuntimeError(f'Temperature payload failed validation: {partial}')
                partial.replace(target)
        entries.append({'year':year,'month':month,'path':target.relative_to(ROOT).as_posix(),
                        'bytes':target.stat().st_size,'sha256':hashlib.sha256(target.read_bytes()).hexdigest(),
                        'request':request})
        print(f'verified {target.name}',flush=True)
    manifest={'dataset':'reanalysis-era5-pressure-levels','retrieved_at_utc':datetime.now(timezone.utc).isoformat(),
              'scope':'all DJF 2011 dates, not event-only dates','files':entries,'failures':[]}
    MANIFEST.write_text(json.dumps(manifest,indent=2),encoding='utf-8')


if __name__=='__main__':
    main()
