"""Small NOAA NCSS annual gauge subsets; resumable, validated, no credentials."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import io
import json
from pathlib import Path
import threading
import time

import numpy as np
import pandas as pd
import requests
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / 'data/raw/cpc_daily_validation'
MANIFEST = ROOT / 'data/manifests/cpc_daily_validation_manifest.json'
LOCK = threading.Lock()
BASE = 'https://psl.noaa.gov/thredds/ncss/grid/Datasets/cpc_global_precip/'
DOCBASE = 'https://ftp.cpc.ncep.noaa.gov/precip/CPC_UNI_PRCP/GAUGE_GLB/DOCU/'


def required_dates(year):
    events = pd.read_csv(ROOT / 'data/processed/events_season_year_terrain_qc.csv.gz', parse_dates=['genesis_time', 'lysis_time'])
    events = events[events.season.eq('DJF') & events.season_year.between(1980, 2023) & events.track_cluster.eq(1)]
    dates = set()
    for ev in events.itertuples():
        start = pd.Timestamp(ev.genesis_time).tz_localize(None).normalize() - pd.Timedelta(days=1)
        end = pd.Timestamp(ev.lysis_time).tz_localize(None) + pd.Timedelta(hours=24) - pd.Timedelta(nanoseconds=1)
        for date in pd.date_range(start, end.normalize() + pd.Timedelta(days=1)):
            if date.year == year:
                dates.add(date)
    return pd.DatetimeIndex(sorted(dates))


def validate(path, year):
    with LOCK, xr.open_dataset(path) as ds:
        expected = required_dates(year)
        actual = pd.DatetimeIndex(ds.time.values)
        if not actual.is_unique or not actual.is_monotonic_increasing or not expected.isin(actual).all():
            raise RuntimeError(f'Incomplete/duplicate daily dates: {year}')
        if ds.precip.attrs.get('units') not in ['mm', 'mm/day']:
            raise RuntimeError('Unrecognised precipitation units')
        for name in ['lat', 'lon']:
            if not np.allclose(np.abs(np.diff(ds[name].values)), .5):
                raise RuntimeError(f'Invalid 0.5 degree {name}')
        if float(ds.lat.min()) > 25.25 or float(ds.lat.max()) < 33.75 or float(ds.lon.min()) > 102.25 or float(ds.lon.max()) < 119.75:
            raise RuntimeError('Incomplete regional domain')
        attrs = dict(ds.attrs)
    return {'year': year, 'path': str(path.relative_to(ROOT)), 'bytes': path.stat().st_size,
            'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'days': len(actual),
            'required_dates': [d.strftime('%Y-%m-%d') for d in expected],
            'global_attributes': {k: str(v) for k, v in attrs.items()}}


def fetch(year):
    path = RAW / f'cpc_precip_{year}.nc'
    params = {'var': 'precip', 'north': 34, 'south': 25, 'west': 102, 'east': 120,
              'horizStride': 1, 'time_start': f'{year}-01-01T00:00:00Z',
              'time_end': f'{year}-12-31T23:59:59Z', 'timeStride': 1, 'accept': 'netcdf4'}
    url = BASE + f'precip.{year}.nc'
    if path.exists():
        result = validate(path, year)
        result.update(url=url, parameters=params, reused=True)
        return result
    part = path.with_suffix('.nc.part')
    if part.exists():
        try:
            result = validate(part, year)
            part.replace(path)
            result.update(path=str(path.relative_to(ROOT)), url=url, parameters=params, reused=True)
            return result
        except Exception:
            part.rename(part.with_name(part.name + f'.invalid_{time.time_ns()}'))
    # Annual NCSS requests timed out in the pilot. Freeze exactly the same
    # annual data, but fetch monthly NetCDF3 pieces and preserve checkpoints.
    pieces = []; monthly_sources = []
    needed = required_dates(year)
    for month in range(1, 13):
        monthdates = needed[needed.month == month]
        if not len(monthdates):
            continue
        start = monthdates.min()
        finish = monthdates.max()
        monthpath = RAW / f'cpc_precip_{year}_{month:02d}.nc'
        request = {**params, 'time_start': start.strftime('%Y-%m-%dT00:00:00Z'),
                   'time_end': finish.strftime('%Y-%m-%dT00:00:00Z'), 'accept': 'netcdf3'}
        expected = pd.date_range(start, finish)
        def read_piece():
            with LOCK, xr.open_dataset(monthpath) as source:
                ds = source.load()
            if not pd.DatetimeIndex(ds.time.values).is_unique or not expected.isin(pd.DatetimeIndex(ds.time.values)).all():
                raise RuntimeError('Invalid monthly dates')
            return ds
        if monthpath.exists():
            ds = read_piece()
        else:
            for attempt in range(3):
                try:
                    response = requests.get(url, params=request, timeout=(15, 45))
                    response.raise_for_status()
                    with xr.open_dataset(io.BytesIO(response.content), engine='scipy') as source:
                        ds = source.load()
                    if not pd.DatetimeIndex(ds.time.values).equals(expected):
                        raise RuntimeError('Monthly response dates incomplete')
                    monthpart = monthpath.with_suffix('.nc.part')
                    monthpart.write_bytes(response.content)
                    monthpart.replace(monthpath)
                    print(f'CPC {year}-{month:02d} monthly checkpoint', flush=True)
                    break
                except Exception:
                    if attempt == 2:
                        raise
                    time.sleep(2)
        pieces.append(ds)
        monthly_sources.append({'url': url, 'parameters': request, 'path': str(monthpath.relative_to(ROOT)),
                                'sha256': hashlib.sha256(monthpath.read_bytes()).hexdigest()})
    combined = xr.concat(pieces, dim='time').sortby('time')
    # netCDF4 writes its own reserved properties attribute; the producer's
    # properties remain in monthly sources, not re-applied to a new container.
    combined.attrs.pop('_NCProperties', None)
    # Server-specific attributes from the first piece are retained; exact
    # per-month requests/checksums, rather than inherited coverage attrs, audit time.
    with LOCK:
        combined.to_netcdf(part, engine='netcdf4', encoding={'precip': {'zlib': True, 'complevel': 2}})
    result = validate(part, year)
    part.replace(path)
    result.update(path=str(path.relative_to(ROOT)), url=url, parameters=params,
                  reused=False, transfer='monthly_netcdf3_assembled_annual', monthly_sources=monthly_sources)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pilot', action='store_true')
    args = parser.parse_args()
    RAW.mkdir(parents=True, exist_ok=True)
    # Preserve producer documentation and the uncertain EOD table for audit.
    for name in ['PRCP_CU_GAUGE_V1.0GLB_0.50deg_EOD.lnx', 'PRCP_CU_GAUGE_V1.0GLB_0.50deg_EOD.lnx.ctl', 'PRCP_CU_GAUGE_V1.0GLB_0.50deg_README.txt']:
        path = RAW / name
        if not path.exists():
            response = requests.get(DOCBASE + name, timeout=(15, 60))
            response.raise_for_status()
            path.write_bytes(response.content)
    eod = np.frombuffer((RAW / 'PRCP_CU_GAUGE_V1.0GLB_0.50deg_EOD.lnx').read_bytes(), dtype='<f4').reshape(360, 720)
    lat = np.arange(-89.75, 90, .5); lon = np.arange(.25, 360, .5)
    eod_region = np.unique(eod[np.ix_((lat >= 25) & (lat <= 34), (lon >= 102) & (lon <= 120))]).tolist()
    if eod_region != [24.0]:
        raise RuntimeError(f'Unexpected EOD; do not silently change alignment: {eod_region}')
    years = [2011] if args.pilot else list(range(1979, 2024))
    target = MANIFEST.with_name('cpc_daily_validation_pilot_manifest.json') if args.pilot else MANIFEST
    manifest = {'status': 'running', 'expected_years': years, 'files': [], 'failures': [], 'eod_region_hours': eod_region,
                'scope': 'All whole-day event windows plus both +/-1-day label shifts; monthly enclosing intervals, not full annual backgrounds',
                'documentation': [{'path': str(p.relative_to(ROOT)), 'url': DOCBASE + p.name,
                                   'sha256': hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(RAW.glob('*.lnx*')) + sorted(RAW.glob('*.txt'))]}
    def checkpoint():
        temp = target.with_suffix('.json.part')
        temp.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
        temp.replace(target)
    checkpoint()
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = {executor.submit(fetch, year): year for year in years}
        for future in as_completed(futures):
            year = futures[future]
            try:
                manifest['files'].append(future.result())
                print(f'CPC {year} verified ({len(manifest["files"])}/{len(years)})', flush=True)
            except Exception as exc:
                manifest['failures'].append({'year': year, 'error': str(exc)})
                print(f'CPC {year} failed: {exc}', flush=True)
            checkpoint()
    manifest['files'].sort(key=lambda e: e['year'])
    manifest['status'] = 'failed' if manifest['failures'] else 'complete'
    checkpoint()
    if manifest['failures']:
        raise RuntimeError('CPC requests failed; inspect checkpoint, do not analyse incomplete data')


if __name__ == '__main__':
    main()
