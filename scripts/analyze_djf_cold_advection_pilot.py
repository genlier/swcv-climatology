"""Event antecedent cold-advection feasibility test, not trend attribution."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from download_djf_cold_advection_pilot import canonical, validate, MONTHS

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'data/processed'
R=6371000.
LAGS=np.arange(-48,25,6)


def advection(t,u,v):
    """Horizontal temperature advection on a spherical regular grid, K/day."""
    dtdy=t.differentiate('lat')/(R*np.pi/180)
    dtdx=t.differentiate('lon')/(R*np.cos(np.deg2rad(t.lat))*np.pi/180)
    return -(u*dtdx+v*dtdy)*86400


def main():
    pieces=[]
    for year,month in MONTHS:
        path=ROOT/f'data/raw/era5_djf_temperature_pilot/era5_t700_{year}{month:02d}.nc'
        if not validate(path,year,month):
            raise RuntimeError(f'Incomplete temperature: {path}')
        with xr.open_dataset(path) as raw:
            t=canonical(raw).t.load()
        with xr.open_dataset(ROOT/f'data/raw/era5_700hpa_6hourly/era5_700hpa_{year}.nc') as raw:
            wind=canonical(raw)[['u','v']].sel(time=t.time,lat=t.lat,lon=t.lon).load()
        with xr.open_dataset(ROOT/f'data/raw/era5_surface_pressure_6hourly/era5_surface_pressure_{year}.nc') as raw:
            sp=canonical(raw).sp.sel(time=t.time,lat=t.lat,lon=t.lon).load()
        safe=sp>=72500
        a=advection(t.where(safe),wind.u.where(safe),wind.v.where(safe))
        box=a.sel(lat=slice(28,32),lon=slice(104,108))
        w=np.cos(np.deg2rad(box.lat))*xr.ones_like(box.isel(time=0))
        coverage=(box.notnull()*w).sum(['lat','lon'])/w.sum(['lat','lon'])
        mean=box.weighted(w).mean(['lat','lon']).where(coverage>=.8)
        pieces.append(pd.DataFrame({'time':t.time.values,'advection_k_day':mean.values,'valid_area_fraction':coverage.values}))
        print(f'advection {year}-{month:02d}',flush=True)
    series=pd.concat(pieces).set_index('time').sort_index()
    series.to_csv(OUT/'djf_cold_advection_pilot_series.csv')
    metadata=pd.read_csv(OUT/'events_terrain_qc_intensity.csv.gz',parse_dates=['genesis_time'])
    metadata['season_year']=metadata.genesis_time.dt.year+metadata.genesis_time.dt.month.eq(12).astype(int)
    winter=metadata[metadata.genesis_time.dt.month.isin([12,1,2])&metadata.season_year.eq(2011)].copy()
    winter['anchor']=pd.to_datetime(winter.genesis_time,utc=True).dt.tz_localize(None).dt.floor('6h')
    selected=winter[winter.track_cluster.eq(1)]
    anchors=pd.DatetimeIndex(winter.anchor)
    values=series.advection_k_day
    records=[]; composites=[]
    for row in selected.itertuples():
        anchor=row.anchor
        targets=pd.DatetimeIndex([anchor+pd.Timedelta(hours=int(h)) for h in LAGS])
        event=values.reindex(targets)
        pre=event.iloc[:8]
        candidates=series.index[(series.index.month==anchor.month)&(series.index.hour==anchor.hour)]
        controls=[]
        for candidate in candidates:
            if np.min(np.abs((anchors-candidate).total_seconds()))<72*3600:
                continue
            times=pd.DatetimeIndex([candidate+pd.Timedelta(hours=int(h)) for h in LAGS])
            control=values.reindex(times)
            if control.notna().all():
                controls.append(control.to_numpy())
        baseline=np.mean(controls,axis=0) if controls else np.full(len(LAGS),np.nan)
        complete=bool(pre.notna().all() and len(controls)>0)
        offset=(pd.Timestamp(row.genesis_time).tz_localize(None)-anchor).total_seconds()/3600
        records.append({'event_id':row.event_id,'anchor':anchor,'genesis_offset_hours':offset,
                        'controls':len(controls),'complete_pre_window':complete,
                        'pre_advection_k_day':float(pre.mean()) if complete else np.nan,
                        'control_pre_advection_k_day':float(baseline[:8].mean()) if complete else np.nan,
                        'pre_difference_k_day':float(pre.mean()-baseline[:8].mean()) if complete else np.nan})
        for lag,ev,co in zip(LAGS,event.to_numpy(),baseline):
            composites.append({'event_id':row.event_id,'lag_hours':int(lag),'event_advection_k_day':ev,'control_advection_k_day':co})
    events=pd.DataFrame(records)
    events.to_csv(OUT/'djf_cold_advection_pilot_events.csv',index=False)
    pd.DataFrame(composites).to_csv(OUT/'djf_cold_advection_pilot_composites.csv',index=False)
    gates={'all_360_times':len(series)==360,'valid_area_time_fraction_ge_95pct':bool(values.notna().mean()>=.95),
           'matched_event_fraction_ge_90pct':bool(len(events)>0 and events.complete_pre_window.mean()>=.9)}
    summary={'scope':'DJF 2011 feasibility only, not long-term or causal inference','times':len(series),
             'selected_cluster1_events':len(events),'valid_time_fraction':float(values.notna().mean()),
             'matched_event_fraction':float(events.complete_pre_window.mean()) if len(events) else 0,
             'mean_event_minus_control_pre_advection_k_day':float(events.pre_difference_k_day.mean()),
             'gates':gates,'feasible_for_full_expansion':all(gates.values()),
             'negative_difference_meaning':'stronger antecedent cooling; descriptive only, controls are reused'}
    (OUT/'djf_cold_advection_pilot_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    main()
