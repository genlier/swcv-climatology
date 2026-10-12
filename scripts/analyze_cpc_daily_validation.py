"""Frozen whole-day, common-event CPC/ERA5 validation. No hourly CPC inference."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import xarray as xr
from statsmodels.stats.multitest import multipletests

from analyze_event_precipitation import REGIONS, canonical, load_djf_events, validate_download_manifest
from download_cpc_daily_validation import validate

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'data/processed'
PREFIX = 'cpc_daily_validation_'
YEARS = np.arange(1980, 2024)


def whole_days(start, end):
    """All UTC days intersecting the half-open original hourly interval."""
    start = pd.Timestamp(start); end = pd.Timestamp(end)
    if end <= start:
        raise ValueError('Non-positive event window')
    return pd.date_range(start.normalize(), (end - pd.Timedelta(nanoseconds=1)).normalize(), freq='D')


def safe_mean(field, region):
    sub = field.sel(lat=slice(region['north'], region['south']) if float(field.lat[0]) > float(field.lat[-1])
                    else slice(region['south'], region['north']), lon=slice(region['west'], region['east']))
    weights = np.cos(np.deg2rad(sub.lat)) * xr.ones_like(sub.isel(time=0))
    fraction = (sub.notnull() * weights).sum(['lat', 'lon']) / weights.sum(['lat', 'lon'])
    return sub.weighted(weights).mean(['lat', 'lon']).where(fraction >= .95)


def load_daily():
    validate_download_manifest(len(load_djf_events()))
    manifest = json.loads((ROOT / 'data/manifests/cpc_daily_validation_manifest.json').read_text())
    if manifest['status'] != 'complete' or len(manifest['files']) != 45 or manifest['failures']:
        raise RuntimeError('Need complete 45-year CPC manifest')
    cpc_parts = []; era_parts = []
    hashes = {}
    for entry in manifest['files']:
        path = ROOT / entry['path']
        current = validate(path, entry['year'])
        if current['sha256'] != entry['sha256']:
            raise RuntimeError('CPC source hash changed')
        hashes[str(path.relative_to(ROOT))] = entry['sha256']
        with xr.open_dataset(path) as ds:
            frame = pd.DataFrame({name: safe_mean(ds.precip.where(ds.precip >= 0), region).values
                                  for name, region in REGIONS.items()}, index=pd.DatetimeIndex(ds.time.values))
        cpc_parts.append(frame)
    for path in sorted((ROOT / 'data/raw/era5_djf_event_precipitation').glob('*.nc')):
        hashes[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
        with xr.open_dataset(path) as source:
            ds = canonical(source)
            frame = pd.DataFrame({name: safe_mean(ds.tp, region).values * 1000
                                  for name, region in REGIONS.items()}, index=pd.DatetimeIndex(ds.time.values))
        era_parts.append(frame)
    cpc = pd.concat(cpc_parts).sort_index()
    hourly = pd.concat(era_parts).sort_index()
    if not cpc.index.is_unique or not hourly.index.is_unique:
        raise RuntimeError('Duplicate product dates/hours')
    # Only complete UTC days, never silently sum a partially downloaded day.
    era = hourly.resample('D').sum(min_count=24)
    era = era.where(hourly.resample('D').count() == 24)
    return cpc, era, hashes


def match(events, cpc, era):
    rows = []
    for ev in events.itertuples():
        start = pd.Timestamp(ev.genesis_time).tz_localize(None)
        end = pd.Timestamp(ev.lysis_time).tz_localize(None) + pd.Timedelta(hours=24)
        dates = whole_days(start, end)
        for region in REGIONS:
            e = era[region].reindex(dates)
            for shift in [-1, 0, 1]:
                c = cpc[region].reindex(dates + pd.Timedelta(days=shift))
                valid = bool(c.notna().all() and e.notna().all())
                rows.append({'event_id': ev.event_id, 'season_year': int(ev.season_year), 'region': region,
                             'cpc_label_shift_days': shift, 'whole_utc_days': len(dates), 'matched': valid,
                             'era5_daily_mean_mm_day': float(e.mean()) if valid else np.nan,
                             'cpc_daily_mean_mm_day': float(c.mean()) if valid else np.nan,
                             'era5_accum_mm': float(e.sum()) if valid else np.nan,
                             'cpc_accum_mm': float(c.sum()) if valid else np.nan,
                             'exclusion_reason': 'ok' if valid else 'incomplete_era5_day' if e.isna().any() else 'incomplete_cpc_day'})
    return pd.DataFrame(rows)


def trend(annual, column, region, shift, period='full'):
    annual = annual.dropna(subset=[column])
    x = sm.add_constant((annual.season_year - annual.season_year.mean()) / 10)
    fit = sm.OLS(annual[column], x).fit(cov_type='HAC', cov_kwds={'maxlags': 2})
    slope = float(fit.params.iloc[1]); mean = float(annual[column].mean())
    ci = fit.conf_int().iloc[1].to_numpy()
    return {'region': region, 'product': column.split('_')[0], 'cpc_label_shift_days': shift, 'period': period,
            'winters': len(annual), 'mean_mm_day': mean, 'slope_mm_day_decade': slope,
            'ci_low_mm_day_decade': float(ci[0]), 'ci_high_mm_day_decade': float(ci[1]),
            'slope_percent_decade': slope / mean * 100, 'p_hac2': float(fit.pvalues.iloc[1])}


def main():
    cpc, era, hashes = load_daily()
    metadata = load_djf_events()
    events = match(metadata, cpc, era)
    events['common_all_shifts'] = events.event_id.map(events.groupby('event_id').matched.all())
    events.to_csv(OUT / (PREFIX + 'events.csv'), index=False)
    statistics = []; annual_parts = []; agreement = []
    for region in REGIONS:
        for shift in [-1, 0, 1]:
            subset = events[events.region.eq(region) & events.cpc_label_shift_days.eq(shift)]
            # Fixed common cohort across both regions and all shifts for fair alignment sensitivity.
            subset = subset[subset.common_all_shifts]
            annual = subset.groupby('season_year')[['era5_daily_mean_mm_day', 'cpc_daily_mean_mm_day']].mean().reindex(YEARS)
            annual.index.name = 'season_year'
            annual = annual.reset_index()
            annual['region'] = region; annual['cpc_label_shift_days'] = shift
            annual_parts.append(annual)
            statistics.append(trend(annual, 'cpc_daily_mean_mm_day', region, shift))
            if shift == 0:
                statistics.append(trend(annual, 'era5_daily_mean_mm_day', region, shift))
                for lo, hi, label in [(1980, 2005, '1980-2005'), (2006, 2023, '2006-2023')]:
                    statistics.append(trend(annual[annual.season_year.between(lo, hi)], 'cpc_daily_mean_mm_day', region, shift, label))
            agreement.append({'region': region, 'cpc_label_shift_days': shift, 'events': len(subset),
                              'event_pearson_descriptive': float(subset.era5_daily_mean_mm_day.corr(subset.cpc_daily_mean_mm_day)),
                              'annual_pearson_descriptive': float(annual.era5_daily_mean_mm_day.corr(annual.cpc_daily_mean_mm_day)),
                              'mean_cpc_minus_era5_mm_day': float((subset.cpc_daily_mean_mm_day-subset.era5_daily_mean_mm_day).mean())})
    stats = pd.DataFrame(statistics)
    stats['family'] = np.where(stats.period.ne('full'), 'exploratory_network_periods',
                               np.where(stats.cpc_label_shift_days.eq(0), 'primary_four_trends', 'exploratory_four_shifts'))
    stats['p_fdr'] = stats.groupby('family').p_hac2.transform(lambda p: multipletests(p, method='fdr_bh')[1])
    stats.to_csv(OUT / (PREFIX + 'statistics.csv'), index=False)
    pd.concat(annual_parts).to_csv(OUT / (PREFIX + 'annual.csv'), index=False)
    primary = events[events.cpc_label_shift_days.eq(0)]
    fraction = float(primary.groupby('event_id').matched.all().mean())
    common = int(events.groupby('event_id').common_all_shifts.first().sum())
    complete_winters = int(events[events.common_all_shifts].season_year.nunique())
    result = {'scope': 'Fixed-box whole-UTC-day total precipitation only, not snowfall or hourly-window verification',
              'input_sha256': hashes, 'events': len(metadata), 'primary_common_match_fraction': fraction,
              'all_shift_common_events': common, 'matched_winters': complete_winters,
              'qc_passed': bool(fraction >= .9 and common / len(metadata) >= .9 and complete_winters == 44),
              'uncertain_eod_requires_shift_sensitivity': True, 'agreement_descriptive': agreement,
              'statistics': stats.to_dict(orient='records'),
              'exclusions': primary.exclusion_reason.value_counts().to_dict(),
              'network_boundary': 'Retrospective 1979-2005 vs real-time 2006 onward; no station-count field in NOAA NetCDF subset'}
    (OUT / (PREFIX + 'summary.json')).write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in result.items() if k != 'input_sha256'}, indent=2), flush=True)


if __name__ == '__main__':
    main()
