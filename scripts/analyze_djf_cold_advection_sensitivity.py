"""Exploratory matching robustness; never replaces the frozen primary analysis."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_djf_cold_advection_full import YEARS, block_interval, nearest_genesis_hours
from analyze_djf_cold_advection_pilot import LAGS

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'data/processed'
PREFIX = 'djf_cold_advection_sensitivity_'
SCHEMES = {'primary_month': None, 'winter_day15': 15, 'winter_day30': 30}


def control_mask(times, anchor, span):
    times = pd.DatetimeIndex(times)
    if span is None:
        return (times.year == anchor.year) & (times.month == anchor.month) & (times.hour == anchor.hour)
    winter = anchor.year + int(anchor.month == 12)
    seasons = times.year + (times.month == 12).astype(int)
    days = np.abs((times.normalize() - anchor.normalize()).days)
    return (seasons == winter) & times.month.isin([12, 1, 2]) & (times.hour == anchor.hour) & (days <= span)


def analyze(series, metadata):
    values = series.advection_k_day
    windows = pd.DataFrame({int(h): values.reindex(values.index + pd.Timedelta(hours=int(h))).to_numpy()
                            for h in LAGS}, index=values.index)
    controls = windows.loc[(nearest_genesis_hours(values.index, metadata.genesis_time) >= 72)
                           & windows.notna().all(axis=1).to_numpy()]
    selected = metadata[metadata.genesis_time.dt.month.isin([12, 1, 2])
                        & metadata.season_year.between(1980, 2023) & metadata.track_cluster.eq(1)]
    rows = []
    for event in selected.itertuples():
        anchor = event.genesis_time.floor('6h')
        pre = windows.loc[anchor, LAGS[LAGS < 0]] if anchor in windows.index else pd.Series(np.nan, index=LAGS[LAGS < 0])
        for scheme, span in SCHEMES.items():
            pool = controls.loc[control_mask(controls.index, anchor, span)]
            matched = bool(pre.notna().all() and len(pool))
            baseline = float(pool.loc[:, LAGS[LAGS < 0]].mean().mean()) if len(pool) else np.nan
            rows.append({'scheme': scheme, 'event_id': event.event_id, 'season_year': event.season_year,
                         'anchor': anchor, 'controls': len(pool), 'matched': matched,
                         'pre_advection_k_day': float(pre.mean()), 'control_pre_advection_k_day': baseline,
                         'pre_difference_k_day': float(pre.mean()) - baseline if matched else np.nan,
                         'exclusion_reason': 'ok' if matched else 'missing_pre_window' if pre.isna().any() else 'no_eligible_controls'})
    return pd.DataFrame(rows)


def main():
    paths = [OUT / 'djf_cold_advection_full_series.csv.gz', OUT / 'events_terrain_qc_intensity.csv.gz',
             OUT / 'djf_cold_advection_full_events.csv', ROOT / 'docs/djf_cold_advection_sensitivity_plan_zh.md']
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    series = pd.read_csv(paths[0], parse_dates=['time']).set_index('time')
    metadata = pd.read_csv(paths[1], parse_dates=['genesis_time'])
    metadata['genesis_time'] = pd.to_datetime(metadata.genesis_time, utc=True).dt.tz_localize(None)
    metadata['season_year'] = metadata.genesis_time.dt.year + metadata.genesis_time.dt.month.eq(12).astype(int)
    events = analyze(series, metadata)
    # Mandatory reproduction check before interpreting exploratory alternatives.
    original = pd.read_csv(paths[2]).set_index('event_id').sort_index()
    reproduced = events[events.scheme.eq('primary_month')].set_index('event_id').sort_index()
    assert original.index.equals(reproduced.index)
    assert np.array_equal(original.controls, reproduced.controls)
    assert np.allclose(original.pre_difference_k_day, reproduced.pre_difference_k_day, equal_nan=True)
    common = events.groupby('event_id').matched.all()
    events['common_all_schemes'] = events.event_id.map(common)
    events.to_csv(OUT / (PREFIX + 'events.csv'), index=False)
    summaries = []; annual_rows = []
    for cohort in ['available', 'common_all_schemes']:
        subset = events if cohort == 'available' else events[events.common_all_schemes]
        for scheme in SCHEMES:
            group = subset[subset.scheme.eq(scheme)]
            annual = group.groupby('season_year').pre_difference_k_day.mean().reindex(YEARS)
            interval = block_interval(annual)
            total = int(events[events.scheme.eq(scheme)].matched.sum())
            summaries.append({'scheme': scheme, 'cohort': cohort, 'events_in_cohort': len(group),
                              'matched_events': int(group.matched.sum()), 'scheme_full_match_fraction': total / len(original),
                              'exploratory_matching_qc_passed': bool(total / len(original) >= .9 and interval['n_matched_winters'] >= 30),
                              **interval})
            for year, value in annual.items():
                annual_rows.append({'scheme': scheme, 'cohort': cohort, 'season_year': year, 'pre_difference_k_day': value})
    pd.DataFrame(annual_rows).to_csv(OUT / (PREFIX + 'annual.csv'), index=False)
    result = {'input_sha256': hashes, 'primary_reproduction_passed': True, 'all_events': len(original),
              'common_events': int(common.sum()), 'schemes_days': SCHEMES, 'genesis_exclusion_hours': 72,
              'results': summaries, 'interpretation': 'Exploratory robustness only; original QC failure and background non-closure remain.'}
    (OUT / (PREFIX + 'summary.json')).write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
