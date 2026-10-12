"""Exploratory day/night audit of frozen, previously failed wind-only proxy."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
from develop_vorticity_only_detector import match_events

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'data/processed'
WINTERS = np.arange(1981, 2024)


def winter_year(time):
    return time.dt.year+time.dt.month.eq(12).astype(int)


def night_mask(time):
    hour = (time.dt.hour+8) % 24
    return (hour >= 20) | (hour < 8)


def boundary_mask(time):
    return (time.dt.month.eq(12) & time.dt.day.eq(31)) | (time.dt.month.eq(1) & time.dt.day.eq(1))


def annual_counts(frame, remove_boundary=False):
    t = frame.genesis_time
    select = (t.dt.month.isin([12, 1, 2]) & winter_year(t).isin(WINTERS) &
              frame.genesis_lat.between(27, 33) & frame.genesis_lon.between(102, 109))
    if remove_boundary:
        select &= ~boundary_mask(t)
    chosen = frame[select].copy()
    chosen['season_year'] = winter_year(chosen.genesis_time)
    chosen['night'] = night_mask(chosen.genesis_time)
    counts = chosen.groupby(['season_year', 'night']).size().unstack(fill_value=0).reindex(index=WINTERS, columns=[False, True], fill_value=0).fillna(0).astype(int)
    counts.columns = ['day', 'night']
    if counts.to_numpy().sum() != len(chosen):
        raise RuntimeError('Count conservation failed')
    return counts, chosen


def matched_identity_audit(detector, reference):
    """Reuse historical matching unchanged; no classification retuning."""
    _, de = annual_counts(detector)
    _, ref = annual_counts(reference)
    point_paths = [OUT/f'vorticity_structure_points_{label}.csv.gz' for label in ['development', 'forward_validation']]
    parts = []
    for label, path in zip(['development', 'forward_validation'], point_paths):
        frame = pd.read_csv(path, parse_dates=['time'])
        frame['detected_id'] = label+':'+frame.detected_id
        parts.append(frame)
    dp = pd.concat(parts, ignore_index=True)
    dp = dp[dp.detected_id.isin(de.detected_id)]
    reference_points = OUT/'track_points.csv.gz'
    rp = pd.read_csv(reference_points)
    rp = rp[rp.event_id.isin(ref.event_id)].copy()
    rp['time'] = pd.to_datetime(rp.time, utc=True).dt.tz_localize(None)
    rp = rp[rp.time.dt.hour.isin([0, 6, 12, 18])]
    if dp.duplicated(['detected_id', 'time']).any() or rp.duplicated(['event_id', 'time']).any():
        raise RuntimeError('Non-unique track timestamp')
    pairs = match_events(de, dp, ref, rp)
    pairs['detector_genesis_time'] = pairs.detected_id.map(de.set_index('detected_id').genesis_time)
    pairs['reference_genesis_time'] = pairs.event_id.map(ref.set_index('event_id').genesis_time)
    pairs['detector_night'] = night_mask(pairs.detector_genesis_time)
    pairs['reference_night'] = night_mask(pairs.reference_genesis_time)
    pairs['genesis_offset_hours'] = (pairs.detector_genesis_time-pairs.reference_genesis_time).dt.total_seconds()/3600
    pairs['season_year'] = winter_year(pairs.detector_genesis_time)
    if not np.array_equal(pairs.season_year, winter_year(pairs.reference_genesis_time)):
        raise RuntimeError('Matching crosses winter definition')
    pairs.to_csv(OUT/'winter_frozen_detector_event_matches.csv', index=False)
    phase_slopes = []
    for (dn, rn), group in pairs.groupby(['detector_night', 'reference_night']):
        counts = group.groupby('season_year').size().reindex(WINTERS, fill_value=0)
        phase_slopes.append({'detector_night': bool(dn), 'reference_night': bool(rn), 'n_pairs': len(group),
                             'descriptive_slope_per_decade': float(np.polyfit((WINTERS-2002)/10, counts.to_numpy(), 1)[0]),
                             'median_genesis_offset_hours': float(group.genesis_offset_hours.median())})
    decomposition = []
    details = []
    for source, frame, identifier in [('frozen_wind_proxy', de, 'detected_id'), ('CNN_all_classes_same_rectangle', ref, 'event_id')]:
        frame = frame.copy()
        frame['matched'] = frame[identifier].isin(pairs[identifier])
        frame['season_year'] = winter_year(frame.genesis_time)
        frame['night'] = night_mask(frame.genesis_time)
        for is_night, period in [(False, 'day'), (True, 'night')]:
            subset = frame[frame.night.eq(is_night)]
            n = len(subset)
            nmatched = int(subset.matched.sum())
            details.append({'source': source, 'period': period, 'n_events': n, 'n_matched': nmatched,
                            'matched_fraction': nmatched/n if n else None})
            for status in [False, True]:
                counts = subset[subset.matched.eq(status)].groupby('season_year').size().reindex(WINTERS, fill_value=0)
                slope = float(np.polyfit((WINTERS-2002)/10, counts.to_numpy(), 1)[0])
                for year, count in counts.items():
                    decomposition.append({'source': source, 'period': period, 'matched': status,
                                          'season_year': int(year), 'n_events': int(count), 'descriptive_slope_per_decade': slope})
    pd.DataFrame(decomposition).to_csv(OUT/'winter_frozen_detector_matching_decomposition.csv', index=False)
    return {'n_detector': len(de), 'n_reference': len(ref), 'n_pairs': len(pairs),
            'event_f1': 2*len(pairs)/(len(de)+len(ref)), 'by_own_genesis_phase': details,
            'clock_phase_mismatch_fraction': float(pairs.detector_night.ne(pairs.reference_night).mean()),
            'phase_pair_descriptive_trends': phase_slopes,
            'phase_pair_counts': [{'detector_night': bool(d), 'reference_night': bool(r), 'n_pairs': int(n)}
                                  for (d, r), n in pairs.groupby(['detector_night', 'reference_night']).size().items()],
            'limits': ['Matched fraction counts partners of either phase; not same-phase recall',
                       'Matched/unmatched trend slopes are descriptive decompositions without added tests',
                       'Post-result matching audit does not validate the frozen detector']}, [*point_paths, reference_points]


def main():
    detector_files = [OUT/f'vorticity_structure_events_{label}.csv.gz' for label in ['development', 'forward_validation']]
    parts = []
    for label, path in zip(['development', 'forward_validation'], detector_files):
        frame = pd.read_csv(path, parse_dates=['genesis_time'])
        frame['detected_id'] = label+':'+frame.detected_id
        parts.append(frame)
    detector = pd.concat(parts, ignore_index=True)
    if detector.detected_id.duplicated().any():
        raise RuntimeError('Duplicate detector identity')
    reference_file = OUT/'events_terrain_qc_intensity.csv.gz'
    reference = pd.read_csv(reference_file)
    reference['genesis_time'] = pd.to_datetime(reference.genesis_time, utc=True).dt.tz_localize(None)
    sources = {'frozen_wind_proxy': detector, 'CNN_all_classes_same_rectangle': reference,
               'CNN_local_class_same_rectangle': reference[reference.track_cluster.eq(1)]}
    rows, count_tables, audits = [], [], []
    design = sm.add_constant((WINTERS-2002)/10)
    for source, frame in sources.items():
        for remove_boundary in [False, True]:
            counts, chosen = annual_counts(frame, remove_boundary)
            variant = 'remove_year_boundary' if remove_boundary else 'complete_winters'
            count_tables.append(counts.reset_index(names='season_year').assign(source=source, variant=variant))
            audits.append({'source': source, 'variant': variant, 'n_events': len(chosen),
                           'night_events': int(counts.night.sum()), 'day_events': int(counts.day.sum()),
                           'utc_genesis_hour_counts': {str(k): int(v) for k, v in chosen.genesis_time.dt.hour.value_counts().sort_index().items()}})
            for endpoint, values in [('night', counts.night), ('day', counts.day), ('night_minus_day', counts.night-counts.day)]:
                fit = sm.OLS(values.to_numpy(), design).fit(cov_type='HAC', cov_kwds={'maxlags': 3})
                ci = fit.conf_int()[1]
                rows.append({'source': source, 'variant': variant, 'endpoint': endpoint,
                             'n_winters': len(WINTERS), 'n_events': len(chosen), 'slope_per_decade': float(fit.params[1]),
                             'ci95_low': float(ci[0]), 'ci95_high': float(ci[1]), 'p_hac3': float(fit.pvalues[1])})
    stats = pd.DataFrame(rows)
    stats['p_fdr_18'] = multipletests(stats.p_hac3, method='fdr_bh')[1]
    stats.to_csv(OUT/'winter_frozen_detector_day_night_trends.csv', index=False)
    pd.concat(count_tables, ignore_index=True).to_csv(OUT/'winter_frozen_detector_day_night_counts.csv', index=False)
    gate_file = OUT/'vorticity_detector_forward_validation.json'
    gate = json.loads(gate_file.read_text(encoding='utf-8'))
    if gate['route_decision'] != 'NO-GO':
        raise RuntimeError('Unexpected historical gate status; re-audit required')
    matching, additional_inputs = matched_identity_audit(detector, reference)
    summary = {'scope': 'Post-result day/night diagnostic of frozen failed proxy; not independent validation',
               'full_winter_period': [1981, 2023], 'n_winters': len(WINTERS), 'series_audits': audits,
               'historical_detector_gate': gate['route_decision'], 'historical_forward_f1': gate['forward_validation_metrics']['event_f1'],
               'trends': stats.to_dict(orient='records'),
               'event_identity_audit': matching,
               'input_sha256': [{'path': p.relative_to(ROOT).as_posix(), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}
                                for p in [*detector_files, reference_file, gate_file, *additional_inputs]],
               'limits': ['Wind classifier used early CNN training labels and shares ERA5',
                          'Low historical event F1 and winter count correlation remain NO-GO',
                          'Calendar-year tracking restarts each January; dropping boundary genesis does not repair tracks',
                          'Rectangular genesis subset is not the basin-local trajectory class',
                          'Forty-three winters, not event count, define statistical replication']}
    (OUT/'winter_frozen_detector_day_night_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(stats.to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
