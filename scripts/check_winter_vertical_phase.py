"""Matched-position/time vertical diurnal convergence diagnostic, not attribution."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
from analyze_winter_genesis_regimes import target_times
from check_winter_diurnal_fixed_mask import load, support_from_pressure, LAT, LON
from extract_winter_local_convergence import spherical_terms

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'data/processed'
HOURS = [0, 6, 12, 18]


def harmonic_components(values):
    """UTC 0/6/12/18 components; positive sine peaks at UTC06."""
    values = np.asarray(values, dtype=float)
    if values.shape[-1] != 4:
        raise ValueError('Four UTC clock samples required')
    return (values[..., 0]-values[..., 2])/2, (values[..., 1]-values[..., 3])/2


def phase_difference_hours(a, b):
    """Signed circular peak difference a minus b in [-12,12)."""
    return (np.asarray(a)-np.asarray(b)+12) % 24-12


def phase_summary(values):
    cosine, sine = harmonic_components(values)
    amplitude = float(np.hypot(cosine, sine))
    return {'amplitude': amplitude,
            'peak_beijing_hour': float((np.arctan2(sine, cosine)*12/np.pi+8) % 24)
            if amplitude > 0 else None}


def paired_phase_interval(values, repeats=1000, block=5, seed=20261005):
    """Circular-block winter bootstrap; preserve paired levels and UTC phases.

    Interval is locally unwrapped around the observed signed phase difference,
    and may extend past the conventional [-12,12) reporting interval.
    """
    values = np.asarray(values, dtype=float)
    if values.ndim != 3 or values.shape[1:] != (2, 4):
        raise ValueError('Expected winters x two levels x four phases')
    n = len(values)
    generator = np.random.default_rng(seed)
    def contrast(curves):
        peaks = [phase_summary(v)['peak_beijing_hour'] for v in curves]
        return float(phase_difference_hours(peaks[0], peaks[1]))
    observed = contrast(values.mean(axis=0))
    samples = []
    for _ in range(repeats):
        starts = generator.integers(0, n, size=int(np.ceil(n/block)))
        indices = ((starts[:, None]+np.arange(block)) % n).ravel()[:n]
        samples.append(observed+float(phase_difference_hours(contrast(values[indices].mean(axis=0)), observed)))
    low, high = np.quantile(samples, [.025, .975])
    return {'estimate_hours': observed, 'ci95_low': float(low), 'ci95_high': float(high),
            'resamples': repeats, 'block_winters': block, 'seed': seed}


def main():
    times = target_times()
    season = times.year+(times.month == 12).astype(int)
    early = times[season <= 2001]
    totals = {level: np.zeros((17, 17)) for level in [700, 850]}
    for year in sorted(set(early.year)):
        wanted = early[early.year == year]
        pressure = load(ROOT/f'data/raw/era5_surface_pressure_6hourly/era5_surface_pressure_{year}.nc', wanted).sp.values
        for level in totals:
            totals[level] += support_from_pressure(pressure, level).sum(axis=0)
    support = (totals[700]/len(early) >= .99) & (totals[850]/len(early) >= .99)
    if not support.any():
        raise RuntimeError('Empty common support')
    weights = np.cos(np.deg2rad(LAT[1:-1]))[:, None]*np.ones((17, 17))
    weights = weights[support]
    rows = []
    for year in sorted(set(times.year)):
        wanted = times[times.year == year]
        pressure = load(ROOT/f'data/raw/era5_surface_pressure_6hourly/era5_surface_pressure_{year}.nc', wanted).sp.values
        fields = {}
        for level in [700, 850]:
            ds = load(ROOT/f'data/raw/era5_{level}hpa_6hourly/era5_{level}hpa_{year}.nc', wanted).sel(level=level)
            u, v = ds.u.values.astype(float), ds.v.values.astype(float)
            safe = pressure >= (level+25)*100
            u[~safe] = np.nan
            v[~safe] = np.nan
            convergence, _ = spherical_terms(u, v, LAT, LON)
            fields[level] = convergence[:, 1:-1, 1:-1][:, support]
        valid = np.isfinite(fields[700]).all(axis=1) & np.isfinite(fields[850]).all(axis=1)
        averages = {level: np.where(valid, np.average(field, axis=1, weights=weights), np.nan)
                    for level, field in fields.items()}
        for index, stamp in enumerate(wanted):
            rows.append({'time': stamp, 'season_year': int(stamp.year+(stamp.month == 12)),
                         'utc_hour': stamp.hour, 'complete': bool(valid[index]),
                         **{f'convergence_{level}': float(averages[level][index]) for level in fields}})
        print(f'matched vertical extraction {year}', flush=True)
    series = pd.DataFrame(rows)
    series.to_csv(OUT/'winter_vertical_phase_series.csv.gz', index=False)
    coverage = series.groupby(['season_year', 'utc_hour']).agg(n_risk=('complete', 'size'), n_complete=('complete', 'sum'))
    coverage.to_csv(OUT/'winter_vertical_phase_coverage.csv')
    if coverage.n_complete.min() < 10:
        raise RuntimeError('Too few matched observations in an annual phase')
    statistics, annual_rows, periods, matrices = [], [], {}, {}
    for level in [700, 850]:
        annual = series.groupby(['season_year', 'utc_hour'])[f'convergence_{level}'].mean().unstack().reindex(columns=HOURS)
        if annual.isna().any().any() or len(annual) != 44:
            raise RuntimeError('Incomplete annual phase matrix')
        matrices[level] = annual.to_numpy()
        cosine, sine = harmonic_components(annual.to_numpy())
        metrics = {'mean': annual.mean(axis=1).to_numpy(), 'amplitude': np.hypot(cosine, sine),
                   'night_minus_day': ((annual[12]+annual[18]-annual[0]-annual[6])/2).to_numpy(),
                   'beijing20_minus08': (annual[12]-annual[0]).to_numpy()}
        for name, response in metrics.items():
            fit = sm.OLS(response, sm.add_constant((annual.index.to_numpy()-2001.5)/10)).fit(cov_type='HAC', cov_kwds={'maxlags': 3})
            ci = fit.conf_int()[1]
            statistics.append({'level': level, 'statistic': name, 'slope_per_decade': float(fit.params[1]),
                               'ci95_low': float(ci[0]), 'ci95_high': float(ci[1]), 'p_hac3': float(fit.pvalues[1]),
                               'early_mean': float(response[:22].mean()), 'late_mean': float(response[22:].mean())})
        for winter, row in annual.iterrows():
            annual_rows.append({'level': level, 'season_year': int(winter), **{f'utc_{h}': float(row[h]) for h in HOURS}})
        for label, subset in [('early', annual.iloc[:22]), ('late', annual.iloc[22:])]:
            curve = subset.mean().to_numpy()
            periods.setdefault(label, {})[str(level)] = {**phase_summary(curve), 'utc_phase_means': curve.tolist()}
    for period in periods.values():
        period['peak_700_minus_850_hours'] = float(phase_difference_hours(period['700']['peak_beijing_hour'], period['850']['peak_beijing_hour']))
    matched = np.stack([matrices[700], matrices[850]], axis=1)
    for label, subset in [('early', matched[:22]), ('late', matched[22:])]:
        periods[label]['paired_phase_interval'] = paired_phase_interval(subset)
    results = pd.DataFrame(statistics)
    results['p_fdr_8'] = multipletests(results.p_hac3, method='fdr_bh')[1]
    results.to_csv(OUT/'winter_vertical_phase_trends.csv', index=False)
    pd.DataFrame(annual_rows).to_csv(OUT/'winter_vertical_phase_annual.csv', index=False)
    summary = {'support_nodes': int(support.sum()), 'matched_timestamp_fraction': float(series.complete.mean()),
               'minimum_phase_count': int(coverage.n_complete.min()), 'trends': results.to_dict(orient='records'),
               'period_mean_harmonics': periods,
               'limits': ['Exploratory same-ERA5 diagnostic, not independent validation or attribution',
                          'Basin core is not all genesis locations', 'Complete timestamps can select weather states',
                          'Four samples per day cannot resolve exact peak times; phase differences are descriptive',
                          'Harmonic of period means differs from mean annual amplitudes; neither is vertical mass flux']}
    summary['raw_input_provenance'] = [
        {'path': path.relative_to(ROOT).as_posix(), 'bytes': path.stat().st_size, 'mtime_ns': path.stat().st_mtime_ns}
        for year in sorted(set(times.year)) for path in [
            ROOT/f'data/raw/era5_surface_pressure_6hourly/era5_surface_pressure_{year}.nc',
            ROOT/f'data/raw/era5_700hpa_6hourly/era5_700hpa_{year}.nc',
            ROOT/f'data/raw/era5_850hpa_6hourly/era5_850hpa_{year}.nc']]
    (OUT/'winter_vertical_phase_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(results.to_string(index=False), flush=True)
    print(json.dumps(periods, indent=2), flush=True)


if __name__ == '__main__':
    main()
