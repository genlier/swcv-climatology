"""Predefined antecedent and background diagnostics, DJF 1980--2023."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import xarray as xr
from statsmodels.stats.multitest import multipletests

from analyze_djf_cold_advection_pilot import advection, LAGS
from download_djf_cold_advection_pilot import canonical
from download_djf_cold_advection_full import MANIFEST, validate, expected_times

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'data/processed'
YEARS=np.arange(1980,2024)
PREFIX='djf_cold_advection_full_'


def load_series(manifest):
    pieces=[]
    for entry in manifest['files']:
        year,month=entry['year'],entry['month']
        path=ROOT/entry['path']
        if not validate(path,year,month,entry['days']) or hashlib.sha256(path.read_bytes()).hexdigest()!=entry['sha256']:
            raise RuntimeError(f'Raw file/manifest failed audit: {path}')
        with xr.open_dataset(path) as raw:
            t=canonical(raw).t.load()
        with xr.open_dataset(ROOT/f'data/raw/era5_700hpa_6hourly/era5_700hpa_{year}.nc') as raw:
            wind=canonical(raw)[['u','v']].sel(time=t.time,lat=t.lat,lon=t.lon).load()
        with xr.open_dataset(ROOT/f'data/raw/era5_surface_pressure_6hourly/era5_surface_pressure_{year}.nc') as raw:
            sp=canonical(raw).sp.sel(time=t.time,lat=t.lat,lon=t.lon).load()
        safe=sp>=72500
        field=advection(t.where(safe),wind.u.where(safe),wind.v.where(safe)).sel(lat=slice(28,32),lon=slice(104,108))
        weights=np.cos(np.deg2rad(field.lat))*xr.ones_like(field.isel(time=0))
        coverage=(field.notnull()*weights).sum(['lat','lon'])/weights.sum(['lat','lon'])
        mean=field.weighted(weights).mean(['lat','lon']).where(coverage>=.8)
        pieces.append(pd.DataFrame({'time':t.time.values,'advection_k_day':mean.values,'valid_area_fraction':coverage.values}))
        print(f'regional advection {year}-{month:02d}',flush=True)
    series=pd.concat(pieces).set_index('time').sort_index()
    if not series.index.is_unique:
        raise RuntimeError('Duplicate regional times')
    return series


def nearest_genesis_hours(times,genesis):
    """Actual genesis, any season/path; exact 72h boundary is allowed."""
    source=np.sort(pd.DatetimeIndex(genesis).as_unit('ns').asi8)
    query=pd.DatetimeIndex(times).as_unit('ns').asi8
    if len(source)==0:
        return np.full(len(query),np.inf)
    positions=np.searchsorted(source,query)
    left=source[np.clip(positions-1,0,len(source)-1)]
    right=source[np.clip(positions,0,len(source)-1)]
    return np.minimum(np.abs(query-left),np.abs(query-right))/3.6e12


def match_events(series,metadata):
    values=series.advection_k_day
    windows=pd.DataFrame({int(lag):values.reindex(values.index+pd.Timedelta(hours=int(lag))).to_numpy() for lag in LAGS},index=values.index)
    distances=nearest_genesis_hours(values.index,metadata.genesis_time)
    eligible=(distances>=72)&windows.notna().all(axis=1).to_numpy()
    controls=windows.loc[eligible]
    selected=metadata[metadata.genesis_time.dt.month.isin([12,1,2])&metadata.season_year.between(1980,2023)&metadata.track_cluster.eq(1)]
    records=[]; composites=[]
    for event in selected.itertuples():
        anchor=event.genesis_time.floor('6h')
        ev=windows.loc[anchor] if anchor in windows.index else pd.Series(np.nan,index=LAGS)
        pool=controls[(controls.index.year==anchor.year)&(controls.index.month==anchor.month)&(controls.index.hour==anchor.hour)]
        baseline=pool.mean(axis=0) if len(pool) else pd.Series(np.nan,index=LAGS)
        complete=bool(ev.loc[LAGS[LAGS<0]].notna().all() and len(pool)>0)
        reason='ok' if complete else 'missing_pre_window' if ev.loc[LAGS[LAGS<0]].isna().any() else 'no_eligible_controls'
        pre=float(ev.loc[LAGS[LAGS<0]].mean()) if complete else np.nan
        control=float(baseline.loc[LAGS[LAGS<0]].mean()) if complete else np.nan
        records.append({'event_id':event.event_id,'season_year':event.season_year,'genesis_time':event.genesis_time,
                        'anchor':anchor,'genesis_offset_hours':(event.genesis_time-anchor).total_seconds()/3600,
                        'controls':len(pool),'complete_pre_window':complete,'exclusion_reason':reason,
                        'pre_advection_k_day':pre,'control_pre_advection_k_day':control,'pre_difference_k_day':pre-control})
        for lag in LAGS:
            composites.append({'event_id':event.event_id,'season_year':event.season_year,'lag_hours':int(lag),
                               'event_advection_k_day':float(ev[lag]),'control_advection_k_day':float(baseline[lag]),'matched_event':complete})
    return pd.DataFrame(records),pd.DataFrame(composites)


def block_interval(annual,seed=20260915,repetitions=2000):
    """Equal winter weights; circular blocks preserve missing-winter positions."""
    x=np.asarray(annual,dtype=float)
    rng=np.random.default_rng(seed)
    estimates=[]
    for _ in range(repetitions):
        starts=rng.integers(0,len(x),size=int(np.ceil(len(x)/5)))
        ids=((starts[:,None]+np.arange(5))%len(x)).ravel()[:len(x)]
        sample=x[ids]
        if np.isfinite(sample).any():
            estimates.append(float(np.nanmean(sample)))
    return {'mean_k_day':float(np.nanmean(x)),'ci_low':float(np.quantile(estimates,.025)),
            'ci_high':float(np.quantile(estimates,.975)),'n_matched_winters':int(np.isfinite(x).sum()),
            'repetitions':repetitions,'block_winters':5,'seed':seed}


def background_table(series,metadata):
    frame=series.loc[series.index.month.isin([12,1,2])].copy()
    frame['season_year']=frame.index.year+(frame.index.month==12).astype(int)
    frame=frame[frame.season_year.between(1980,2023)]
    reference=frame[frame.season_year.between(1981,2010)]
    thresholds=reference.groupby(reference.index.month).advection_k_day.quantile(.1).to_dict()
    counts=metadata[metadata.genesis_time.dt.month.isin([12,1,2])&metadata.season_year.between(1980,2023)&metadata.track_cluster.eq(1)].groupby('season_year').size()
    rows=[]
    for winter,group in frame.groupby('season_year'):
        valid=group.advection_k_day.notna()
        q=np.array([thresholds[int(month)] for month in group.index.month])
        strong=(group.advection_k_day.to_numpy()<=q)&(group.advection_k_day.to_numpy()<0)
        rows.append({'season_year':int(winter),'expected_times':len(group),'valid_times':int(valid.sum()),
                     'valid_time_fraction':float(valid.mean()),'mean_cooling_k_day':float(-group.advection_k_day.mean()),
                     'strong_cold_advection_fraction':float(strong[valid].mean()),'cluster1_frequency':int(counts.get(winter,0))})
    return pd.DataFrame(rows),{str(int(k)):float(v) for k,v in thresholds.items()}


def closure_statistics(table):
    year=(table.season_year-table.season_year.mean())/10
    offset=np.log(table.valid_time_fraction)
    base_design=sm.add_constant(pd.DataFrame({'year_decade':year}))
    base=sm.GLM(table.cluster1_frequency,base_design,offset=offset,family=sm.families.Poisson()).fit(cov_type='HAC',cov_kwds={'maxlags':3})
    rows=[]
    for predictor in ['mean_cooling_k_day','strong_cold_advection_fraction']:
        x=table[predictor];sd=float(x.std(ddof=1))
        if not np.isfinite(sd) or sd<=0:
            raise RuntimeError(f'Degenerate background predictor: {predictor}')
        trend=sm.OLS(x,base_design).fit(cov_type='HAC',cov_kwds={'maxlags':3})
        design=sm.add_constant(pd.DataFrame({'year_decade':year,'index_z':(x-x.mean())/sd}))
        joint=sm.GLM(table.cluster1_frequency,design,offset=offset,family=sm.families.Poisson()).fit(cov_type='HAC',cov_kwds={'maxlags':3})
        implied=float(trend.params.year_decade/sd*joint.params.index_z)
        attenuation=float(100*(base.params.year_decade-joint.params.year_decade)/base.params.year_decade)
        rows.append({'predictor':predictor,'trend_raw_per_decade':float(trend.params.year_decade),
                     'trend_ci_low':float(trend.conf_int().loc['year_decade',0]),'trend_ci_high':float(trend.conf_int().loc['year_decade',1]),
                     'trend_p_hac3':float(trend.pvalues.year_decade),'count_log_rate_per_sd':float(joint.params.index_z),
                     'count_association_p_hac3':float(joint.pvalues.index_z),'implied_log_rate_per_decade':implied,
                     'base_log_rate_per_decade':float(base.params.year_decade),'adjusted_log_rate_per_decade':float(joint.params.year_decade),
                     'year_attenuation_percent':attenuation,'direction_consistent':bool(np.sign(implied)==np.sign(base.params.year_decade))})
    result=pd.DataFrame(rows)
    result['trend_p_fdr']=multipletests(result.trend_p_hac3,method='fdr_bh')[1]
    result['association_p_fdr']=multipletests(result.count_association_p_hac3,method='fdr_bh')[1]
    result['passes_statistical_trend_closure']=(result.trend_p_fdr<.05)&(result.association_p_fdr<.05)&result.direction_consistent&(result.year_attenuation_percent>0)
    return result


def main():
    manifest=json.loads(MANIFEST.read_text(encoding='utf-8'))
    if manifest['status']!='complete' or manifest['winters']!=[1980,2023] or len(manifest['files'])!=220 or manifest['failures']:
        raise RuntimeError('Expected complete 220-file full-period temperature manifest')
    expected=pd.DatetimeIndex(np.concatenate([expected_times(e['year'],e['month'],e['days']).values for e in manifest['files']])).sort_values()
    series=load_series(manifest)
    if not series.index.equals(expected) or len(series)!=4147*4:
        raise RuntimeError('Regional time-series does not exactly match frozen 4147-date plan')
    series.to_csv(OUT/(PREFIX+'series.csv.gz'))
    metadata=pd.read_csv(OUT/'events_terrain_qc_intensity.csv.gz',parse_dates=['genesis_time'])
    metadata['genesis_time']=pd.to_datetime(metadata.genesis_time,utc=True).dt.tz_localize(None)
    metadata['season_year']=metadata.genesis_time.dt.year+metadata.genesis_time.dt.month.eq(12).astype(int)
    events,composites=match_events(series,metadata)
    events.to_csv(OUT/(PREFIX+'events.csv'),index=False)
    composites.to_csv(OUT/(PREFIX+'composites.csv.gz'),index=False)
    annual=events.groupby('season_year').pre_difference_k_day.mean().reindex(YEARS)
    antecedent=block_interval(annual)
    table,thresholds=background_table(series,metadata)
    table['matched_event_pre_difference_k_day']=annual.reindex(table.season_year).to_numpy()
    table['matched_events']=events[events.complete_pre_window].groupby('season_year').size().reindex(table.season_year,fill_value=0).to_numpy()
    table.to_csv(OUT/(PREFIX+'annual.csv'),index=False)
    if len(table)!=44 or (table.valid_times==0).any():
        raise RuntimeError('Invalid winter background coverage')
    statistics=closure_statistics(table)
    statistics.to_csv(OUT/(PREFIX+'closure_statistics.csv'),index=False)
    gates={'all_44_winters':len(table)==44,'every_winter_valid_times_ge_95pct':bool((table.valid_time_fraction>=.95).all()),
           'matched_event_fraction_ge_90pct':bool(events.complete_pre_window.mean()>=.9),
           'at_least_30_matched_winters':antecedent['n_matched_winters']>=30,
           'reference_thresholds_negative':all(q<0 for q in thresholds.values())}
    passed=statistics.loc[statistics.passes_statistical_trend_closure,'predictor'].tolist()
    support=bool(all(gates.values()) and antecedent['ci_high']<0 and len(passed)>0)
    summary={'scope':'DJF 1980-2023, horizontal cold-advection statistical diagnostics, not causal attribution',
             'events':len(events),'matched_events':int(events.complete_pre_window.sum()),'matched_event_fraction':float(events.complete_pre_window.mean()),
             'monthly_1981_2010_q10_advection_k_day':thresholds,'antecedent_equal_winter_block_interval':antecedent,
             'gates':gates,'qc_passed':all(gates.values()),'passing_background_predictors':passed,
             'joint_statistical_mechanism_support':support,'interpretation':'statistical support, not proof of causality' if support else 'winter mechanism not closed by the predefined cold-advection tests',
             'exclusions':events.exclusion_reason.value_counts().to_dict()}
    (OUT/(PREFIX+'summary.json')).write_text(json.dumps(summary,indent=2),encoding='utf-8')
    text=f"# 全期冬季冷平流诊断\n\n44 个冬季；{len(events)} 个本地型事件，{int(events.complete_pre_window.sum())} 个匹配事件。\n\n"
    text+=f"等冬季权重前期事件减对照差值：{antecedent['mean_k_day']:+.4f} K/日，5 冬循环区组 95% 区间 [{antecedent['ci_low']:+.4f}, {antecedent['ci_high']:+.4f}]。负值表示生成前相对更强冷却。\n\n"
    text+=f"QC 通过：{all(gates.values())}；背景趋势闭合指标：{', '.join(passed) if passed else '无'}；联合统计支持：{support}。\n\n"
    text+='未通过联合标准时，冬季频数增长机制仍未闭合；通过也不证明因果。完整指标、排除和检验清单均保存在 processed 输出中，不能只选择显著结果。\n'
    (ROOT/'docs/djf_cold_advection_full_results_zh.md').write_text(text,encoding='utf-8')
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    main()
