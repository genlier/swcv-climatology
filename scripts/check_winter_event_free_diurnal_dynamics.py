"""Exclude catalogued vortex lifetimes plus buffers from actual-clock dynamics.

This is an exploratory selected-background sensitivity, not an unconditioned
climate trend or independent evidence. All track classes are excluded so
circulation from passing systems is not labelled vortex-free.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
from check_winter_diurnal_fixed_mask import harmonic

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'data/processed'


def interval_mask(times,genesis,lysis,buffer_hours):
    starts=pd.DatetimeIndex(genesis-pd.Timedelta(hours=buffer_hours))
    stops=pd.DatetimeIndex(lysis+pd.Timedelta(hours=buffer_hours))
    a=times.searchsorted(starts,side='left')
    b=times.searchsorted(stops,side='right')
    change=np.zeros(len(times)+1,dtype=int)
    np.add.at(change,a,1);np.add.at(change,b,-1)
    return np.cumsum(change[:-1])>0


def main():
    series=pd.read_csv(OUT/'winter_fixed_support_dynamics_series.csv.gz',parse_dates=['time'])
    series=series[(series.region=='basin')&series.metric.str.startswith('convergence')].copy()
    meta=pd.read_csv(OUT/'events_terrain_qc_intensity.csv.gz')
    genesis=pd.to_datetime(meta.genesis_time,utc=True).dt.tz_localize(None)
    lysis=pd.to_datetime(meta.lysis_time,utc=True).dt.tz_localize(None)
    times=pd.DatetimeIndex(sorted(series.time.unique()))
    outputs=[];counts=[];annuals=[]
    for scope,select in [('basin_local_lifetimes',meta.track_cluster.eq(1)),('all_swcv_lifetimes',np.ones(len(meta),dtype=bool))]:
        for buffer in [0,12,24]:
            blocked=interval_mask(times,genesis[select],lysis[select],buffer)
            free=pd.Series(~blocked,index=times)
            subset=series[series.time.map(free)].copy()
            for metric,g in subset.groupby('metric'):
                g=g.dropna(subset=['complete_value'])
                phase_count=g.groupby(['season_year','utc_hour']).size().unstack().reindex(index=np.arange(1980,2024),columns=[0,6,12,18],fill_value=0)
                if phase_count.min().min()<10:raise RuntimeError('Too few event-free phase samples')
                annual=g.groupby(['season_year','utc_hour']).complete_value.mean().unstack().reindex(columns=[0,6,12,18])
                for statistic,values in [('mean',annual.mean(axis=1).to_numpy()),('amplitude',harmonic(annual.to_numpy()))]:
                    fit=sm.OLS(values,sm.add_constant((annual.index.to_numpy()-2001.5)/10)).fit(cov_type='HAC',cov_kwds={'maxlags':3})
                    ci=fit.conf_int()[1]
                    outputs.append({'scope':scope,'buffer_hours':buffer,'metric':metric,'statistic':statistic,
                        'slope_per_decade':float(fit.params[1]),'ci95_low':float(ci[0]),'ci95_high':float(ci[1]),
                        'p_hac3':float(fit.pvalues[1]),'early_mean':float(values[:22].mean()),'late_mean':float(values[22:].mean()),
                        'minimum_phase_boxes_per_winter':int(phase_count.min().min())})
                for wy,phase in phase_count.iterrows():
                    counts.append({'scope':scope,'buffer_hours':buffer,'metric':metric,'season_year':int(wy),
                        **{f'utc_{h}_boxes':int(phase[h]) for h in [0,6,12,18]}})
                for wy,phase in annual.iterrows():
                    annuals.append({'scope':scope,'buffer_hours':buffer,'metric':metric,'season_year':int(wy),
                        **{f'utc_{h}':float(phase[h]) for h in [0,6,12,18]}})
    table=pd.DataFrame(outputs)
    table['p_fdr_24']=multipletests(table.p_hac3,method='fdr_bh')[1]
    table.to_csv(OUT/'winter_event_free_diurnal_dynamics_trends.csv',index=False)
    pd.DataFrame(counts).to_csv(OUT/'winter_event_free_diurnal_dynamics_coverage.csv',index=False)
    pd.DataFrame(annuals).to_csv(OUT/'winter_event_free_diurnal_dynamics_annual.csv',index=False)
    obj={'scope':'Exclude measured catalogue lifetimes plus 0/12/24h buffers; selected backgrounds only',
        'trends':table.to_dict(orient='records'),
        'limits':['Undetected and excluded-original-catalogue vortices remain',
                  'Changing event frequency changes conditional background selection',
                  'This does not remove all weather transients or prove a thermal mechanism',
                  'All-class exclusion uses retained catalogue only, not independent validation']}
    (OUT/'winter_event_free_diurnal_dynamics_summary.json').write_text(json.dumps(obj,indent=2),encoding='utf-8')
    print(table[table.statistic=='amplitude'].to_string(index=False))


if __name__=='__main__':main()
