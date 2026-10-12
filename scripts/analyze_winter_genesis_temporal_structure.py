"""Winter local genesis timing and event clustering; no causal interpretation."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'data/processed'
YEARS=np.arange(1980,2024)


def main():
    frame=pd.read_csv(OUT/'events_terrain_qc_intensity.csv.gz')
    frame['genesis']=pd.to_datetime(frame.genesis_time,utc=True).dt.tz_localize(None)
    frame['season_year']=frame.genesis.dt.year+frame.genesis.dt.month.eq(12).astype(int)
    frame=frame[frame.track_cluster.eq(1)&frame.genesis.dt.month.isin([12,1,2])&frame.season_year.between(1980,2023)].copy()
    frame['beijing_hour']=(frame.genesis.dt.hour+8)%24
    frame['night_20_08']=frame.beijing_hour.ge(20)|frame.beijing_hour.lt(8)
    rows=[]
    design=sm.add_constant(pd.DataFrame({'decade':(YEARS-YEARS.mean())/10}))
    for threshold in [24,48,72]:
        for year in YEARS:
            g=frame[frame.season_year.eq(year)].sort_values('genesis')
            episodes=0 if len(g)==0 else 1+int((g.genesis.diff().dt.total_seconds()/3600>threshold).sum())
            rows.append({'season_year':int(year),'gap_hours':threshold,'events':len(g),'episodes':episodes,
                         'events_per_episode':len(g)/episodes if episodes else np.nan,
                         'night_events':int(g.night_20_08.sum()),'day_events':int((~g.night_20_08).sum()),
                         'median_observed_hours':float(g.observed_hours.median()),
                         'median_duration_hours':float(g.duration_hours.median())})
    annual=pd.DataFrame(rows)
    annual.to_csv(OUT/'winter_genesis_temporal_structure_annual.csv',index=False)
    records=[]
    for gap in [24,48,72]:
        sub=annual[annual.gap_hours.eq(gap)]
        cols=['episodes','events_per_episode']
        if gap==48:
            cols+=['events','night_events','day_events','median_observed_hours','median_duration_hours']
        for col in cols:
            x=sub[col].to_numpy()
            fit=sm.OLS(x,design).fit(cov_type='HAC',cov_kwds={'maxlags':3})
            ci=fit.conf_int().loc['decade']
            records.append({'gap_hours':gap,'metric':col,'early_mean':float(x[:22].mean()),
                            'late_mean':float(x[22:].mean()),'ols_change_per_decade':float(fit.params.decade),
                            'ci95_low':float(ci.iloc[0]),'ci95_high':float(ci.iloc[1]),'p_hac_unadjusted':float(fit.pvalues.decade)})
    result=pd.DataFrame(records)
    result.to_csv(OUT/'winter_genesis_temporal_structure_statistics.csv',index=False)
    base=annual[annual.gap_hours.eq(48)]
    contrast=sm.OLS((base.night_events-base.day_events).to_numpy(),design).fit(cov_type='HAC',cov_kwds={'maxlags':3})
    # Paired winter counts: formal night/day slope contrast, not separate p values.
    ci=contrast.conf_int().loc['decade']
    parts=[]
    for night,col in [(0,'day_events'),(1,'night_events')]:
        parts.append(pd.DataFrame({'count':base[col].to_numpy(),'decade':design.decade.to_numpy(),
                                  'night':night,'season_year':YEARS}))
    long=pd.concat(parts,ignore_index=True)
    long['interaction']=long.decade*long.night
    fit=sm.GLM(long['count'],sm.add_constant(long[['decade','night','interaction']]),
               family=sm.families.Poisson()).fit(cov_type='cluster',cov_kwds={'groups':long.season_year})
    tests=[float(contrast.pvalues.decade),float(fit.pvalues.interaction)]
    adjusted=multipletests(tests,method='fdr_bh')[1]
    hourly=frame.groupby(['season_year','beijing_hour']).size().unstack(fill_value=0).reindex(index=YEARS,columns=range(24),fill_value=0)
    hourly.to_csv(OUT/'winter_genesis_beijing_hour_counts.csv')
    summary={'scope':'Post-first-pass exploratory night/day follow-up, not preregistered',
             'night_definition':'Beijing civil time 20:00 inclusive to 08:00 exclusive',
             'night_ols_share_of_total_slope':float(result.loc[result.metric.eq('night_events'),'ols_change_per_decade'].iloc[0]/result.loc[result.metric.eq('events'),'ols_change_per_decade'].iloc[0]),
             'paired_absolute_slope_contrast':{'night_minus_day_events_per_decade':float(contrast.params.decade),
                                              'ci95_low':float(ci.iloc[0]),'ci95_high':float(ci.iloc[1]),'p_fdr_two_followups':float(adjusted[0])},
             'relative_trend_interaction':{'night_vs_day_log_rate_per_decade':float(fit.params.interaction),
                                            'ci95':fit.conf_int().loc['interaction'].tolist(),'p_fdr_two_followups':float(adjusted[1])},
             'limitation':'Exploratory two-test correction does not cover overall research selection; cluster Poisson does not handle between-winter serial dependence'}
    timing=[]
    for definition,start,clock in [('civil_18_06',18,frame.beijing_hour.to_numpy()),
                                   ('civil_20_08',20,frame.beijing_hour.to_numpy()),
                                   ('civil_22_10',22,frame.beijing_hour.to_numpy()),
                                   ('mean_solar_20_08',20,(frame.genesis.dt.hour.to_numpy()+frame.genesis_lon.to_numpy()/15)%24)]:
        night=((clock-start)%24)<12
        tmp=frame.assign(night=night)
        annual_n=tmp.groupby('season_year').night.sum().reindex(YEARS,fill_value=0).to_numpy()
        annual_total=tmp.groupby('season_year').size().reindex(YEARS,fill_value=0).to_numpy()
        contrast_fit=sm.OLS(2*annual_n-annual_total,design).fit(cov_type='HAC',cov_kwds={'maxlags':3})
        nfit=sm.OLS(annual_n,design).fit(cov_type='HAC',cov_kwds={'maxlags':3})
        interval=contrast_fit.conf_int().loc['decade'].to_numpy()
        timing.append({'definition':definition,'night_slope_per_decade':float(nfit.params.decade),
                       'night_minus_day_slope_per_decade':float(contrast_fit.params.decade),
                       'contrast_ci95':interval.tolist(),'p_unadjusted':float(contrast_fit.pvalues.decade)})
    corrected=multipletests([r['p_unadjusted'] for r in timing],method='fdr_bh')[1]
    for row,adj in zip(timing,corrected):row['p_fdr_four_timing_sensitivities']=float(adj)
    summary['night_definition_sensitivity']=timing
    (OUT/'winter_genesis_temporal_structure_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(result.to_string(index=False))
    print(json.dumps(summary,indent=2))
    print(json.dumps({'events':len(frame),'note':'Exploratory HAC intervals, no multiplicity correction; Beijing civil time, not solar time'}))


if __name__=='__main__':main()
