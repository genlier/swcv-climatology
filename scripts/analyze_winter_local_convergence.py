"""Exploratory precursor associations and continuous background trends."""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'data/processed'
METRICS=['convergence_700','convergence_850','stretching_700','stretching_850']


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--single-lag',type=int,choices=[6,12,24])
    parser.add_argument('--transition',action='store_true',help='Exploratory -12 to -6 h difference')
    args=parser.parse_args()
    lags=[args.single_lag] if args.single_lag else [24,18,12,6]
    prefix='winter_local_convergence_' if args.single_lag is None else f'winter_local_convergence_lag{args.single_lag}_'
    if args.transition:
        if args.single_lag is not None:parser.error('Transition and single lag are mutually exclusive')
        lags=[6,12]
        prefix='winter_local_convergence_transition_'
    raw=pd.read_csv(OUT/'winter_local_convergence_series.csv.gz',parse_dates=['time']).set_index('time')
    frame=pd.read_csv(OUT/'winter_genesis_regimes_times.csv.gz',parse_dates=['time'])
    frame['night']=frame.utc_hour.isin([12,18])
    frame['decade']=(frame.season_year-2001.5)/10
    for metric in METRICS:
        precursor=np.stack([raw[metric].reindex(pd.DatetimeIndex(frame.time)-pd.Timedelta(hours=lag)).to_numpy()
                            for lag in lags])
        frame[metric]=precursor[0]-precursor[1] if args.transition else np.mean(precursor,axis=0)
    frame.to_csv(OUT/(prefix+'precursors.csv.gz'),index=False)
    records=[]
    annual_rows=[]
    for metric in METRICS:
        for night in [False,True]:
            subset=frame[frame.night] if night else frame
            thresholds=subset[subset.season_year.le(2001)].groupby(['month','utc_hour'])[metric].quantile(.9)
            subset=subset.copy()
            subset['threshold']=[thresholds.loc[(m,h)] for m,h in zip(subset.month,subset.utc_hour)]
            subset['strong']=np.where(subset[metric].notna(),subset[metric]>subset.threshold,np.nan)
            annual=subset.groupby('season_year').agg(mean=(metric,'mean'),strong_fraction=('strong','mean'),
                                                     valid=(metric,'count'),expected=(metric,'size'))
            for y,r in annual.iterrows():annual_rows.append({'metric':metric,'night_only':night,'season_year':int(y),**r.to_dict()})
            design=sm.add_constant((annual.index.to_numpy()-2001.5)/10)
            for statistic in ['mean','strong_fraction']:
                fit=sm.OLS(annual[statistic].to_numpy(),design).fit(cov_type='HAC',cov_kwds={'maxlags':3})
                ci=fit.conf_int()[1]
                records.append({'metric':metric,'night_only':night,'statistic':statistic,
                                'change_per_decade':float(fit.params[1]),'ci95_low':float(ci[0]),
                                'ci95_high':float(ci[1]),'p_hac':float(fit.pvalues[1]),
                                'early_mean':float(annual[statistic].iloc[:22].mean()),
                                'late_mean':float(annual[statistic].iloc[22:].mean()),
                                'min_winter_valid_fraction':float((annual.valid/annual.expected).min())})
    trends=pd.DataFrame(records)
    for statistic in ['mean','strong_fraction']:
        ids=trends.statistic.eq(statistic)
        trends.loc[ids,'p_fdr_eight']=multipletests(trends.loc[ids,'p_hac'],method='fdr_bh')[1]
    trends.to_csv(OUT/(prefix+'background_trends.csv'),index=False)
    pd.DataFrame(annual_rows).to_csv(OUT/(prefix+'annual.csv'),index=False)
    model_rows=[]
    for metric in METRICS:
        sub=frame.dropna(subset=[metric]).copy()
        early=sub.season_year.le(2001)
        scale=float(sub.loc[early,metric].std())
        z=(sub[metric]-sub.loc[early,metric].mean())/scale
        cats=pd.get_dummies(sub[['regime','month','utc_hour']].astype(str),drop_first=True,dtype=float)
        design=sm.add_constant(pd.concat([sub[['decade']],cats],axis=1)).astype(float)
        base=sm.GLM(sub.events,design,family=sm.families.Poisson()).fit(cov_type='cluster',cov_kwds={'groups':sub.season_year})
        joint_design=design.assign(precursor_z=z)
        fit=sm.GLM(sub.events,joint_design,family=sm.families.Poisson()).fit(cov_type='cluster',cov_kwds={'groups':sub.season_year})
        interval=fit.conf_int().loc['precursor_z']
        model_rows.append({'metric':metric,'n_boxes':len(sub),'n_events':int(sub.events.sum()),
                           'log_rate_per_early_sd':float(fit.params.precursor_z),
                           'ci95_low':float(interval.iloc[0]),'ci95_high':float(interval.iloc[1]),
                           'p_cluster':float(fit.pvalues.precursor_z),
                           'base_year_log_rate_per_decade':float(base.params.decade),
                           'adjusted_year_log_rate_per_decade':float(fit.params.decade),
                           'year_coefficient_attenuation_percent':float(100*(base.params.decade-fit.params.decade)/base.params.decade)})
    models=pd.DataFrame(model_rows)
    models['p_fdr_four']=multipletests(models.p_cluster,method='fdr_bh')[1]
    models.to_csv(OUT/(prefix+'conditional_models.csv'),index=False)
    print(trends.to_string(index=False),flush=True)
    print(models.to_string(index=False),flush=True)
    summary={'scope':'Exploratory precursor diagnostics, not closed budget or causal mediation',
             'lags_hours':[-lag for lag in lags],'metrics':METRICS,
             'precursor_definition':'value(-6h) minus value(-12h)' if args.transition else 'mean at listed lags',
             'background':trends.to_dict(orient='records'),'conditional_models':models.to_dict(orient='records'),
             'uncertainty':'HAC3 annual trends; winter-cluster Poisson does not account for between-winter serial dependence'}
    (OUT/(prefix+'summary.json')).write_text(json.dumps(summary,indent=2),encoding='utf-8')


if __name__=='__main__':main()
