"""Predeclared surface precursor diagnostics, full reporting, exploratory only."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
from check_winter_diurnal_fixed_mask import harmonic

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'data/processed'
METRICS=['basin_t2m_k','west_minus_basin_t2m_k','basin_skin_minus_air_k',
         'basin_log_blh','basin_dewpoint_depression_k','basin_convergence10_s']


def main():
    series=pd.read_csv(OUT/'winter_surface_mechanism_series.csv.gz',parse_dates=['time']).set_index('time')
    risk=pd.read_csv(OUT/'winter_genesis_regimes_times.csv.gz',parse_dates=['time'])
    risk['decade']=(risk.season_year-2001.5)/10
    rows=[];trends=[];coverage=[]
    # Actual-clock thermal/wind phase changes, distinct from event precursor lags.
    actual=series.reset_index()
    actual['season_year']=actual.time.dt.year+actual.time.dt.month.eq(12).astype(int)
    actual['utc_hour']=actual.time.dt.hour
    actual=actual[actual.time.dt.month.isin([12,1,2])&actual.season_year.between(1980,2023)]
    harmonic_rows=[];phase_rows=[]
    for metric in METRICS:
        annual=actual.groupby(['season_year','utc_hour'])[metric].mean().unstack().reindex(columns=[0,6,12,18])
        if annual.shape!=(44,4) or annual.isna().any().any():raise RuntimeError('Incomplete surface clock-phase series')
        for statistic,values in [('mean',annual.mean(axis=1).to_numpy()),('amplitude',harmonic(annual.to_numpy()))]:
            fit=sm.OLS(values,sm.add_constant((annual.index.to_numpy()-2001.5)/10)).fit(cov_type='HAC',cov_kwds={'maxlags':3})
            ci=fit.conf_int()[1]
            harmonic_rows.append({'metric':metric,'statistic':statistic,'change_per_decade':float(fit.params[1]),
                'ci95_low':float(ci[0]),'ci95_high':float(ci[1]),'p_hac3':float(fit.pvalues[1]),
                'early_mean':float(values[:22].mean()),'late_mean':float(values[22:].mean())})
        for hour in [0,6,12,18]:
            fit=sm.OLS(annual[hour].to_numpy(),sm.add_constant((annual.index.to_numpy()-2001.5)/10)).fit(cov_type='HAC',cov_kwds={'maxlags':3})
            ci=fit.conf_int()[1]
            phase_rows.append({'metric':metric,'utc_hour':hour,'change_per_decade':float(fit.params[1]),
                'ci95_low':float(ci[0]),'ci95_high':float(ci[1]),'p_hac3':float(fit.pvalues[1]),
                'early_mean':float(annual[hour].iloc[:22].mean()),'late_mean':float(annual[hour].iloc[22:].mean())})
    harmonics=pd.DataFrame(harmonic_rows)
    harmonics['p_fdr_12']=multipletests(harmonics.p_hac3,method='fdr_bh')[1]
    phases=pd.DataFrame(phase_rows)
    phases['p_fdr_24']=multipletests(phases.p_hac3,method='fdr_bh')[1]
    harmonics.to_csv(OUT/'winter_surface_actual_clock_harmonic_trends.csv',index=False)
    phases.to_csv(OUT/'winter_surface_actual_clock_phase_trends.csv',index=False)
    for precursor in ['lag6','transition12to6']:
        frame=risk.copy()
        for metric in METRICS:
            at6=series[metric].reindex(pd.DatetimeIndex(frame.time)-pd.Timedelta(hours=6)).to_numpy()
            at12=series[metric].reindex(pd.DatetimeIndex(frame.time)-pd.Timedelta(hours=12)).to_numpy()
            frame[metric]=at6 if precursor=='lag6' else at6-at12
        frame.to_csv(OUT/f'winter_surface_{precursor}_precursors.csv.gz',index=False)
        for metric in METRICS:
            # Monthly/hour early-climatology anomalies preserve trends and phase strata.
            climat=frame[frame.season_year.le(2001)].groupby(['month','utc_hour'])[metric].mean()
            frame['anomaly']=frame[metric]-np.array([climat.loc[(m,h)] for m,h in zip(frame.month,frame.utc_hour)])
            for hour,g in frame.groupby('utc_hour'):
                annual=g.groupby('season_year').anomaly.mean()
                design=sm.add_constant((annual.index.to_numpy()-2001.5)/10)
                fit=sm.OLS(annual.to_numpy(),design).fit(cov_type='HAC',cov_kwds={'maxlags':3})
                ci=fit.conf_int()[1]
                trends.append({'precursor':precursor,'metric':metric,'utc_risk_hour':int(hour),
                               'change_per_decade':float(fit.params[1]),'ci95_low':float(ci[0]),'ci95_high':float(ci[1]),
                               'p_hac3':float(fit.pvalues[1])})
            night=frame[frame.utc_hour.isin([12,18])].dropna(subset=['anomaly']).copy()
            early=night.season_year.le(2001)
            center=float(night.loc[early,'anomaly'].mean());scale=float(night.loc[early,'anomaly'].std())
            if not np.isfinite(scale) or scale<=0:raise RuntimeError('Degenerate surface predictor')
            night['x']=(night.anomaly-center)/scale
            fixed=pd.get_dummies(night[['regime','month','utc_hour']].astype(str),drop_first=True,dtype=float)
            design=sm.add_constant(pd.concat([night[['decade']],fixed],axis=1)).astype(float)
            base=sm.GLM(night.events,design,family=sm.families.Poisson()).fit(cov_type='cluster',cov_kwds={'groups':night.season_year})
            fit=sm.GLM(night.events,design.assign(surface_z=night.x),family=sm.families.Poisson()).fit(cov_type='cluster',cov_kwds={'groups':night.season_year})
            ci=fit.conf_int().loc['surface_z']
            rows.append({'precursor':precursor,'metric':metric,'n_events':int(night.events.sum()),'n_boxes':len(night),
                         'log_rate_per_early_anomaly_sd':float(fit.params.surface_z),'ci95_low':float(ci.iloc[0]),'ci95_high':float(ci.iloc[1]),
                         'p_winter_cluster':float(fit.pvalues.surface_z),'base_year_log_rate':float(base.params.decade),
                         'adjusted_year_log_rate':float(fit.params.decade),
                         'year_coefficient_attenuation_percent':float(100*(base.params.decade-fit.params.decade)/base.params.decade)})
            coverage.append({'precursor':precursor,'metric':metric,'valid_fraction':float(frame[metric].notna().mean())})
    model=pd.DataFrame(rows)
    model['p_fdr_12']=multipletests(model.p_winter_cluster,method='fdr_bh')[1]
    trend=pd.DataFrame(trends)
    trend['p_fdr_48']=multipletests(trend.p_hac3,method='fdr_bh')[1]
    model.to_csv(OUT/'winter_surface_mechanism_conditional_models.csv',index=False)
    trend.to_csv(OUT/'winter_surface_mechanism_phase_trends.csv',index=False)
    result={'scope':'Exploratory surface thermal/BLH/wind proxies, not causal attribution',
            'coverage':coverage,'models':model.to_dict(orient='records'),'phase_trends':trend.to_dict(orient='records'),
            'actual_clock_harmonics':harmonics.to_dict(orient='records'),'actual_clock_phases':phases.to_dict(orient='records'),
            'limits':['BLH and skin temperature are model-derived','No surface heat flux or pressure-gradient attribution',
                      'Catalogue-conditioned event counts','No independent-product validation','Post-first-round exploratory hypotheses']}
    (OUT/'winter_surface_mechanism_summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(model.to_string(index=False),flush=True)


if __name__=='__main__':main()
