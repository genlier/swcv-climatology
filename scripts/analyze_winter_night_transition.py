"""Exploratory night-only association of precursor transition and genesis."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'data/processed'
METRICS=['convergence_700','convergence_850','stretching_700','stretching_850']


def main():
    data=pd.read_csv(OUT/'winter_local_convergence_transition_precursors.csv.gz')
    frame=data[data.utc_hour.isin([12,18])].copy()
    rows=[]
    for metric in METRICS:
        sub=frame.dropna(subset=[metric]).copy()
        scale=float(sub.loc[sub.season_year.le(2001),metric].std())
        mean=float(sub.loc[sub.season_year.le(2001),metric].mean())
        sub['x']=(sub[metric]-mean)/scale
        cats=pd.get_dummies(sub[['regime','month','utc_hour']].astype(str),drop_first=True,dtype=float)
        design=sm.add_constant(pd.concat([sub[['decade']],cats],axis=1)).astype(float)
        base=sm.GLM(sub.events,design,family=sm.families.Poisson()).fit(cov_type='cluster',cov_kwds={'groups':sub.season_year})
        fit=sm.GLM(sub.events,design.assign(transition_z=sub.x),family=sm.families.Poisson()).fit(cov_type='cluster',cov_kwds={'groups':sub.season_year})
        ci=fit.conf_int().loc['transition_z']
        annual=sub.groupby('season_year').x.mean()
        trend=sm.OLS(annual.to_numpy(),sm.add_constant((annual.index.to_numpy()-2001.5)/10)).fit(cov_type='HAC',cov_kwds={'maxlags':3})
        rows.append({'metric':metric,'n_boxes':len(sub),'n_events':int(sub.events.sum()),
                     'log_rate_per_early_sd':float(fit.params.transition_z),'ci95_low':float(ci.iloc[0]),'ci95_high':float(ci.iloc[1]),
                     'p_cluster':float(fit.pvalues.transition_z),'precursor_trend_sd_per_decade':float(trend.params[1]),
                     'base_year_log_rate_per_decade':float(base.params.decade),'adjusted_year_log_rate_per_decade':float(fit.params.decade),
                     'year_coefficient_attenuation_percent':float(100*(base.params.decade-fit.params.decade)/base.params.decade),
                     'trend_times_association_log_rate_per_decade':float(trend.params[1]*fit.params.transition_z)})
    table=pd.DataFrame(rows)
    table['p_fdr_four']=multipletests(table.p_cluster,method='fdr_bh')[1]
    table.to_csv(OUT/'winter_night_transition_conditional_models.csv',index=False)
    print(table.to_string(index=False))
    summary={'scope':'Post-hoc hypothesis generation: night-only -12 to -6h transition. Not causal mediation.',
             'models':table.to_dict(orient='records'),
             'limitations':['Residual weather differences within broad regimes','ERA5 observing-system homogeneity',
                            'Catalogue-conditioned counts','No independent replication','No cross-winter serial correction in cluster GLM',
                            'Model coefficient attenuation is not attributable fraction']}
    (OUT/'winter_night_transition_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')


if __name__=='__main__':main()
