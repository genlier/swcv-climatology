"""Explore broad-regime residuals using continuous, antecedent-only PC scores."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'data/processed'


def main():
    frame=pd.read_csv(OUT/'winter_local_convergence_transition_precursors.csv.gz',parse_dates=['time'])
    original=pd.read_csv(OUT/'winter_genesis_regimes_times.csv.gz',parse_dates=['time'])
    classification=np.load(OUT/'winter_genesis_regimes_classification.npz')
    scores=classification['scores']
    if len(scores)!=len(original) or not original.time.equals(frame.time):
        raise RuntimeError('Antecedent PC rows not aligned with transition risk rows')
    early=original.season_year.le(2001).to_numpy()
    scores=(scores-scores[early].mean(axis=0))/scores[early].std(axis=0,ddof=1)
    for k in range(8):frame[f'pc{k+1}']=scores[:,k]
    rows=[]
    for metric in ['convergence_700','convergence_850','stretching_700','stretching_850']:
        sub=frame[frame.utc_hour.isin([12,18])].dropna(subset=[metric]).copy()
        early=sub.season_year.le(2001)
        x=(sub[metric]-sub.loc[early,metric].mean())/sub.loc[early,metric].std()
        fixed=pd.get_dummies(sub[['regime','month','utc_hour']].astype(str),drop_first=True,dtype=float)
        design=sm.add_constant(pd.concat([sub[['decade']],fixed],axis=1)).astype(float)
        original_base=None
        for control in ['regimes_only','plus_linear_pc8','plus_linear_and_squared_pc8']:
            covariates=design.copy()
            if control!='regimes_only':
                for k in range(8):covariates[f'pc{k+1}']=sub[f'pc{k+1}']
            if control=='plus_linear_and_squared_pc8':
                for k in range(8):covariates[f'pc{k+1}_sq']=sub[f'pc{k+1}']**2
            base=sm.GLM(sub.events,covariates,family=sm.families.Poisson()).fit(
                cov_type='cluster',cov_kwds={'groups':sub.season_year},maxiter=100)
            fit=sm.GLM(sub.events,covariates.assign(transition=x),family=sm.families.Poisson()).fit(
                cov_type='cluster',cov_kwds={'groups':sub.season_year},maxiter=100)
            if not base.converged or not fit.converged:raise RuntimeError('Count model did not converge')
            if original_base is None:original_base=float(base.params.decade)
            ci=fit.conf_int().loc['transition'];year_ci=base.conf_int().loc['decade']
            rows.append({'metric':metric,'control':control,'n_events':int(sub.events.sum()),'n_boxes':len(sub),
                'n_parameters_base':len(base.params),'base_year_log_rate':float(base.params.decade),
                'base_year_ci95_low':float(year_ci.iloc[0]),'base_year_ci95_high':float(year_ci.iloc[1]),
                'circulation_control_year_attenuation_percent':float(100*(original_base-base.params.decade)/original_base),
                'joint_year_log_rate':float(fit.params.decade),'transition_log_rate_per_early_sd':float(fit.params.transition),
                'transition_ci95_low':float(ci.iloc[0]),'transition_ci95_high':float(ci.iloc[1]),
                'transition_p_cluster':float(fit.pvalues.transition),
                'transition_additional_year_attenuation_percent':float(100*(base.params.decade-fit.params.decade)/base.params.decade),
                'design_condition_number':float(np.linalg.cond(covariates.to_numpy()))})
            print(metric,control,'year:',base.params.decade,'transition:',fit.params.transition,flush=True)
    table=pd.DataFrame(rows)
    table['transition_p_fdr_12']=multipletests(table.transition_p_cluster,method='fdr_bh')[1]
    table.to_csv(OUT/'winter_continuous_circulation_control_models.csv',index=False)
    result={'scope':'Exploratory sensitivity to within-regime continuous antecedent circulation; not causal attribution',
        'antecedent_hours':24,'pc_components':8,'early_training_period':'1980–2001',
        'models':table.to_dict(orient='records'),
        'limits':['PCs capture only 84.2% of early peripheral feature variance',
                  'Linear and squared terms do not fully control all circulation differences',
                  'Winter cluster SE does not correct between-winter serial dependence',
                  'No independent temporal holdout; models are exploratory',
                  'Coefficient attenuation is not a causal or GHG-attributable fraction']}
    (OUT/'winter_continuous_circulation_controls_summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(table.to_string(index=False))


if __name__=='__main__':main()
