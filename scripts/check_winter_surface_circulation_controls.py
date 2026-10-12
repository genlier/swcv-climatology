"""Fixed surface candidate set with antecedent continuous circulation controls.

Run only after full-period extraction; report all candidates and both windows.
Joint models are descriptive adjustments, not mediation or causal attribution.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
from analyze_winter_surface_mechanism import METRICS

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'data/processed'


def main():
    manifest=json.loads((ROOT/'data/manifests/winter_surface_mechanism_manifest.json').read_text())
    if manifest.get('status')!='complete':raise RuntimeError('Surface download not complete')
    original=pd.read_csv(OUT/'winter_genesis_regimes_times.csv.gz',parse_dates=['time'])
    scores=np.load(OUT/'winter_genesis_regimes_classification.npz')['scores']
    early=original.season_year.le(2001).to_numpy()
    scores=(scores-scores[early].mean(axis=0))/scores[early].std(axis=0,ddof=1)
    rows=[];jointrows=[];jointcoeff=[]
    for window in ['lag6','transition12to6']:
        frame=pd.read_csv(OUT/f'winter_surface_{window}_precursors.csv.gz',parse_dates=['time'])
        if not frame.time.equals(original.time):raise RuntimeError('Surface/PC time misalignment')
        for k in range(8):frame[f'pc{k+1}']=scores[:,k]
        frame=frame[frame.utc_hour.isin([12,18])].copy()
        if frame[METRICS].isna().any().any():raise RuntimeError('Unexpected missing surface candidate')
        frame['decade']=(frame.season_year-2001.5)/10
        fixed=pd.get_dummies(frame[['regime','month','utc_hour']].astype(str),drop_first=True,dtype=float)
        design=sm.add_constant(pd.concat([frame[['decade']],fixed],axis=1)).astype(float)
        for k in range(8):
            design[f'pc{k+1}']=frame[f'pc{k+1}']
            design[f'pc{k+1}_sq']=frame[f'pc{k+1}']**2
        predictors=pd.DataFrame(index=frame.index)
        for metric in METRICS:
            climate=frame[frame.season_year.le(2001)].groupby(['month','utc_hour'])[metric].mean()
            anomaly=frame[metric]-np.array([climate.loc[(m,h)] for m,h in zip(frame.month,frame.utc_hour)])
            early=frame.season_year.le(2001)
            predictors[metric]=(anomaly-anomaly[early].mean())/anomaly[early].std()
        def fit(x):
            model=sm.GLM(frame.events,x,family=sm.families.Poisson()).fit(
                cov_type='cluster',cov_kwds={'groups':frame.season_year},maxiter=100)
            if not model.converged:raise RuntimeError('Surface count model did not converge')
            return model
        baseline=fit(design)
        for metric in METRICS:
            model=fit(design.assign(surface_z=predictors[metric]))
            ci=model.conf_int().loc['surface_z']
            rows.append({'window':window,'metric':metric,'n_events':int(frame.events.sum()),
                'log_rate_per_early_anomaly_sd':float(model.params.surface_z),
                'ci95_low':float(ci.iloc[0]),'ci95_high':float(ci.iloc[1]),'p_cluster':float(model.pvalues.surface_z),
                'base_year_log_rate':float(baseline.params.decade),'adjusted_year_log_rate':float(model.params.decade),
                'attenuation_percent':float(100*(baseline.params.decade-model.params.decade)/baseline.params.decade)})
        full=pd.concat([design,predictors],axis=1)
        joint=fit(full)
        year_ci=joint.conf_int().loc['decade']
        jointrows.append({'window':window,'n_events':int(frame.events.sum()),'n_boxes':len(frame),
            'n_parameters':len(joint.params),'base_year_log_rate':float(baseline.params.decade),
            'joint_year_log_rate':float(joint.params.decade),'joint_year_ci95':year_ci.tolist(),
            'attenuation_percent':float(100*(baseline.params.decade-joint.params.decade)/baseline.params.decade),
            'design_condition_number':float(np.linalg.cond(full.to_numpy())),
            'pearson_dispersion':float(joint.pearson_chi2/joint.df_resid)})
        predictors.corr().to_csv(OUT/f'winter_surface_{window}_candidate_correlations.csv')
        for metric in METRICS:
            ci=joint.conf_int().loc[metric]
            jointcoeff.append({'window':window,'metric':metric,'log_rate_per_early_anomaly_sd':float(joint.params[metric]),
                'ci95_low':float(ci.iloc[0]),'ci95_high':float(ci.iloc[1]),'p_cluster':float(joint.pvalues[metric])})
    single=pd.DataFrame(rows);single['p_fdr_12']=multipletests(single.p_cluster,method='fdr_bh')[1]
    coefficients=pd.DataFrame(jointcoeff);coefficients['p_fdr_12']=multipletests(coefficients.p_cluster,method='fdr_bh')[1]
    single.to_csv(OUT/'winter_surface_continuous_control_models.csv',index=False)
    coefficients.to_csv(OUT/'winter_surface_joint_candidate_coefficients.csv',index=False)
    pd.DataFrame(jointrows).to_csv(OUT/'winter_surface_joint_candidate_year_models.csv',index=False)
    obj={'scope':'Full-period exploratory surface candidates; 24h antecedent regime + linear/squared PC8 controls',
        'individual_models':single.to_dict(orient='records'),'joint_models':jointrows,
        'joint_coefficients':coefficients.to_dict(orient='records'),
        'limits':['Statistical attenuation is not a mediated or attributable fraction',
                  'Joint thermal/BLH/wind proxies may be collinear',
                  'No physical pressure-gradient or heat-flux attribution',
                  'Cluster SE lacks cross-winter serial correction',
                  'Model exploration lacks an independent holdout']}
    (OUT/'winter_surface_continuous_controls_summary.json').write_text(json.dumps(obj,indent=2),encoding='utf-8')
    print(single.to_string(index=False));print(pd.DataFrame(jointrows).to_string(index=False))


if __name__=='__main__':main()
