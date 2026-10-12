"""Colocated fixed-cell count sensitivity; all listed candidates reported."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import xarray as xr
from scipy.stats import norm
from statsmodels.stats.multitest import multipletests
from threadpoolctl import threadpool_limits
from spatial_profile_poisson import fit

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'data/processed'
SURFACE=['t2m','skin_air','dewpoint_depression','log_blh','convergence10']
DYNAMICS=['convergence_700','stretching_700','convergence_850','stretching_850']


def main():
    with xr.open_dataset(OUT/'winter_spatial_mechanism.nc') as raw:cube=raw.load()
    with xr.open_dataset(OUT/'winter_spatial_genesis_counts.nc') as raw:counts=raw.load()
    risk=pd.read_csv(OUT/'winter_genesis_regimes_times.csv.gz',parse_dates=['time'])
    if not pd.DatetimeIndex(counts.time.values).equals(pd.DatetimeIndex(risk.time)):raise RuntimeError('Spatial risk grid not aligned')
    scores=np.load(OUT/'winter_genesis_regimes_classification.npz')['scores']
    early=risk.season_year.le(2001).to_numpy()
    scores=(scores-scores[early].mean(axis=0))/scores[early].std(axis=0,ddof=1)
    night=risk.utc_hour.isin([12,18]).to_numpy()
    frame=risk[night].reset_index(drop=True)
    time=pd.DatetimeIndex(frame.time)
    response=counts.events.sel(time=time).values.reshape(-1)
    cells=np.tile(np.arange(42),len(frame))
    years=np.repeat(frame.season_year.to_numpy(),42)
    cats=pd.get_dummies(frame[['regime','month','utc_hour']].astype(str),drop_first=True,dtype=float)
    common=np.column_stack([(frame.season_year.to_numpy()-2001.5)/10,cats.to_numpy(),scores[night],scores[night]**2])
    base=np.repeat(common,42,axis=0)
    family=[];joint=[];coverage=[];coefficients=[]
    for window in ['lag6','transition12to6']:
        values={}
        for metric in SURFACE+DYNAMICS:
            at6=cube[metric].sel(time=time-pd.Timedelta(hours=6)).values
            at12=cube[metric].sel(time=time-pd.Timedelta(hours=12)).values
            field=at6 if window=='lag6' else at6-at12
            # Each cell, month and risk-clock phase has an early mean; only
            # early data determine normalization, not the event labels.
            anomaly=field.copy()
            for month in [12,1,2]:
                for hour in [12,18]:
                    select=(frame.month.to_numpy()==month)&(frame.utc_hour.to_numpy()==hour)
                    training=select&(frame.season_year.to_numpy()<=2001)
                    with np.errstate(invalid='ignore'):
                        valid=np.isfinite(field[training])
                        denom=valid.sum(axis=0)
                        centre=np.divide(np.nansum(field[training],axis=0),denom,out=np.full(42,np.nan),where=denom>0)
                    anomaly[select]-=centre
            flat=anomaly.reshape(-1)
            training=(years<=2001)&np.isfinite(flat)
            scale=np.std(flat[training],ddof=1)
            if not np.isfinite(scale) or scale<=0:raise RuntimeError(f'Invalid precursor scale {metric}')
            values[metric]=flat/scale
        for metric in SURFACE+DYNAMICS:
            mask=np.isfinite(values[metric])
            baseline=fit(base[mask],response[mask],cells[mask],years[mask])
            model=fit(np.column_stack([base[mask],values[metric][mask]]),response[mask],cells[mask],years[mask])
            coef=float(model['params'][-1]);se=float(np.sqrt(model['covariance'][-1,-1]))
            yearcoef=float(model['params'][0]);yse=float(np.sqrt(model['covariance'][0,0]))
            family.append({'window':window,'metric':metric,'n_events':model['n_events'],'n_rows':model['n_rows'],
                'n_cells':model['n_informative_cells'],'base_year_log_rate':float(baseline['params'][0]),
                'adjusted_year_log_rate':yearcoef,'adjusted_year_ci95_low':yearcoef-1.96*yse,'adjusted_year_ci95_high':yearcoef+1.96*yse,
                'coefficient_per_early_cell_anomaly_sd':coef,'ci95_low':coef-1.96*se,'ci95_high':coef+1.96*se,
                'p_hac3':float(2*norm.sf(abs(coef/se))),'attenuation_percent':float(100*(baseline['params'][0]-yearcoef)/baseline['params'][0]),
                'condition_number':model['information_condition_number'],'max_scaled_gradient':model['max_scaled_gradient']})
            coverage.append({'window':window,'metric':metric,'risk_fraction':float(mask.mean()),
                'excluded_events':int(response[~mask].sum()),'all_night_events':int(response.sum())})
            print(window,metric,'events',model['n_events'],'year',yearcoef,'candidate',coef,flush=True)
        sets={'surface_joint':SURFACE,'700_dynamic_joint':DYNAMICS[:2],
              'surface_700_joint':SURFACE+DYNAMICS[:2]}
        for label,metrics in sets.items():
            predictors=np.column_stack([values[metric] for metric in metrics])
            mask=np.isfinite(predictors).all(axis=1)
            baseline=fit(base[mask],response[mask],cells[mask],years[mask])
            model=fit(np.column_stack([base[mask],predictors[mask]]),response[mask],cells[mask],years[mask])
            yearcoef=float(model['params'][0]);yse=float(np.sqrt(model['covariance'][0,0]))
            joint.append({'window':window,'model':label,'n_events':model['n_events'],'n_rows':model['n_rows'],
                'base_year_log_rate':float(baseline['params'][0]),'adjusted_year_log_rate':yearcoef,
                'year_ci95_low':yearcoef-1.96*yse,'year_ci95_high':yearcoef+1.96*yse,
                'attenuation_percent':float(100*(baseline['params'][0]-yearcoef)/baseline['params'][0]),
                'condition_number':model['information_condition_number'],'max_scaled_gradient':model['max_scaled_gradient']})
            for k,metric in enumerate(metrics):
                coef=float(model['params'][base.shape[1]+k]);se=float(np.sqrt(model['covariance'][base.shape[1]+k,base.shape[1]+k]))
                coefficients.append({'window':window,'model':label,'metric':metric,'coefficient':coef,
                    'ci95_low':coef-1.96*se,'ci95_high':coef+1.96*se,'p_hac3':float(2*norm.sf(abs(coef/se)))})
            print(window,label,'joint year',yearcoef,flush=True)
    single=pd.DataFrame(family);single['p_fdr_18']=multipletests(single.p_hac3,method='fdr_bh')[1]
    coef=pd.DataFrame(coefficients);coef['p_fdr_all']=multipletests(coef.p_hac3,method='fdr_bh')[1]
    single.to_csv(OUT/'winter_spatial_candidate_models.csv',index=False)
    pd.DataFrame(joint).to_csv(OUT/'winter_spatial_joint_models.csv',index=False)
    coef.to_csv(OUT/'winter_spatial_joint_coefficients.csv',index=False)
    pd.DataFrame(coverage).to_csv(OUT/'winter_spatial_candidate_coverage.csv',index=False)
    result={'scope':'Exploratory fixed-cell profile-Poisson sensitivity, annual-score HAC3, no causal attribution',
        'n_winters':44,'n_spatial_cells':42,'night_events_in_region':int(response.sum()),
        'single_models':single.to_dict(orient='records'),'joint_models':joint,'joint_coefficients':coef.to_dict(orient='records'),
        'coverage':coverage,'limits':['Spatial environmental averages remain proxies',
            'Level-dependent eligibility changes the target population','No independent catalogue or reanalysis replication',
            'HAC3 sandwich and normal intervals are approximate for 44 winters',
            'Coefficient attenuation is not a mediated or attributable fraction','Multiple exploratory rounds are not preregistered']}
    (OUT/'winter_spatial_mechanism_summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(pd.DataFrame(joint).to_string(index=False))


if __name__=='__main__':
    with threadpool_limits(limits=1):main()
