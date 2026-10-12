"""Exploratory direct interaction contrasts, not differences in significance."""
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


def main():
    with xr.open_dataset(OUT/'winter_spatial_mechanism.nc') as raw:cube=raw.load()
    with xr.open_dataset(OUT/'winter_spatial_genesis_counts.nc') as raw:counts=raw.load()
    risk=pd.read_csv(OUT/'winter_genesis_regimes_times.csv.gz',parse_dates=['time'])
    scores=np.load(OUT/'winter_genesis_regimes_classification.npz')['scores']
    early=risk.season_year.le(2001).to_numpy()
    scores=(scores-scores[early].mean(axis=0))/scores[early].std(axis=0,ddof=1)
    cats=pd.get_dummies(risk[['regime','month','utc_hour']].astype(str),drop_first=True,dtype=float)
    decade=(risk.season_year.to_numpy()-2001.5)/10
    night=risk.utc_hour.isin([12,18]).to_numpy().astype(float)
    common=np.column_stack([decade,cats.to_numpy(),scores,scores**2,decade*night])
    x=np.repeat(common,42,axis=0);isnight=np.repeat(night,42)
    y=counts.events.values.reshape(-1);cell=np.tile(np.arange(42),len(risk))
    years=np.repeat(risk.season_year.to_numpy(),42)
    baseline=fit(x,y,cell,years)
    contrast_index=common.shape[1]-1
    def contrast(model,k):
        value=float(model['params'][k]);se=float(np.sqrt(model['covariance'][k,k]))
        return {'estimate':value,'ci95_low':value-1.96*se,'ci95_high':value+1.96*se,'p_hac3':float(2*norm.sf(abs(value/se)))}
    result={'scope':'Exploratory all-time fixed-cell direct interactions; annual-score HAC3',
        'n_events':int(y.sum()),'n_winters':44,'year_night_vs_day':contrast(baseline,contrast_index),
        'candidate_interactions':[],'limits':['Interactions are post-spatial-result follow-ups',
            'Not a thermal or anthropogenic causal test','Same catalogue and ERA5; no independent replication']}
    for metric in ['t2m','dewpoint_depression']:
        field=cube[metric].sel(time=pd.DatetimeIndex(risk.time)-pd.Timedelta(hours=6)).values
        anomaly=field.copy()
        for month in [12,1,2]:
            for hour in [0,6,12,18]:
                mask=(risk.month.to_numpy()==month)&(risk.utc_hour.to_numpy()==hour)
                anomaly[mask]-=field[mask&early].mean(axis=0)
        scaled=(anomaly/anomaly[early].std(ddof=1)).reshape(-1)
        model=fit(np.column_stack([x,scaled,scaled*isnight]),y,cell,years)
        result['candidate_interactions'].append({'metric':metric,'night_minus_day_candidate':contrast(model,-1),
            'day_candidate':contrast(model,-2),'year_night_vs_day':contrast(model,contrast_index)})
        print(metric,json.dumps(result['candidate_interactions'][-1]),flush=True)
    p=[row['night_minus_day_candidate']['p_hac3'] for row in result['candidate_interactions']]
    corrected=multipletests(p,method='fdr_bh')[1]
    for row,q in zip(result['candidate_interactions'],corrected):row['night_minus_day_candidate']['p_fdr_2']=float(q)
    (OUT/'winter_spatial_day_night_interaction_summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print('Baseline year interaction',result['year_night_vs_day'])


if __name__=='__main__':
    with threadpool_limits(limits=1):main()
