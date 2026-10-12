"""Bounded sensitivity of direct day/night contrasts: scale, time, safety, HAC."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import xarray as xr
from scipy.stats import norm
from threadpoolctl import threadpool_limits
from spatial_profile_poisson import fit,hac_covariance

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'data/processed'


def main():
    with xr.open_dataset(OUT/'winter_spatial_mechanism.nc') as raw:cube=raw.load()
    with xr.open_dataset(OUT/'winter_spatial_genesis_counts.nc') as raw:counts=raw.load()
    original=pd.read_csv(OUT/'winter_genesis_regimes_times.csv.gz',parse_dates=['time'])
    pc=np.load(OUT/'winter_genesis_regimes_classification.npz')['scores']
    early=original.season_year.le(2001).to_numpy()
    pc=(pc-pc[early].mean(axis=0))/pc[early].std(axis=0,ddof=1)
    results=[]
    for variant in ['full_1degree','early_1degree','late_1degree','700_eligible_1degree','coarse_2degree']:
        selection=np.ones(len(original),dtype=bool)
        if variant=='early_1degree':selection=early
        if variant=='late_1degree':selection=~early
        frame=original[selection].reset_index(drop=True)
        time=pd.DatetimeIndex(frame.time)
        y=counts.events.sel(time=time).values
        fields={m:cube[m].sel(time=time-pd.Timedelta(hours=6)).values for m in ['t2m','dewpoint_depression']}
        ncell=42
        allowed=np.ones_like(y,dtype=bool)
        if variant=='700_eligible_1degree':allowed=np.isfinite(cube.convergence_700.sel(time=time-pd.Timedelta(hours=6)).values)
        if variant=='coarse_2degree':
            # Twelve disjoint groups over the same fixed domain; easternmost
            # groups have width 1 degree, all others 2 degrees. No events lost.
            group=(np.arange(42)//7//2)*4+(np.arange(42)%7//2)
            weights=np.cos(np.deg2rad(cube.cell_lat.values))
            y=np.column_stack([y[:,group==g].sum(axis=1) for g in range(12)])
            fields={m:np.column_stack([np.average(f[:,group==g],axis=1,weights=weights[group==g]) for g in range(12)]) for m,f in fields.items()}
            ncell=12;allowed=np.ones_like(y,dtype=bool)
        cats=pd.get_dummies(frame[['regime','month','utc_hour']].astype(str),drop_first=True,dtype=float)
        decade=(frame.season_year.to_numpy()-2001.5)/10
        night=frame.utc_hour.isin([12,18]).to_numpy().astype(float)
        common=np.column_stack([decade,cats.to_numpy(),pc[selection],pc[selection]**2,decade*night])
        x=np.repeat(common,ncell,axis=0);isnight=np.repeat(night,ncell)
        response=y.reshape(-1);cell=np.tile(np.arange(ncell),len(frame));years=np.repeat(frame.season_year.to_numpy(),ncell)
        keep=allowed.reshape(-1);idx=common.shape[1]-1
        for metric in ['baseline','t2m','dewpoint_depression']:
            design=x
            if metric!='baseline':
                # Normalization always comes from the full early period, not
                # a late-only fit. Early/late subset models retain units.
                trainfield=cube[metric].sel(time=pd.DatetimeIndex(original.time[early])-pd.Timedelta(hours=6)).values
                if ncell==12:trainfield=np.column_stack([np.average(trainfield[:,group==g],axis=1,weights=weights[group==g]) for g in range(12)])
                train=original[early].reset_index(drop=True)
                anomaly=fields[metric].copy();training_anomaly=trainfield.copy()
                for month in [12,1,2]:
                    for hour in [0,6,12,18]:
                        target=(frame.month.to_numpy()==month)&(frame.utc_hour.to_numpy()==hour)
                        training=(train.month.to_numpy()==month)&(train.utc_hour.to_numpy()==hour)
                        centre=trainfield[training].mean(axis=0)
                        anomaly[target]-=centre;training_anomaly[training]-=centre
                scaled=(anomaly/training_anomaly.std(ddof=1)).reshape(-1)
                design=np.column_stack([x,scaled,scaled*isnight])
            model=fit(design[keep],response[keep],cell[keep],years[keep])
            for lag in [0,1,3,5,7]:
                cov=hac_covariance(model['information_inverse'],model['annual_efficient_scores'],lag)
                for contrast,k in [('year_night_minus_day',idx)]+([] if metric=='baseline' else [('candidate_night_minus_day',design.shape[1]-1)]):
                    value=float(model['params'][k]);se=float(np.sqrt(cov[k,k]))
                    results.append({'variant':variant,'candidate':metric,'contrast':contrast,'annual_hac_lags':lag,
                        'estimate':value,'ci95_low':value-1.96*se,'ci95_high':value+1.96*se,
                        'p_hac':float(2*norm.sf(abs(value/se))),'n_events':model['n_events'],
                        'n_winters':model['n_winters'],'n_cells_informative':model['n_informative_cells'],
                        'condition_number':model['information_condition_number']})
            print(variant,metric,'events',model['n_events'],'contrast',model['params'][idx],flush=True)
            pd.DataFrame(results).to_csv(OUT/'winter_spatial_interaction_robustness.csv',index=False)
    obj={'scope':'Exploratory robustness, not independent temporal validation or causal attribution',
        'results':results,'limits':['Time splits chosen after full-record exploration',
            'Coarse cells average a different spatial support; eastern edge is 1 degree wide',
            'Safety subset changes event population','Intervals are sensitivity diagnostics, no family correction across redundant bandwidths']}
    (OUT/'winter_spatial_robustness_summary.json').write_text(json.dumps(obj,indent=2),encoding='utf-8')


if __name__=='__main__':
    with threadpool_limits(limits=1):main()
