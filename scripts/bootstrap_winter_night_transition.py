"""Grouped exposure-offset count models, paired circular-winter block uncertainty."""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
import statsmodels.api as sm

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'data/processed'
METRICS=['convergence_700','convergence_850','stretching_700','stretching_850']


def grouped(frame,metric):
    sub=frame[frame.utc_hour.isin([12,18])].dropna(subset=[metric]).copy()
    edges=sub.loc[sub.season_year.le(2001),metric].quantile([.2,.4,.6,.8]).to_numpy()
    sub['bin']=np.searchsorted(edges,sub[metric].to_numpy(),side='right')
    table=sub.groupby(['season_year','month','utc_hour','regime','bin']).agg(events=('events','sum'),exposure=('events','size')).reset_index()
    table['decade']=(table.season_year-2001.5)/10
    fixed=pd.get_dummies(table[['month','utc_hour','regime']].astype(str),drop_first=True,dtype=float)
    bins=pd.get_dummies(table['bin'].astype(str),prefix='bin',drop_first=True,dtype=float)
    base=sm.add_constant(pd.concat([table[['decade']],fixed],axis=1)).to_numpy()
    joint=np.column_stack([base,bins.to_numpy()])
    return table,base,joint,edges


def fit(y,design,offset,weights=None,start=None):
    return sm.GLM(y,design,offset=offset,family=sm.families.Poisson(),freq_weights=weights).fit(start_params=start,maxiter=60,disp=0)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--repetitions',type=int,default=300)
    args=parser.parse_args()
    frame=pd.read_csv(OUT/'winter_local_convergence_transition_precursors.csv.gz')
    results=[]
    for metric in METRICS:
        table,base,joint,edges=grouped(frame,metric)
        y=table.events.to_numpy();offset=np.log(table.exposure.to_numpy())
        bfit=fit(y,base,offset);jfit=fit(y,joint,offset)
        estimate=100*(bfit.params[1]-jfit.params[1])/bfit.params[1]
        for block in [3,5]:
            rng=np.random.default_rng(20261004)
            samples=[]
            for repetition in range(args.repetitions):
                starts=rng.integers(0,44,size=int(np.ceil(44/block)))
                ids=((starts[:,None]+np.arange(block))%44).ravel()[:44]
                mult=np.bincount(ids,minlength=44)
                weights=mult[table.season_year.to_numpy()-1980]
                keep=weights>0
                try:
                    b=fit(y[keep],base[keep],offset[keep],weights[keep],bfit.params)
                    j=fit(y[keep],joint[keep],offset[keep],weights[keep],jfit.params)
                    if b.converged and j.converged and abs(b.params[1])>.0001:
                        samples.append([b.params[1],j.params[1],100*(b.params[1]-j.params[1])/b.params[1]])
                except (ValueError,np.linalg.LinAlgError):pass
            if len(samples)<.95*args.repetitions:raise RuntimeError('Too many invalid bootstrap fits')
            ci=np.quantile(samples,[.025,.975],axis=0)
            row={'metric':metric,'block_winters':block,'n_events':int(y.sum()),'n_groups':len(table),
                 'early_quintile_edges':edges.tolist(),'base_year_log_rate':float(bfit.params[1]),
                 'adjusted_year_log_rate':float(jfit.params[1]),'attenuation_percent':float(estimate),
                 'attenuation_ci95':ci[:,2].tolist(),'base_year_ci95':ci[:,0].tolist(),
                 'adjusted_year_ci95':ci[:,1].tolist(),'valid_repetitions':len(samples)}
            results.append(row)
            print(json.dumps(row),flush=True)
            (OUT/'winter_night_transition_block_bootstrap.json').write_text(json.dumps({'scope':'Descriptive coefficient attenuation, not attributable fraction',
                'repetitions':args.repetitions,'results':results},indent=2),encoding='utf-8')


if __name__=='__main__':main()
