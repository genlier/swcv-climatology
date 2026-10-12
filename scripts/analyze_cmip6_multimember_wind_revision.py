"""Compute paired-member, equal-model responses only after all 48 wind groups pass QC."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from extract_cmip6_multimember_wind_revision import STATUS,ROOT,METRICS,valid_part

def main():
    status=json.loads(STATUS.read_text())
    if status['status']!='complete' or len(status['files'])!=48 or status['failures']:
        raise RuntimeError('All 48 QC-passing wind groups required; no partial-ensemble conclusion')
    rows=[]
    for item in status['files']:
        path=ROOT/item['path']
        if not valid_part(path): raise RuntimeError(f'Invalid annual part: {path}')
        data=pd.read_csv(path)
        row={key:item[key] for key in ['model','experiment','member']}
        for metric in METRICS:
            early=data.loc[data.year.between(1951,1980),metric].mean()
            late=data.loc[data.year.between(1985,2014),metric].mean()
            row[metric+'_change']=late-early
        rows.append(row)
    response=pd.DataFrame(rows)
    expected=pd.read_csv(ROOT/'data/processed/cmip6_multimember_z850_responses.csv')
    cols=['model','experiment','member']
    if set(map(tuple,response[cols].values))!=set(map(tuple,expected[cols].values)):
        raise RuntimeError('Wind cohort differs from frozen height cohort')
    metrics=[x+'_change' for x in METRICS]
    models=response.groupby(['model','experiment'])[metrics].mean().reset_index()
    rng=np.random.default_rng(20260915); summaries=[]
    for experiment,group in models.groupby('experiment'):
        assert len(group)==8
        for metric in metrics:
            values=group[metric].to_numpy()
            draws=rng.choice(values,size=(10000,8),replace=True).mean(axis=1)
            low,high=np.quantile(draws,[.025,.975])
            summaries.append({'experiment':experiment,'metric':metric,'models':8,'members_per_model':3,
                'mean':values.mean(),'median':np.median(values),'positive_models':int((values>0).sum()),
                'model_bootstrap_ci_low':low,'model_bootstrap_ci_high':high,
                'scope':'Available-model sampling interval, not control-noise detection or member-uncertainty interval'})
    out=ROOT/'data/processed'
    response.to_csv(out/'cmip6_multimember_ua_revision_responses.csv',index=False)
    models.to_csv(out/'cmip6_multimember_ua_revision_model_summary.csv',index=False)
    pd.DataFrame(summaries).to_csv(out/'cmip6_multimember_ua_revision_multimodel_summary.csv',index=False)
    print(pd.DataFrame(summaries).to_string(index=False))

if __name__=='__main__':main()
