"""Frozen classifier, common risk-set lag sensitivity; no outcome-guided fitting."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from analyze_winter_genesis_regimes import analyze

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'data/processed'


def main():
    frame=pd.read_csv(OUT/'winter_genesis_regimes_times.csv.gz',parse_dates=['time'])
    labels=frame.set_index('time').regime
    shifted={lag:labels.reindex(pd.DatetimeIndex(frame.time)+pd.Timedelta(hours=24-lag)).to_numpy()
             for lag in [12,24,48]}
    common=np.logical_and.reduce([np.isfinite(x) for x in shifted.values()])
    rows=[]
    for lag in [12,24,48]:
        current=frame.loc[common].copy()
        current['regime']=shifted[lag][common].astype(int)
        for strata in [['regime'],['regime','month','utc_hour']]:
            _,summary=analyze(current,strata,block=5)
            row={'lag_hours':lag,'strata':strata,'n_common_boxes':len(current),
                 'n_common_events':int(current.events.sum()),'n_omitted_events':int(frame.events.sum()-current.events.sum()),
                 'summary':summary}
            rows.append(row)
            print(json.dumps(row),flush=True)
    (OUT/'winter_genesis_regimes_lag_sensitivity.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')


if __name__=='__main__':main()
