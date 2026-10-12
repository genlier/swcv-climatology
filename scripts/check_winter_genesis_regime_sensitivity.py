"""Predefined class-count and block-length sensitivities of descriptive decomposition."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from analyze_winter_genesis_regimes import analyze,SEED

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'data/processed'


def main():
    frozen=np.load(OUT/'winter_genesis_regimes_classification.npz')
    scores,early=frozen['scores'],frozen['early']
    frame=pd.read_csv(OUT/'winter_genesis_regimes_times.csv.gz',parse_dates=['time'])
    results=[]
    for k in [3,4,5]:
        current=frame.copy()
        km=KMeans(n_clusters=k,n_init=30,random_state=SEED).fit(scores[early])
        current['regime']=km.predict(scores)
        for block in [3,5]:
            for strata in [['regime'],['regime','month','utc_hour']]:
                _,summary=analyze(current,strata,block=block)
                results.append({'classes':k,'block_winters':block,'strata':strata,'summary':summary})
                print(json.dumps(results[-1]),flush=True)
    (OUT/'winter_genesis_regimes_sensitivity.json').write_text(json.dumps(results,indent=2),encoding='utf-8')


if __name__=='__main__':main()
