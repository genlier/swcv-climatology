"""Report nocturnal growth by early-fixed event strength and duration thresholds.

Peak event diagnostics are six-hour sampled lifetime summaries, not genesis
intensity; thresholded frequency is not a homogeneous detection validation.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'data/processed'
YEARS=np.arange(1980,2024)


def trend(counts):
    fit=sm.OLS(counts.to_numpy(),sm.add_constant((YEARS-2001.5)/10)).fit(cov_type='HAC',cov_kwds={'maxlags':3})
    ci=fit.conf_int()[1]
    return {'slope_per_decade':float(fit.params[1]),'ci95_low':float(ci[0]),'ci95_high':float(ci[1]),'p_hac3':float(fit.pvalues[1])}


def main():
    data=pd.read_csv(OUT/'events_terrain_qc_intensity.csv.gz')
    data['genesis']=pd.to_datetime(data.genesis_time,utc=True).dt.tz_localize(None)
    data['season_year']=data.genesis.dt.year+data.genesis.dt.month.eq(12).astype(int)
    data=data[data.track_cluster.eq(1)&data.genesis.dt.month.isin([12,1,2])&data.season_year.between(1980,2023)].copy()
    hour=(data.genesis.dt.hour+8)%24
    data['night']=hour.ge(20)|hour.lt(8)
    if len(data)!=732 or data.night.sum()!=423:raise RuntimeError('Target event population mismatch')
    early=data.season_year.le(2001)
    metrics={'peak_vorticity700':'max_vorticity_s_1_700_c25',
             'peak_height_deficit700':'max_height_deficit_gpm_700_c25'}
    criteria={'all_events':pd.Series(True,index=data.index),'duration_ge12h':data.duration_hours.ge(12),
              'duration_ge24h':data.duration_hours.ge(24)}
    thresholds={}
    partitions={}
    for label,column in metrics.items():
        for quantile in [.5,.75]:
            cutoff=float(data.loc[early,column].quantile(quantile))
            key=f'{label}_ge_early_q{int(100*quantile)}'
            thresholds[key]={'column':column,'cutoff':cutoff,'early_valid_events':int(data.loc[early,column].notna().sum())}
            criteria[key]=data[column].ge(cutoff)
        key=f'{label}_ge_early_q50'
        partitions[label]={'below':data[column].lt(thresholds[key]['cutoff']),
                           'above':criteria[key],'missing':data[column].isna()}
    records=[];annuals=[]
    for criterion,select in criteria.items():
        for clock,mask in [('night',data.night),('day',~data.night)]:
            chosen=data[select&mask]
            counts=chosen.groupby('season_year').size().reindex(YEARS,fill_value=0)
            records.append({'criterion':criterion,'clock':clock,'events':len(chosen),
                'early_events_per_winter':float(counts.iloc[:22].mean()),'late_events_per_winter':float(counts.iloc[22:].mean()),**trend(counts)})
            for wy,n in counts.items():annuals.append({'criterion':criterion,'clock':clock,'season_year':int(wy),'events':int(n)})
    partitionrows=[]
    for label,parts in partitions.items():
        total=np.zeros(44)
        for category,mask in parts.items():
            chosen=data[mask&data.night]
            counts=chosen.groupby('season_year').size().reindex(YEARS,fill_value=0)
            total+=counts.to_numpy()
            partitionrows.append({'metric':label,'category':category,'night_events':len(chosen),**trend(counts)})
        reference=data[data.night].groupby('season_year').size().reindex(YEARS,fill_value=0).to_numpy()
        np.testing.assert_array_equal(total,reference)
    result=pd.DataFrame(records)
    result['p_fdr_14']=multipletests(result.p_hac3,method='fdr_bh')[1]
    result.to_csv(OUT/'winter_night_event_strength_threshold_trends.csv',index=False)
    pd.DataFrame(annuals).to_csv(OUT/'winter_night_event_strength_threshold_annual.csv',index=False)
    pd.DataFrame(partitionrows).to_csv(OUT/'winter_night_event_strength_partition_trends.csv',index=False)
    obj={'scope':'Exploratory strength/duration sensitivity of catalogue nocturnal growth',
         'thresholds':thresholds,'trends':result.to_dict(orient='records'),'night_partitions':partitionrows,
         'limits':['Lifetime sampled peaks are not genesis intensity or full lifetime hourly maxima',
                   'Six-hour intensity sampling can differ in coverage and phase',
                   'Longer events have more opportunities to attain a peak',
                   'Thresholded trends do not establish stable detection or correct missing catalogues',
                   'Partition p values are descriptive; thresholds overlap and are not independent']}
    (OUT/'winter_night_event_strength_summary.json').write_text(json.dumps(obj,indent=2),encoding='utf-8')
    print(result.to_string(index=False))
    print(pd.DataFrame(partitionrows).to_string(index=False))


if __name__=='__main__':main()
