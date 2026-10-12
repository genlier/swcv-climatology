"""Actual-time phase and first-harmonic changes of continuous basin dynamics."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'data/processed'
METRICS=['convergence_700','convergence_850','stretching_700','stretching_850']


def main():
    table=pd.read_csv(OUT/'winter_local_convergence_series.csv.gz',parse_dates=['time'])
    t=table.time
    table['season_year']=t.dt.year+t.dt.month.eq(12).astype(int)
    table['utc_hour']=t.dt.hour
    table=table[t.dt.month.isin([12,1,2])&table.season_year.between(1980,2023)]
    rows=[];statistics=[];comparisons=[]
    for metric in METRICS:
        annual=table.groupby(['season_year','utc_hour'])[metric].mean().unstack().reindex(columns=[0,6,12,18])
        mean=annual.mean(axis=1)
        cosine=(annual[0]-annual[12])/2
        sine=(annual[6]-annual[18])/2
        amplitude=np.hypot(cosine,sine)
        for year in annual.index:
            rows.append({'metric':metric,'season_year':int(year),'mean':float(mean.loc[year]),
                         'cosine':float(cosine.loc[year]),'sine':float(sine.loc[year]),
                         'amplitude':float(amplitude.loc[year]),
                         'harmonic_peak_utc_hour':float(np.arctan2(sine.loc[year],cosine.loc[year])*12/np.pi%24),
                         **{f'utc_{hour}':float(annual.loc[year,hour]) for hour in [0,6,12,18]}})
        design=sm.add_constant((annual.index.to_numpy()-2001.5)/10)
        for name,series in [('mean',mean),('cosine',cosine),('sine',sine),('amplitude',amplitude)]:
            fit=sm.OLS(series.to_numpy(),design).fit(cov_type='HAC',cov_kwds={'maxlags':3})
            ci=fit.conf_int()[1]
            statistics.append({'metric':metric,'statistic':name,'slope_per_decade':float(fit.params[1]),
                                'ci95_low':float(ci[0]),'ci95_high':float(ci[1]),'p_hac3':float(fit.pvalues[1]),
                                'early_mean':float(series.iloc[:22].mean()),'late_mean':float(series.iloc[22:].mean())})
        # Harmonic of the period mean, distinguish from mean yearly amplitudes.
        early=annual.iloc[:22].mean().to_numpy();late=annual.iloc[22:].mean().to_numpy()
        def harmonic(vals):
            a=(vals[0]-vals[2])/2;b=(vals[1]-vals[3])/2
            return float(np.hypot(a,b)),float(np.arctan2(b,a)*12/np.pi%24)
        amp0,phase0=harmonic(early);amp1,phase1=harmonic(late)
        comparisons.append({'metric':metric,'early_harmonic_amplitude':amp0,'late_harmonic_amplitude':amp1,
                            'early_peak_beijing_hour':(phase0+8)%24,'late_peak_beijing_hour':(phase1+8)%24,
                            'note':'Four samples/day cannot resolve precise peak times or distinguish higher harmonics.'})
    pd.DataFrame(rows).to_csv(OUT/'winter_diurnal_dynamics_annual.csv',index=False)
    stats=pd.DataFrame(statistics)
    stats['p_fdr_16']=multipletests(stats.p_hac3,method='fdr_bh')[1]
    stats.to_csv(OUT/'winter_diurnal_dynamics_trends.csv',index=False)
    (OUT/'winter_diurnal_dynamics_summary.json').write_text(json.dumps({'scope':'Continuous actual-clock-time diagnostics, not event-lag phases',
        'trends':stats.to_dict(orient='records'),'period_harmonics':comparisons},indent=2),encoding='utf-8')
    print(stats.to_string(index=False))
    print(json.dumps(comparisons,indent=2))


if __name__=='__main__':main()
