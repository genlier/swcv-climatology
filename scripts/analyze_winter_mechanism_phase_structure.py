"""Phase-resolved, month/regime-stratified winter genesis follow-up."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'data/processed'


def trend(series):
    design=sm.add_constant((series.index.to_numpy()-2001.5)/10)
    fit=sm.OLS(series.to_numpy(),design).fit(cov_type='HAC',cov_kwds={'maxlags':3})
    ci=fit.conf_int()[1]
    return {'slope_per_decade':float(fit.params[1]),'ci95_low':float(ci[0]),'ci95_high':float(ci[1]),'p_hac3':float(fit.pvalues[1])}


def main():
    frame=pd.read_csv(OUT/'winter_local_convergence_transition_precursors.csv.gz')
    years=np.arange(1980,2024)
    records=[]
    for column in ['utc_hour','month','regime']:
        for category,g in frame.groupby(column):
            annual=g.groupby('season_year').events.sum().reindex(years,fill_value=0)
            records.append({'stratification':column,'category':int(category),'early_events_per_winter':float(annual.iloc[:22].mean()),
                            'late_events_per_winter':float(annual.iloc[22:].mean()),**trend(annual)})
    counts=pd.DataFrame(records)
    counts['p_fdr_all_11']=multipletests(counts.p_hac3,method='fdr_bh')[1]
    counts.to_csv(OUT/'winter_phase_genesis_count_trends.csv',index=False)
    dynamics=[]
    for hour,g in frame.groupby('utc_hour'):
        for metric in ['convergence_700','convergence_850','stretching_700','stretching_850']:
            annual=g.groupby('season_year')[metric].mean()
            dynamics.append({'utc_hour':int(hour),'risk_bin_beijing_start':(int(hour)+8)%24,
                             'metric':metric,'early_mean':float(annual.iloc[:22].mean()),
                             'late_mean':float(annual.iloc[22:].mean()),**trend(annual)})
    dyn=pd.DataFrame(dynamics)
    dyn['p_fdr_all_16']=multipletests(dyn.p_hac3,method='fdr_bh')[1]
    dyn.to_csv(OUT/'winter_phase_transition_background_trends.csv',index=False)
    robustness=[]
    total=frame.groupby('season_year').events.sum().reindex(years,fill_value=0)
    night=frame[frame.utc_hour.isin([12,18])].groupby('season_year').events.sum().reindex(years,fill_value=0)
    for name,series in [('total',total),('night',night)]:
        for start in range(1980,2020):
            subset=series.loc[~series.index.to_series().between(start,start+4).to_numpy()]
            robustness.append({'response':name,'removed_start':start,'removed_end':start+4,**trend(subset)})
        for part in ['early','late']:
            subset=series.iloc[:22] if part=='early' else series.iloc[22:]
            robustness.append({'response':name,'period':part,**trend(subset)})
    pd.DataFrame(robustness).to_csv(OUT/'winter_genesis_time_window_sensitivity.csv',index=False)
    # Non-event backgrounds: catalogued local generation at the risk time is absent.
    noevent=frame[frame.events.eq(0)]
    checks=[]
    for hour,g in noevent.groupby('utc_hour'):
        for metric in ['convergence_700','convergence_850']:
            checks.append({'utc_hour':int(hour),'metric':metric,**trend(g.groupby('season_year')[metric].mean())})
    pd.DataFrame(checks).to_csv(OUT/'winter_phase_non_genesis_background_trends.csv',index=False)
    print(counts.to_string(index=False))
    print(dyn.to_string(index=False))
    print('Five-year exclusion night minimum/maximum slope:',pd.DataFrame(robustness).query("response=='night' and removed_start.notna()").slope_per_decade.agg(['min','max']).to_dict())
    (OUT/'winter_phase_structure_summary.json').write_text(json.dumps({'scope':'exploratory full phase reporting',
        'counts':counts.to_dict(orient='records'),'dynamics':dyn.to_dict(orient='records'),
        'time_window_checks':robustness,'non_genesis_background':checks},indent=2),encoding='utf-8')


if __name__=='__main__':main()
