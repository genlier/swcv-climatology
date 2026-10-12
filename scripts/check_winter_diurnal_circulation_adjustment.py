"""Descriptive phase-specific adjustment for antecedent peripheral circulation.

Only measured circulation contributions are subtracted; year and month terms
remain. Annual adjusted-harmonic uncertainty treats nuisance coefficients as
estimated and fixed, so intervals are sensitivity diagnostics, not definitive
attribution uncertainty.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
from check_winter_diurnal_fixed_mask import harmonic

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'data/processed'


def main():
    series=pd.read_csv(OUT/'winter_fixed_support_dynamics_series.csv.gz',parse_dates=['time'])
    series=series[(series.region=='basin')&series.metric.str.startswith('convergence')]
    risk=pd.read_csv(OUT/'winter_genesis_regimes_times.csv.gz',parse_dates=['time'])
    scores=np.load(OUT/'winter_genesis_regimes_classification.npz')['scores']
    early=risk.season_year.le(2001).to_numpy()
    scores=(scores-scores[early].mean(axis=0))/scores[early].std(axis=0,ddof=1)
    for k in range(8):risk[f'pc{k+1}']=scores[:,k]
    frame=series.merge(risk[['time','regime','month']+[f'pc{k+1}' for k in range(8)]],on='time',validate='many_to_one')
    if len(frame)!=2*len(risk):raise RuntimeError('Incomplete circulation/dynamic timestamp join')
    frame['decade']=(frame.season_year-2001.5)/10
    summaries=[];phasefits=[];annuals=[]
    for metric,g in frame.groupby('metric'):
        for control in ['unadjusted','regimes_pc8','regimes_pc8_squared']:
            pieces=[]
            for hour,sub in g.groupby('utc_hour'):
                sub=sub.dropna(subset=['complete_value']).copy()
                sub['adjusted']=sub.complete_value
                if control!='unadjusted':
                    months=pd.get_dummies(sub.month.astype(str),prefix='month',drop_first=True,dtype=float)
                    regimes=pd.get_dummies(sub.regime.astype(str),prefix='regime',drop_first=True,dtype=float)
                    circulation=regimes.copy()
                    for k in range(8):circulation[f'pc{k+1}']=sub[f'pc{k+1}']
                    if control=='regimes_pc8_squared':
                        for k in range(8):circulation[f'pc{k+1}_sq']=sub[f'pc{k+1}']**2
                    design=sm.add_constant(pd.concat([sub[['decade']],months,circulation],axis=1)).astype(float)
                    fit=sm.OLS(sub.complete_value*1e6,design).fit(cov_type='cluster',cov_kwds={'groups':sub.season_year})
                    fitted=(circulation*fit.params[circulation.columns]).sum(axis=1)*1e-6
                    sub['adjusted']=sub.complete_value-fitted+fitted[sub.season_year.le(2001)].mean()
                    ci=fit.conf_int().loc['decade']
                    phasefits.append({'metric':metric,'control':control,'utc_hour':int(hour),
                        'conditional_slope_per_decade':float(fit.params.decade)*1e-6,
                        'ci95_low':float(ci.iloc[0])*1e-6,'ci95_high':float(ci.iloc[1])*1e-6,
                        'p_cluster':float(fit.pvalues.decade),'n_boxes':len(sub)})
                pieces.append(sub[['season_year','utc_hour','adjusted']])
            annual=pd.concat(pieces).groupby(['season_year','utc_hour']).adjusted.mean().unstack().reindex(columns=[0,6,12,18])
            for statistic,values in [('mean',annual.mean(axis=1).to_numpy()),('amplitude',harmonic(annual.to_numpy()))]:
                fit=sm.OLS(values,sm.add_constant((annual.index.to_numpy()-2001.5)/10)).fit(cov_type='HAC',cov_kwds={'maxlags':3})
                ci=fit.conf_int()[1]
                summaries.append({'metric':metric,'control':control,'statistic':statistic,
                    'slope_per_decade':float(fit.params[1]),'ci95_low':float(ci[0]),'ci95_high':float(ci[1]),
                    'p_hac3':float(fit.pvalues[1]),'early_mean':float(values[:22].mean()),'late_mean':float(values[22:].mean())})
            for wy,values in annual.iterrows():
                annuals.append({'metric':metric,'control':control,'season_year':int(wy),
                    **{f'utc_{h}':float(values[h]) for h in [0,6,12,18]}})
    table=pd.DataFrame(summaries)
    table['p_fdr_12']=multipletests(table.p_hac3,method='fdr_bh')[1]
    phases=pd.DataFrame(phasefits)
    phases['p_fdr_16']=multipletests(phases.p_cluster,method='fdr_bh')[1]
    table.to_csv(OUT/'winter_diurnal_circulation_adjusted_trends.csv',index=False)
    phases.to_csv(OUT/'winter_diurnal_circulation_adjusted_phase_models.csv',index=False)
    pd.DataFrame(annuals).to_csv(OUT/'winter_diurnal_circulation_adjusted_annual.csv',index=False)
    result={'scope':'Exploratory descriptive circulation adjustment of complete fixed-support basin dynamics',
        'antecedent_hours':24,'trends':table.to_dict(orient='records'),'phase_models':phases.to_dict(orient='records'),
        'limits':['Nuisance-fit uncertainty not propagated to annual harmonic CIs',
                  'Phase models use winter clusters, not cross-winter serial covariance',
                  'Antecedent PCs do not identify local or anthropogenic forcing',
                  'Subtraction can remove genuine forced circulation responses',
                  'Only four clock samples and one reanalysis product']}
    (OUT/'winter_diurnal_circulation_adjustment_summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(table.to_string(index=False))


if __name__=='__main__':main()
