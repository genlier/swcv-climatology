"""Audit genesis location versus the predeclared basin diagnostic support."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'data/processed'


def main():
    data=pd.read_csv(OUT/'events_terrain_qc_intensity.csv.gz')
    time=pd.to_datetime(data.genesis_time,utc=True)
    wy=time.dt.year+time.dt.month.eq(12)
    data=data[data.track_cluster.eq(1)&time.dt.month.isin([12,1,2])&wy.between(1980,2023)].copy()
    data['season_year']=wy.loc[data.index]
    bh=(time.loc[data.index].dt.hour+8)%24
    data['night']=bh.ge(20)|bh.lt(8)
    core=data.genesis_lat.between(28,32)&data.genesis_lon.between(104,108)
    data['region']='outside_diagnostic_core'
    north=data.genesis_lat.gt(30);east=data.genesis_lon.gt(106)
    for name,mask in [('NW',north&~east),('NE',north&east),('SW',~north&~east),('SE',~north&east)]:
        data.loc[core&mask,'region']=name
    years=np.arange(1980,2024);rows=[]
    for clock,mask in [('night',data.night),('day',~data.night)]:
        reconstructed=np.zeros(44,dtype=int)
        for region in ['NW','NE','SW','SE','outside_diagnostic_core']:
            chosen=data[mask&data.region.eq(region)]
            count=chosen.groupby('season_year').size().reindex(years,fill_value=0)
            reconstructed+=count.to_numpy()
            fit=sm.OLS(count.to_numpy(),sm.add_constant((years-2001.5)/10)).fit(cov_type='HAC',cov_kwds={'maxlags':3})
            ci=fit.conf_int()[1]
            rows.append({'clock':clock,'region':region,'n_events':len(chosen),
                'early_events_per_winter':float(count.iloc[:22].mean()),'late_events_per_winter':float(count.iloc[22:].mean()),
                'slope_per_decade':float(fit.params[1]),'ci95_low':float(ci[0]),'ci95_high':float(ci[1]),'p_hac3':float(fit.pvalues[1])})
        np.testing.assert_array_equal(reconstructed,data[mask].groupby('season_year').size().reindex(years,fill_value=0).to_numpy())
    table=pd.DataFrame(rows)
    table['p_fdr_10']=multipletests(table.p_hac3,method='fdr_bh')[1]
    table.to_csv(OUT/'winter_genesis_spatial_support_trends.csv',index=False)
    obj={'scope':'Exploratory spatial support check; lifetime track cluster is not a genesis polygon',
        'diagnostic_rectangle':'28–32N,104–108E','core_events':int(core.sum()),'all_events':len(data),
        'core_night_events':int((core&data.night).sum()),'all_night_events':int(data.night.sum()),
        'trends':table.to_dict(orient='records'),
        'limits':['Regional means are environmental proxies, not values at every genesis centre',
                  'A lifetime track-based cluster can include genesis outside the diagnostic core',
                  'Quadrants are exploratory and do not follow physical basin boundaries',
                  'Spatial correspondence is not proof of mechanism']}
    (OUT/'winter_genesis_spatial_support_summary.json').write_text(json.dumps(obj,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in obj.items() if k!='trends'},indent=2))
    print(table.to_string(index=False))


if __name__=='__main__':main()
