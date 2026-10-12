"""Check diurnal dynamics with early-defined, spatially fixed terrain supports.

Derivatives still mask unsafe pressure levels at each timestamp. The fixed
support is defined without late-period data or event labels; missing-value
coverage and a complete-timestamp variant are reported, not silently filled.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import xarray as xr
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
from analyze_winter_genesis_regimes import target_times
from extract_winter_local_convergence import spherical_terms

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'data/processed'
LAT=np.arange(27.75,32.26,.25)
LON=np.arange(103.75,108.26,.25)


def load(path,times):
    with xr.open_dataset(path) as raw:
        ds=raw.rename({a:b for a,b in [('valid_time','time'),('latitude','lat'),
            ('longitude','lon'),('pressure_level','level')] if a in raw.coords})
        return ds.sortby('lat').sortby('lon').sel(time=times,lat=LAT,lon=LON).load()


def support_from_pressure(pressure,level):
    safe=pressure>=(level+25)*100
    # Both derivative neighbours must be safe, not only the centre cell.
    valid=(safe[:,1:-1,1:-1]&safe[:,:-2,1:-1]&safe[:,2:,1:-1]
           &safe[:,1:-1,:-2]&safe[:,1:-1,2:])
    return valid


def harmonic(values):
    return np.hypot((values[...,0]-values[...,2])/2,
                    (values[...,1]-values[...,3])/2)


def main():
    times=target_times()
    season=times.year+(times.month==12).astype(int)
    early=times[season<=2001]
    totals={level:np.zeros((17,17),dtype=float) for level in [700,850]}
    for year in sorted(set(early.year)):
        wanted=early[early.year==year]
        sp=load(ROOT/f'data/raw/era5_surface_pressure_6hourly/era5_surface_pressure_{year}.nc',wanted).sp.values
        for level in totals:totals[level]+=support_from_pressure(sp,level).sum(axis=0)
    supports={level:totals[level]/len(early)>=.99 for level in totals}
    weight=np.cos(np.deg2rad(LAT[1:-1]))[:,None]*np.ones((17,17))
    la,lo=np.meshgrid(LAT[1:-1],LON[1:-1],indexing='ij')
    regions={'basin':np.ones((17,17),dtype=bool),'west':lo<=106,'east':lo>106,
             'south':la<=30,'north':la>30}
    rows=[];grids=[]
    for year in sorted(set(times.year)):
        wanted=times[times.year==year]
        sy=wanted.year+(wanted.month==12).astype(int)
        sp=load(ROOT/f'data/raw/era5_surface_pressure_6hourly/era5_surface_pressure_{year}.nc',wanted).sp.values
        for level in supports:
            ds=load(ROOT/f'data/raw/era5_{level}hpa_6hourly/era5_{level}hpa_{year}.nc',wanted).sel(level=level)
            u,v=ds.u.values.astype(float),ds.v.values.astype(float)
            safe=sp>=(level+25)*100
            u[~safe]=np.nan;v[~safe]=np.nan
            fields=spherical_terms(u,v,LAT,LON)
            for name,field in zip(['convergence','stretching'],fields):
                f=field[:,1:-1,1:-1]
                for region,geometry in regions.items():
                    support=supports[level]&geometry
                    w=weight*support
                    valid=np.isfinite(f)
                    denominator=(valid*w).sum(axis=(1,2))
                    coverage=denominator/w.sum()
                    mean=np.nansum(f*w,axis=(1,2))/np.where(denominator>0,denominator,np.nan)
                    for stamp,wy,value,cov in zip(wanted,sy,mean,coverage):
                        rows.append({'time':stamp,'season_year':int(wy),'utc_hour':stamp.hour,
                            'metric':f'{name}_{level}','region':region,'value':value,'coverage':cov,
                            'complete_value':value if cov>=1-1e-12 else np.nan})
                if name=='convergence':
                    for wy in np.unique(sy):
                        for hour in [0,6,12,18]:
                            select=(sy==wy)&(wanted.hour==hour)
                            for iy,ix in zip(*np.where(supports[level])):
                                cell=f[select,iy,ix]
                                grids.append({'season_year':int(wy),'utc_hour':hour,'level':level,
                                    'lat':float(la[iy,ix]),'lon':float(lo[iy,ix]),
                                    'value':float(np.nanmean(cell)),'valid_fraction':float(np.isfinite(cell).mean())})
        print(f'fixed-support extraction {year}',flush=True)
    series=pd.DataFrame(rows)
    series.to_csv(OUT/'winter_fixed_support_dynamics_series.csv.gz',index=False)
    grid=pd.DataFrame(grids).groupby(['season_year','utc_hour','level','lat','lon'],as_index=False).mean()
    grid.to_csv(OUT/'winter_fixed_support_convergence_grid.csv.gz',index=False)
    stats=[]
    for (metric,region),g in series.groupby(['metric','region']):
        for variant in ['value','complete_value']:
            annual=g.groupby(['season_year','utc_hour'])[variant].mean().unstack().reindex(columns=[0,6,12,18])
            if annual.isna().any().any():raise RuntimeError(f'Missing annual phase: {metric} {region} {variant}')
            for statistic,response in [('mean',annual.mean(axis=1).to_numpy()),('amplitude',harmonic(annual.to_numpy()))]:
                fit=sm.OLS(response,sm.add_constant((annual.index.to_numpy()-2001.5)/10)).fit(cov_type='HAC',cov_kwds={'maxlags':3})
                ci=fit.conf_int()[1]
                stats.append({'metric':metric,'region':region,'variant':variant,'statistic':statistic,
                    'slope_per_decade':float(fit.params[1]),'ci95_low':float(ci[0]),'ci95_high':float(ci[1]),
                    'p_hac3':float(fit.pvalues[1]),'early_mean':float(response[:22].mean()),
                    'late_mean':float(response[22:].mean()),'minimum_timestamp_coverage':float(g.coverage.min()),
                    'complete_timestamp_fraction':float(g.complete_value.notna().mean())})
    result=pd.DataFrame(stats)
    result['p_fdr_80']=multipletests(result.p_hac3,method='fdr_bh')[1]
    result.to_csv(OUT/'winter_fixed_support_dynamics_trends.csv',index=False)
    summary={'scope':'Early static support >=99% pressure-safe derivative stencils; dynamic safety retained',
        'support_nodes':{str(level):int(support.sum()) for level,support in supports.items()},
        'trends':result.to_dict(orient='records'),
        'limits':['Fixed spatial eligibility is not fixed temporal completeness',
                  'Complete-timestamp sensitivity can itself select weather states',
                  'Spatial subregions are exploratory, not independent replications',
                  'Four times daily resolve only a coarse first harmonic']}
    (OUT/'winter_fixed_support_dynamics_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(result[(result.region=='basin')&(result.statistic=='amplitude')].to_string(index=False))


if __name__=='__main__':main()
