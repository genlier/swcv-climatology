"""Post-review common-corridor, paired-coordinate and cohort-definition audits."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'data/processed'


def main():
    all_events=pd.read_csv(OUT/'events_terrain_qc_intensity.csv.gz')
    lifecycle=pd.read_csv(OUT/'path_intensity_fixed_window_analysis_table.csv.gz')
    strict=all_events.loc[all_events.center_within_25hpa_700_fraction.eq(0)&all_events.valid_point_fraction_700_c25.gt(0)]
    valid=lifecycle.loc[lifecycle.max_vorticity_s_1_700_c25.notna()]
    within=strict.loc[strict.event_id.isin(lifecycle.event_id)]
    outside=strict.loc[~strict.event_id.isin(lifecycle.event_id)]
    extra=valid.loc[~valid.event_id.isin(strict.event_id)]
    definition={'strict_1979_2023':len(strict),'lifecycle_valid':len(valid),
        'strict_within_lifecycle':len(within),'strict_outside_lifecycle':len(outside),
        'lifecycle_valid_but_not_strict':len(extra),
        'outside_by_calendar_year':{str(k):int(v) for k,v in outside.groupby('year').size().items()},
        'interpretation':'Different predicates and temporal windows, not a single omitted event'}
    assert len(strict)-len(outside)+len(extra)==len(valid)
    pd.concat([outside.assign(reason='strict_outside_lifecycle_window'),extra.assign(reason='valid_once_not_strict')]).to_csv(
        OUT/'major_revision_sample_definition_events.csv',index=False)
    table=lifecycle.copy()
    table['lon_bin']=np.floor(table.genesis_lon).astype(int)
    table['lat_bin']=np.floor(table.genesis_lat).astype(int)
    table['early_valid']=table.first_height.notna()&table.first_vorticity.notna()
    points=pd.read_csv(OUT/'terrain_qc_intensity_points.csv.gz')
    points=points.merge(table[['event_id','genesis_time']],on='event_id',validate='many_to_one')
    points['lag_h']=(pd.to_datetime(points.time,utc=True)-pd.to_datetime(points.genesis_time,utc=True)).dt.total_seconds()/3600
    early=points.loc[points.lag_h.between(0,6)].copy()
    early['safe_valid']=(~early.center_within_25hpa_700.astype(bool)) & early.height_deficit_gpm_700_c25.notna() & early.max_vorticity_s_1_700_c25.notna()
    safety=early.groupby('event_id').safe_valid.all()
    table['early_positions_safe_valid']=table.event_id.map(safety).fillna(False).astype(bool)
    coverage=table.groupby(['lat_bin','lon_bin','track_cluster']).agg(events=('event_id','size'),valid=('early_valid','sum')).reset_index()
    coverage['fraction_valid']=coverage.valid/coverage.events
    pivot=coverage.pivot(index=['lat_bin','lon_bin'],columns='track_cluster',values='events').reindex(columns=[0,1,2]).fillna(0)
    fractions=coverage.pivot(index=['lat_bin','lon_bin'],columns='track_cluster',values='fraction_valid').reindex(columns=[0,1,2])
    eligible=pivot.ge(5).all(axis=1)&fractions.eq(1).all(axis=1)
    coverage['common_corridor']=list(zip(coverage.lat_bin,coverage.lon_bin))
    selected=set(eligible[eligible].index.tolist())
    safe_fraction=table.groupby(['lat_bin','lon_bin','track_cluster']).early_positions_safe_valid.mean().unstack().reindex(columns=[0,1,2])
    safe_selected=set(eligible[eligible & safe_fraction.eq(1).all(axis=1)].index.tolist())
    coverage['strict_early_position_corridor']=pd.MultiIndex.from_frame(coverage[['lat_bin','lon_bin']]).isin(safe_selected)
    coverage['common_corridor']=coverage.common_corridor.isin(selected)
    coverage.to_csv(OUT/'major_revision_corridor_coverage.csv',index=False)
    table=table.loc[pd.MultiIndex.from_frame(table[['lat_bin','lon_bin']]).isin(selected)].copy()
    table.to_csv(OUT/'major_revision_corridor_events.csv',index=False)
    rows=[]
    for metric in ['height','vorticity']:
        sample=table.dropna(subset=[f'first_{metric}','eastward_reach_deg','genesis_lat','genesis_lon','season_year']).copy()
        for column in [f'first_{metric}','eastward_reach_deg','genesis_lat','genesis_lon']:
            sample['z_'+column]=(sample[column]-sample[column].mean())/sample[column].std(ddof=0)
        for family,formula,term in [
            ('continuous',f'z_first_{metric} ~ z_eastward_reach_deg + C(season) + z_genesis_lat + z_genesis_lon','z_eastward_reach_deg'),
            ('cluster',f'z_first_{metric} ~ C(track_cluster, Treatment(reference=1)) + C(season) + z_genesis_lat + z_genesis_lon','C(track_cluster, Treatment(reference=1))[T.2]')]:
            fit=smf.ols(formula,sample).fit(cov_type='cluster',cov_kwds={'groups':sample.season_year,'use_correction':True})
            ci=fit.conf_int().loc[term]
            rows.append({'metric':metric,'family':family,'term':term,'coefficient_sd':fit.params[term],
                'ci_low':ci.iloc[0],'ci_high':ci.iloc[1],'p_year_cluster':fit.pvalues[term],
                'events':len(sample),'years':sample.season_year.nunique(),'formula':formula,
                'covariance_rank':int(np.linalg.matrix_rank(fit.cov_params())),'parameters':len(fit.params)})
    models=pd.DataFrame(rows)
    models['p_fdr']=multipletests(models.p_year_cluster,method='fdr_bh')[1]
    models.to_csv(OUT/'major_revision_corridor_models.csv',index=False)
    native=pd.read_csv(OUT/'native_sigma085_djf_events.csv.gz').dropna(subset=['max_vorticity_s_1_native','max_vorticity_s_1_700_c25'])
    annual=native.groupby('season_year')[['max_vorticity_s_1_native','max_vorticity_s_1_700_c25']].quantile(.9)*1e5
    annual['difference_q90_700_minus_sigma']=annual.max_vorticity_s_1_700_c25-annual.max_vorticity_s_1_native
    fit=sm.OLS(annual.difference_q90_700_minus_sigma,sm.add_constant(annual.index.to_numpy()-annual.index.to_numpy().mean())).fit(cov_type='HAC',cov_kwds={'maxlags':2})
    annual.to_csv(OUT/'major_revision_paired_coordinate_annual.csv')
    paired={'events':len(native),'winters':len(annual),'unit':'1e-5 s^-1 per decade',
        'q90_difference_slope':float(fit.params.iloc[1]*10),'ci_low':float(fit.conf_int().iloc[1,0]*10),
        'ci_high':float(fit.conf_int().iloc[1,1]*10),'p_hac2':float(fit.pvalues.iloc[1]),
        'interpretation':'Post-review annual quantile-contrast trend, not mean eventwise difference or physical origin test'}
    summary={'sample_definitions':definition,'corridor_bins':[list(x) for x in sorted(selected)],
        'corridor_events':len(table),'corridor_cluster_counts':{str(k):int(v) for k,v in table.groupby('track_cluster').size().items()},
        'corridor_scope':'100% early metric availability in observed sample; not universal terrain safety',
        'strict_early_position_corridor_bins':[list(x) for x in sorted(safe_selected)],
        'corridor_early_positions_all_safe_valid_events':int(table.early_positions_safe_valid.sum()),
        'paired_coordinate':paired,'analysis_status':'Post-review exploratory diagnostics; mechanisms unresolved'}
    (OUT/'major_revision_existing_data_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2)); print(models.to_string(index=False))


if __name__=='__main__':main()
