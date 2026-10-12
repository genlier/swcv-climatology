"""Measured spatial-matching evidence and conservative, reproducible handover."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib
import json
import numpy as np
import pandas as pd
import xarray as xr
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'data/processed'


def main():
    result=json.loads((OUT/'winter_spatial_mechanism_summary.json').read_text())
    extraction=json.loads((OUT/'winter_spatial_mechanism_extraction.json').read_text())
    with xr.open_dataset(OUT/'winter_spatial_mechanism.nc') as raw:cube=raw.load()
    with xr.open_dataset(OUT/'winter_spatial_genesis_counts.nc') as raw:counts=raw.load()
    risk=pd.read_csv(OUT/'winter_genesis_regimes_times.csv.gz',parse_dates=['time'])
    night=risk[risk.utc_hour.isin([12,18])]
    early_night=night[night.season_year.le(2001)]
    eventweights=counts.events.sel(time=pd.DatetimeIndex(early_night.time)).sum('time').values.astype(float)
    eventweights/=eventweights.sum()
    trends=[]
    for metric in ['t2m','skin_air','dewpoint_depression','log_blh','convergence10']:
        field=cube[metric].sel(time=pd.DatetimeIndex(night.time)-pd.Timedelta(hours=6)).values
        for weighting,w in [('area_fixed',np.cos(np.deg2rad(cube.cell_lat.values))),('early_event_fixed',eventweights)]:
            values=np.average(field,axis=1,weights=w)
            annual=pd.Series(values,index=night.season_year).groupby(level=0).mean()
            fit=sm.OLS(annual.to_numpy(),sm.add_constant((annual.index.to_numpy()-2001.5)/10)).fit(cov_type='HAC',cov_kwds={'maxlags':3})
            ci=fit.conf_int()[1]
            trends.append({'metric':metric,'weighting':weighting,'slope_per_decade':float(fit.params[1]),
                'ci95_low':float(ci[0]),'ci95_high':float(ci[1]),'p_hac3':float(fit.pvalues[1]),
                'early_mean':float(annual.iloc[:22].mean()),'late_mean':float(annual.iloc[22:].mean())})
    background=pd.DataFrame(trends)
    background['p_fdr_10']=multipletests(background.p_hac3,method='fdr_bh')[1]
    background.to_csv(OUT/'winter_spatial_lag6_background_trends.csv',index=False)
    lines=['# 冬季夜间局地西南涡：空间匹配机制检验','',
        f'生成UTC：{datetime.now(timezone.utc).isoformat()}','',
        '## 本轮回答的问题','',
        '不再把单一盆地平均值代表所有生成中心，而是在27–33°N、102–109°E固定42个1°格区中保留所有六小时风险时次，以格区固定截距控制静态空间差异。731/732事件纳入；26.5°N的1992_007在固定域外，仅此1例。所有423个夜间事件在域内。', '',
        '近地层全风险集有333564个格区×夜间风险箱，但推断层级为44冬，不是333564个独立重复。700 hPa候选保留410个夜间事件、850 hPa保留377例；水平少于4个早期安全导数模板或时次有效面积不足80%的样本剔除，未插补。', '',
        '## 夜间单候选模型','',
        '控制月份、UTC时次、外围天气类及提前24小时八PC线性/平方项；每候选的基准模型使用相同有效样本。候选用本格区×月份×时次的早期异常标准化。年度有效得分先投影去除格区截距，再用HAC(3)协方差；双侧正态近似95%区间与18项BH校正。','',
        '| 窗口 | 候选 | 事件数 | 每早期异常SD的log-rate | 95% CI | BH p | 年份系数衰减 |',
        '|---|---|---:|---:|---:|---:|---:|']
    for row in result['single_models']:
        lines.append(f"| {row['window']} | {row['metric']} | {row['n_events']} | {row['coefficient_per_early_cell_anomaly_sd']:+.3f} | [{row['ci95_low']:+.3f}, {row['ci95_high']:+.3f}] | {row['p_fdr_18']:.3g} | {row['attenuation_percent']:+.2f}% |")
    lines+=['','## 候选共同加入，是否吸收趋势？','',
        '| 窗口 | 模型 | 事件数 | 基准年份log-rate/decade | 调整后 | 调整后95% CI | 描述衰减 |',
        '|---|---|---:|---:|---:|---:|---:|']
    for row in result['joint_models']:
        lines.append(f"| {row['window']} | {row['model']} | {row['n_events']} | {row['base_year_log_rate']:.3f} | {row['adjusted_year_log_rate']:.3f} | [{row['year_ci95_low']:.3f}, {row['year_ci95_high']:.3f}] | {row['attenuation_percent']:+.2f}% |")
    lines+=['','两项700 hPa动力候选共同加入仅使年份系数减少约1.65%；加入近地层候选后年份系数反而增加。空间匹配改善了代表性，但没有得到“这些候选解释夜间增加”的证据。衰减不是机制贡献或因果中介比例，也不把负衰减当作可定量的热力抑制归因。','',
        '## 对应空间背景是否沿有利方向变化？','',
        '对−6小时连续夜间风险背景固定加权：一组按格区面积；一组按早期夜间生成格区分布。早期权重固定，不让后期生成位置变化偷偷改变背景样本。','',
        '| 候选 | 权重 | 趋势/decade | HAC 95% CI | 10项BH p |','|---|---|---:|---:|---:|']
    for row in background.to_dict(orient='records'):
        lines.append(f"| {row['metric']} | {row['weighting']} | {row['slope_per_decade']:+.4g} | [{row['ci95_low']:+.4g}, {row['ci95_high']:+.4g}] | {row['p_fdr_10']:.3g} |")
    lines+=['','按早期生成位置固定加权，提前6小时背景温度增加约0.259°C/decade、露点差增加约0.288 K/decade。夜间生成却更常伴随较低温度和较小露点差；相应联合模型未吸收增加。这说明列定的平均热力背景变化与夜间计数增长并非简单同向对应，不能把增暖直接转译为生成增加。露点差是近地层干湿代理，不等同整层水汽或相对湿度，不能概括为整个大气柱干化。']
    interaction=OUT/'winter_spatial_day_night_interaction_summary.json'
    if interaction.exists():
        obj=json.loads(interaction.read_text());row=obj['year_night_vs_day']
        lines+=['','## 后续探索：直接昼夜交互','',
            f"全部731事件中，年份×夜间的log-rate差/decade为{row['estimate']:+.3f}，95%区间[{row['ci95_low']:+.3f}, {row['ci95_high']:+.3f}]，双侧HAC p={row['p_hac3']:.3g}。该检验比较效应差，而不是比较两组各自的显著性。",'',
            '| 候选 | 夜间减白天关联差/SD | 95% CI | 两项BH p |','|---|---:|---:|---:|']
        for row in obj['candidate_interactions']:
            contrast=row['night_minus_day_candidate']
            lines.append(f"| {row['metric']} | {contrast['estimate']:+.3f} | [{contrast['ci95_low']:+.3f}, {contrast['ci95_high']:+.3f}] | {contrast['p_fdr_2']:.3g} |")
        lines+=['','交互检验在看到夜间空间匹配结果后提出，明确是探索性，不是独立验证或预注册主检验。']
    lines+=['','## 可以写什么，不能写什么','',
        '- 可写：增加具有明显昼夜结构；空间匹配后仍无法由列定的近地层/局地动力候选吸收，负结果不是由单一盆地矩形的代表性问题完全造成。',
        '- 可写：较冷、露点差较小的格区时次与夜间生成正相关，而这不等于长期冷却/增湿驱动；必须结合上述连续背景趋势与联合模型。',
        '- 不可写：增暖、夜间冷却、边界层变浅或地形热力环流已被确认是增长原因。',
        '- 不可写：网格×时次是独立样本；系数衰减是可归因事件比例；不显著说明机制不存在。','',
        '## 复现及下一步','',
        'extract_winter_spatial_mechanism.py → analyze_winter_spatial_mechanism.py → check_winter_spatial_day_night_interaction.py → summarize_winter_spatial_mechanism.py。',
        '数值测试检验格区边界、计数守恒、解析梯度、与固定截距GLM拟合一致性，以及去除截距干扰参数后协方差与全GLM聚类协方差的一致性。',
        '下一步应集中于可证伪的日变化垂直结构和检测一致性，而不是继续从同一风险集筛选更多显著热力指标。论文主线可从过程审查转向“夜间生成增长与日变化动力增强，但其增长并非列定的局地热力代理所解释”，仍保留归因未决的边界。','']
    target=ROOT/'docs/winter_local_mechanism_spatial_results_zh.md'
    target.write_text('\n'.join(lines),encoding='utf-8')
    paths=list(OUT.glob('winter_spatial*'))+[target,ROOT/'docs/winter_local_mechanism_spatial_plan_zh.md']
    paths+=list((ROOT/'scripts').glob('*winter_spatial*'))+[ROOT/'scripts/spatial_profile_poisson.py']
    paths+=[ROOT/'scripts/export_winter_mechanism_tables.py',
        ROOT/'manuscript/winter_mechanism_tables.tex',OUT/'winter_mechanism_software_versions.json']
    paths+=[ROOT/'data/manifests/winter_surface_mechanism_manifest.json',
        OUT/'events_terrain_qc_intensity.csv.gz',OUT/'winter_genesis_regimes_times.csv.gz',
        OUT/'winter_genesis_regimes_classification.npz',ROOT/'tests/test_winter_spatial_mechanism.py']
    manifest={'generated_utc':datetime.now(timezone.utc).isoformat(),
        'artifacts':[{'path':p.relative_to(ROOT).as_posix(),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in paths if p.is_file()],
        'scope':'Spatial analysis artifacts and source code; raw sources referenced by extraction and surface manifests'}
    (ROOT/'data/manifests/winter_spatial_mechanism_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(target,flush=True);print(background.to_string(index=False))


if __name__=='__main__':main()
