"""Source-grounded first-round mechanism report and broad exploratory FDR audit."""
from pathlib import Path
import hashlib
import json
import pandas as pd
import numpy as np
from statsmodels.stats.multitest import multipletests

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'data/processed'


def main():
    families=[]; associations=[]
    for label,prefix in [('mean','winter_local_convergence_'),('lag6','winter_local_convergence_lag6_'),
                         ('lag12','winter_local_convergence_lag12_'),('lag24','winter_local_convergence_lag24_'),
                         ('transition','winter_local_convergence_transition_')]:
        t=pd.read_csv(OUT/(prefix+'background_trends.csv'))
        t['analysis']=label
        families.append(t)
        a=pd.read_csv(OUT/(prefix+'conditional_models.csv'))
        a['analysis']=label
        associations.append(a)
    t=pd.concat(families,ignore_index=True)
    t['p_fdr_all_80_exploratory_background_tests']=multipletests(t.p_hac,method='fdr_bh')[1]
    t.to_csv(OUT/'winter_mechanism_all_background_tests.csv',index=False)
    night=pd.read_csv(OUT/'winter_night_transition_conditional_models.csv')
    night['analysis']='night_transition'
    associations.append(night)
    a=pd.concat(associations,ignore_index=True)
    a['p_fdr_all_24_exploratory_associations']=multipletests(a.p_cluster,method='fdr_bh')[1]
    a.to_csv(OUT/'winter_mechanism_all_association_tests.csv',index=False)
    temporal=json.loads((OUT/'winter_genesis_temporal_structure_summary.json').read_text())
    regimes=json.loads((OUT/'winter_genesis_regimes_summary.json').read_text())
    sensitivity=json.loads((OUT/'winter_genesis_regimes_sensitivity.json').read_text())
    text=['# 冬季盆地局地西南涡为何增加：第一轮发现与机制证据', '',
          '日期：2026-10-04。状态：完成现有数据首轮分析；夜间增量规律及天气型分解已出现，主驱动机制尚未闭合。', '',
          '## 1. 数据与可复现入口', '',
          '732例目录盆地局地事件、44个冬季、15,884个六小时背景箱。早期1980–2001，后期2002–2023。',
          '方案与限制：`docs/winter_local_genesis_mechanism_plan_zh.md`。源论文和现有主分析未改。', '',
          '按顺序执行：',
          '```powershell',
          '.venv\\Scripts\\python.exe scripts\\analyze_winter_genesis_regimes.py',
          '.venv\\Scripts\\python.exe scripts\\check_winter_genesis_regime_sensitivity.py',
          '.venv\\Scripts\\python.exe scripts\\check_winter_genesis_regime_lags.py',
          '.venv\\Scripts\\python.exe scripts\\analyze_winter_genesis_temporal_structure.py',
          '.venv\\Scripts\\python.exe scripts\\extract_winter_local_convergence.py',
          '.venv\\Scripts\\python.exe scripts\\analyze_winter_local_convergence.py',
          '.venv\\Scripts\\python.exe scripts\\analyze_winter_local_convergence.py --single-lag 6',
          '.venv\\Scripts\\python.exe scripts\\analyze_winter_local_convergence.py --single-lag 12',
          '.venv\\Scripts\\python.exe scripts\\analyze_winter_local_convergence.py --single-lag 24',
          '.venv\\Scripts\\python.exe scripts\\analyze_winter_local_convergence.py --transition',
          '.venv\\Scripts\\python.exe scripts\\analyze_winter_night_transition.py',
          '.venv\\Scripts\\python.exe scripts\\summarize_winter_genesis_mechanism.py',
          '.venv\\Scripts\\python.exe -m unittest discover -s tests -p test_winter_genesis_mechanism.py',
          '```', '',
          '## 2. 新的活动规律：增量集中于夜间生成', '',
          '北京时间20–08时夜间事件增加1.358次/十年，白天增加0.175次/十年。',
          f'夜间贡献拟合总增量的{temporal["night_ols_share_of_total_slope"]:.1%}，这是点估计，非已确立的因果份额。',
          '正式配对斜率差为1.183次/十年，HAC95%区间[0.592,1.774]；不是用一显著一不显著证明差异。',
          '18–06、20–08、22–10及当地平均太阳时20–08四种时段定义中，夜间减白天的斜率差均为正、区间均不跨零。',
          '平均太阳时未计均时差；目录生成时间可能受识别阈值及观测同质性影响。', '',
          '## 3. 环境出现时长与类型内生成率', '',
          '| 量 | 后期减前期，次/冬 |', '|---|---:|']
    for name,title in [('total','局地事件总增加'),('exposure','外围天气类暴露贡献'),('conditional_rate','类型内生成率贡献')]:
        text.append(f'| {title} | {regimes["decomposition"][name]["estimate"]:+.3f} |')
    rate=[r['summary']['conditional_rate']['estimate'] for r in sensitivity]
    text += ['',f'3/4/5类及月份×时次分层下，生成率贡献点估计为{min(rate):.3f}–{max(rate):.3f}次/冬；3/5年块区间均为正。',
             '12/24/48小时前外围类敏感性采用同一冻结分类器和同一风险集；保留724例、遗漏边缘8例，各滞后结论方向一致。',
             '这只能说明所定义粗天气类的频率变化解释有限。类型内残差仍可能包含环境强度变化、地理改变、观测变化和遗漏变量，不是纯局地物理效率，更不是增暖效应。', '',
             '## 4. 候选动力过程：夜间生成前的辐合转变', '',
             '24小时平均的区域辐合/伸展没有解释趋势；生成前6小时较强辐合与更高生成率关联，但平均强度的长期变化仍未闭合。',
             '探索性−12至−6小时变化量在夜间有明确正趋势，且与夜间生成率正关联。',
             '| 候选量 | 夜间变化量趋势/十年 | 80项背景检验FDR p | 夜间生成关联，log率/SD | 24项关联FDR p | 年份系数衰减 |',
             '|---|---:|---:|---:|---:|---:|']
    for metric in ['convergence_700','convergence_850','stretching_700','stretching_850']:
        tr=t[(t.analysis=='transition')&t.night_only&t.statistic.eq('mean')&t.metric.eq(metric)].iloc[0]
        ar=a[(a.analysis=='night_transition')&a.metric.eq(metric)].iloc[0]
        text.append(f'| {metric} | {tr.change_per_decade:+.3e} | {tr.p_fdr_all_80_exploratory_background_tests:.3g} | {ar.log_rate_per_early_sd:.3f} | {ar.p_fdr_all_24_exploratory_associations:.3g} | {ar.year_coefficient_attenuation_percent:.1f}% |')
    text += ['', '辐合变化单位s⁻¹，伸展变化单位s⁻²；表中均为6小时差值的年代际趋势。',
             '700 hPa夜间转变指标的均值由1.079×10⁻⁶增至1.889×10⁻⁶ s⁻¹。850 hPa由−5.875×10⁻⁷增至−0.767×10⁻⁷ s⁻¹，表示辐合减弱程度缓和，不可统一描述成实际增强。',
             '加入夜间转变量后，年份系数只衰减4.3%–8.5%，大部分增长仍未解释。此值没有因果中介含义，亦无衰减置信区间。',
             '80/24项FDR是本轮动力探索的宽口径审计，不覆盖所有历史研究选择；事后设计和同源ERA5不能因此变成确认性因果发现。', '',
             '## 5. 排除简单计数拆分解释的初步检查', '',
             '以相邻生成间隔24/48/72小时分组，过程数趋势均为正；48小时过程数早期11.23、后期13.36次/冬。',
             '事件持续时间中位数未检出趋势，暂未显示增加完全由同一过程被拆成更多短事件造成；链式时间分组仍非独立天气过程识别。', '',
             '## 6. 下一步直接接手', '',
             '1. 独立复核夜间规律：取得独立目录的实际初生时刻；年度汇总不能验证日变化。解决134例目录规则差异。',
             '2. 将夜间转变细分为具体UTC相位，检查空间位置、地形掩膜和天气类内变化；做年度块重采样及模型系数衰减区间。',
             '3. 为热力解释提取连续地表/近地层温度、边界层及通量资料。当前风场已支持动力转变诊断，但不能区分坡地热力、背景风或观测系统因素。',
             '4. 候选过程得到独立确认后，补足垂直层、垂直速度和加热项，检查涡度预算。再设计配对增暖区域试验。', '',
             '当前可写科学结论：冬季目录局地事件增长表现为显著的夜间生成偏向；在粗外围天气类保持一致后仍存在类型内生成率提高；夜间生成前的辐合转变提供一个支持有限、尚未独立确认的动力候选。']
    (ROOT/'docs/winter_local_genesis_mechanism_results_zh.md').write_text('\n'.join(text),encoding='utf-8')
    paths=sorted(set(list(OUT.glob('winter_genesis_*'))+list(OUT.glob('winter_local_convergence_*'))+
                     list(OUT.glob('winter_night_transition_*'))+list(OUT.glob('winter_mechanism_all_*'))+
                     list((ROOT/'scripts').glob('*winter*genesis*.py'))+
                     list((ROOT/'scripts').glob('*winter*convergence*.py'))+
                     [ROOT/'scripts/analyze_winter_night_transition.py',
                      ROOT/'tests/test_winter_genesis_mechanism.py',
                      ROOT/'docs/winter_local_genesis_mechanism_plan_zh.md',
                      ROOT/'docs/winter_local_genesis_mechanism_results_zh.md']))
    manifest={'date':'2026-10-04','stage':'existing-data exploratory mechanism discovery',
              'catalogue_input':{'path':'data/processed/events_terrain_qc_intensity.csv.gz',
                                 'sha256':hashlib.sha256((OUT/'events_terrain_qc_intensity.csv.gz').read_bytes()).hexdigest()},
              'files':[{'path':p.relative_to(ROOT).as_posix(),'bytes':p.stat().st_size,
                        'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in paths],
              'raw_input_provenance':'winter_genesis_regimes_summary.json records path, bytes and mtime for background wind/surface pressure; no full raw-file checksum re-audit in this round',
              'next_required':'Independent nocturnal genesis validation; phase/location/mask robustness; surface thermal process diagnostics'}
    (ROOT/'data/manifests/winter_local_genesis_mechanism_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print('Saved integrated first-round report and 80/24-test exploratory audit.')


if __name__=='__main__':main()
