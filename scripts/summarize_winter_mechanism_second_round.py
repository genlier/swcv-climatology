"""Rebuild the second-round research handover from measured outputs."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib
import json
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'data/processed'


def read(name):
    return json.loads((OUT/name).read_text(encoding='utf-8'))


def main():
    phase=read('winter_phase_structure_summary.json')
    dynamics=read('winter_diurnal_dynamics_summary.json')
    bootstrap=read('winter_night_transition_block_bootstrap.json')
    manifest=json.loads((ROOT/'data/manifests/winter_surface_mechanism_manifest.json').read_text(encoding='utf-8'))
    lines=['# 冬季局地西南涡增加：第二轮机制诊断','',
        f'报告生成（UTC）：{datetime.now(timezone.utc).isoformat()}','',
        '## 结论边界','',
        '目前发现的是目录中的夜间生成增加，以及连续ERA5盆地辐合日变化增强。尚未证明由增暖、坡地冷却或温室气体强迫造成；大部分计数趋势仍未被候选动力指标吸收。所有新机制检验均为探索性分析。','',
        '样本：1980–2023共44个完整DJF；732个盆地局地事件。早期1980–2001、后期2002–2023各22冬。北京时间20–08为夜间；时刻指六小时计数风险箱的起点，不能当作精确生成时刻。','',
        '## 1. 增加集中于两个夜间时段','',
        '| 北京时间风险箱 | events/winter/decade | HAC(3) 95% CI | 11项BH调整p |',
        '|---|---:|---:|---:|']
    labels={0:'08–14',6:'14–20',12:'20–02',18:'02–08'}
    for row in phase['counts']:
        if row['stratification']!='utc_hour':continue
        lines.append(f"| {labels[row['category']]} | {row['slope_per_decade']:+.3f} | [{row['ci95_low']:+.3f}, {row['ci95_high']:+.3f}] | {row['p_fdr_all_11']:.3g} |")
    sensitivity=pd.read_csv(OUT/'winter_genesis_time_window_sensitivity.csv')
    removed=sensitivity[(sensitivity.response=='night')&sensitivity.removed_start.notna()]
    lines+=['',f"删除任一连续五年窗口后，夜间趋势仍为正：{removed.slope_per_decade.min():.3f}–{removed.slope_per_decade.max():.3f} events/winter/decade。这是敏感性范围，不是置信区间。",'',
        '## 2. 与其说日均辐合增强，更像日变化调制增强','',
        '以下振幅是每个冬季四时次均值的一阶谐波振幅，再对22冬取平均；不等于22冬平均曲线的振幅。时次稀疏，不报告精确峰值时间。','',
        '| 指标 | 早期均值 | 后期均值 | 振幅改变 | 振幅趋势BH p（16项） |',
        '|---|---:|---:|---:|---:|']
    for row in dynamics['trends']:
        if row['statistic']!='amplitude':continue
        change=100*(row['late_mean']/row['early_mean']-1)
        lines.append(f"| {row['metric']} | {row['early_mean']:.4g} | {row['late_mean']:.4g} | {change:+.1f}% | {row['p_fdr_16']:.3g} |")
    lines+=['', '700 hPa日均辐合无检测到的长期增加；850 hPa日均也不支持增加。两个气压面的日变化并不同步，不能写成整层夜间辐合同步增强。','',
        '## 3. 统计解释量仍有限','',
        '下表为生成前−12至−6小时变化量的早期五分位分箱模型，控制年份、月份、时次与外围天气类，并采用暴露offset。年份系数衰减仅是描述统计，不是归因比例；与首轮连续变量模型数值不同。圆周五冬块重采样300次，区间仍属初步。','',
        '| 转变量 | 年份系数衰减 | 五冬块95%区间 |', '|---|---:|---:|']
    for row in bootstrap['results']:
        if row['block_winters']!=5:continue
        lo,hi=row['attenuation_ci95']
        lines.append(f"| {row['metric']} | {row['attenuation_percent']:.2f}% | [{lo:.2f}, {hi:.2f}]% |")
    fixed=OUT/'winter_fixed_support_dynamics_summary.json'
    if fixed.exists():
        obj=read(fixed.name)
        lines+=['','## 4. 地形掩膜与空间敏感性','',
            f"固定支撑按早期至少99%时次安全的中心及四个导数邻点定义：{obj['support_nodes']}。晚期不参与网格筛选。每个时次仍剔除不安全层；另报告支撑内全部有效时次的敏感性。",'',
            '| 指标 | 取样 | 年均振幅早后期改变 | HAC 95%趋势区间 | 80项BH p |', '|---|---|---:|---:|---:|']
        for row in obj['trends']:
            if row['region']!='basin' or row['statistic']!='amplitude':continue
            lines.append(f"| {row['metric']} | {row['variant']} | {100*(row['late_mean']/row['early_mean']-1):+.1f}% | [{row['ci95_low']:.3g}, {row['ci95_high']:.3g}] | {row['p_fdr_80']:.3g} |")
        lines+=['','固定支撑不是固定时间完整性；完整时次子集也可能选择天气状态。东西/南北分区属于探索性空间定位，不是独立验证。']
        lines+=['','850 hPa振幅变化的分区定位（完整时次；描述性、不作因果解释）：','',
            '| 分区 | 年均振幅改变 | 80项BH p |','|---|---:|---:|']
        for row in obj['trends']:
            if row['metric']!='convergence_850' or row['variant']!='complete_value' or row['statistic']!='amplitude' or row['region']=='basin':continue
            lines.append(f"| {row['region']} | {100*(row['late_mean']/row['early_mean']-1):+.1f}% | {row['p_fdr_80']:.3g} |")
    strength=OUT/'winter_night_event_strength_summary.json'
    if strength.exists():
        obj=read(strength.name)
        lines+=['','### 事件强度和持续时间复核','',
            '阈值由早期所有盆地局地事件的有效生命周期峰值固定；700 hPa指标按25 hPa地形安全标准。不是生成强度，也不是逐小时真实最大值。全样本有228/732例无有效安全层峰值，其中夜间147/423例；不能忽略缺测，也不能把有峰值子样本代表全部事件。','',
            '| 夜间事件筛选 | 事件数 | 频数趋势/decade | HAC 95% CI | 14项BH p |','|---|---:|---:|---:|---:|']
        for row in obj['trends']:
            if row['clock']!='night':continue
            lines.append(f"| {row['criterion']} | {row['events']} | {row['slope_per_decade']:+.3f} | [{row['ci95_low']:+.3f}, {row['ci95_high']:+.3f}] | {row['p_fdr_14']:.3g} |")
        lines+=['','至少12小时事件及超过早期涡度/高度亏损阈值的夜间事件也增加，因此现有目录增量不只出现在最短、最弱候选中。但长期事件达到峰值的机会更多，安全层有效性与地形位置相关，尚不能证明检测机制稳定。持续24小时以上夜间仅33例，区间跨零，不把它写成“长寿命事件不增加”。']
    circulation=OUT/'winter_continuous_circulation_controls_summary.json'
    if circulation.exists():
        obj=read(circulation.name)
        lines+=['','### 类内部连续环流复核','',
            '早期训练的外围700 hPa八个PC（生成前24小时），增加线性及平方项控制；不重新使用事件训练分类。八PC仅覆盖早期约84.2%特征方差，不代表控制所有环流变化。', '',
            '| 指标 | 环流控制 | 夜间年份log-rate/decade | 转变量每早期SD的log-rate | 转变量BH p |', '|---|---|---:|---:|---:|']
        for row in obj['models']:
            if row['control']=='regimes_only':continue
            lines.append(f"| {row['metric']} | {row['control']} | {row['base_year_log_rate']:.3f} | {row['transition_log_rate_per_early_sd']:.3f} | {row['transition_p_fdr_12']:.3g} |")
        lines+=['','连续环流控制没有吸收夜间增加；动力转变量关联仍在但减弱。这仍不能排除遗漏环流、目录变化或再分析非均一性。冬季聚类协方差未修正跨冬序列相关。']
    adjusted=OUT/'winter_diurnal_circulation_adjustment_summary.json'
    if adjusted.exists():
        obj=read(adjusted.name)
        lines+=['','### 连续背景辐合的环流调整','',
            '按四时次分别拟合年份、月份、天气类和八PC，减去拟合环流项后再计算年度振幅；使用固定支撑全部有效的时次。以下是描述敏感性，年度HAC区间未传播环流参数估计误差。', '',
            '| 指标 | 调整 | 振幅趋势/decade | HAC 95% CI | 12项BH p |', '|---|---|---:|---:|---:|']
        for row in obj['trends']:
            if row['statistic']!='amplitude':continue
            lines.append(f"| {row['metric']} | {row['control']} | {row['slope_per_decade']:.3g} | [{row['ci95_low']:.3g}, {row['ci95_high']:.3g}] | {row['p_fdr_12']:.3g} |")
        lines+=['','该调整不是反事实归因；可能同时减去真实强迫引起的环流响应，且外围提前24小时PC不能代表同时刻全部天气状态。']
    free=OUT/'winter_event_free_diurnal_dynamics_summary.json'
    if free.exists():
        obj=read(free.name)
        lines+=['','### 剔除目录活动时段后的背景检查','',
            '在固定支撑全部有效时次中，排除所有路径类别的西南涡生命周期及前后24小时。保留的背景中振幅趋势如下；0/12小时缓冲与只排除局地类别的所有结果另存CSV。','',
            '| 指标 | 振幅趋势/decade | HAC 95% CI | 24项BH p | 最少年度相位时次 |','|---|---:|---:|---:|---:|']
        for row in obj['trends']:
            if row['scope']!='all_swcv_lifetimes' or row['buffer_hours']!=24 or row['statistic']!='amplitude':continue
            lines.append(f"| {row['metric']} | {row['slope_per_decade']:.3g} | [{row['ci95_low']:.3g}, {row['ci95_high']:.3g}] | {row['p_fdr_24']:.3g} | {row['minimum_phase_boxes_per_winter']} |")
        lines+=['','增强并非只在目录活动时段出现。但剔除依据目录，本身随频数变化选择天气状态；未检测的涡旋仍可能存在，不是独立机制验证。']
    spatial=OUT/'winter_genesis_spatial_support_summary.json'
    if spatial.exists():
        obj=read(spatial.name)
        lines+=['','### 生成位置与诊断区域代表性','',
            f"预定盆地矩形28–32°N、104–108°E含{obj['core_events']}/{obj['all_events']}个生成中心，夜间{obj['core_night_events']}/{obj['all_night_events']}个。路径聚类不是生成区边界，因此区域平均只作为环境代理，不能称为所有生成中心的局地诊断。",'',
            '| 夜间生成分区 | 事件数 | 频数趋势/decade | HAC 95% CI | 10项BH p |','|---|---:|---:|---:|---:|']
        for row in obj['trends']:
            if row['clock']!='night':continue
            lines.append(f"| {row['region']} | {row['n_events']} | {row['slope_per_decade']:+.3f} | [{row['ci95_low']:+.3f}, {row['ci95_high']:+.3f}] | {row['p_fdr_10']:.3g} |")
        lines+=['','四分区只用于位置复核，不等同地形物理分区；空间重叠不能证明因果机制。']
    lines+=['','## 5. 近地层热力机制：数据状态','',
        f"清单状态：{manifest['status']}；已校验{len(manifest['files'])}/45年；处理状态：{manifest.get('processing','等待完整下载')}。",'',
        '新资料包括2m气温/露点、地表皮肤温度、边界层高度及10m风；比较西侧矩形与盆地温差、皮肤减气温、湿度、边界层高度和近地层辐合。矩形不是经海拔确认的坡面；温差不等于压强梯度，BLH不是观测逆温，皮肤温差不是感热通量。']
    surface=OUT/'winter_surface_mechanism_summary.json'
    if manifest.get('processing')=='complete' and surface.exists():
        result=read(surface.name)
        lines+=['','完整期热力候选条件模型结果（12项BH校正；负衰减表示年份系数增加）：','',
            '| 窗口 | 指标 | 每早期SD的log-rate | 95% CI | BH p | 年份系数衰减 |','|---|---|---:|---:|---:|---:|']
        for row in result['models']:
            lines.append(f"| {row['precursor']} | {row['metric']} | {row['log_rate_per_early_anomaly_sd']:+.3f} | [{row['ci95_low']:+.3f}, {row['ci95_high']:+.3f}] | {row['p_fdr_12']:.3g} | {row['year_coefficient_attenuation_percent']:+.2f}% |")
        lines+=['','这些结果需要结合48项时次趋势和物理符号一起判断，不因单个关联显著就确认机制。不同探索性表使用不同检验族，不是整轮统一家族错误控制。']
        if 'actual_clock_harmonics' in result:
            lines+=['','近地层实际时钟背景趋势（与事件提前窗口分开；12项BH校正）：','',
                '| 指标 | 统计量 | 趋势/decade | HAC 95% CI | BH p |','|---|---|---:|---:|---:|']
            for row in result['actual_clock_harmonics']:
                lines.append(f"| {row['metric']} | {row['statistic']} | {row['change_per_decade']:+.4g} | [{row['ci95_low']:+.4g}, {row['ci95_high']:+.4g}] | {row['p_fdr_12']:.3g} |")
        controls=OUT/'winter_surface_continuous_controls_summary.json'
        if controls.exists():
            obj=read(controls.name)
            lines+=['','控制连续外围八PC线性/平方项后，六项近地层候选共同加入的结果：','',
                '| 窗口 | 控制天气后年份系数 | 加全部候选后年份系数 | 聚类95% CI | 描述衰减 | 设计条件数 |','|---|---:|---:|---:|---:|---:|']
            for row in obj['joint_models']:
                lo,hi=row['joint_year_ci95']
                lines.append(f"| {row['window']} | {row['base_year_log_rate']:.3f} | {row['joint_year_log_rate']:.3f} | [{lo:.3f}, {hi:.3f}] | {row['attenuation_percent']:+.2f}% | {row['design_condition_number']:.1f} |")
            lines+=['','单候选12项及联合模型中12项系数分别做BH校正，完整结果见winter_surface_continuous_control_models.csv及winter_surface_joint_candidate_coefficients.csv。共线性相关矩阵另存；这是统计调整，不是热力机制归因。']
    else:
        lines+=['','不能用已到达的少数年份估计年代际热力机制；必须等45年通过校验。后台控制器将在下载和提取完成后自动执行条件分析并更新本报告。']
    lines+=['','## 6. 研究图与复现','',
        '研究图：`figures/winter_mechanism/winter_nocturnal_genesis_discovery.png`（同时有PDF/SVG与source-data）。图表示已有动力诊断，不包含未完成的近地层结果。',
        '静态预检、面板对齐、PDF字号和碰撞检查已通过；PNG是研究预览，不声称期刊提交合规。随机数仅用于观测冬季块重采样，不生成观测数据。','',
        '执行顺序：下载 → extract_winter_surface_mechanism.py → analyze_winter_surface_mechanism.py → check_winter_surface_circulation_controls.py → summarize_winter_mechanism_second_round.py。控制器为scripts/continue_winter_mechanism_after_surface.py，状态见logs/winter_mechanism_controller_status.json；最多三次断点恢复。', '',
        '## 尚未解决、不可越过的限制','',
        '- 目录元数据9379与原始9513之间134例剔除出处未明；夜间趋势缺独立产品复核。',
        '- 候选指标可能包含生成前涡旋本身，提前6小时仍不能排除反向关联。',
        '- 背景环流类内部强度与形态变化可能遗漏；再分析日变化变化也可能受观测系统影响。',
        '- 无闭合涡度/能量预算、无强迫反事实，不能确认人为增暖归因。','']
    target=ROOT/'docs/winter_local_mechanism_second_round_results_zh.md'
    target.write_text('\n'.join(lines),encoding='utf-8')
    paths=list(OUT.glob('winter_*dynamics*'))+list(OUT.glob('winter_phase_*'))+list(OUT.glob('winter_*circulation*'))+list(OUT.glob('winter_night_event_strength*'))+[OUT/'winter_night_transition_block_bootstrap.json',target]
    paths+=list(OUT.glob('winter_genesis_spatial_support*'))
    if manifest.get('processing')=='complete':paths+=list(OUT.glob('winter_surface_*'))
    audit={'generated_utc':datetime.now(timezone.utc).isoformat(),
           'outputs':[{'path':p.relative_to(ROOT).as_posix(),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in paths if p.is_file()],
           'scope':'Existing second-round diagnostic artifacts; not a full raw-data integrity audit'}
    (ROOT/'data/manifests/winter_mechanism_second_round_manifest.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    print(target,flush=True)


if __name__=='__main__':main()
