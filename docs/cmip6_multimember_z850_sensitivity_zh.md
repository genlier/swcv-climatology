# CMIP6/DAMIP 多成员 Z850 响应敏感性

更新日期：2026-09-13

## 目的与证据边界

本分析直接回应审稿意见 R3-M1：原稿每个模式只使用一个成员，内部大气变率可能掩盖或制造涡动环流响应。这里对可获得成对成员的 8 个 CMIP6 模式分别提取 3 个 historical 和 3 个 hist-GHG 成员，比较成员分布、模式内集合均值和跨模式结果。

该检验针对大尺度 WNPAC 背景指标，不是西南涡频数、强度或路径的直接归因。三成员集合能降低单成员抽样误差，但仍小于完整大集合；8 模式的自助区间反映模式抽样，不等同于正式最优指纹检测的不确定性。

## 数据获取与质量控制

- 模式：BCC-CSM2-MR、CNRM-CM6-1、CanESM5、GISS-E2-1-G、HadGEM3-GC31-LL、IPSL-CM6A-LR、MIROC6、MRI-ESM2-0。成员按 realization number 自然排序后取最低三个共同标签；除 MRI-ESM2-0 为 r1/r3/r5 外，其余均为 r1/r2/r3。
- 实验：historical 与 hist-GHG；每个模式取 3 对相同 variant label 的成员，共 48 个数据集。
- 48/48 个数据集通过身份、1951--2014 年 768 个连续月份、850 hPa 层和经纬度坐标检查。CNRM-CM6-1 r3 hist-GHG 的云端验证版本为 v20190308，NCSS 使用同一数据集的后续发布版 v20201208；两个版本均覆盖完整时段，manifest 分别记录验证版本和实际提取版本。
- ESGF 文件清单包含 216 个源文件；所有模式--实验--成员组均完整覆盖两个响应时段，无时间缺口。
- 为避免下载全部 19 个压力层，使用 ESGF/THREDDS NCSS 服务端切片，只缓存 850 hPa、15--25°N、全部经度，以及 1951--1980 和 1985--2014 两个 30 年时段。
- 最终年表为 48 × 60 = 2,880 行；失败清单为空。原始区域切片保存在被 Git 忽略的 `data/raw/cmip6_multimember_z850_subsets/`，可由 manifest 重建。

## 指标与统计

区域原始高度是 15--25°N、115--150°E 的余弦纬度加权 JJA 850-hPa 位势高度；纬向平均高度使用同一纬带的全部经度；涡动高度为两者之差。响应定义为 1985--2014 气候态减 1951--1980 气候态，中间留出 1981--1984 四年。

先对每个模式、实验和指标的 3 个成员求集合均值，再在 8 个模式间求等权平均。跨模式均值区间使用固定随机种子 20260912 的 10,000 次模式自助抽样。图中空心圆为成员，菱形为模式内或多模式均值。

## 主要结果

| 实验 | 指标 | 8模式平均（m） | 中位数（m） | 模式bootstrap 95%区间（m） | 正响应模式 |
|---|---:|---:|---:|---:|---:|
| hist-GHG | 区域原始Z850 | 4.19 | 3.70 | 3.16 至 5.28 | 8/8 |
| hist-GHG | 纬向平均Z850 | 4.19 | 4.16 | 3.40 至 5.05 | 8/8 |
| hist-GHG | 涡动Z850 | -0.004 | -0.11 | -0.39 至 0.51 | 2/8 |
| historical | 区域原始Z850 | 3.91 | 3.75 | 3.20 至 4.78 | 8/8 |
| historical | 纬向平均Z850 | 2.94 | 2.96 | 2.58 至 3.31 | 8/8 |
| historical | 涡动Z850 | 0.97 | 0.91 | 0.35 至 1.65 | 7/8 |

hist-GHG 中，模式集合平均涡动响应从单成员子样本的 +0.56 m 降至 -0.004 m，且只有 2/8 个模式为正；8 个模式的 95% 自助区间跨零。这加强了“原始区域高度上升主要是纬向一致的热力/厚度响应，而非稳健 WNPAC 动力增强”的结论。8 个模式中有 5 个的 hist-GHG 涡动响应在单个 r1 成员与三成员均值之间翻转符号，说明审稿人指出的成员抽样问题确实存在，但集合平均后并未显现一致的温室气体动力响应。

historical 的涡动响应则为 +0.97 m，7/8 个模式为正，模式自助区间不跨零。它不能归因于温室气体：historical 同时包含温室气体、气溶胶、自然强迫及残余内部变率，而且这里只比较两个时期。该对照说明不能把“hist-GHG 涡动响应不一致”扩展为所有历史强迫组合下均无区域动力变化。

重新提取的 r1 指标与既有单成员结果高度一致。16 个模式--实验的涡动响应最大绝对差为 0.117 m，来自服务端切片边界/网格处理的细小差异；这不会改变符号分裂或集合平均结论。

## 可复现入口与产物

依次运行：

```powershell
powershell.exe -ExecutionPolicy Bypass -File scripts\run_reviewer_data_revision.ps1 -Stage EValidate
powershell.exe -ExecutionPolicy Bypass -File scripts\run_reviewer_data_revision.ps1 -Stage EUrls
powershell.exe -ExecutionPolicy Bypass -File scripts\run_reviewer_data_revision.ps1 -Stage EExtract
powershell.exe -ExecutionPolicy Bypass -File scripts\run_reviewer_data_revision.ps1 -Stage EAnalyze
```

主要产物：

- `data/manifests/cmip6_multimember_zg_opendap_files.csv`：216 个源文件及覆盖时段。
- `data/processed/cmip6_multimember_zarr_validation.csv`：48 个数据集的变量级验证。
- `data/processed/cmip6_multimember_z850_jja_annual.csv.gz`：2,880 行年度指标。
- `data/processed/cmip6_multimember_z850_responses.csv`：48 个成员响应。
- `data/processed/cmip6_multimember_z850_model_summary.csv`：模式内三成员汇总。
- `data/processed/cmip6_multimember_z850_multimodel_summary.csv`：跨模式均值和区间。
- `data/processed/cmip6_multimember_z850_selected_comparison.csv`：既有单成员、重算 r1 与三成员均值比较。
- `figures/cmip6_multimember_z850_responses.pdf`：投稿级矢量图；对齐、字体和碰撞三项 QA 均通过。

## 论文回填原则

论文可以删除“尚未构建多成员均值”的限制，并明确报告 8 模式三成员敏感性。仍保留三项边界：只有 3 个成员、只覆盖具有配对成员的 8 个模式、没有对西南涡事件本身做强迫归因。hist-GHG 结果支持弱而不一致的涡动响应；historical 的正响应应单独报告，不解释为温室气体效应。
