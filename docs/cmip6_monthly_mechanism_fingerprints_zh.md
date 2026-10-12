# 阶段6C–6D：CMIP6/DAMIP月平均机制指纹、控制噪声与WNPAC动力敏感性

更新日期：2026-09-12

## 1. 阶段定位

本阶段检验与观测机制链一致的大尺度背景量是否对外强迫呈现跨模式一致响应。它不直接识别模式中的西南涡，因此不是西南涡频数、强度或路径的检测归因。阶段6B的700 hPa纯风场事件识别器未通过前向验证后，高频直接归因路线已按预注册规则停止；本阶段是保留下来的低成本、低风险机制指纹路线。

## 2. 数据与冻结定义

- 模式：BCC-CSM2-MR、CNRM-CM6-1、CanESM5、GFDL-ESM4、IPSL-CM6A-LR；每个模式—实验使用一个共同成员。
- 实验：historical、hist-GHG、hist-aer、hist-nat，以及用于内部变率参照的piControl。
- 变量：月平均近地面气温`tas`、海表温度`tos`、850 hPa位势高度`zg`。
- 时间：提取1950–2014年月值，按JJA平均；变化定义为1985–2014均值减1951–1980均值，中间保留4年间隔以减少两个30年气候态的边界依赖。
- GMST：全球`tas`余弦纬度加权平均。
- IOBM：20°S–20°N、40°–100°E的`tos`区域平均。
- WNPAC背景量：15°–25°N、115°–150°E的850 hPa `zg`区域平均，与观测阶段固定框一致。

ESGF文件检索共解析225个目标数据集和1,575个逻辑文件，其中1,563个文件带NCSS服务。Pangeo云目录覆盖15个候选模式中的14个，但只有上述5个模式同时具备五类实验（含后续piControl）和三变量的完整云端组合。区域子集优先通过ESGF NCSS获得；曲线海洋网格接口返回500时，使用Pangeo CMIP6 Google Cloud Zarr副本，并自动识别规则或二维经纬度坐标。当前保存156个区域NetCDF子集（0.44 GiB）和60个紧凑月指标文件。

piControl文件级审计显示15个候选模式的三变量共同控制长度均至少500年。五模式原型每个指标取300年，先估计并移除线性控制漂移，再构造237个滑动窗口；每个窗口比较前30年与间隔4年后的后30年，和强迫响应的统计量完全同形。IPSL云端Zarr保存1850–3049段，而当前ESGF文件副本保存3050–3849段；为避免约1.4 GiB全文件在限速节点上的数小时下载，IPSL的`tas/tos`噪声取云端早期300年，`zg`噪声取ESGF后期300年。三个指标均作单变量噪声比较；`tas/tos`保持同期，可计算IOBM−GMST，未计算Z850与温度的跨变量控制协方差。

## 3. 五模式原型结果

下表先给出跨模式平均原始变化；括号内为同号模式数。相对于piControl噪声的比较见下一小节。

| 实验 | JJA GMST变化（K） | JJA IOBM变化（K） | IOBM−GMST变化（K） | WNPAC Z850变化（m） |
|---|---:|---:|---:|---:|
| hist-GHG | +0.712（5/5正） | +0.586（5/5正） | −0.126（5/5负） | +5.631（5/5正） |
| hist-aer | −0.130（4/5负） | −0.140（5/5负） | −0.010（3/5正、2/5负） | −1.287（5/5负） |
| hist-nat | −0.040（4/5负） | −0.032（4/5负） | +0.007（3/5正、2/5负） | −0.208（3/5负） |
| historical | +0.498（5/5正） | +0.375（5/5正） | −0.123（5/5负） | +4.863（5/5正） |

当前最清楚的结果是强迫方向分离：温室气体和综合历史强迫对应GMST、印度洋SST和WNPAC框850 hPa高度共同增加；人为气溶胶对应印度洋SST和WNPAC框高度在5个模式中一致降低；自然强迫的跨模式变化较小且符号一致性弱。historical和hist-GHG中，印度洋绝对增暖虽一致，但均弱于同期GMST增暖，因此不能把结果表述为“印度洋相对全球增暖”。

原始高度结果在绝对量方向上与观测阶段“较暖IOBM—较强WNPAC背景—较高JJA西南涡频数”的统计链表面相容，但6D动力敏感性检验表明，其中大部分是纬向一致的热力高度抬升，不能据此声称动力型WNPAC增强。它没有证明温室气体导致西南涡频数增加，也没有给出频数、强度或路径的强迫贡献比例。

### 3.1 相对于piControl内部变率

信噪比定义为强迫响应除以各模式去漂移piControl滑动差值的标准差。`|S/N|≥2`仅作为直观可检测性门槛，不等同于正式最优指纹检测。

| 实验 | GMST中位S/N（≥2模式数） | IOBM中位S/N（≥2模式数） | WNPAC框Z850中位S/N（≥2模式数） |
|---|---:|---:|---:|
| hist-GHG | +6.33（5/5） | +8.55（5/5） | +2.85（4/5） |
| hist-aer | −1.19（1/5） | −1.96（2/5） | −0.84（0/5） |
| hist-nat | −0.63（0/5） | −0.08（0/5） | −0.10（0/5） |
| historical | +4.13（5/5） | +5.33（5/5） | +2.48（4/5） |

因此，五模式原型中温室气体和综合历史强迫的GMST、IOBM信号均清楚超出各模式控制噪声；WNPAC框Z850也以正方向为主，但可检测性和幅度的模式差异更大。气溶胶响应虽然在IOBM和Z850上都是5/5负方向，单模式可检测性较弱；自然强迫没有指标达到跨模式稳定的2σ水平。IOBM−GMST在hist-GHG和historical均为5/5负方向，但仅2/5达到2σ，说明“印度洋绝对增暖”比“印度洋相对全球异常增暖/降温”更稳健。

300年控制漂移总体较小：GMST为−0.0046至+0.0520 K/世纪，IOBM为+0.0013至+0.0323 K/世纪，Z850为−0.452至+0.681 m/世纪；零分布已逐模式、逐指标线性去漂移。

### 3.2 阶段6D：去纬向均值的动力敏感性

位势高度同时包含热力和动力效应，增暖造成的整层厚度增加可主导原始高度变化；已有检测归因工作据此将平均高度成分与动力异常分开。WNP副高投影研究也指出传统位势高度指数受纬向一致增暖显著影响，并以减去纬向平均的涡动位势高度表征动力结构。因此，本阶段冻结动力代理为：

`eddy Z850 = 15°–25°N、115°–150°E区域平均Z850 − 15°–25°N纬向平均Z850`

区域框沿用观测分析中常用的15°–25°N、115°–150°E WNPAC高度框。方法依据见[Christidis and Stott (2015)](https://agupubs.onlinelibrary.wiley.com/doi/10.1002/2015GL066669)、[Projected changes in the western North Pacific subtropical high under six global warming targets](https://www.tandfonline.com/doi/pdf/10.1080/16742834.2020.1696649)及[WNPAC高度框应用示例](https://pmc.ncbi.nlm.nih.gov/articles/PMC3581955/)。

| 实验 | 原始Z850平均变化（m） | 纬向平均Z850变化（m） | 涡动Z850平均变化（m） | 涡动Z850中位S/N（≥2模式数） |
|---|---:|---:|---:|---:|
| hist-GHG | +5.631（5/5正） | +4.632（5/5正） | +0.999（4/5正） | +0.18（0/5） |
| hist-aer | −1.287（5/5负） | −0.895（5/5负） | −0.392（4/5负） | −0.30（0/5） |
| hist-nat | −0.208（3/5负） | −0.400（3/5负） | +0.192（3/5正） | +0.39（0/5） |
| historical | +4.863（5/5正） | +3.264（5/5正） | +1.599（4/5正） | +1.55（1/5） |

结果改变了原始Z850的物理解读。hist-GHG原始Z850为5/5正且4/5超过2σ，但其纬向平均成分为5/5正、5/5超过2σ；去除该成分后，涡动Z850仅4/5正、中位变化+0.163 m、中位S/N +0.18，0/5超过2σ。historical也从原始4/5超过2σ降为涡动1/5，且有1个模式转为负值。气溶胶试验的涡动响应仍以负方向为主，但0/5超过2σ。故五模式原型支持“温室气体导致显著的大尺度热力高度抬升”，不支持“温室气体导致可检测的动力型WNPAC普遍增强”。

逐模式经验尾概率同样没有给出hist-GHG涡动高度的显著模式：最小值为0.067（GFDL-ESM4），其余均不小于0.105；这些概率来自高度重叠控制窗口，仅作模式内诊断。该敏感性结果也意味着，不能用原始WNPAC框Z850把“IOBM强迫响应”桥接为“WNPAC动力响应”，更不能继续桥接为西南涡频数响应。

### 3.3 低层风独立复核与15模式扩展

为避免结论依赖单一高度指标，进一步冻结风场型WNPAC指数为：

`WNPAC wind = U850(22.5°–32.5°N, 110°–140°E) − U850(5°–15°N, 90°–130°E)`

正值表示更强的WNPAC型反气旋切变。南减北的原始Wang–Fan指数用于东亚夏季风强度；这里按WNPAC反气旋方向反号。定义依据见[Wang et al. (2008)](https://journals.ametsoc.org/doi/10.1175/2008JCLI2183.1)、[WNPAC近期应用](https://journals.ametsoc.org/view/journals/clim/36/6/JCLI-D-22-0271.1.xml)和[反号WFI诊断定义](https://journals.ametsoc.org/view/journals/clim/33/5/jcli-d-18-0808.1.xml)。指标框和符号在查看结果前固定。

15个模式均有完整300年piControl风指标，并各构造237个去漂移滑动零分布。historical、hist-GHG和hist-nat均为15模式；CESM2 hist-aer的`r1i1p1f1`缺1950–1999、`r3i1p1f1`缺2000–2014，禁止跨成员拼接，因此该实验为14模式。

| 实验 | 模式数 | 风指数平均变化（m s⁻¹） | 中位变化（m s⁻¹） | 正/负模式 | 中位S/N | `|S/N|≥2` |
|---|---:|---:|---:|---:|---:|---:|
| hist-GHG | 15 | +0.233 | +0.290 | 10/5 | +0.40 | 2/15 |
| hist-aer | 14 | +0.166 | +0.067 | 7/7 | +0.11 | 0/14 |
| hist-nat | 15 | +0.030 | −0.071 | 7/8 | −0.13 | 2/15 |
| historical | 15 | +0.258 | +0.459 | 10/5 | +1.13 | 1/15 |

扩展后的风场结果与五模式涡动高度结论一致：hist-GHG和historical的跨模式平均方向偏正，但模式间分歧大，典型响应没有达到2个控制噪声标准差。hist-GHG中BCC-CSM2-MR为+2.20σ，而CNRM-CM6-1为−2.44σ，两个大信号方向相反；不能用少数显著模式声称普遍动力增强。historical只有ACCESS-ESM1-5达到−2.65σ，且方向与跨模式平均相反。低层风复核因此强化的是“动力响应不稳健”的证伪性结论，而不是新的正归因证据。

### 3.4 15模式联合扩展的实际有效样本

温度—海温—高度联合提取已完成文件级质控。GMST四类强迫均为15模式；IOBM有效模式数为hist-GHG 15、hist-aer 13、hist-nat 10、historical 15。对原始/纬向平均/涡动Z850进一步要求对应300年piControl控制序列后，涡动高度的有效比较样本为hist-GHG 11、hist-aer 10、hist-nat 10、historical 13。

在这些有效样本中，hist-GHG涡动Z850平均仅+0.20 m、中位S/N +0.09且0/11超过2σ；historical为+0.94 m、中位S/N +0.50且2/13超过2σ；hist-aer为+0.36 m、中位S/N +0.39且0/10超过2σ；hist-nat中位响应为−0.22 m、中位S/N −0.09。样本不完整来自曲线海温接口和两个piControl高度接口，不能解释为物理缺测，更不能用零值或跨成员拼接修补。

## 4. 质量控制与限制

- 15模式扩展的文件完整性已逐指标审计：tas为60/60个瞬变组合；tos原始文件虽可收集60组，但7组曲线网格在ESGF接口返回常数0，已移入`data/processed/cmip6_invalid_tos`并从统计中排除；Z850有1个CESM2 hist-aer成员本身仅覆盖2000–2014，严格排除。
- piControl目前tas/tos各14模式、Z850区域指标13模式；E3SM-2-0和HadGEM3-GC31-LL的控制高度仍缺少可用接口。所有缺口均按完整JJA和非恒定序列硬门槛处理，不跨成员拼接、不用零值替代。
- 空间口径：所有来源使用同一固定框和纬度权重；海温提取器支持`lat/lon`、`latitude/longitude`及`nav_lat/nav_lon`。
- 成员限制：当前每个模式只使用一个成员，未表示初值集合离散度。
- 模式数限制：低层风已扩展至15模式；温度GMST为15模式，IOBM按实验为10–15模式，原始/涡动Z850的有效控制比较按实验为10–13模式。因此“15模式”只适用于风和GMST，不应误写成所有联合指标均为15模式。
- 内部变率限制：每模式使用300年piControl并得到237个高度重叠的滑动窗口；这些窗口不是237个独立样本，经验尾概率只作模式内诊断。
- 动力解释限制：去纬向均值涡动高度和低层风南北框差两个独立指标均显示动力响应不稳健，但二者仍不能完整表示反气旋的位置和面积。
- 归一化限制：当GMST变化接近零时，`Z850变化/GMST变化`不稳定；该比值只作诊断，不作为主要跨强迫比较量。

## 5. 下一阶段

1. 对仍缺的E3SM/HadGEM控制高度尝试独立NCSS/OPeNDAP回退；若服务持续不可用，固定当前有效样本并发布缺口清单。
2. 对控制窗口步长、控制段选择及去漂移方法做敏感性检验，避免把重叠窗口数量误当有效样本量。
3. 若数据允许，加入多个初值成员，分别估计模式内与模式间不确定性。
4. 最终只将阶段6写成“外强迫下大尺度机制背景的相容性及其证伪性敏感性”，与观测频数—强度—路径结果并列，不越界为直接西南涡归因。

## 6. 可复现文件

- `scripts/audit_cmip6_cloud_monthly.py`：云目录完整性审计。
- `scripts/build_cmip6_monthly_file_manifest.py`：ESGF文件及服务清单。
- `scripts/extract_cmip6_monthly_indices_ncss.py`：NCSS区域子集和月指标提取。
- `scripts/extract_cmip6_tas_from_zarr.py`：云端全球GMST后备提取。
- `scripts/extract_cmip6_tos_from_zarr.py`：规则/曲线网格IOBM后备提取。
- `scripts/extract_cmip6_local_monthly_index.py`：HTTPS原始NetCDF回退的tas/tos区域指标提取。
- `scripts/analyze_cmip6_monthly_mechanism_fingerprints.py`：JJA变化量、跨模式汇总和图件。
- `scripts/audit_cmip6_picontrol_spans.py`：控制试验三变量共同年份审计。
- `scripts/extract_cmip6_picontrol_indices_ncss.py`、`scripts/extract_cmip6_picontrol_indices_zarr.py`：300年控制指标提取。
- `scripts/analyze_cmip6_picontrol_noise.py`：控制漂移、滑动零分布、信噪比和经验尾概率。
- `scripts/extract_cmip6_z850_zonal_mean_ncss.py`：15°–25°N全经度Z850提取。
- `scripts/analyze_cmip6_eddy_z850_sensitivity.py`：涡动高度、强迫响应、控制零分布及原始—动力敏感性图。
- `scripts/build_cmip6_monthly_ua_manifest.py`：15模式月平均`ua`数据集与文件清单。
- `scripts/extract_cmip6_wnpac_wind_ncss.py`、`scripts/extract_cmip6_wnpac_wind_zarr.py`、`scripts/extract_cmip6_wnpac_wind_opendap.py`：风指标的主提取与两类回退路径。
- `scripts/analyze_cmip6_wnpac_wind_fingerprints.py`：15模式风响应、piControl零分布和信噪比。
- `data/processed/cmip6_monthly_mechanism_responses_1951_1980_vs_1985_2014.csv`：模式级响应。
- `data/processed/cmip6_monthly_mechanism_multimodel_summary.csv`：多模式摘要。
- `data/processed/cmip6_mechanism_forced_vs_picontrol.csv`：逐模式强迫响应—控制噪声比较。
- `data/processed/cmip6_mechanism_forced_vs_picontrol_summary.csv`：五模式信噪比摘要。
- `figures/cmip6_monthly_mechanism_fingerprints.png`：五模式指纹图。
- `figures/cmip6_mechanism_signal_to_control_noise.png`：强迫响应相对控制噪声图。
- `data/processed/cmip6_eddy_z850_forced_responses.csv`：原始、纬向平均和涡动Z850逐模式响应。
- `data/processed/cmip6_eddy_z850_forced_vs_control.csv`：三类高度指标逐模式信噪比及经验尾概率。
- `data/processed/cmip6_eddy_z850_multimodel_summary.csv`：阶段6D按实验有效样本摘要（10–13模式）。
- `figures/cmip6_raw_vs_eddy_z850_response.png`：去纬向均值前后的响应与信噪比对照。
- `data/processed/cmip6_wnpac_wind_multimodel_summary.csv`：15模式低层风摘要。
- `data/processed/cmip6_tos_exclusions.csv`：曲线网格海温异常常数序列及排除原因。
- `data/manifests/cmip6_monthly_ua_member_exceptions.csv`：成员覆盖例外及禁止拼接决策。
- `figures/cmip6_wnpac_wind_fingerprints.png`：15模式低层风响应与控制噪声图。

数据访问依据ESGF Search/THREDDS服务与[Pangeo CMIP6云数据说明](https://pangeo-data.github.io/pangeo-cmip6-cloud/accessing_data.html)。
