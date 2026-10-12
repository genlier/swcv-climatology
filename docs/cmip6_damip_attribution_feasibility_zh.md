# CMIP6/DAMIP外强迫归因可行性审计与预注册设计

更新日期：2026-09-06

## 1. 阶段定位

本阶段不是把观测GMST回归改名为“归因”，而是检查真正的检测归因是否具备资料和识别条件。研究设计依据CMIP6总体框架[Eyring et al. (2016)](https://doi.org/10.5194/gmd-9-1937-2016)、DAMIP单强迫试验设计[Gillett et al. (2016)](https://doi.org/10.5194/gmd-9-3685-2016)，统计框架参考最优指纹理论[Allen & Stott (2003)](https://doi.org/10.1007/s00382-003-0313-9)和正则化最优指纹[Ribes et al. (2013)](https://doi.org/10.1007/s00382-013-1735-7)。两篇开放设计论文已从出版商下载并保存哈希。

本轮通过官方ESGF Search API审计`historical`、`hist-GHG`、`hist-aer`、`hist-nat`和`piControl`。查询的是最新逻辑数据集；由于分布式索引在`replica=false`下仍返回部分同一实例的多节点副本，又按`instance_id`二次去重。审计日期为2026-09-06，ESGF后续发布或撤回数据时结果可能变化。

## 2. 核心发现

### 2.1 完整高度—风场逐涡旋路线目前不可行

直接复刻现有动力识别至少需要6小时`zg+ua+va`气压层场。ESGF中：

| 试验 | zg模式数 | ua模式数 | va模式数 |
|---|---:|---:|---:|
| historical | 13 | 27 | 27 |
| hist-GHG | 1 | 5 | 4 |
| hist-aer | 1 | 4 | 4 |
| hist-nat | **0** | 4 | 3 |
| piControl | 9 | 17 | 18 |

`hist-nat`没有索引到任何`6hrPlevPt/zg`，所以不存在覆盖四个瞬变试验和控制试验的`zg+ua+va`模式。GISS‑E2‑1‑G虽有hist-GHG和hist-aer高度场，但缺hist-nat高度场，不能完成因子分解。由此得到明确的no-go结论：不能按当前高度亏损定义直接开展完整DAMIP归因。

### 2.2 700 hPa风场—涡度路线有三个候选模式

若只要求`ua+va`并从风场计算700 hPa相对涡度，以下模式在五个试验中均至少有一个完整成员：

| 模式 | historical | hist-GHG | hist-aer | hist-nat | piControl | 四个瞬变试验最小一成员原始索引体量 |
|---|---:|---:|---:|---:|---:|---:|
| IPSL-CM6A-LR | 31 | 6 | 6 | 6 | 2 | 131.4 GiB |
| MIROC6 | 10 | 3 | 3 | 3 | 1 | 1298.6 GiB |
| MRI-ESM2-0 | 10 | 5 | 5 | 5 | 1 | 471.5 GiB |

体量是ESGF全空间、全气压层数据集的索引大小，不是区域子集估计。已实测THREDDS NetCDF Subset Service可按经纬度、850 hPa层和时间返回小型NetCDF，说明区域/单层下载技术上可行；但节点服务是否对三套模式全部稳定仍需逐一验证。

该路线改变了西南涡定义：没有闭合等高线或高度亏损约束时，普通切变涡度中心可能成为假阳性。因此三个模式“有数据”不等于“可归因”，必须先用ERA5进行独立校准。

### 2.3 预计算rv850不能解决问题

`rv850`在hist-GHG、hist-aer和hist-nat各只有一个模式来源，且没有同一模式跨全部试验闭合。更重要的是，本项目ERA5地形审计显示850 hPa强度仅约21%的事件可用。该变量只可作为低海拔敏感性，不能成为西南涡主识别层。

### 2.4 月平均机制指纹路线资料充足，但不是涡旋归因

若目标改为月平均850 hPa WNPAC高度、全球tas增暖和热带印度洋tos，15个模式同时具有四个瞬变试验和piControl：ACCESS-CM2、ACCESS-ESM1-5、BCC-CSM2-MR、CESM2、CNRM-CM6-1、CanESM5、E3SM-2-0、FGOALS-g3、GFDL-ESM4、GISS-E2-1-G、HadGEM3-GC31-LL、IPSL-CM6A-LR、MIROC6、MRI-ESM2-0和NorESM2-LM。

这足以研究“温室气体和气溶胶是否改变IOBM/WNPAC背景”，并能与已经得到的IOBM—WNPAC—JJA频数机制链对接。但从模式WNPAC变化推断西南涡频数，仍依赖观测回归关系跨模式、跨气候态稳定这一额外假设。因此它只能称为机制指纹或物理一致性检验，不能替代西南涡频数、强度和路径的直接归因。

## 3. 路线决策

本项目采用两层方案：

1. **主路线：700 hPa涡度事件识别。** 先在ERA5现有6小时风场上构建不使用位势高度的候选检测器；通过前向验证后，再进入IPSL‑CM6A‑LR、MIROC6和MRI‑ESM2‑0。
2. **支持路线：月平均机制指纹。** 使用15模式的zg/tas/tos评估WNPAC、IOBM和GMST对不同外强迫的响应。无论结果多显著，都不把它写成“西南涡已被归因”。

完整`zg+ua+va`路线和`rv850`主路线当前停止。这个选择避免因资料缺口事后删除hist-nat，或只用一个GISS模式的GHG/aerosol结果冒充完整归因。

## 4. 700 hPa涡度识别器的预注册门槛

### 4.1 样本划分

- 开发期：1980–2001年，仅用于确定涡度阈值、局地极值尺度、持续时间和跟踪参数。
- 前向验证期：2002–2023年，开发完成前不查看汇总技巧分数。
- 另将ERA5降格到三个候选模式的近似网格，避免只在0.25°ERA5上成功、到模式粗网格即失效。

### 4.2 固定事件定义

- 生成候选区沿用26–33°N、100–109°E，并保留地形排除。
- 从700 hPa `ua/va`计算相对涡度，寻找空间平滑后的正涡度局地极大。
- 相邻6小时中心最大连接距离不超过400 km，持续至少12小时。
- 与现有事件匹配要求至少两个6小时时次中心距离不超过300 km，生成时刻相差不超过12小时。

### 4.3 go/no-go门槛

在2002–2023年验证期：

- 事件级F1不低于0.60；
- DJF和JJA年度频数相关均不低于0.60；
- 平均频数绝对偏差不超过参照的30%；
- ERA5降格后至少两个候选模式网格仍满足上述门槛。

若门槛失败，将停止直接西南涡归因，正式报告“高频DAMIP资料存在，但无法在不改变事件定义的前提下识别目标现象”，只保留月平均机制指纹。不得通过查看DAMIP强迫结果后回头修改识别阈值。

## 5. 通过门槛后的统计设计

### 5.1 共同时间窗和端点

CMIP6 historical通常止于2014，而DAMIP单强迫可延伸至2020。观测对模式的主检测窗固定为1980–2014，避免把SSP延伸只接到all-forcing而造成试验不对称。端点顺序为：

1. DJF频数趋势；
2. JJA频数趋势；
3. DJF年P90 700 hPa涡度；
4. 东移逸出比例和净位移；
5. JJA WNPAC/IOBM机制指纹。

模型内使用相对自身1980–1999气候态的异常或百分比变化，避免绝对频数模式偏差支配结果。路径必须在同一检测器生成的模式轨迹上计算。

### 5.2 内部变率和模式不确定性

- 从piControl切取与观测等长、互不重叠或移动的35年片段，去除控制漂移，估计无强迫趋势分布和指纹协方差。
- 先检验all-forcing信号能否被检测，再考虑GHG与aerosol的双信号分离。
- 模式作为独立层级、成员嵌套于模式；多模式平均按模式等权，不能让CanESM5或GISS的大成员数压倒其他模式。
- 最优指纹缩放因子区间排除0定义为检测；与1相容仅表示模式信号幅度与观测相容，不自动证明唯一因果来源。
- 三个直接候选模式样本很小，GHG/aerosol双信号高度共线时不强行分解；应报告不可辨识或宽区间。

## 6. 反证检查

最强反对意见是：涡度识别器与原CNN/闭合高度定义不是同一个现象。该问题不能靠统计显著性解决，只能靠前向事件匹配和降格网格验证。第二个问题是模式分辨率可能不足以解析青藏高原东侧中尺度低涡；因此即使三模式平均有信号，也必须同时报告每模式气候态偏差与检测技巧。第三个问题是月平均WNPAC变化可能成立而西南涡频数关系在模式气候中失效，故代理路线不能承担主结论。

当前没有学术诚信或数据许可障碍；主要风险是现象可识别性和资料稀疏性。AI参与了检索、ESGF元数据解析、代码和设计整理；最终识别阈值及归因解释仍需领域专家审查。

## 7. 本阶段产物

- `scripts/audit_cmip6_damip_availability.py`
- `data/manifests/cmip6_damip_esgf_query_manifest.json`
- `data/processed/cmip6_damip_esgf_dataset_inventory.csv`
- `data/processed/cmip6_damip_availability_summary.csv`
- `data/processed/cmip6_damip_pathway_eligibility.csv`
- `data/processed/cmip6_damip_monthly_proxy_intersection.csv`
- `data/processed/cmip6_damip_route_decision.json`
- `figures/cmip6_damip_availability_audit.png`
- `refs/literature/cmip6_attribution/download_manifest.json`

下一子阶段原定为6B：在ERA5上开发并前向验证700 hPa涡度事件识别器。在通过go/no-go门槛前，不启动数百GiB级DAMIP高频下载。

## 8. 阶段6B事后状态更新（2026-09-07）

6B已按上述冻结规则完成。2002–2023前向验证事件F1为0.418，DJF/JJA年频数r为0.295/0.606；事件F1和DJF相关未达到0.6。因此700 hPa纯风场直接识别路线判定为NO-GO，不进行模式网格降格或高频下载。下一阶段只保留15模式月平均机制指纹，详见`docs/vorticity_detector_forward_validation_zh.md`。
