[English](README.md) | 简体中文

# 西南涡活动的年代际变化与增暖响应

围绕西南涡的频数、强度和路径开展联合分析，重点研究冬季盆地局地型的昼夜变化，并利用 CMIP6/DAMIP 试验分析区域环流背景的增暖响应。

## 核心程序

| 程序 | 分析内容 | 主要结果 |
| --- | --- | --- |
| `analyze_tracks.py` | 轨迹特征、K-means 分型、年频数及温度关联 | 事件表、路径群统计、年代际变化图 |
| `analyze_joint_response.py` | 频数—强度—路径联合统计 | 年度联合指标、强事件与东移事件统计 |
| `analyze_path_intensity_fixed_windows.py` | 初生后固定窗口的强度与东移延伸关联 | 标准化回归系数及置信区间 |
| `analyze_winter_genesis_temporal_structure.py` | 冬季局地生成的昼夜趋势与事件聚集 | 夜间和白昼趋势、增长差异、逐小时频数 |
| `analyze_winter_night_transition.py` | 夜间生成与前期辐合、涡度拉伸变化的关联 | Poisson 回归系数、调整后的年份系数 |
| `analyze_cmip6_multimember_responses.py` | 温室气体与历史全强迫下的高度响应 | 原始、纬向平均及涡动高度的多模式比较 |

`audit_panel_alignment.py` 为 CMIP6 绘图提供子图矩形与标签检查。

## 环境

使用 Python 3.13。在仓库根目录运行：

```sh
python -m venv .venv
# 按所用终端激活虚拟环境
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

## 数据准备

轨迹采用 He & Cao 发布的 CNN/ERA5 西南涡目录：[Zenodo 数据集](https://doi.org/10.5281/zenodo.15073815)。将逐年 `YYYY.txt` 文件放在 `data/raw/swcv_zenodo_15073815/`。每行依次为轨迹步数、纬度、经度和 UTC 时间（`YYYYMMDDHH`），步数为 0 表示一条新轨迹开始。

全球平均温度采用 [NASA GISTEMP](https://data.giss.nasa.gov/gistemp/)，保存为 `data/raw/gistemp_v4_global.csv`，使用包含 `Year` 和 `J-D` 列、首行为说明的原始年表格式。

强度、冬季环境和 CMIP6 分析读取预处理表，统一放在 `data/processed/`。各表名称和字段见下方。结果写入 `data/processed/`，图片写入 `figures/`。

## 运行方法

### 1. 轨迹与年度统计

```sh
python scripts/analyze_tracks.py
```

读取轨迹和 GISTEMP，计算初生与消亡位置、持续时间、净位移等特征，标准化后开展 K-means 分型。生成 `events.csv`、`track_points.csv.gz`、`annual_metrics.csv`、`track_cluster_summary.csv` 和温度关联表。

年度统计中的季节频数按日历年汇总；后续冬季专题采用跨年冬季，12 月归入次年的冬季年份。东移到达判据采用轨迹最大经度达到 110°E；路径聚类通过多项轨迹特征分组。

### 2. 频数—强度—路径联合分析

另准备 `era5_event_intensity.csv`：每个事件一行，以 `event_id` 与轨迹匹配，包含 `max_height_deficit_gpm`、`max_vorticity_s_1` 等强度字段。强度表来自沿轨迹配准的 ERA5 场。使用当前分型的三路径群（0：偏南局地型；1：盆地局地型；2：长寿命东移型）。

```sh
python scripts/analyze_joint_response.py
```

输出 `annual_joint_fip_metrics.csv`、`joint_warming_associations_hac.csv` 和 `events_with_era5_intensity.csv.gz`。高度亏损与涡度分别标准化后构成联合强度指标，强事件采用该指标的上十分位阈值。

### 3. 路径与初生早期强度

准备三张表：

- `events_season_year_terrain_qc.csv.gz`：包含 `event_id`、`genesis_time`、`lysis_time`、`season_year`、`season`、`track_cluster`、`genesis_lat`、`genesis_lon`、`lysis_lat`、`lysis_lon`、`max_lon` 的地形质控事件表。
- `terrain_qc_intensity_points.csv.gz`：逐轨迹点的 `event_id`、`time`、`height_deficit_gpm_700_c25` 与 `max_vorticity_s_1_700_c25`。
- `jja_event_moisture_lag_event_level.csv.gz`：夏季事件的 `event_id`、`lag_hours` 与 `area_convergence_anom_kg_m2_s`。

```sh
python scripts/analyze_path_intensity_fixed_windows.py
```

比较初生 0–6 小时、0–12 小时、0–24 小时的强度指标，拟合连续东移延伸与路径群对比，并纳入有效采样次数等协变量。输出 `path_intensity_fixed_window_models.csv` 与回归系数图。完整字段按脚本中的事件筛选和回归公式准备。

### 4. 冬季局地生成的昼夜变化

准备 `events_terrain_qc_intensity.csv.gz`，包含 `genesis_time`、`genesis_lon`、`track_cluster`、`observed_hours`、`duration_hours`。脚本筛选路径群 1 的局地事件及 1980–2023 年完整冬季，夜间定义为北京时间 20:00–08:00。

```sh
python scripts/analyze_winter_genesis_temporal_structure.py
```

输出 `winter_genesis_temporal_structure_annual.csv`、`winter_genesis_temporal_structure_statistics.csv`、`winter_genesis_beijing_hour_counts.csv` 与汇总 JSON，计算夜间、白昼趋势及两者增长差异。

### 5. 夜间生成的前期动力条件

准备 `winter_local_convergence_transition_precursors.csv.gz`，每行对应一个风险时段，包含 `utc_hour`、`season_year`、`regime`、`month`、`decade`、`events`，以及 `convergence_700`、`convergence_850`、`stretching_700`、`stretching_850`。动力字段表示生成前 −12 至 −6 小时的变化量；`decade` 采用 `(season_year - 2001.5) / 10`。

```sh
python scripts/analyze_winter_night_transition.py
```

采用 Poisson 模型，控制环流类别、月份与时段，以早期年份的标准差标准化前期动力条件。输出 `winter_night_transition_conditional_models.csv`，含每标准差的 log-rate 系数、置信区间及加入前期条件后的年份系数。

### 6. CMIP6 多成员高度响应

准备 `cmip6_multimember_z850_responses.csv`：包含 `model`、`experiment`、`member`、`raw_z850_m_change`、`zonal_z850_m_change`、`eddy_z850_m_change`，对应 8 个模式、`hist-GHG` 与 `historical` 两种试验、每模式每试验 3 个成员。区域为 15–25°N、115–150°E，指标为夏季 850 hPa 高度；变化量采用 1985–2014 年均值减去 1951–1980 年均值，以米计。

模式为 BCC-CSM2-MR、CNRM-CM6-1、CanESM5、GISS-E2-1-G、HadGEM3-GC31-LL、IPSL-CM6A-LR、MIROC6 和 MRI-ESM2-0。

另准备 `cmip6_eddy_z850_forced_responses.csv`，包含单成员比较所需的 `source`、`experiment`、`member` 及上述三个变化字段。

```sh
python scripts/analyze_cmip6_multimember_responses.py
```

先平均各模式的三个成员，再汇总模式间响应，输出 `cmip6_multimember_z850_model_summary.csv`、`cmip6_multimember_z850_multimodel_summary.csv`、试验对比表与高度响应图。涡动高度由区域高度减去同纬度的纬向平均高度得到，用于描述区域相对高度变化。

## 方法与参考文献

文献条目见 `refs/`。ERA5 数据来自 Copernicus Climate Data Store，CMIP6/DAMIP 数据来自 ESGF；数据使用遵循各提供方的许可。项目代码的复用许可可联系仓库维护者。
