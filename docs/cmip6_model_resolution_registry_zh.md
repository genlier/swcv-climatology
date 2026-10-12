# CMIP6/DAMIP 15 模式标称水平分辨率与网格注册表

本注册表响应审稿人意见 R3-m1。15 个模式的模式、机构、成员、网格标签、标称分辨率和网格描述均由脚本读取仓库内 ESGF/Pangeo NetCDF 子集保留的全局属性并与预期表核对。CESM2 使用冻结目录指定的 Zarr 数据集生成最小元数据子集；该文件只用于属性核验，不进入科学分析。数据集标识符是检索定位记录，近似经纬间距是整理字段。

| 模式 (Model) | 机构 (Institution) | 选定成员 (Variant) | 网格标记 (Grid) | 标称分辨率 (Nominal Resolution) | 近似网格间距 (Approx. Grid) | ESGF 定位记录 (Locator) |
|---|---|---|---|---|---|---|
| ACCESS-CM2 | CSIRO-ARCCSS | r1i1p1f1 | gn | 250 km | 1.25° × 1.875° | `ESGF CMIP6.CMIP.CSIRO-ARCCSS.ACCESS-CM2.historical.r1i1p1f1.Amon.zg.gn.v20191108` |
| ACCESS-ESM1-5 | CSIRO | r1i1p1f1 | gn | 250 km | 1.25° × 1.875° | `ESGF CMIP6.CMIP.CSIRO.ACCESS-ESM1-5.historical.r1i1p1f1.Amon.zg.gn.v20191115` |
| BCC-CSM2-MR | BCC | r1i1p1f1 | gn | 100 km | 1.125° × 1.125° | `ESGF CMIP6.CMIP.BCC.BCC-CSM2-MR.historical.r1i1p1f1.Amon.zg.gn.v20181126` |
| CanESM5 | CCCma | r1i1p1f1 | gn | 500 km | 2.81° × 2.81° | `ESGF CMIP6.CMIP.CCCma.CanESM5.historical.r1i1p1f1.Amon.zg.gn.v20190429` |
| CESM2 | NCAR | r1i1p1f1 | gn | 100 km | 0.9° × 1.25° | `ESGF CMIP6.CMIP.NCAR.CESM2.historical.r1i1p1f1.Amon.zg.gn.v20190308` |
| CNRM-CM6-1 | CNRM-CERFACS | r1i1p1f2 | gr | 250 km | 1.4° × 1.4° | `ESGF CMIP6.CMIP.CNRM-CERFACS.CNRM-CM6-1.historical.r1i1p1f2.Amon.zg.gr.v20180917` |
| E3SM-2-0 | E3SM-Project | r1i1p1f1 | gr | 100 km | 1.0° × 1.0° | `ESGF CMIP6.CMIP.E3SM-Project.E3SM-2-0.historical.r1i1p1f1.Amon.zg.gr.v20220830` |
| FGOALS-g3 | CAS | r1i1p1f1 | gn | 250 km | 2.0° × 2.0° | `ESGF CMIP6.CMIP.CAS.FGOALS-g3.historical.r1i1p1f1.Amon.zg.gn.v20191029` |
| GFDL-ESM4 | NOAA-GFDL | r1i1p1f1 | gr1 | 100 km | 1.0° × 1.25° | `ESGF CMIP6.CMIP.NOAA-GFDL.GFDL-ESM4.historical.r1i1p1f1.Amon.zg.gr1.v20180701` |
| GISS-E2-1-G | NASA-GISS | r1i1p1f1 | gn | 250 km | 2.0° × 2.5° | `ESGF CMIP6.CMIP.NASA-GISS.GISS-E2-1-G.historical.r1i1p1f1.Amon.zg.gn.v20180827` |
| HadGEM3-GC31-LL | MOHC | r1i1p1f3 | gn | 250 km | 1.25° × 1.875° | `ESGF CMIP6.CMIP.MOHC.HadGEM3-GC31-LL.historical.r1i1p1f3.Amon.zg.gn.v20190624` |
| IPSL-CM6A-LR | IPSL | r1i1p1f1 | gr | 250 km | 1.27° × 2.5° | `ESGF CMIP6.CMIP.IPSL.IPSL-CM6A-LR.historical.r1i1p1f1.Amon.zg.gr.v20180803` |
| MIROC6 | MIROC | r1i1p1f1 | gn | 250 km | 1.4° × 1.4° | `ESGF CMIP6.CMIP.MIROC.MIROC6.historical.r1i1p1f1.Amon.zg.gn.v20181212` |
| MRI-ESM2-0 | MRI | r1i1p1f1 | gn | 100 km | 1.12° × 1.125° | `ESGF CMIP6.CMIP.MRI.MRI-ESM2-0.historical.r1i1p1f1.Amon.zg.gn.v20190222` |
| NorESM2-LM | NCC | r1i1p1f1 | gn | 250 km | 1.9° × 2.5° | `ESGF CMIP6.CMIP.NCC.NorESM2-LM.historical.r1i1p1f1.Amon.zg.gn.v20190815` |

CSV 记录核验状态、`tracking_id` 和核验路径。15 个模式均标记为 `verified_local_netcdf`。CESM2 最小元数据文件的来源 Zarr、大小和 SHA-256 另见 `data/manifests/cesm2_registry_metadata.json`。
