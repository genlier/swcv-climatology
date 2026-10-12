"""Approximate terrain-following SWCV intensity on the sigma=0.85 surface.

ERA5 winds are linearly interpolated in pressure between 700 and 850 hPa at
each grid cell to p=0.85*surface pressure. Relative vorticity is then computed
on that spatially varying surface. Results are retained only where the target
pressure is bracketed by the two pressure levels.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import xarray as xr
from scipy.stats import pearsonr, spearmanr


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw"
OUT = ROOT / "data/processed"
DOC = ROOT / "docs/sigma085_intensity_sensitivity_zh.md"
EARTH_R = 6_371_000.0
SIGMA = 0.85
MIN_VALID_FRACTION = 0.80


def normalize(ds: xr.Dataset, level: int | None = None) -> xr.Dataset:
    rename = {}
    for old, new in (("valid_time", "time"), ("pressure_level", "level")):
        if old in ds.coords and new not in ds.coords:
            rename[old] = new
    ds = ds.rename(rename)
    if level is not None and "level" in ds.dims:
        ds = ds.sel(level=level, drop=True)
    return ds.sortby("latitude").sortby("longitude")


def relative_vorticity(u: xr.DataArray, v: xr.DataArray) -> xr.DataArray:
    dy_per_degree = EARTH_R * np.deg2rad(1.0)
    dx_per_degree = EARTH_R * np.cos(np.deg2rad(u.latitude)) * np.deg2rad(1.0)
    return v.differentiate("longitude") / dx_per_degree - u.differentiate("latitude") / dy_per_degree


def point_metric(ds700: xr.Dataset, ds850: xr.Dataset, dssp: xr.Dataset, time: np.datetime64, lat: float, lon: float) -> dict[str, float]:
    bounds = {"latitude": slice(lat - 2.0, lat + 2.0), "longitude": slice(lon - 2.0, lon + 2.0)}
    low = ds700[["u", "v"]].sel(time=time, method="nearest").sel(**bounds)
    high = ds850[["u", "v"]].sel(time=time, method="nearest").sel(**bounds)
    sp = dssp["sp"].sel(time=time, method="nearest").sel(latitude=low.latitude, longitude=low.longitude) / 100.0
    target = SIGMA * sp
    valid = (target >= 700.0) & (target <= 850.0)
    alpha = (target - 700.0) / 150.0
    u_sigma = (low.u + alpha * (high.u - low.u)).where(valid)
    v_sigma = (low.v + alpha * (high.v - low.v)).where(valid)
    zeta = relative_vorticity(u_sigma, v_sigma)
    radius = np.sqrt((zeta.latitude - lat) ** 2 + ((zeta.longitude - lon) * np.cos(np.deg2rad(lat))) ** 2)
    core = radius <= 1.5
    valid_fraction = float(zeta.notnull().where(core).mean(skipna=True))
    return {
        "sigma085_target_pressure_hpa": float(target.sel(latitude=lat, longitude=lon, method="nearest")),
        "sigma085_core_valid_fraction": valid_fraction,
        "sigma085_max_vorticity_s_1": float(zeta.where(core).max(skipna=True)) if valid_fraction >= MIN_VALID_FRACTION else np.nan,
        "sigma085_mean_vorticity_s_1": float(zeta.where(core).mean(skipna=True)) if valid_fraction >= MIN_VALID_FRACTION else np.nan,
    }


def collocate() -> pd.DataFrame:
    points = pd.read_csv(OUT / "track_points.csv.gz", parse_dates=["time"])
    points = points.loc[points.time.dt.year.between(1979, 2023) & points.time.dt.hour.isin([0, 6, 12, 18])].copy()
    rows = []
    for year, sample in points.groupby(points.time.dt.year):
        paths = {
            700: RAW / "era5_700hpa_6hourly" / f"era5_700hpa_{year}.nc",
            850: RAW / "era5_850hpa_6hourly" / f"era5_850hpa_{year}.nc",
            "sp": RAW / "era5_surface_pressure_6hourly" / f"era5_surface_pressure_{year}.nc",
        }
        if any(not path.exists() for path in paths.values()):
            raise FileNotFoundError(f"Missing sigma input for {year}")
        with xr.open_dataset(paths[700]) as raw700, xr.open_dataset(paths[850]) as raw850, xr.open_dataset(paths["sp"]) as rawsp:
            ds700, ds850, dssp = normalize(raw700, 700), normalize(raw850, 850), normalize(rawsp)
            for row in sample.itertuples():
                record = {"event_id": row.event_id, "time": row.time, "lat": row.lat, "lon": row.lon}
                record.update(point_metric(ds700, ds850, dssp, row.time.to_datetime64(), row.lat, row.lon))
                rows.append(record)
        print(f"sigma=0.85 collocation {year}: {len(sample)} points", flush=True)
    result = pd.DataFrame(rows)
    result.to_csv(OUT / "sigma085_intensity_points.csv.gz", index=False)
    return result


def fit_year_effect(frame: pd.DataFrame, response: str) -> dict[str, float]:
    work = frame.dropna(subset=[response]).copy()
    year_c = work.year - work.year.mean()
    season_dummies = pd.get_dummies(work.season, prefix="season", drop_first=True, dtype=float)
    design = pd.concat([pd.DataFrame({"const": 1.0, "year_c": year_c, "log_duration": np.log1p(work.duration_hours)}), season_dummies], axis=1)
    fit = sm.OLS(work[response] * 1e5, design).fit(cov_type="cluster", cov_kwds={"groups": work.year})
    return {
        "n_events": int(len(work)),
        "trend_1e5_s_1_per_decade": float(fit.params["year_c"] * 10.0),
        "trend_p_year_clustered": float(fit.pvalues["year_c"]),
        "trend_ci_low": float(fit.conf_int().loc["year_c", 0] * 10.0),
        "trend_ci_high": float(fit.conf_int().loc["year_c", 1] * 10.0),
    }


def fit_winter_q90(frame: pd.DataFrame, response: str) -> dict[str, float]:
    winter = frame.loc[frame.season.eq("DJF") & frame.season_year.between(1980, 2023)].dropna(subset=[response])
    annual = winter.groupby("season_year")[response].quantile(0.9).reset_index()
    year_c = annual.season_year - annual.season_year.mean()
    fit = sm.OLS(annual[response] * 1e5, sm.add_constant(year_c)).fit(cov_type="HAC", cov_kwds={"maxlags": 2})
    return {
        "n_winters": int(len(annual)),
        "n_events": int(len(winter)),
        "trend_1e5_s_1_per_decade": float(fit.params.iloc[1] * 10.0),
        "trend_p_hac": float(fit.pvalues.iloc[1]),
        "trend_ci_low": float(fit.conf_int().iloc[1, 0] * 10.0),
        "trend_ci_high": float(fit.conf_int().iloc[1, 1] * 10.0),
    }


def summarize(points: pd.DataFrame) -> None:
    event = points.groupby("event_id").agg(
        sampled_points=("time", "size"),
        valid_point_fraction_sigma085=("sigma085_max_vorticity_s_1", lambda x: x.notna().mean()),
        max_vorticity_s_1_sigma085=("sigma085_max_vorticity_s_1", "max"),
        mean_vorticity_s_1_sigma085=("sigma085_mean_vorticity_s_1", "mean"),
        median_target_pressure_hpa=("sigma085_target_pressure_hpa", "median"),
    ).reset_index()
    metadata = pd.read_csv(OUT / "events_terrain_qc_intensity.csv.gz", parse_dates=["genesis_time"])
    month = metadata.genesis_time.dt.month
    metadata["season"] = np.select(
        [month.isin([12, 1, 2]), month.isin([3, 4, 5]), month.isin([6, 7, 8])],
        ["DJF", "MAM", "JJA"],
        default="SON",
    )
    metadata["season_year"] = metadata.year + month.eq(12).astype(int)
    keep = ["event_id", "year", "season", "season_year", "duration_hours", "track_cluster", "max_vorticity_s_1_700_c25"]
    event = metadata[keep].merge(event, on="event_id", how="left", validate="one_to_one")
    event.to_csv(OUT / "events_sigma085_intensity.csv.gz", index=False)

    common = event.dropna(subset=["max_vorticity_s_1_sigma085", "max_vorticity_s_1_700_c25"])
    pearson = pearsonr(common.max_vorticity_s_1_sigma085, common.max_vorticity_s_1_700_c25)
    spearman = spearmanr(common.max_vorticity_s_1_sigma085, common.max_vorticity_s_1_700_c25)
    trends = {
        "sigma085_all_valid": fit_year_effect(event, "max_vorticity_s_1_sigma085"),
        "sigma085_common_with_700": fit_year_effect(common, "max_vorticity_s_1_sigma085"),
        "700_c25_common_with_sigma": fit_year_effect(common, "max_vorticity_s_1_700_c25"),
    }
    winter_q90_trends = {
        "sigma085_all_valid": fit_winter_q90(event, "max_vorticity_s_1_sigma085"),
        "sigma085_common_with_700": fit_winter_q90(common, "max_vorticity_s_1_sigma085"),
        "700_c25_common_with_sigma": fit_winter_q90(common, "max_vorticity_s_1_700_c25"),
    }
    summary = {
        "coordinate": "sigma=0.85 approximated by linear pressure interpolation between 700 and 850 hPa",
        "minimum_core_valid_fraction": MIN_VALID_FRACTION,
        "events_total": int(len(event)),
        "events_sigma_valid": int(event.max_vorticity_s_1_sigma085.notna().sum()),
        "events_common_sigma_700": int(len(common)),
        "sigma_event_retention_fraction": float(event.max_vorticity_s_1_sigma085.notna().mean()),
        "common_pearson_r": float(pearson.statistic),
        "common_pearson_p": float(pearson.pvalue),
        "common_spearman_rho": float(spearman.statistic),
        "common_spearman_p": float(spearman.pvalue),
        "trends": trends,
        "winter_q90_trends": winter_q90_trends,
    }
    (OUT / "sigma085_intensity_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    sigma_trend = trends["sigma085_common_with_700"]
    fixed_trend = trends["700_c25_common_with_sigma"]
    sigma_winter = winter_q90_trends["sigma085_common_with_700"]
    fixed_winter = winter_q90_trends["700_c25_common_with_sigma"]
    direction = "一致" if np.sign(sigma_trend["trend_1e5_s_1_per_decade"]) == np.sign(fixed_trend["trend_1e5_s_1_per_decade"]) else "不一致"
    DOC.write_text(
        f"""# 近似地形跟随 sigma=0.85 涡度复核（R1-M2）

利用已有 ERA5 700/850 hPa 逐 6 小时风场和地面气压，在每个网格将风线性插值到 $p=0.85p_s$，随后在该空间变化的压力面上计算相对涡度。目标压力必须位于 700--850 hPa，且涡心 1.5° 圆域至少 80% 网格有效。位势高度差没有沿用，因为 sigma 面几何高度会直接随地形变化，容易把地形坡度误判为气旋高度亏损。

## 结果

- 全部 {len(event):,} 个 1979--2023 年事件中，{summary['events_sigma_valid']:,} 个具有有效 sigma=0.85 最大涡度（保留率 {100*summary['sigma_event_retention_fraction']:.1f}%）。
- 与地形质控 700 hPa 指标共有 {len(common):,} 个事件；Pearson $r={pearson.statistic:.3f}$，Spearman $\\rho={spearman.statistic:.3f}$。
- 在同一共同样本并控制季节与事件时长后，sigma 指标的年代际斜率为 {sigma_trend['trend_1e5_s_1_per_decade']:+.3f} ×10$^{{-5}}$ s$^{{-1}}$ decade$^{{-1}}$（年份聚类 $p={sigma_trend['trend_p_year_clustered']:.3g}$），700 hPa 指标为 {fixed_trend['trend_1e5_s_1_per_decade']:+.3f}（$p={fixed_trend['trend_p_year_clustered']:.3g}$）；方向{direction}。
- 与论文冬季上尾结论直接对应的共同样本 DJF 年度 90 分位趋势：sigma 指标为 {sigma_winter['trend_1e5_s_1_per_decade']:+.3f} ×10$^{{-5}}$ s$^{{-1}}$ decade$^{{-1}}$（HAC $p={sigma_winter['trend_p_hac']:.3g}$），700 hPa 指标为 {fixed_winter['trend_1e5_s_1_per_decade']:+.3f}（$p={fixed_winter['trend_p_hac']:.3g}$）。

## 边界

这是一项由两个标准压力层线性插值得到的近似 sigma 坐标敏感性，而非 ERA5 原生模式层诊断。它避免了固定 700 hPa 在高地穿地的问题，但只保留 $0.85p_s$ 被 700/850 hPa 夹住的位置，并不能恢复边界层完整垂直结构。结果用于检验涡度趋势方向是否依赖固定等压面，不用于替代原生模式层动力分析。

ECMWF 官方垂直坐标说明指出，ERA5 原生大气场使用 137 个混合 sigma/pressure 模式层，半层压力由 $p=a+b p_s$ 给出，而标准压力层产品来自垂直插值。因此本分析明确标记为双压力层近似；原生模式层复核仍是后续工作。来源：https://confluence.ecmwf.int/pages/viewpage.action?pageId=226496389
""",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


def main() -> None:
    points_path = OUT / "sigma085_intensity_points.csv.gz"
    points = pd.read_csv(points_path, parse_dates=["time"]) if points_path.exists() else collocate()
    summarize(points)


if __name__ == "__main__":
    main()
