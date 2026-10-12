"""Diagnose large-scale winter circulation covariates of Sichuan-basin SWCV counts."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import xarray as xr
from scipy.stats import pearsonr, spearmanr


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "era5_djf_mechanism_monthly"
OUT = ROOT / "data" / "processed"
FIG = ROOT / "figures"
DOC = ROOT / "docs" / "djf_mechanism_diagnostics_zh.md"
YEARS = np.arange(1980, 2024)


def canonical(ds: xr.Dataset) -> xr.Dataset:
    rename = {}
    for old, new in (("valid_time", "time"), ("pressure_level", "level"), ("latitude", "lat"), ("longitude", "lon")):
        if old in ds.coords or old in ds.dims:
            rename[old] = new
    return ds.rename(rename)


def load_seasonal() -> xr.Dataset:
    datasets = []
    for path in sorted(RAW.glob("era5_djf_mechanism_monthly_*.nc")):
        with xr.open_dataset(path) as source:
            datasets.append(canonical(source).load())
    if len(datasets) != 46:
        raise RuntimeError(f"Expected 46 verified annual files, found {len(datasets)}")
    monthly = xr.concat(datasets, dim="time").sortby("time")
    season_year = monthly.time.dt.year + xr.where(monthly.time.dt.month == 12, 1, 0)
    monthly = monthly.assign_coords(season_year=("time", season_year.values.astype(int)))
    seasonal = monthly.groupby("season_year").mean("time").sel(season_year=YEARS)
    counts = monthly.time.groupby("season_year").count().sel(season_year=YEARS)
    if not bool((counts == 3).all()):
        raise RuntimeError("One or more DJF seasons do not contain exactly three monthly means")
    return seasonal


def box(field: xr.DataArray, lat0: float, lat1: float, lon0: float, lon1: float) -> xr.DataArray:
    lat_slice = slice(lat1, lat0) if float(field.lat[0]) > float(field.lat[-1]) else slice(lat0, lat1)
    return field.sel(lat=lat_slice, lon=slice(lon0, lon1))


def area_mean(field: xr.DataArray, lat0: float, lat1: float, lon0: float, lon1: float) -> xr.DataArray:
    selected = box(field, lat0, lat1, lon0, lon1)
    weights = np.cos(np.deg2rad(selected.lat))
    return selected.weighted(weights).mean(("lat", "lon"))


def build_indices(ds: xr.Dataset) -> pd.DataFrame:
    u200 = ds.u.sel(level=200)
    u850, v850 = ds.u.sel(level=850), ds.v.sel(level=850)
    u700, v700 = ds.u.sel(level=700), ds.v.sel(level=700)
    t850, t700 = ds.t.sel(level=850), ds.t.sel(level=700)
    theta850 = t850 * (1000.0 / 850.0) ** 0.286
    theta700 = t700 * (1000.0 / 700.0) ** 0.286
    low_shear = np.hypot(u700 - u850, v700 - v850)
    deep_shear = np.hypot(u200 - u850, ds.v.sel(level=200) - v850)

    jet_band = box(u200, 20, 45, 80, 120).mean("lon")
    jet_axis = jet_band.lat.isel(lat=jet_band.argmax("lat"))
    result = pd.DataFrame(
        {
            "season_year": ds.season_year.values.astype(int),
            "east_asia_jet_max_u200_ms": jet_band.max("lat").values,
            "east_asia_jet_axis_lat": jet_axis.values,
            "south_branch_u200_ms": area_mean(u200, 20, 35, 80, 105).values,
            "basin_northerly_v850_ms": (-area_mean(v850, 30, 36, 102, 110)).values,
            "basin_t850_k": area_mean(t850, 27, 33, 102, 109).values,
            "basin_static_stability_k": area_mean(theta700 - theta850, 27, 33, 102, 109).values,
            "basin_low_shear_ms": area_mean(low_shear, 27, 33, 102, 109).values,
            "basin_deep_shear_ms": area_mean(deep_shear, 27, 33, 102, 109).values,
        }
    )
    return result


def bh_adjust(values: pd.Series) -> pd.Series:
    order = values.sort_values()
    n = len(order)
    adjusted = (order * n / np.arange(1, n + 1)).iloc[::-1].cummin().iloc[::-1].clip(upper=1)
    return adjusted.reindex(values.index)


def statistics(table: pd.DataFrame) -> pd.DataFrame:
    response = "cluster1_frequency"
    predictors = [column for column in table if column not in {"season_year", response, "frequency"}]
    rows = []
    year = table.season_year - table.season_year.mean()
    for predictor in predictors:
        x = table[predictor]
        pearson = pearsonr(x, table[response])
        spearman = spearmanr(x, table[response])
        standardized = (x - x.mean()) / x.std(ddof=1)
        design = sm.add_constant(pd.DataFrame({"index_z": standardized, "year": year}))
        fit = sm.GLM(table[response], design, family=sm.families.Poisson()).fit(
            cov_type="HAC", cov_kwds={"maxlags": 3}
        )
        rows.append(
            {
                "response": response,
                "predictor": predictor,
                "pearson_r": pearson.statistic,
                "pearson_p": pearson.pvalue,
                "spearman_rho": spearman.statistic,
                "spearman_p": spearman.pvalue,
                "poisson_log_rate_per_sd_adjusted_for_year": fit.params["index_z"],
                "poisson_rate_percent_per_sd": 100 * np.expm1(fit.params["index_z"]),
                "poisson_p_hac3": fit.pvalues["index_z"],
                "year_log_rate_per_decade": fit.params["year"] * 10,
                "n_winters": int(fit.nobs),
            }
        )
    result = pd.DataFrame(rows)
    result["pearson_p_fdr"] = bh_adjust(result.pearson_p)
    result["poisson_p_fdr"] = bh_adjust(result.poisson_p_hac3)
    return result.sort_values("poisson_p_hac3")


def composite_fields(ds: xr.Dataset, table: pd.DataFrame) -> xr.Dataset:
    ranked = table.sort_values("cluster1_frequency")
    n = len(ranked) // 3
    low_years = ranked.head(n).season_year.to_numpy()
    high_years = ranked.tail(n).season_year.to_numpy()
    difference = ds.sel(season_year=high_years).mean("season_year") - ds.sel(season_year=low_years).mean("season_year")
    difference.attrs["high_years"] = ",".join(map(str, high_years))
    difference.attrs["low_years"] = ",".join(map(str, low_years))
    return difference


def plot_summary(table: pd.DataFrame, stats: pd.DataFrame, composite: xr.Dataset) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    best = stats.iloc[0].predictor
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    axes[0, 0].plot(table.season_year, table.cluster1_frequency, color="#222222")
    axes[0, 0].set(title="Sichuan-basin local DJF vortices", xlabel="DJF year", ylabel="Events")
    axes[0, 1].scatter(table[best], table.cluster1_frequency, color="#2b6cb0")
    fit = np.polyfit(table[best], table.cluster1_frequency, 1)
    xx = np.linspace(table[best].min(), table[best].max())
    axes[0, 1].plot(xx, np.polyval(fit, xx), color="#c43c39")
    axes[0, 1].set(title=best.replace("_", " "), xlabel=best, ylabel="Events")
    u200 = composite.u.sel(level=200)
    v850 = composite.v.sel(level=850)
    for ax, field, title, cmap in (
        (axes[1, 0], u200, "High minus low vortex winters: U200 (m s$^{-1}$)", "RdBu_r"),
        (axes[1, 1], v850, "High minus low vortex winters: V850 (m s$^{-1}$)", "RdBu_r"),
    ):
        limit = float(np.nanpercentile(np.abs(field.values), 98))
        image = ax.contourf(field.lon, field.lat, field, levels=np.linspace(-limit, limit, 17), cmap=cmap, extend="both")
        ax.set(xlabel="Longitude", ylabel="Latitude", title=title)
        fig.colorbar(image, ax=ax, shrink=0.85)
    fig.savefig(FIG / "djf_mechanism_summary.png", dpi=220)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    seasonal = load_seasonal()
    indices = build_indices(seasonal)
    counts = pd.read_csv(OUT / "djf_change_structure_annual.csv")
    table = counts[["season_year", "frequency", "cluster1_frequency"]].merge(
        indices, on="season_year", how="inner", validate="one_to_one"
    )
    if len(table) != len(YEARS):
        raise RuntimeError(f"Expected {len(YEARS)} winters after merge, found {len(table)}")
    stats = statistics(table)
    composite = composite_fields(seasonal, table)
    table.to_csv(OUT / "djf_mechanism_indices.csv", index=False)
    stats.to_csv(OUT / "djf_mechanism_statistics.csv", index=False)
    composite.to_netcdf(OUT / "djf_mechanism_high_minus_low_composites.nc")
    plot_summary(table, stats, composite)

    best = stats.iloc[0]
    summary = {
        "winters": [int(YEARS.min()), int(YEARS.max())],
        "n_winters": int(len(table)),
        "response": "cluster1_frequency",
        "strongest_adjusted_predictor": best.predictor,
        "rate_percent_per_sd": float(best.poisson_rate_percent_per_sd),
        "p_hac3": float(best.poisson_p_hac3),
        "p_fdr": float(best.poisson_p_fdr),
    }
    DOC.write_text(
        "# DJF 冬季动力机制诊断\n\n"
        "本文件由 `scripts/analyze_djf_mechanism.py` 自动生成。结果在写入论文前需检查空间合成和多重检验。\n\n"
        f"- 冬季范围：{YEARS.min()}--{YEARS.max()}（以 1 月年份标记）\n"
        f"- 主要响应：Sichuan-basin local cluster 1 frequency\n"
        f"- 时间趋势校正后最强候选指标：`{best.predictor}`\n"
        f"- 每 1 s.d. 的频数变化：{best.poisson_rate_percent_per_sd:.2f}%\n"
        f"- HAC p={best.poisson_p_hac3:.4g}，FDR p={best.poisson_p_fdr:.4g}\n\n"
        "完整结果见 `data/processed/djf_mechanism_statistics.csv`；高低频冬季合成见 "
        "`data/processed/djf_mechanism_high_minus_low_composites.nc` 和 `figures/djf_mechanism_summary.png`。\n",
        encoding="utf-8",
    )
    (OUT / "djf_mechanism_status.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
