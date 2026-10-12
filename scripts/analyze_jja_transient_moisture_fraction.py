"""Decompose same-level 6-hourly ERA5 JJA qv into mean and transient terms."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import xarray as xr


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/era5_jja_qv_6hourly"
MANIFEST = ROOT / "data/manifests/era5_jja_qv_6hourly_manifest.json"
OUT = ROOT / "data/processed"
DOC = ROOT / "docs/jja_transient_moisture_flux_zh.md"
YEARS = range(1980, 2024)
LEVELS = (700, 850)


def canonical(ds: xr.Dataset) -> xr.Dataset:
    rename = {}
    for old, new in (("valid_time", "time"), ("latitude", "lat"), ("longitude", "lon"), ("pressure_level", "level")):
        if old in ds.coords or old in ds.dims:
            rename[old] = new
    return ds.rename(rename)


def weighted_space_mean(field: xr.DataArray) -> float:
    weights = np.cos(np.deg2rad(field.lat))
    return float(field.weighted(weights).mean(("lat", "lon"), skipna=True))


def validate_manifest() -> None:
    if not MANIFEST.exists():
        raise FileNotFoundError(f"Missing completed manifest: {MANIFEST}")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if manifest.get("failures") or len(manifest.get("files", [])) != len(tuple(YEARS)):
        raise RuntimeError("The same-level q/v download manifest is incomplete")


def decompose_year(year: int) -> tuple[list[dict[str, float]], list[dict[str, float]]]:
    path = RAW / f"era5_jja_qv_6hourly_{year}.nc"
    with xr.open_dataset(path) as source:
        ds = canonical(source).load()
    seasonal_rows, monthly_rows = [], []
    for level in LEVELS:
        level_ds = ds.sel(level=level)
        month_fields = []
        for month in (6, 7, 8):
            part = level_ds.sel(time=level_ds.time.dt.month == month)
            q, v = part.q.astype("float64"), part.v.astype("float64")
            total = (q * v).mean("time")
            mean_flow = q.mean("time") * v.mean("time")
            transient = ((q - q.mean("time")) * (v - v.mean("time"))).mean("time")
            closure = total - mean_flow - transient
            row = {
                "season_year": year,
                "month": month,
                "level_hpa": level,
                "n_times": int(part.sizes["time"]),
                "total_qv": weighted_space_mean(total),
                "monthly_mean_flow_qv": weighted_space_mean(mean_flow),
                "submonthly_transient_qv": weighted_space_mean(transient),
                "closure_error": weighted_space_mean(closure),
            }
            row["transient_fraction"] = row["submonthly_transient_qv"] / row["total_qv"]
            monthly_rows.append(row)
            month_fields.append((part.sizes["time"], total, mean_flow, transient, closure))

        n_total = sum(item[0] for item in month_fields)
        aggregate = []
        for field_index in range(1, 5):
            aggregate.append(sum(item[0] * item[field_index] for item in month_fields) / n_total)
        total, mean_flow, transient, closure = aggregate

        # Sensitivity around one seasonal mean includes the monthly-mean
        # progression as well as submonthly variability.
        q, v = level_ds.q.astype("float64"), level_ds.v.astype("float64")
        seasonal_transient = ((q - q.mean("time")) * (v - v.mean("time"))).mean("time")
        seasonal_mean_flow = q.mean("time") * v.mean("time")
        values = {
            "season_year": year,
            "level_hpa": level,
            "n_times": int(level_ds.sizes["time"]),
            "total_qv": weighted_space_mean(total),
            "monthly_mean_flow_qv": weighted_space_mean(mean_flow),
            "submonthly_transient_qv": weighted_space_mean(transient),
            "closure_error": weighted_space_mean(closure),
            "seasonal_mean_flow_qv": weighted_space_mean(seasonal_mean_flow),
            "seasonal_transient_qv": weighted_space_mean(seasonal_transient),
        }
        values["transient_fraction"] = values["submonthly_transient_qv"] / values["total_qv"]
        values["absolute_partition_fraction"] = abs(values["submonthly_transient_qv"]) / (
            abs(values["monthly_mean_flow_qv"]) + abs(values["submonthly_transient_qv"])
        )
        seasonal_rows.append(values)
    ds.close()
    return seasonal_rows, monthly_rows


def hac_trend(values: pd.Series, years: pd.Series) -> dict[str, float]:
    x = years - years.mean()
    fit = sm.OLS(values, sm.add_constant(x)).fit(cov_type="HAC", cov_kwds={"maxlags": 2})
    return {
        "trend_per_decade": float(fit.params.iloc[1] * 10.0),
        "trend_p_hac": float(fit.pvalues.iloc[1]),
        "trend_ci_low": float(fit.conf_int().iloc[1, 0] * 10.0),
        "trend_ci_high": float(fit.conf_int().iloc[1, 1] * 10.0),
    }


def moving_block_ratio_ci(numerator: np.ndarray, denominator: np.ndarray, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    block, n = 5, len(numerator)
    # Circular blocks give every year equal probability of appearing, including
    # the record endpoints, while preserving five-year local dependence.
    starts = np.arange(n)
    means = np.empty(5000)
    for i in range(len(means)):
        picks = rng.choice(starts, int(np.ceil(n / block)), replace=True)
        indices = np.concatenate([(np.arange(j, j + block) % n) for j in picks])[:n]
        means[i] = numerator[indices].mean() / denominator[indices].mean()
    return tuple(np.quantile(means, [0.025, 0.975]))


def main() -> None:
    validate_manifest()
    seasonal_rows, monthly_rows = [], []
    for year in YEARS:
        seasonal, monthly = decompose_year(year)
        seasonal_rows.extend(seasonal)
        monthly_rows.extend(monthly)
        print(f"decomposed {year}", flush=True)
    annual = pd.DataFrame(seasonal_rows)
    monthly = pd.DataFrame(monthly_rows)
    if annual.closure_error.abs().max() > 1e-9 or monthly.closure_error.abs().max() > 1e-9:
        raise RuntimeError("qv decomposition failed the numerical closure check")

    summaries = []
    for level, group in annual.groupby("level_hpa"):
        fraction_ci = moving_block_ratio_ci(
            group.submonthly_transient_qv.to_numpy(), group.total_qv.to_numpy(), seed=20260913 + int(level)
        )
        row = {
            "level_hpa": int(level),
            "n_years": int(len(group)),
            "mean_total_qv": float(group.total_qv.mean()),
            "mean_monthly_mean_flow_qv": float(group.monthly_mean_flow_qv.mean()),
            "mean_submonthly_transient_qv": float(group.submonthly_transient_qv.mean()),
            "mean_transient_fraction": float(group.transient_fraction.mean()),
            "ratio_of_means_transient_fraction": float(group.submonthly_transient_qv.mean() / group.total_qv.mean()),
            "median_transient_fraction": float(group.transient_fraction.median()),
            "transient_fraction_ci_low": float(fraction_ci[0]),
            "transient_fraction_ci_high": float(fraction_ci[1]),
            "max_abs_closure_error": float(group.closure_error.abs().max()),
        }
        for metric in ("total_qv", "monthly_mean_flow_qv", "submonthly_transient_qv", "transient_fraction"):
            row.update({f"{metric}_{key}": value for key, value in hac_trend(group[metric], group.season_year).items()})
        summaries.append(row)
    summary = pd.DataFrame(summaries)
    annual.to_csv(OUT / "jja_qv_transient_decomposition_annual.csv", index=False)
    monthly.to_csv(OUT / "jja_qv_transient_decomposition_monthly.csv", index=False)
    summary.to_csv(OUT / "jja_qv_transient_decomposition_summary.csv", index=False)

    lines = [
        "# JJA 同层 6 小时水汽通量分解（R2-m1）\n",
        "ERA5 1980--2023 年 JJA 逐 6 小时比湿 $q$ 与北向风 $v$ 在 20--27.5°N、100--115°E 上进行完全同层、同单位和同掩膜分解。主定义先在每个自然月计算 $\\overline{qv}=\\overline q\\,\\overline v+\\overline{q'v'}$，再按该月时次数加权汇总至 JJA；因此瞬变项不混入月平均态的季节推进。\n",
        "## 核心结果\n",
    ]
    for row in summaries:
        lines.append(
            f"- {row['level_hpa']} hPa：瞬变项多年平均为 {row['mean_submonthly_transient_qv']:+.6g} kg kg⁻¹ m s⁻¹，"
            f"占多年平均总 $qv$ 的 {100*row['ratio_of_means_transient_fraction']:+.1f}%（5 年循环移动区组 bootstrap 95% 区间 "
            f"{100*row['transient_fraction_ci_low']:+.1f}% 至 {100*row['transient_fraction_ci_high']:+.1f}%）；"
            f"占比趋势为每十年 {100*row['transient_fraction_trend_per_decade']:+.2f} 个百分点"
            f"（HAC $p={row['transient_fraction_trend_p_hac']:.3g}$）。\n"
        )
    lines.extend([
        "\n## 解释边界\n",
        "这里的 $qv$ 是压力层比湿与风速的乘积，不是质量加权的全柱积分水汽输送；700 与 850 hPa 结果应分别解释，不能相加为柱积分。6 小时时间分辨率可量化天气尺度与次月尺度协方差，但会漏掉 6 小时以内的波动。因此结果回答月平均乘积遗漏了多少已解析瞬变项，而不是所有尺度的完整瞬变水汽通量。\n",
    ])
    DOC.write_text("".join(lines), encoding="utf-8")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
