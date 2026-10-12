"""Analyse native ERA5 sigma=0.85 intensity for DJF 1980--2023."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import xarray as xr

from analyze_native_model_level_sigma_pilot import canonical, correlation, point_metrics
from download_era5_model_level_sigma_full import selection


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw"
OUT = ROOT / "data/processed"
MANIFEST = ROOT / "data/manifests/era5_model_level_sigma_djf_manifest.json"
POINTS_OUT = OUT / "native_sigma085_djf_points.csv.gz"
EVENTS_OUT = OUT / "native_sigma085_djf_events.csv.gz"
ANNUAL_OUT = OUT / "native_sigma085_djf_annual_q90.csv"
SUMMARY_OUT = OUT / "native_sigma085_djf_summary.json"
DOC = ROOT / "docs/native_sigma085_djf_trend_zh.md"


def fit_q90(frame: pd.DataFrame, response: str) -> tuple[dict[str, float | int], pd.DataFrame]:
    work = frame.dropna(subset=[response]).copy()
    annual = work.groupby("season_year")[response].quantile(0.9).reset_index()
    if len(annual) < 10:
        return {
            "n_events": int(len(work)),
            "n_winters": int(len(annual)),
            "trend_1e5_s_1_per_decade": np.nan,
            "trend_p_hac": np.nan,
            "trend_ci_low": np.nan,
            "trend_ci_high": np.nan,
        }, annual
    year_c = annual.season_year - annual.season_year.mean()
    fit = sm.OLS(annual[response] * 1e5, sm.add_constant(year_c)).fit(
        cov_type="HAC", cov_kwds={"maxlags": 2}
    )
    result = {
        "n_events": int(len(work)),
        "n_winters": int(len(annual)),
        "trend_1e5_s_1_per_decade": float(fit.params.iloc[1] * 10.0),
        "trend_p_hac": float(fit.pvalues.iloc[1]),
        "trend_ci_low": float(fit.conf_int().iloc[1, 0] * 10.0),
        "trend_ci_high": float(fit.conf_int().iloc[1, 1] * 10.0),
    }
    return result, annual


def load_points(manifest: dict[str, object]) -> pd.DataFrame:
    points, _ = selection()
    coefficients = {int(key): value for key, value in manifest["half_level_coefficients"].items()}
    rows: list[pd.DataFrame] = []
    files = {int(item["year"]): ROOT / item["path"] for item in manifest["files"]}
    for year, group in points.groupby(points.time.dt.year):
        year = int(year)
        if year not in files:
            raise FileNotFoundError(f"Manifest/file absent for calendar year {year}")
        surface_path = RAW / f"era5_surface_pressure_6hourly/era5_surface_pressure_{year}.nc"
        with xr.open_dataset(files[year]) as raw_model, xr.open_dataset(surface_path) as raw_surface:
            model = canonical(raw_model).load()
            surface = canonical(raw_surface).load()
        try:
            rows.append(point_metrics(model, {year: surface}, group, coefficients))
        finally:
            model.close()
            surface.close()
        print(f"analysed {year}: {len(group)} track points", flush=True)
    return pd.concat(rows, ignore_index=True)


def main() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if manifest.get("failures") or len(manifest.get("files", [])) != 45:
        raise RuntimeError("Full native-model-level manifest is incomplete")
    native = load_points(manifest)
    approx = pd.read_csv(OUT / "sigma085_intensity_points.csv.gz", parse_dates=["time"])[
        ["event_id", "time", "sigma085_max_vorticity_s_1"]
    ]
    fixed = pd.read_csv(OUT / "terrain_qc_intensity_points.csv.gz", parse_dates=["time"])[
        ["event_id", "time", "max_vorticity_s_1_700_c25"]
    ]
    for frame in (approx, fixed):
        if frame.time.dt.tz is not None:
            frame["time"] = frame.time.dt.tz_localize(None)
    native = native.merge(approx, on=["event_id", "time"], how="left", validate="one_to_one")
    native = native.merge(fixed, on=["event_id", "time"], how="left", validate="one_to_one")
    native.to_csv(POINTS_OUT, index=False)

    metadata = pd.read_csv(
        OUT / "events_terrain_qc_intensity.csv.gz",
        parse_dates=["genesis_time"],
        usecols=["event_id", "genesis_time", "track_cluster", "duration_hours"],
    )
    metadata["season_year"] = metadata.genesis_time.dt.year + metadata.genesis_time.dt.month.eq(12).astype(int)
    events = native.groupby("event_id").agg(
        sampled_points=("time", "size"),
        valid_point_fraction_native=("max_vorticity_s_1_native", lambda x: x.notna().mean()),
        max_vorticity_s_1_native=("max_vorticity_s_1_native", "max"),
        max_vorticity_s_1_approx=("sigma085_max_vorticity_s_1", "max"),
        max_vorticity_s_1_700_c25=("max_vorticity_s_1_700_c25", "max"),
        median_target_pressure_hpa=("target_pressure_hpa_native", "median"),
        mean_core_bracket_fraction=("core_bracket_fraction_native", "mean"),
    ).reset_index().merge(metadata, on="event_id", how="left", validate="one_to_one")
    events.to_csv(EVENTS_OUT, index=False)

    native_approx = correlation(events, "max_vorticity_s_1_native", "max_vorticity_s_1_approx")
    native_700 = correlation(events, "max_vorticity_s_1_native", "max_vorticity_s_1_700_c25")
    native_all, native_annual = fit_q90(events, "max_vorticity_s_1_native")
    common_approx = events.dropna(subset=["max_vorticity_s_1_native", "max_vorticity_s_1_approx"])
    native_common_approx, _ = fit_q90(common_approx, "max_vorticity_s_1_native")
    approx_common_native, _ = fit_q90(common_approx, "max_vorticity_s_1_approx")
    common_700 = events.dropna(subset=["max_vorticity_s_1_native", "max_vorticity_s_1_700_c25"])
    native_common_700, _ = fit_q90(common_700, "max_vorticity_s_1_native")
    fixed700_common_native, _ = fit_q90(common_700, "max_vorticity_s_1_700_c25")
    cluster_trends = {}
    annual_parts = [native_annual.assign(metric="native_all", track_cluster="all")]
    for cluster, group in events.groupby("track_cluster"):
        trend, annual = fit_q90(group, "max_vorticity_s_1_native")
        cluster_trends[str(int(cluster))] = trend
        annual_parts.append(annual.assign(metric="native_cluster", track_cluster=int(cluster)))
    pd.concat(annual_parts, ignore_index=True).to_csv(ANNUAL_OUT, index=False)

    event_valid_fraction = float(events.max_vorticity_s_1_native.notna().mean())
    point_valid_fraction = float(native.max_vorticity_s_1_native.notna().mean())
    bracket_fraction = float(native.core_bracket_fraction_native.mean())
    summary = {
        "scope": "DJF season-years 1980-2023",
        "events": int(len(events)),
        "track_points": int(len(native)),
        "event_valid_fraction": event_valid_fraction,
        "point_valid_fraction": point_valid_fraction,
        "mean_core_bracket_fraction": bracket_fraction,
        "target_pressure_hpa_range": [float(native.target_pressure_hpa_native.min()), float(native.target_pressure_hpa_native.max())],
        "correlations": {
            "event_native_vs_approx": native_approx,
            "event_native_vs_700_c25": native_700,
        },
        "winter_q90_trends": {
            "native_all": native_all,
            "native_common_with_approx": native_common_approx,
            "approx_common_with_native": approx_common_native,
            "native_common_with_700": native_common_700,
            "700_common_with_native": fixed700_common_native,
            "native_by_track_cluster": cluster_trends,
        },
        "coefficient_source": manifest["coefficient_source"],
        "request_documentation": manifest["request_documentation"],
    }
    SUMMARY_OUT.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    ntrend = native_all
    atrend = approx_common_native
    ftrend = fixed700_common_native
    DOC.write_text(
        f"""# ERA5 原生模式层 sigma=0.85 冬季强度复核

## 数据与方法

对 1980--2023 年 DJF 的 {len(events):,} 个事件和 {len(native):,} 个六小时轨迹点，使用 ERA5 原生模式层 110--115。逐格根据 L137 混合坐标和同一时次地面气压计算全层压力，将 `u/v` 线性插值到 $p=0.85p_s$，随后计算涡心 1.5° 圆域最大相对涡度。年度强度定义为各冬季事件最大涡度的 90 分位，线性趋势使用 HAC（2 阶滞后）不确定性。

## 质量控制与一致性

- 原生指标事件有效率 {100*event_valid_fraction:.1f}%，点有效率 {100*point_valid_fraction:.1f}%，核心区平均夹层覆盖率 {100*bracket_fraction:.1f}%。
- 原生与双压力层近似的事件级 Spearman $\\rho={native_approx['spearman_rho']:.3f}$（$n={native_approx['n']}$）；原生与地形质控 700 hPa 为 $\\rho={native_700['spearman_rho']:.3f}$（$n={native_700['n']}$）。

## 冬季上尾趋势

- 原生 sigma 全有效样本：{ntrend['trend_1e5_s_1_per_decade']:+.3f} $\\times10^{{-5}}$ s$^{{-1}}$ decade$^{{-1}}$（95% CI {ntrend['trend_ci_low']:+.3f} 至 {ntrend['trend_ci_high']:+.3f}，HAC $p={ntrend['trend_p_hac']:.3g}$；{ntrend['n_events']} 个事件、{ntrend['n_winters']} 个冬季）。
- 与原生共同样本上的双压力层近似：{atrend['trend_1e5_s_1_per_decade']:+.3f}（$p={atrend['trend_p_hac']:.3g}$）。
- 与原生共同样本上的 700 hPa 地形质控指标：{ftrend['trend_1e5_s_1_per_decade']:+.3f}（$p={ftrend['trend_p_hac']:.3g}$）。

该结果用于判断论文中的冬季上尾增强能否脱离固定 700 hPa 坐标成立。任何主文修改必须同时报告原生全样本和共同样本，不能只选择方向有利的指标。

系数定义：{manifest['coefficient_source']}

ERA5 Complete 请求说明：{manifest['request_documentation']}
""",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
