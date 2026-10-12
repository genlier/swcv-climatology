"""Analyse the DJF 2011 ERA5 native-model-level sigma=0.85 pilot."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from scipy.stats import pearsonr, spearmanr


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw"
OUT = ROOT / "data/processed"
MANIFEST = ROOT / "data/manifests/era5_model_level_sigma_pilot_manifest.json"
MODEL_FILE = RAW / "era5_model_level_sigma_pilot/era5_model_level_sigma_pilot_DJF2011.nc"
POINTS_OUT = OUT / "native_sigma085_pilot_points.csv.gz"
EVENTS_OUT = OUT / "native_sigma085_pilot_events.csv"
SUMMARY_OUT = OUT / "native_sigma085_pilot_summary.json"
DOC = ROOT / "docs/native_sigma085_pilot_zh.md"
SIGMA = 0.85
EARTH_R = 6_371_000.0
CORE_RADIUS_DEG = 1.5
MIN_CORE_VALID = 0.80


def canonical(ds: xr.Dataset) -> xr.Dataset:
    rename = {}
    for old, new in (
        ("valid_time", "time"),
        ("latitude", "lat"),
        ("longitude", "lon"),
        ("hybrid", "level"),
        ("model_level", "level"),
    ):
        if (old in ds.coords or old in ds.dims) and new not in ds.coords and new not in ds.dims:
            rename[old] = new
    ds = ds.rename(rename)
    return ds.sortby("lat").sortby("lon")


def selected_points() -> pd.DataFrame:
    events = pd.read_csv(OUT / "events_terrain_qc_intensity.csv.gz", parse_dates=["genesis_time"])
    events["season_year"] = events.year + events.genesis_time.dt.month.eq(12).astype(int)
    ids = set(
        events.loc[
            events.genesis_time.dt.month.isin([12, 1, 2]) & events.season_year.eq(2011),
            "event_id",
        ]
    )
    points = pd.read_csv(OUT / "track_points.csv.gz", parse_dates=["time"])
    points = points.loc[points.event_id.isin(ids) & points.time.dt.hour.isin([0, 6, 12, 18])].copy()
    return points.sort_values(["time", "event_id"])


def relative_vorticity(u: xr.DataArray, v: xr.DataArray) -> xr.DataArray:
    dy_per_degree = EARTH_R * np.deg2rad(1.0)
    dx_per_degree = EARTH_R * np.cos(np.deg2rad(u.lat)) * np.deg2rad(1.0)
    return v.differentiate("lon") / dx_per_degree - u.differentiate("lat") / dy_per_degree


def full_pressure_pa(sp: xr.DataArray, level: int, coefficients: dict[int, dict[str, float]]) -> xr.DataArray:
    upper = coefficients[level - 1]
    lower = coefficients[level]
    return 0.5 * (
        upper["a_pa"]
        + lower["a_pa"]
        + (upper["b"] + lower["b"]) * sp
    )


def interpolate_sigma(
    wind: xr.Dataset, sp: xr.DataArray, coefficients: dict[int, dict[str, float]]
) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray, xr.DataArray]:
    levels = [int(value) for value in wind.level.values]
    pressures = xr.concat(
        [full_pressure_pa(sp, level, coefficients) for level in levels],
        dim=pd.Index(levels, name="level"),
    )
    target = SIGMA * sp
    u_sigma = xr.full_like(sp, np.nan, dtype=float)
    v_sigma = xr.full_like(sp, np.nan, dtype=float)
    bracket = xr.full_like(sp, np.nan, dtype=float)
    for low_level, high_level in zip(levels[:-1], levels[1:]):
        p0 = pressures.sel(level=low_level)
        p1 = pressures.sel(level=high_level)
        valid = (target >= p0) & (target <= p1)
        alpha = (target - p0) / (p1 - p0)
        u_here = wind.u.sel(level=low_level) + alpha * (
            wind.u.sel(level=high_level) - wind.u.sel(level=low_level)
        )
        v_here = wind.v.sel(level=low_level) + alpha * (
            wind.v.sel(level=high_level) - wind.v.sel(level=low_level)
        )
        u_sigma = xr.where(valid, u_here, u_sigma)
        v_sigma = xr.where(valid, v_here, v_sigma)
        bracket = xr.where(valid, float(low_level), bracket)
    return u_sigma, v_sigma, target, bracket


def point_metrics(
    model: xr.Dataset,
    surface_by_year: dict[int, xr.Dataset],
    points: pd.DataFrame,
    coefficients: dict[int, dict[str, float]],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    model_times = set(pd.to_datetime(model.time.values))
    for row in points.itertuples():
        timestamp = pd.Timestamp(row.time).tz_localize(None)
        if timestamp not in model_times:
            raise KeyError(f"Native model-level time absent: {timestamp}")
        bounds = {"lat": slice(row.lat - 2.0, row.lat + 2.0), "lon": slice(row.lon - 2.0, row.lon + 2.0)}
        wind = model[["u", "v"]].sel(time=timestamp).sel(**bounds)
        surface = surface_by_year[timestamp.year]
        if timestamp not in set(pd.to_datetime(surface.time.values)):
            raise KeyError(f"Surface-pressure time absent: {timestamp}")
        sp = surface.sp.sel(time=timestamp).sel(lat=wind.lat, lon=wind.lon)
        u_sigma, v_sigma, target, bracket = interpolate_sigma(wind, sp, coefficients)
        zeta = relative_vorticity(u_sigma, v_sigma)
        radius = np.sqrt((zeta.lat - row.lat) ** 2 + ((zeta.lon - row.lon) * np.cos(np.deg2rad(row.lat))) ** 2)
        core = radius <= CORE_RADIUS_DEG
        core_count = int(core.sum())
        bracket_fraction = float(bracket.notnull().where(core).sum() / core_count)
        vort_fraction = float(zeta.notnull().where(core).sum() / core_count)
        center_target = float(target.sel(lat=row.lat, lon=row.lon, method="nearest") / 100.0)
        center_bracket = bracket.sel(lat=row.lat, lon=row.lon, method="nearest")
        rows.append(
            {
                "event_id": row.event_id,
                "time": timestamp,
                "lat": row.lat,
                "lon": row.lon,
                "target_pressure_hpa_native": center_target,
                "center_bracket_lower_level": float(center_bracket) if np.isfinite(center_bracket) else np.nan,
                "core_bracket_fraction_native": bracket_fraction,
                "core_valid_fraction_native": vort_fraction,
                "max_vorticity_s_1_native": float(zeta.where(core).max(skipna=True)) if vort_fraction >= MIN_CORE_VALID else np.nan,
                "mean_vorticity_s_1_native": float(zeta.where(core).mean(skipna=True)) if vort_fraction >= MIN_CORE_VALID else np.nan,
            }
        )
    return pd.DataFrame(rows)


def correlation(frame: pd.DataFrame, x: str, y: str) -> dict[str, float | int]:
    common = frame.dropna(subset=[x, y])
    result: dict[str, float | int] = {"n": int(len(common))}
    if len(common) < 3 or common[x].nunique() < 2 or common[y].nunique() < 2:
        result.update({"pearson_r": np.nan, "pearson_p": np.nan, "spearman_rho": np.nan, "spearman_p": np.nan})
        return result
    pearson = pearsonr(common[x], common[y])
    spearman = spearmanr(common[x], common[y])
    result.update(
        {
            "pearson_r": float(pearson.statistic),
            "pearson_p": float(pearson.pvalue),
            "spearman_rho": float(spearman.statistic),
            "spearman_p": float(spearman.pvalue),
        }
    )
    return result


def main() -> None:
    if not MODEL_FILE.exists():
        raise FileNotFoundError(f"Run HPilotDownload first: {MODEL_FILE}")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    coefficients = {int(key): value for key, value in manifest["half_level_coefficients"].items()}
    points = selected_points()
    with xr.open_dataset(MODEL_FILE) as raw_model:
        model = canonical(raw_model).load()
    surface_by_year: dict[int, xr.Dataset] = {}
    try:
        for year in sorted(points.time.dt.year.unique()):
            path = RAW / f"era5_surface_pressure_6hourly/era5_surface_pressure_{year}.nc"
            surface_by_year[int(year)] = canonical(xr.open_dataset(path)).load()
        native = point_metrics(model, surface_by_year, points, coefficients)
    finally:
        for ds in surface_by_year.values():
            ds.close()
        model.close()

    approx = pd.read_csv(OUT / "sigma085_intensity_points.csv.gz", parse_dates=["time"])[
        ["event_id", "time", "sigma085_max_vorticity_s_1", "sigma085_mean_vorticity_s_1"]
    ]
    fixed = pd.read_csv(OUT / "terrain_qc_intensity_points.csv.gz", parse_dates=["time"])[
        ["event_id", "time", "max_vorticity_s_1_700_c25"]
    ]
    for frame in (approx, fixed):
        frame["time"] = frame.time.dt.tz_localize(None) if frame.time.dt.tz is not None else frame.time
    native = native.merge(approx, on=["event_id", "time"], how="left", validate="one_to_one")
    native = native.merge(fixed, on=["event_id", "time"], how="left", validate="one_to_one")
    native.to_csv(POINTS_OUT, index=False)

    metadata = pd.read_csv(OUT / "events_terrain_qc_intensity.csv.gz", usecols=["event_id", "track_cluster"])
    events = native.groupby("event_id").agg(
        sampled_points=("time", "size"),
        valid_point_fraction_native=("max_vorticity_s_1_native", lambda x: x.notna().mean()),
        max_vorticity_s_1_native=("max_vorticity_s_1_native", "max"),
        max_vorticity_s_1_approx=("sigma085_max_vorticity_s_1", "max"),
        max_vorticity_s_1_700_c25=("max_vorticity_s_1_700_c25", "max"),
        median_target_pressure_hpa=("target_pressure_hpa_native", "median"),
    ).reset_index().merge(metadata, on="event_id", how="left", validate="one_to_one")
    events.to_csv(EVENTS_OUT, index=False)

    point_native_approx = correlation(native, "max_vorticity_s_1_native", "sigma085_max_vorticity_s_1")
    event_native_approx = correlation(events, "max_vorticity_s_1_native", "max_vorticity_s_1_approx")
    event_native_700 = correlation(events, "max_vorticity_s_1_native", "max_vorticity_s_1_700_c25")
    event_valid_fraction = float(events.max_vorticity_s_1_native.notna().mean())
    point_valid_fraction = float(native.max_vorticity_s_1_native.notna().mean())
    bracket_fraction = float(native.core_bracket_fraction_native.mean())
    exact_times = len(native) == len(points)
    gate_checks = {
        "all_requested_times_exactly_matched": bool(exact_times),
        "mean_core_bracket_fraction_ge_0.99": bool(bracket_fraction >= 0.99),
        "point_metric_valid_fraction_ge_0.95": bool(point_valid_fraction >= 0.95),
        "event_metric_valid_fraction_ge_0.95": bool(event_valid_fraction >= 0.95),
        "native_approx_event_spearman_ge_0.50": bool(event_native_approx["spearman_rho"] >= 0.50),
    }
    expand = all(gate_checks.values())
    summary = {
        "scope": "DJF 2011 native ERA5 model-level pilot; not a long-term trend estimate",
        "events": int(len(events)),
        "track_points": int(len(native)),
        "native_valid_events": int(events.max_vorticity_s_1_native.notna().sum()),
        "event_valid_fraction": event_valid_fraction,
        "point_valid_fraction": point_valid_fraction,
        "mean_core_bracket_fraction": bracket_fraction,
        "target_pressure_hpa_range": [float(native.target_pressure_hpa_native.min()), float(native.target_pressure_hpa_native.max())],
        "center_bracket_counts": {
            str(int(key)): int(value)
            for key, value in native.center_bracket_lower_level.value_counts(dropna=True).sort_index().items()
        },
        "native_event_q90_1e5_s_1": float(events.max_vorticity_s_1_native.quantile(0.9) * 1e5),
        "approx_event_q90_1e5_s_1": float(events.max_vorticity_s_1_approx.quantile(0.9) * 1e5),
        "fixed700_event_q90_1e5_s_1": float(events.max_vorticity_s_1_700_c25.quantile(0.9) * 1e5),
        "correlations": {
            "point_native_vs_approx": point_native_approx,
            "event_native_vs_approx": event_native_approx,
            "event_native_vs_700_c25": event_native_700,
        },
        "gate_checks": gate_checks,
        "expand_to_1980_2023": expand,
        "coefficient_source": manifest["coefficient_source"],
        "request_documentation": manifest["request_documentation"],
    }
    SUMMARY_OUT.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    verdict = "通过，可进入 1980--2023 年分批扩展" if expand else "未通过，须先修正层范围或插值/QC"
    DOC.write_text(
        f"""# ERA5 原生模式层 sigma=0.85 一冬试验

## 设计

以事件数等于长期中位数的 DJF 2011 为代表冬季。对 19 个事件的 46 个六小时轨迹点，使用 ERA5 原生模式层 111--115 的风，依据 L137 半层系数和同一时次地面气压计算全层压力；逐格线性插值到 $p=0.85p_s$ 后，再计算相对涡度。该试验只验证坐标方法和数据工程，不估计年代际趋势。

## 质量控制

- 46/46 个请求轨迹时次精确匹配；目标气压范围为 {summary['target_pressure_hpa_range'][0]:.1f}--{summary['target_pressure_hpa_range'][1]:.1f} hPa。
- 涡心 1.5° 圆域平均夹层覆盖率为 {100*bracket_fraction:.1f}%；点指标有效率为 {100*point_valid_fraction:.1f}%，事件指标有效率为 {100*event_valid_fraction:.1f}%（{summary['native_valid_events']}/{len(events)}）。
- 原生指标与双压力层近似的事件级 Spearman $\\rho={event_native_approx['spearman_rho']:.3f}$（$n={event_native_approx['n']}$）；与地形质控 700 hPa 指标为 $\\rho={event_native_700['spearman_rho']:.3f}$（$n={event_native_700['n']}$）。
- 原生、双压力层近似和 700 hPa 的事件最大涡度 90 分位分别为 {summary['native_event_q90_1e5_s_1']:.2f}、{summary['approx_event_q90_1e5_s_1']:.2f} 和 {summary['fixed700_event_q90_1e5_s_1']:.2f} $\\times10^{{-5}}$ s$^{{-1}}$。

## 门槛结论

**{verdict}。** 这里的通过仅表示模式层范围、时间匹配、核心区覆盖和事件排序足以支持长期扩展；不能据此宣称原生 sigma 强度存在或不存在长期趋势。45 年扩展后仍须重新计算年度 DJF 90 分位趋势、HAC 不确定性、路径分型和共同样本敏感性。

系数定义：{manifest['coefficient_source']}

ERA5 Complete 请求说明：{manifest['request_documentation']}
""",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
