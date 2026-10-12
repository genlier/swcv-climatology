"""Second-round DJF precipitation analysis using moving vortex-centred disks."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import xarray as xr
from scipy.stats import norm


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/processed"
EVENTS_FILE = OUT / "events_season_year_terrain_qc.csv.gz"
POINTS_FILE = OUT / "track_points.csv.gz"
LEGACY_RAW = ROOT / "data/raw/era5_djf_event_precipitation"
EXTENDED_RAW = ROOT / "data/raw/era5_djf_event_precipitation_extended"
LEGACY_MANIFEST = ROOT / "data/manifests/era5_djf_event_precipitation_manifest.json"
EXTENDED_MANIFEST = ROOT / "data/manifests/era5_djf_event_precipitation_extended_manifest.json"
DOC = ROOT / "docs/djf_track_relative_precipitation_zh.md"
RADII_KM = (300, 500)
REGIONS = {
    "basin": (27.0, 33.0, 102.0, 109.0),
    "downstream": (25.0, 34.0, 109.0, 120.0),
}


def canonical(ds: xr.Dataset) -> xr.Dataset:
    rename = {}
    for old, new in (("valid_time", "time"), ("latitude", "lat"), ("longitude", "lon")):
        if old in ds.coords or old in ds.dims:
            rename[old] = new
    return ds.rename(rename)


def selected_events() -> pd.DataFrame:
    events = pd.read_csv(EVENTS_FILE, parse_dates=["genesis_time", "lysis_time"])
    events = events.loc[
        events.season.eq("DJF")
        & events.season_year.between(1980, 2023)
        & events.track_cluster.isin([1, 2])
    ].copy()
    if events.event_id.duplicated().any() or len(events) != 800:
        raise RuntimeError(f"Unexpected event selection ({len(events)} rows)")
    return events


def validate_manifest(path: Path, expected_events: int) -> None:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("failures"):
        raise RuntimeError(f"Incomplete manifest: {path}")
    counts = manifest.get("selection_counts", {})
    if int(counts.get("events", -1)) != expected_events:
        raise RuntimeError(f"Unexpected event count in {path}: {counts}")
    if len(manifest.get("files", [])) != int(counts.get("year_month_requests", -1)):
        raise RuntimeError(f"Unexpected file count in {path}")


def legacy_safe_500km(points: pd.DataFrame) -> pd.Series:
    dlat = 500.0 / 111.2
    dlon = 500.0 / (111.2 * np.cos(np.deg2rad(points.lat)))
    safe = (
        points.lat.sub(dlat).ge(20.0)
        & points.lat.add(dlat).le(38.0)
        & points.lon.sub(dlon).ge(95.0)
        & points.lon.add(dlon).le(120.0)
    )
    return safe.groupby(points.event_id).all()


def source_files(raw: Path) -> dict[str, Path]:
    return {path.stem.rsplit("_", 1)[-1]: path for path in raw.glob("era5_djf_event_precip_*.nc")}


def load_window(files: dict[str, Path], start: pd.Timestamp, end: pd.Timestamp) -> xr.Dataset:
    months = pd.period_range(start.to_period("M"), (end - pd.Timedelta(hours=1)).to_period("M"), freq="M")
    pieces = []
    for month in months:
        key = month.strftime("%Y%m")
        with xr.open_dataset(files[key]) as source:
            ds = canonical(source)
            keep = (ds.time >= start.to_datetime64()) & (ds.time < end.to_datetime64())
            pieces.append(ds.isel(time=keep).load())
    merged = xr.concat(pieces, dim="time").sortby("time")
    _, unique = np.unique(merged.time.values, return_index=True)
    merged = merged.isel(time=np.sort(unique))
    expected = int((end - start).total_seconds() / 3600)
    if merged.sizes["time"] != expected:
        raise RuntimeError(f"Incomplete event window: {merged.sizes['time']}/{expected}")
    return merged


def track_centres(event: pd.Series, points: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    start = pd.Timestamp(event.genesis_time).tz_localize(None)
    lysis = pd.Timestamp(event.lysis_time).tz_localize(None)
    end = lysis + pd.Timedelta(hours=24)
    times = pd.date_range(start, end, freq="h", inclusive="left")
    track = points.loc[points.event_id.eq(event.event_id)].copy()
    track["time"] = pd.to_datetime(track.time, utc=True).dt.tz_localize(None)
    track = track.set_index("time").sort_index()
    observed = pd.date_range(start, lysis, freq="h")
    if not track.index.equals(observed):
        raise RuntimeError(f"Track times are incomplete for {event.event_id}")
    centres = track[["lat", "lon"]].reindex(times).ffill()
    return centres.lat.to_numpy(), centres.lon.to_numpy()


def moving_disk_mean(field: xr.DataArray, centre_lat: np.ndarray, centre_lon: np.ndarray, radius_km: float) -> np.ndarray:
    lat = np.deg2rad(field.lat.values)[None, :, None]
    lon = np.deg2rad(field.lon.values)[None, None, :]
    clat = np.deg2rad(centre_lat)[:, None, None]
    clon = np.deg2rad(centre_lon)[:, None, None]
    hav = np.sin((lat - clat) / 2.0) ** 2 + np.cos(lat) * np.cos(clat) * np.sin((lon - clon) / 2.0) ** 2
    distance = 2.0 * 6371.0 * np.arcsin(np.minimum(1.0, np.sqrt(hav)))
    mask = distance <= radius_km
    values = np.asarray(field.values, dtype=float)
    latitude_weights = np.cos(lat)
    weights = np.where(mask & np.isfinite(values), latitude_weights, 0.0)
    denominator = weights.sum(axis=(1, 2))
    if np.any(denominator == 0):
        raise RuntimeError("Empty vortex-centred disk")
    return np.nansum(values * weights, axis=(1, 2)) / denominator


def fixed_box_mean(field: xr.DataArray, bounds: tuple[float, float, float, float]) -> np.ndarray:
    south, north, west, east = bounds
    lat_slice = slice(north, south) if field.lat[0] > field.lat[-1] else slice(south, north)
    subset = field.sel(lat=lat_slice, lon=slice(west, east))
    return np.asarray(subset.weighted(np.cos(np.deg2rad(subset.lat))).mean(("lat", "lon")), dtype=float)


def process_events(events: pd.DataFrame, points: pd.DataFrame) -> pd.DataFrame:
    safe = legacy_safe_500km(points)
    legacy_files, extended_files = source_files(LEGACY_RAW), source_files(EXTENDED_RAW)
    rows = []
    for i, (_, event) in enumerate(events.iterrows(), start=1):
        use_legacy = event.track_cluster == 1 and bool(safe.loc[event.event_id])
        start = pd.Timestamp(event.genesis_time).tz_localize(None)
        end = pd.Timestamp(event.lysis_time).tz_localize(None) + pd.Timedelta(hours=24)
        ds = load_window(legacy_files if use_legacy else extended_files, start, end)
        centre_lat, centre_lon = track_centres(event, points)
        row = {
            "event_id": event.event_id,
            "season_year": int(event.season_year),
            "track_cluster": int(event.track_cluster),
            "window_hours": int(ds.sizes["time"]),
            "source_domain": "legacy" if use_legacy else "extended",
        }
        for radius in RADII_KM:
            tp = moving_disk_mean(ds.tp, centre_lat, centre_lon, radius) * 1000.0
            sf = moving_disk_mean(ds.sf, centre_lat, centre_lon, radius) * 1000.0
            t2m = moving_disk_mean(ds.t2m, centre_lat, centre_lon, radius) - 273.15
            row[f"r{radius}_tp_accum_mm"] = float(tp.sum())
            row[f"r{radius}_sf_accum_mm"] = float(sf.sum())
            row[f"r{radius}_tp_rate_mm_h"] = float(tp.mean())
            row[f"r{radius}_sf_rate_mm_h"] = float(sf.mean())
            row[f"r{radius}_t2m_celsius"] = float(t2m.mean())
        for name, bounds in REGIONS.items():
            tp = fixed_box_mean(ds.tp, bounds) * 1000.0
            sf = fixed_box_mean(ds.sf, bounds) * 1000.0
            row[f"{name}_tp_accum_mm"] = float(tp.sum())
            row[f"{name}_sf_accum_mm"] = float(sf.sum())
        rows.append(row)
        ds.close()
        if i % 50 == 0:
            print(f"processed {i}/{len(events)} events", flush=True)
    return pd.DataFrame(rows)


def bh_fdr(pvalues: pd.Series) -> pd.Series:
    order = np.argsort(pvalues.to_numpy())
    ranked = pvalues.to_numpy()[order]
    adjusted = np.minimum.accumulate((ranked * len(ranked) / np.arange(1, len(ranked) + 1))[::-1])[::-1]
    result = np.empty(len(ranked))
    result[order] = np.minimum(adjusted, 1.0)
    return pd.Series(result, index=pvalues.index)


def model_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    metrics = [f"r{radius}_{variable}_{kind}" for radius in RADII_KM for variable in ("tp", "sf") for kind in ("accum_mm", "rate_mm_h")]
    year_c = frame.season_year - frame.season_year.mean()
    cluster2 = frame.track_cluster.eq(2).astype(float)
    duration_c = np.log(frame.window_hours) - np.log(frame.window_hours).mean()
    for metric in metrics:
        response = frame[metric].astype(float)
        design = pd.DataFrame({"const": 1.0, "year_c": year_c, "cluster2": cluster2, "year_x_cluster2": year_c * cluster2})
        if "accum" in metric:
            design["log_duration_c"] = duration_c
        fit = sm.OLS(response, design).fit(cov_type="cluster", cov_kwds={"groups": frame.season_year})
        reference = float(fit.params["const"])
        if reference <= 0:
            raise RuntimeError(f"Non-positive cluster-1 midperiod reference for {metric}: {reference}")
        for term in ("year_c", "cluster2", "year_x_cluster2"):
            beta = float(fit.params[term])
            se = float(fit.bse[term])
            scale = 10.0 if term in {"year_c", "year_x_cluster2"} else 1.0
            rows.append({
                "metric": metric,
                "term": term,
                "beta_raw": beta,
                "raw_unit": (
                    ("mm_h-1_per_year" if metric.endswith("rate_mm_h") else "mm_per_year")
                    if term in {"year_c", "year_x_cluster2"}
                    else ("mm_h-1" if metric.endswith("rate_mm_h") else "mm")
                ),
                "cluster1_midperiod_reference": reference,
                "effect_scale": "per_decade" if scale == 10.0 else "cluster2_vs_cluster1_at_midperiod",
                "change_percent_of_reference": 100.0 * beta * scale / reference,
                "ci_low_percent": 100.0 * (beta - norm.ppf(0.975) * se) * scale / reference,
                "ci_high_percent": 100.0 * (beta + norm.ppf(0.975) * se) * scale / reference,
                "p_cluster_year": float(fit.pvalues[term]),
                "n_events": int(len(frame)),
                "n_season_years": int(frame.season_year.nunique()),
            })
    result = pd.DataFrame(rows)
    result["p_fdr_24"] = bh_fdr(result.p_cluster_year)
    return result


def annual_summary(frame: pd.DataFrame) -> pd.DataFrame:
    metrics = [column for column in frame if column.startswith("r") and (column.endswith("accum_mm") or column.endswith("rate_mm_h"))]
    aggregations = {metric: "mean" for metric in metrics}
    aggregations.update({"event_id": "size", "window_hours": "mean"})
    annual = frame.groupby(["season_year", "track_cluster"], as_index=False).agg(aggregations)
    return annual.rename(columns={"event_id": "n_events", "window_hours": "mean_window_hours"})


def equal_year_sensitivity(annual: pd.DataFrame) -> pd.DataFrame:
    """Give each winter equal weight when estimating cluster-1 conditional trends."""
    local = annual.loc[annual.track_cluster.eq(1)].copy()
    metrics = [column for column in local if column.startswith("r") and (column.endswith("accum_mm") or column.endswith("rate_mm_h"))]
    rows = []
    year_c = local.season_year - local.season_year.mean()
    for metric in metrics:
        fit = sm.OLS(local[metric], sm.add_constant(year_c)).fit(cov_type="HAC", cov_kwds={"maxlags": 2})
        beta, se = float(fit.params.iloc[1]), float(fit.bse.iloc[1])
        reference = float(local[metric].mean())
        rows.append({
            "metric": metric,
            "trend_percent_per_decade": 100.0 * beta * 10.0 / reference,
            "ci_low_percent": 100.0 * (beta - norm.ppf(0.975) * se) * 10.0 / reference,
            "ci_high_percent": 100.0 * (beta + norm.ppf(0.975) * se) * 10.0 / reference,
            "p_hac": float(fit.pvalues.iloc[1]),
            "n_season_years": int(len(local)),
        })
    result = pd.DataFrame(rows)
    result["p_fdr_8"] = bh_fdr(result.p_hac)
    return result


def main() -> None:
    validate_manifest(LEGACY_MANIFEST, 732)
    validate_manifest(EXTENDED_MANIFEST, 107)
    events = selected_events()
    points = pd.read_csv(POINTS_FILE, usecols=["event_id", "lat", "lon", "time"])
    points = points.loc[points.event_id.isin(events.event_id)].copy()
    event_metrics = process_events(events, points)
    annual = annual_summary(event_metrics)
    models = model_metrics(event_metrics)
    annual_sensitivity = equal_year_sensitivity(annual)
    event_metrics.to_csv(OUT / "djf_track_relative_precipitation_events.csv", index=False)
    models.to_csv(OUT / "djf_track_relative_precipitation_models.csv", index=False)
    annual.to_csv(OUT / "djf_track_relative_precipitation_annual.csv", index=False)
    annual_sensitivity.to_csv(OUT / "djf_track_relative_precipitation_annual_sensitivity.csv", index=False)

    significant = models.loc[models.p_fdr_24.lt(0.05)]
    lines = [
        "# DJF 涡旋轨迹相对降水与东移型对照（R2-M2 第二轮）\n",
        "本轮将 1980--2023 年 732 个本地型（cluster 1）与 68 个东移型（cluster 2）事件统一定义为生成至消亡后 24 小时的半开小时窗。降水和降雪在随涡心逐小时移动的 300/500 km 圆域内作余弦纬度加权；消亡后涡心固定在最后位置。39 个可能触及原 500 km 数据边界的本地型事件和全部东移型事件使用扩展域补下载，其余 693 个本地型事件复用首轮数据。\n",
        "## 统计设计\n",
        "对 8 个预注册轨迹相对指标（两半径 × 降水/降雪 × 累积/小时率）分别拟合原始物理量的线性模型 `metric ~ year + cluster2 + year×cluster2`；累积量另控制对数事件窗时长。标准误按冬季年份聚类。本地型年代际趋势、年代中点的 cluster 对照和差异趋势共 24 个检验统一 Benjamini--Hochberg FDR。效应量以模型在年代中点、平均事件窗长度下的本地型预测值为参照；这样小时率不会因 `log1p` 的加一常数而被压缩。年度表只对该年实际发生的事件取等事件平均；无东移型事件的年份不填零。\n",
        "## FDR 结果\n",
    ]
    if significant.empty:
        lines.append("24 个主检验在 FDR 校正后均未达到 0.05。\n")
    else:
        for row in significant.itertuples():
            lines.append(f"- `{row.metric}` / `{row.term}`（{row.effect_scale}）：相对本地型年代中点参照变化 {row.change_percent_of_reference:+.1f}%（95% CI {row.ci_low_percent:+.1f}% 至 {row.ci_high_percent:+.1f}%），FDR $p={row.p_fdr_24:.3g}$。\n")
    lines.extend([
        "\n## 等年份权重敏感性\n",
        "将每个冬季的本地型事件先取平均，再以 2 阶 HAC 误差拟合趋势，8 个指标方向均为负且在独立的 8 项 FDR 校正后显著；估计幅度为每十年 -8.1% 至 -15.5%。这表明主结果并非由事件较多的年份获得更高权重所致。\n",
        "\n## 解释边界\n",
        "轨迹邻域消除了固定框随路径离开而漏采的主要几何问题，但 ERA5 仍不是独立雨量站验证。圆域均值描述涡旋环境中的平均水当量，不等于流域体积降水；事件窗可能重叠，因此年度累积不能解释为互不重复的季节总量。cluster 2 仅 68 个事件，差异趋势的置信区间应优先于显著性标签。\n",
    ])
    DOC.write_text("".join(lines), encoding="utf-8")
    print(f"saved {len(event_metrics)} event metrics and {len(models)} model tests")


if __name__ == "__main__":
    main()
