"""Analyze vortex-associated precipitation and snowfall for DJF events (R2-M2).

Separates a frequency-sensitive event-window burden index from per-event and
duration-normalised metrics in the Sichuan Basin and a fixed downstream box.
"""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "era5_djf_event_precipitation"
MANIFEST = ROOT / "data" / "manifests" / "era5_djf_event_precipitation_manifest.json"
OUT = ROOT / "data" / "processed"
DOC = ROOT / "docs" / "djf_event_precipitation_zh.md"
EVENTS_FILE = OUT / "events_season_year_terrain_qc.csv.gz"
YEARS = np.arange(1980, 2024)

REGIONS = {
    "basin": {"south": 27.0, "north": 33.0, "west": 102.0, "east": 109.0},
    "downstream": {"south": 25.0, "north": 34.0, "west": 109.0, "east": 120.0},
}
BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_BLOCK_YEARS = 5


def load_djf_events() -> pd.DataFrame:
    events = pd.read_csv(EVENTS_FILE, parse_dates=["genesis_time", "lysis_time"])
    selected = events.loc[
        events.season.eq("DJF")
        & events.season_year.between(1980, 2023)
        & events.track_cluster.eq(1)
    ].copy()
    if selected.event_id.duplicated().any():
        raise RuntimeError("Duplicate event_id values in DJF cluster-1 selection")
    return selected


def validate_download_manifest(expected_events: int) -> None:
    """Refuse analysis unless the downloader recorded a complete successful run."""
    if not MANIFEST.exists():
        raise FileNotFoundError(f"Completed download manifest not found: {MANIFEST}")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    selection = manifest.get("selection_counts", {})
    files = manifest.get("files", [])
    failures = manifest.get("failures", [])
    expected_requests = int(selection.get("year_month_requests", -1))
    if int(selection.get("events", -1)) != expected_events:
        raise RuntimeError("Download manifest event count does not match the current event selection")
    if failures or expected_requests < 1 or len(files) != expected_requests:
        raise RuntimeError(
            f"Incomplete download manifest: {len(files)}/{expected_requests} files, "
            f"{len(failures)} failures"
        )
    missing = [row.get("path", "") for row in files if not (ROOT / row.get("path", "")).is_file()]
    if missing:
        raise RuntimeError(f"Manifest references {len(missing)} missing files")


def canonical(ds: xr.Dataset) -> xr.Dataset:
    rename = {}
    for old, new in (("valid_time", "time"), ("latitude", "lat"), ("longitude", "lon")):
        if old in ds.coords or old in ds.dims:
            rename[old] = new
    return ds.rename(rename)


def region_mean(ds: xr.Dataset, region: dict[str, float]) -> xr.Dataset:
    lat_slice = slice(region["north"], region["south"]) if float(ds.lat[0]) > float(ds.lat[-1]) else slice(region["south"], region["north"])
    selected = ds.sel(lat=lat_slice, lon=slice(region["west"], region["east"]))
    if selected.sizes.get("lat", 0) == 0 or selected.sizes.get("lon", 0) == 0:
        raise RuntimeError(f"Empty spatial selection for {region}")
    return selected.weighted(np.cos(np.deg2rad(selected.lat))).mean(("lat", "lon"))


def moving_block_trend_ci(
    values: np.ndarray, year_c: np.ndarray, rng: np.random.Generator
) -> tuple[float, float]:
    """Residual moving-block bootstrap CI for the linear decadal change."""
    design = np.column_stack([np.ones(len(year_c)), year_c])
    coefficients = np.linalg.lstsq(design, values, rcond=None)[0]
    fitted = design @ coefficients
    residuals = values - fitted
    starts = np.arange(len(values) - BOOTSTRAP_BLOCK_YEARS + 1)
    blocks_needed = int(np.ceil(len(values) / BOOTSTRAP_BLOCK_YEARS))
    slopes = np.empty(BOOTSTRAP_REPLICATES)
    denominator = float(np.sum(year_c**2))
    for i in range(BOOTSTRAP_REPLICATES):
        sampled_starts = rng.choice(starts, size=blocks_needed, replace=True)
        sampled = np.concatenate(
            [residuals[start : start + BOOTSTRAP_BLOCK_YEARS] for start in sampled_starts]
        )[: len(values)]
        response = fitted + sampled
        slopes[i] = float(np.sum(year_c * (response - response.mean())) / denominator) * 10.0
    return tuple(np.quantile(slopes, [0.025, 0.975]))


def process_event_precipitation() -> pd.DataFrame:
    events = load_djf_events()
    nc_files = list(RAW.glob("era5_djf_event_precip_*.nc"))
    if not nc_files:
        raise FileNotFoundError(f"No precipitation netCDF files found in {RAW}")

    # Build an index of available datasets by year-month
    datasets = {}
    for p in nc_files:
        stem = p.stem  # era5_djf_event_precip_YYYYMM
        ym = stem.replace("era5_djf_event_precip_", "")
        datasets[ym] = p

    event_records = []
    for ev in events.itertuples():
        t_start = pd.Timestamp(ev.genesis_time).tz_localize(None)
        # Half-open interval: duration equals the number of expected hourly samples.
        t_end_exclusive = pd.Timestamp(ev.lysis_time).tz_localize(None) + pd.Timedelta(hours=24)
        
        # Determine year-months needed
        months_needed = pd.period_range(
            t_start.to_period("M"),
            (t_end_exclusive - pd.Timedelta(hours=1)).to_period("M"),
            freq="M",
        ).strftime("%Y%m")
        
        slices = []
        for ym in months_needed:
            if ym in datasets:
                with xr.open_dataset(datasets[ym]) as source:
                    ds = canonical(source)
                    times = pd.DatetimeIndex(ds.time.values)
                    mask = (times >= t_start) & (times < t_end_exclusive)
                    if mask.any():
                        slices.append(ds.isel(time=mask).load())

        if not slices:
            continue

        merged_ds = xr.concat(slices, dim="time").sortby("time")
        _, unique_indices = np.unique(merged_ds.time.values, return_index=True)
        merged_ds = merged_ds.isel(time=np.sort(unique_indices))
        window_hours = int((t_end_exclusive - t_start).total_seconds() / 3600.0)
        if merged_ds.sizes["time"] != window_hours:
            raise RuntimeError(
                f"Incomplete event window for {ev.event_id}: expected {window_hours}, "
                f"found {merged_ds.sizes['time']} hours"
            )

        record = {
            "event_id": ev.event_id,
            "season_year": int(ev.season_year),
            "precipitation_window_hours": window_hours,
            "downloaded_hours": int(merged_ds.sizes["time"]),
        }
        for name, region in REGIONS.items():
            spatial = region_mean(merged_ds, region)
            tp = float(spatial["tp"].sum("time") * 1000.0)
            sf = float(spatial["sf"].sum("time") * 1000.0)
            record[f"{name}_tp_accum_mm"] = tp
            record[f"{name}_sf_accum_mm"] = sf
            record[f"{name}_tp_rate_mm_h"] = tp / window_hours
            record[f"{name}_sf_rate_mm_h"] = sf / window_hours
            record[f"{name}_snow_fraction"] = sf / tp if tp > 0 else np.nan
            record[f"{name}_t2m_celsius"] = float(spatial["t2m"].mean("time") - 273.15)
        event_records.append(record)

    return pd.DataFrame(event_records)


def analyze_annual_precip(event_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    annual_records = []
    for yr in YEARS:
        sub = event_df[event_df.season_year == yr]
        n_events = len(sub)
        if n_events == 0:
            annual_records.append({
                "season_year": int(yr),
                "event_count": 0,
                "summed_event_window_basin_tp_mm": 0.0,
                "summed_event_window_basin_sf_mm": 0.0,
                "summed_event_window_downstream_tp_mm": 0.0,
                "summed_event_window_downstream_sf_mm": 0.0,
                "mean_basin_tp_per_event_mm": 0.0,
                "mean_basin_sf_per_event_mm": 0.0,
                "p90_basin_tp_per_event_mm": 0.0,
                "p90_basin_sf_per_event_mm": 0.0,
                "mean_basin_tp_rate_mm_h": 0.0,
                "mean_basin_sf_rate_mm_h": 0.0,
                "mean_downstream_tp_per_event_mm": 0.0,
                "mean_downstream_sf_per_event_mm": 0.0,
                "p90_downstream_tp_per_event_mm": 0.0,
                "p90_downstream_sf_per_event_mm": 0.0,
                "mean_downstream_tp_rate_mm_h": 0.0,
                "mean_downstream_sf_rate_mm_h": 0.0,
                "mean_precipitation_window_h": 0.0,
            })
        else:
            annual_records.append({
                "season_year": int(yr),
                "event_count": n_events,
                # Overlapping event windows are intentionally not deduplicated.
                # These sums are burden indices, not unique seasonal totals.
                "summed_event_window_basin_tp_mm": float(sub.basin_tp_accum_mm.sum()),
                "summed_event_window_basin_sf_mm": float(sub.basin_sf_accum_mm.sum()),
                "summed_event_window_downstream_tp_mm": float(sub.downstream_tp_accum_mm.sum()),
                "summed_event_window_downstream_sf_mm": float(sub.downstream_sf_accum_mm.sum()),
                "mean_basin_tp_per_event_mm": float(sub.basin_tp_accum_mm.mean()),
                "mean_basin_sf_per_event_mm": float(sub.basin_sf_accum_mm.mean()),
                "p90_basin_tp_per_event_mm": float(sub.basin_tp_accum_mm.quantile(0.90)),
                "p90_basin_sf_per_event_mm": float(sub.basin_sf_accum_mm.quantile(0.90)),
                "mean_basin_tp_rate_mm_h": float(sub.basin_tp_rate_mm_h.mean()),
                "mean_basin_sf_rate_mm_h": float(sub.basin_sf_rate_mm_h.mean()),
                "mean_downstream_tp_per_event_mm": float(sub.downstream_tp_accum_mm.mean()),
                "mean_downstream_sf_per_event_mm": float(sub.downstream_sf_accum_mm.mean()),
                "p90_downstream_tp_per_event_mm": float(sub.downstream_tp_accum_mm.quantile(0.90)),
                "p90_downstream_sf_per_event_mm": float(sub.downstream_sf_accum_mm.quantile(0.90)),
                "mean_downstream_tp_rate_mm_h": float(sub.downstream_tp_rate_mm_h.mean()),
                "mean_downstream_sf_rate_mm_h": float(sub.downstream_sf_rate_mm_h.mean()),
                "mean_precipitation_window_h": float(sub.precipitation_window_hours.mean()),
            })

    annual_df = pd.DataFrame(annual_records)
    
    # Statistical trends
    rows = []
    year_c = annual_df.season_year - annual_df.season_year.mean()
    endpoints = [
        "summed_event_window_basin_tp_mm",
        "summed_event_window_basin_sf_mm",
        "summed_event_window_downstream_tp_mm",
        "summed_event_window_downstream_sf_mm",
        "mean_basin_tp_per_event_mm",
        "mean_basin_sf_per_event_mm",
        "p90_basin_tp_per_event_mm",
        "p90_basin_sf_per_event_mm",
        "mean_basin_tp_rate_mm_h",
        "mean_basin_sf_rate_mm_h",
        "mean_downstream_tp_per_event_mm",
        "mean_downstream_sf_per_event_mm",
        "p90_downstream_tp_per_event_mm",
        "p90_downstream_sf_per_event_mm",
        "mean_downstream_tp_rate_mm_h",
        "mean_downstream_sf_rate_mm_h",
    ]
    rng = np.random.default_rng(20260912)
    for col in endpoints:
        fit = sm.OLS(annual_df[col], sm.add_constant(year_c)).fit(cov_type="HAC", cov_kwds={"maxlags": 2})
        mean_val = float(annual_df[col].mean())
        slope = float(fit.params.iloc[1])
        decadal_change = slope * 10.0
        hac_ci = fit.conf_int().iloc[1].to_numpy(dtype=float) * 10.0
        block_ci = moving_block_trend_ci(
            annual_df[col].to_numpy(dtype=float), year_c.to_numpy(dtype=float), rng
        )
        pct_change = (decadal_change / mean_val * 100.0) if mean_val > 1e-6 else 0.0
        p_val = float(fit.pvalues.iloc[1])
        rows.append({
            "variable": col,
            "climatological_mean": mean_val,
            "slope_per_year": slope,
            "change_per_decade": decadal_change,
            "change_per_decade_hac_ci95_low": float(hac_ci[0]),
            "change_per_decade_hac_ci95_high": float(hac_ci[1]),
            "change_per_decade_block_ci95_low": float(block_ci[0]),
            "change_per_decade_block_ci95_high": float(block_ci[1]),
            "pct_change_per_decade": pct_change,
            "pct_change_block_ci95_low": float(block_ci[0] / mean_val * 100.0),
            "pct_change_block_ci95_high": float(block_ci[1] / mean_val * 100.0),
            "p_value_hac": p_val,
        })

    stats_df = pd.DataFrame(rows)
    order = stats_df.p_value_hac.sort_values().index
    n_tests = len(stats_df)
    adjusted = (
        stats_df.loc[order, "p_value_hac"] * n_tests / np.arange(1, n_tests + 1)
    ).iloc[::-1].cummin().iloc[::-1].clip(upper=1)
    stats_df["p_value_fdr"] = adjusted.reindex(stats_df.index)
    return annual_df, stats_df


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    expected_events = len(load_djf_events())
    validate_download_manifest(expected_events)
    event_df = process_event_precipitation()
    if event_df.empty:
        print("No event records processed (waiting for downloads)")
        return
    if len(event_df) != expected_events:
        raise RuntimeError(
            f"Incomplete precipitation coverage: processed {len(event_df)} of {expected_events} events"
        )

    annual_df, stats_df = analyze_annual_precip(event_df)
    
    event_df.to_csv(OUT / "djf_event_precipitation_events.csv", index=False)
    annual_df.to_csv(OUT / "djf_event_precipitation_annual.csv", index=False)
    stats_df.to_csv(OUT / "djf_event_precipitation_statistics.csv", index=False)
    summary = {
        "events": int(len(event_df)),
        "winters": [int(YEARS.min()), int(YEARS.max())],
        "window": "[genesis, lysis + 24 h)",
        "complete_hourly_coverage": bool(
            (event_df.downloaded_hours == event_df.precipitation_window_hours).all()
        ),
        "regions": REGIONS,
        "annual_sum_interpretation": "event-window burden index; overlapping windows are not deduplicated",
        "inferential_family": "16 symmetric region-by-hydrometeor burden, mean, p90 and rate trends",
        "trend_uncertainty": "HAC maxlags=2 and 2,000-replicate 5-year residual moving-block bootstrap",
    }
    (OUT / "djf_event_precipitation_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    report = [
        "# DJF 涡旋事件窗降水与降雪诊断",
        "",
        "更新日期：2026-09-12",
        "",
        "## 设计边界",
        "",
        "本诊断包含1980--2023年732个四川盆地本地型（cluster 1）DJF事件，使用半开事件窗 "
        "`[genesis, lysis + 24 h)`，并逐事件要求小时覆盖完整。四川盆地固定区为27--33°N、"
        "102--109°E，下游固定区为25--34°N、109--120°E。逐年事件窗之和会重复计入重叠事件，"
        "故仅称为事件窗负荷指数，不能解释为唯一的季节总降水。",
        "",
        "降水和降雪在两个区域均对称报告事件窗负荷、逐事件均值、90分位数和时长归一化率。"
        "16项趋势统一进行BH-FDR控制；不确定性同时使用HAC（最大滞后2年）和2,000次、"
        "5年残差移动区组bootstrap。",
        "",
        "## 趋势结果",
        "",
        "| 指标 | 每十年变化（相对气候均值） | 5年区组95% CI | FDR p |",
        "|---|---:|---:|---:|",
    ]
    for row in stats_df.itertuples():
        report.append(
            f"| `{row.variable}` | {row.pct_change_per_decade:+.2f}% | "
            f"{row.pct_change_block_ci95_low:+.2f}% 至 {row.pct_change_block_ci95_high:+.2f}% | "
            f"{row.p_value_fdr:.4f} |"
        )
    report.extend(
        [
            "",
            "## 解释",
            "",
            "两区降水事件窗负荷均无显著增加。盆地单事件平均降水及其时长归一化率下降；"
            "盆地单事件平均降雪和降雪率也下降。下游区最稳健的变化集中在降雪，"
            "包括单事件均值、90分位数和降雪率。结果与‘更多本地涡旋机械地增加负荷、"
            "但单事件水文量下降’的抵消结构一致。",
            "",
            "该结果不等同于涡旋造成的降水量，因为固定事件窗包含背景天气，且不同事件窗可能重叠。"
            "首轮也不包括东移型事件的轨迹邻域降水，不能据此断言长江流域洪水风险不变。",
            "",
        ]
    )
    DOC.write_text("\n".join(report), encoding="utf-8")
    print(f"Processed {len(event_df)} events across {len(annual_df)} winters")
    print(stats_df.to_string(index=False))


if __name__ == "__main__":
    main()
