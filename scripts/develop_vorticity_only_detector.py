"""Develop and forward-validate a 700-hPa vorticity-only SWCV detector.

The detector deliberately does not use geopotential height.  Hyperparameters
are selected on 1980--2001 and then frozen before the 2002--2023 evaluation.
Intermediate centre candidates are cached so the expensive ERA5 scan is
restartable.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from scipy.ndimage import correlate, gaussian_filter, maximum_filter, uniform_filter
from scipy.optimize import linear_sum_assignment
from scipy.stats import pearsonr


ROOT = Path(__file__).resolve().parents[1]
ERA5 = ROOT / "data" / "raw" / "era5_700hpa_6hourly"
PS = ROOT / "data" / "raw" / "era5_surface_pressure_6hourly"
OUT = ROOT / "data" / "processed"
FIG = ROOT / "figures"
CACHE = OUT / "vorticity_detector_cache"
EARTH_R = 6_371_000.0
DEV_YEARS = range(1980, 2002)
VAL_YEARS = range(2002, 2024)
SOURCE_BOX = (26.0, 33.0, 99.0, 109.0)


@dataclass(frozen=True)
class DetectorConfig:
    resolution_deg: float
    smooth_deg: float
    peak_separation_deg: float
    threshold_s_1: float
    vortex_ratio_min: float
    min_steps: int
    max_gap_steps: int
    link_km: float = 400.0


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))


def normalize(ds: xr.Dataset) -> xr.Dataset:
    rename = {}
    for old, new in [("valid_time", "time"), ("pressure_level", "level")]:
        if old in ds.coords and new not in ds.coords:
            rename[old] = new
    ds = ds.rename(rename)
    if "level" in ds.dims:
        ds = ds.sel(level=700, drop=True)
    return ds.sortby("latitude").sortby("longitude")


def wind_derivatives(u: np.ndarray, v: np.ndarray, lat: np.ndarray, lon: np.ndarray):
    """Return vorticity and total horizontal strain on a spherical grid."""
    dlat = math.radians(float(np.median(np.diff(lat))))
    dlon = math.radians(float(np.median(np.diff(lon))))
    dvd_lambda = np.gradient(v, dlon, axis=-1, edge_order=2)
    dud_lambda = np.gradient(u, dlon, axis=-1, edge_order=2)
    dud_phi = np.gradient(u, dlat, axis=-2, edge_order=2)
    dvd_phi = np.gradient(v, dlat, axis=-2, edge_order=2)
    coslat = np.cos(np.deg2rad(lat))[None, :, None]
    dvdx = dvd_lambda / (EARTH_R * coslat)
    dudx = dud_lambda / (EARTH_R * coslat)
    dudy = dud_phi / EARTH_R
    dvdy = dvd_phi / EARTH_R
    zeta = dvdx - dudy
    strain_sq = (dudx - dvdy) ** 2 + (dvdx + dudy) ** 2
    return zeta, strain_sq


def static_terrain_mask(target_lat: np.ndarray, target_lon: np.ndarray) -> np.ndarray:
    """Fixed observed-terrain mask: 1980 mean surface pressure >=725 hPa.

    This ancillary mask is independent of atmospheric experiment and can be
    applied identically to ERA5 and CMIP grids.  It prevents below/near-ground
    700-hPa extrema from seeding tracks without requiring ps at run time.
    """
    path = PS / "era5_surface_pressure_1980.nc"
    with xr.open_dataset(path) as raw:
        ds = normalize(raw)
        name = "sp" if "sp" in ds else next(iter(ds.data_vars))
        mean_ps = ds[name].mean("time") / 100.0
        mask = mean_ps.interp(latitude=target_lat, longitude=target_lon, method="nearest") >= 725.0
        return np.asarray(mask.values, dtype=bool)


def target_axis(values: np.ndarray, resolution: float) -> np.ndarray:
    lo, hi = float(values.min()), float(values.max())
    return np.arange(lo, hi + resolution / 2, resolution)


def candidate_cache_path(year: int, resolution: float, smooth: float, peak_sep: float) -> Path:
    tag = f"v3_r{resolution:g}_s{smooth:g}_p{peak_sep:g}".replace(".", "p")
    return CACHE / tag / f"centres_{year}.csv.gz"


def scan_year(year: int, resolution: float, smooth_deg: float, peak_sep_deg: float, force: bool = False) -> Path:
    cache = candidate_cache_path(year, resolution, smooth_deg, peak_sep_deg)
    if cache.exists() and not force:
        return cache
    cache.parent.mkdir(parents=True, exist_ok=True)
    with xr.open_dataset(ERA5 / f"era5_700hpa_{year}.nc") as raw:
        ds = normalize(raw)[["u", "v"]]
        if resolution > 0.251:
            lat_new = target_axis(ds.latitude.values, resolution)
            lon_new = target_axis(ds.longitude.values, resolution)
            ds = ds.interp(latitude=lat_new, longitude=lon_new)
        lat, lon = ds.latitude.values, ds.longitude.values
        u = np.asarray(ds.u.values, dtype=np.float32)
        v = np.asarray(ds.v.values, dtype=np.float32)
        sigma = max(smooth_deg / resolution, 0.0)
        if sigma > 0:
            u = gaussian_filter(u, sigma=(0, sigma, sigma), mode="nearest")
            v = gaussian_filter(v, sigma=(0, sigma, sigma), mode="nearest")
        zeta, strain_sq = wind_derivatives(u, v, lat, lon)
        vortex_ratio = zeta ** 2 / (strain_sq + 1e-12)
        core_cells = max(3, int(round(1.5 / resolution)) * 2 + 1)
        core_mean_zeta = uniform_filter(zeta, size=(1, core_cells, core_cells), mode="nearest")
        speed = np.hypot(u, v)
        surrounding_speed = uniform_filter(speed, size=(1, core_cells, core_cells), mode="nearest")
        wind_speed_deficit = surrounding_speed - speed
        radius_cells = max(1, int(round(1.5 / resolution)))
        yy, xx = np.mgrid[-radius_cells:radius_cells + 1, -radius_cells:radius_cells + 1]
        rr = np.hypot(xx, yy)
        ring = (rr >= max(1, radius_cells * 0.5)) & (rr <= radius_cells)
        norm = max(int(ring.sum()), 1)
        kx = np.where(ring, xx / np.maximum(rr, 1), 0.0) / norm
        ky = np.where(ring, yy / np.maximum(rr, 1), 0.0) / norm
        circulation = correlate(v, kx[None, :, :], mode="nearest") - correlate(u, ky[None, :, :], mode="nearest")
        footprint = max(3, int(round(peak_sep_deg / resolution)) * 2 + 1)
        local_max = zeta == maximum_filter(zeta, size=(1, footprint, footprint), mode="nearest")
        terrain = static_terrain_mask(lat, lon)
        valid = local_max & terrain[None, :, :] & (zeta >= 1.0e-5)
        ti, yi, xi = np.nonzero(valid)
        frame = pd.DataFrame({
            "time": pd.to_datetime(ds.time.values[ti]),
            "lat": lat[yi],
            "lon": lon[xi],
            "vorticity_s_1": zeta[ti, yi, xi].astype(np.float32),
            "vortex_ratio": vortex_ratio[ti, yi, xi].astype(np.float32),
            "core_mean_vorticity_s_1": core_mean_zeta[ti, yi, xi].astype(np.float32),
            "center_wind_ms": speed[ti, yi, xi].astype(np.float32),
            "wind_speed_deficit_ms": wind_speed_deficit[ti, yi, xi].astype(np.float32),
            "circulation_ms": circulation[ti, yi, xi].astype(np.float32),
        })
    frame.to_csv(cache, index=False)
    print(f"scan {year}: {len(frame):,} candidate maxima -> {cache}", flush=True)
    return cache


def load_candidates(years, config: DetectorConfig, force: bool = False) -> pd.DataFrame:
    paths = [scan_year(y, config.resolution_deg, config.smooth_deg, config.peak_separation_deg, force) for y in years]
    frames = [pd.read_csv(p, parse_dates=["time"]) for p in paths]
    out = pd.concat(frames, ignore_index=True)
    return out.loc[(out.vorticity_s_1 >= config.threshold_s_1) & (out.vortex_ratio >= config.vortex_ratio_min)].copy()


def in_source(lat, lon) -> bool:
    return SOURCE_BOX[0] <= lat <= SOURCE_BOX[1] and SOURCE_BOX[2] <= lon <= SOURCE_BOX[3]


def track_candidates(candidates: pd.DataFrame, config: DetectorConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    all_tracks: list[dict] = []
    next_id = 0
    for year, annual in candidates.groupby(candidates.time.dt.year, sort=True):
        active: list[dict] = []
        finished: list[dict] = []
        times = pd.date_range(f"{year}-01-01", f"{year}-12-31 18:00", freq="6h")
        by_time = {t: x for t, x in annual.groupby("time")}
        for time in times:
            cand = by_time.get(time, annual.iloc[0:0]).sort_values("vorticity_s_1", ascending=False).reset_index(drop=True)
            used = set()
            matched_active = set()
            if active and len(cand):
                cost = np.full((len(active), len(cand)), 1e6, dtype=float)
                for i, tr in enumerate(active):
                    gap_hours = (time - tr["last_time"]).total_seconds() / 3600
                    if gap_hours <= 6 * (config.max_gap_steps + 1):
                        cost[i] = haversine_km(tr["last_lat"], tr["last_lon"], cand.lat.values, cand.lon.values)
                ri, ci = linear_sum_assignment(cost)
                for i, j in zip(ri, ci):
                    if cost[i, j] <= config.link_km:
                        row = cand.iloc[j]
                        active[i]["points"].append((time, row.lat, row.lon, row.vorticity_s_1))
                        active[i].update(last_time=time, last_lat=row.lat, last_lon=row.lon, misses=0)
                        matched_active.add(i); used.add(j)
            survivors = []
            for i, tr in enumerate(active):
                if i not in matched_active:
                    tr["misses"] += 1
                if tr["misses"] <= config.max_gap_steps:
                    survivors.append(tr)
                else:
                    finished.append(tr)
            active = survivors
            for j, row in cand.iterrows():
                if j not in used and in_source(row.lat, row.lon):
                    active.append({
                        "track_id": f"V{next_id:06d}", "points": [(time, row.lat, row.lon, row.vorticity_s_1)],
                        "last_time": time, "last_lat": row.lat, "last_lon": row.lon, "misses": 0,
                    })
                    next_id += 1
        finished.extend(active)
        all_tracks.extend(tr for tr in finished if len(tr["points"]) >= config.min_steps)
    event_rows, point_rows = [], []
    for tr in all_tracks:
        pts = tr["points"]
        event_rows.append({
            "detected_id": tr["track_id"], "genesis_time": pts[0][0], "lysis_time": pts[-1][0],
            "genesis_lat": pts[0][1], "genesis_lon": pts[0][2], "n_steps": len(pts),
            "max_vorticity_s_1": max(x[3] for x in pts),
        })
        point_rows.extend({"detected_id": tr["track_id"], "time": x[0], "lat": x[1], "lon": x[2], "vorticity_s_1": x[3]} for x in pts)
    return pd.DataFrame(event_rows), pd.DataFrame(point_rows)


def reference_data(years) -> tuple[pd.DataFrame, pd.DataFrame]:
    years = set(years)
    events = pd.read_csv(OUT / "events.csv", parse_dates=["genesis_time", "lysis_time"])
    events["genesis_time"] = events.genesis_time.dt.tz_localize(None)
    events["lysis_time"] = events.lysis_time.dt.tz_localize(None)
    events = events.loc[events.year.isin(years)].copy()
    points = pd.read_csv(OUT / "track_points.csv.gz", parse_dates=["time"])
    points["time"] = points.time.dt.tz_localize(None)
    points = points.loc[points.event_id.isin(events.event_id) & points.time.dt.hour.isin([0, 6, 12, 18])].copy()
    return events, points


def match_events(det_events: pd.DataFrame, det_points: pd.DataFrame, ref_events: pd.DataFrame, ref_points: pd.DataFrame):
    ref_groups = {k: g.set_index("time") for k, g in ref_points.groupby("event_id")}
    det_groups = {k: g.set_index("time") for k, g in det_points.groupby("detected_id")}
    candidates = []
    for d in det_events.itertuples():
        possible = ref_events.loc[(ref_events.genesis_time - d.genesis_time).abs() <= pd.Timedelta(hours=12)]
        dg = det_groups[d.detected_id]
        for r in possible.itertuples():
            rg = ref_groups.get(r.event_id)
            if rg is None:
                continue
            common = dg.index.intersection(rg.index)
            if len(common) < 2:
                continue
            dist = haversine_km(rg.loc[common, "lat"].values, rg.loc[common, "lon"].values,
                                dg.loc[common, "lat"].values, dg.loc[common, "lon"].values)
            nmatch = int(np.sum(dist <= 300.0))
            if nmatch >= 2:
                candidates.append((d.detected_id, r.event_id, nmatch, float(np.mean(dist[dist <= 300.0]))))
    if not candidates:
        return pd.DataFrame(columns=["detected_id", "event_id", "matched_centres", "mean_distance_km"])
    pairs = pd.DataFrame(candidates, columns=["detected_id", "event_id", "matched_centres", "mean_distance_km"])
    dids, rids = pairs.detected_id.unique(), pairs.event_id.unique()
    di, ri = {v: i for i, v in enumerate(dids)}, {v: i for i, v in enumerate(rids)}
    cost = np.full((len(dids), len(rids)), 1e6)
    for p in pairs.itertuples():
        cost[di[p.detected_id], ri[p.event_id]] = -p.matched_centres + p.mean_distance_km / 100000.0
    rows, cols = linear_sum_assignment(cost)
    chosen = []
    for i, j in zip(rows, cols):
        if cost[i, j] < 0:
            chosen.append(pairs.loc[(pairs.detected_id == dids[i]) & (pairs.event_id == rids[j])]
                          .sort_values(["matched_centres", "mean_distance_km"], ascending=[False, True]).iloc[0])
    return pd.DataFrame(chosen).reset_index(drop=True)


def season_and_year(time: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    month = time.dt.month
    season = np.select([month.isin([12, 1, 2]), month.isin([3, 4, 5]), month.isin([6, 7, 8])],
                       ["DJF", "MAM", "JJA"], default="SON")
    return season, time.dt.year.to_numpy()


def evaluate(det_events, det_points, ref_events, ref_points, period: str) -> tuple[dict, pd.DataFrame]:
    pairs = match_events(det_events, det_points, ref_events, ref_points)
    tp, nd, nr = len(pairs), len(det_events), len(ref_events)
    precision = tp / nd if nd else 0.0
    recall = tp / nr if nr else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    counts = []
    for label, events in [("detected", det_events.rename(columns={"genesis_time": "time"})),
                          ("reference", ref_events.rename(columns={"genesis_time": "time"}))]:
        e = events.copy(); e["season"], e["count_year"] = season_and_year(e.time)
        c = e.groupby(["count_year", "season"]).size().rename(label)
        counts.append(c)
    count_table = pd.concat(counts, axis=1).fillna(0).reset_index()
    seasonal = {}
    for season in ["DJF", "JJA"]:
        q = count_table.loc[count_table.season == season]
        r = float(pearsonr(q.detected, q.reference).statistic) if len(q) >= 3 and q.detected.std() > 0 else np.nan
        bias = float((q.detected.mean() - q.reference.mean()) / q.reference.mean())
        seasonal[season] = {"pearson_r": r, "mean_bias_fraction": bias,
                            "detected_mean": float(q.detected.mean()), "reference_mean": float(q.reference.mean())}
    metrics = {"period": period, "n_detected": nd, "n_reference": nr, "n_matched": tp,
               "precision": precision, "recall": recall, "event_f1": f1, "seasonal": seasonal}
    return metrics, count_table


def calibration_configs(resolution: float) -> list[DetectorConfig]:
    return [DetectorConfig(resolution, 0.5, 0.75, threshold, ratio, steps, gap)
            for threshold in [3e-5, 5e-5, 7e-5, 1e-4]
            for ratio in [0.5, 1.0, 2.0] for steps in [2, 3] for gap in [0, 1]]


def config_score(metrics: dict) -> tuple:
    s = metrics["seasonal"]
    gates = int(metrics["event_f1"] >= 0.6)
    gates += int(s["DJF"]["pearson_r"] >= 0.6) + int(s["JJA"]["pearson_r"] >= 0.6)
    gates += int(abs(s["DJF"]["mean_bias_fraction"]) <= 0.3)
    gates += int(abs(s["JJA"]["mean_bias_fraction"]) <= 0.3)
    return gates, metrics["event_f1"], min(s["DJF"]["pearson_r"], s["JJA"]["pearson_r"]), -max(abs(s["DJF"]["mean_bias_fraction"]), abs(s["JJA"]["mean_bias_fraction"]))


def run_resolution(resolution: float, force_scan: bool = False) -> dict:
    base = calibration_configs(resolution)[0]
    dev_candidates = load_candidates(DEV_YEARS, base, force_scan)
    ref_dev, refp_dev = reference_data(DEV_YEARS)
    rows, products = [], {}
    for cfg in calibration_configs(resolution):
        selected = dev_candidates.loc[(dev_candidates.vorticity_s_1 >= cfg.threshold_s_1) &
                                      (dev_candidates.vortex_ratio >= cfg.vortex_ratio_min)]
        de, dp = track_candidates(selected, cfg)
        met, counts = evaluate(de, dp, ref_dev, refp_dev, "1980-2001 development")
        rows.append({**asdict(cfg), **{k: v for k, v in met.items() if k != "seasonal"},
                     **{f"{sea}_{k}": v for sea, vals in met["seasonal"].items() for k, v in vals.items()}})
        products[cfg] = (de, dp, met, counts)
        print(f"calibrate r={resolution:g} t={cfg.threshold_s_1:.0e} ratio={cfg.vortex_ratio_min:g} "
              f"steps={cfg.min_steps} gap={cfg.max_gap_steps}: F1={met['event_f1']:.3f}", flush=True)
    best = max(products, key=lambda x: config_score(products[x][2]))
    de, dp, dev_met, dev_counts = products[best]
    val_candidates = load_candidates(VAL_YEARS, best, force_scan)
    ve, vp = track_candidates(val_candidates, best)
    ref_val, refp_val = reference_data(VAL_YEARS)
    val_met, val_counts = evaluate(ve, vp, ref_val, refp_val, "2002-2023 forward validation")
    tag = f"r{resolution:g}".replace(".", "p")
    pd.DataFrame(rows).to_csv(OUT / f"vorticity_detector_calibration_{tag}.csv", index=False)
    de.to_csv(OUT / f"vorticity_detector_events_dev_{tag}.csv.gz", index=False)
    dp.to_csv(OUT / f"vorticity_detector_points_dev_{tag}.csv.gz", index=False)
    ve.to_csv(OUT / f"vorticity_detector_events_validation_{tag}.csv.gz", index=False)
    vp.to_csv(OUT / f"vorticity_detector_points_validation_{tag}.csv.gz", index=False)
    pd.concat([dev_counts.assign(period="development"), val_counts.assign(period="validation")]).to_csv(
        OUT / f"vorticity_detector_seasonal_counts_{tag}.csv", index=False)
    return {"resolution_deg": resolution, "selected_config": asdict(best), "development": dev_met, "forward_validation": val_met}


def make_figure(results: list[dict]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    x = np.arange(len(results)); labels = [f"{r['resolution_deg']:g}°" for r in results]
    axes[0].bar(x - .17, [r["development"]["event_f1"] for r in results], .34, label="Development")
    axes[0].bar(x + .17, [r["forward_validation"]["event_f1"] for r in results], .34, label="Validation")
    axes[0].axhline(.6, color="k", ls="--", lw=1); axes[0].set(ylabel="Event F1", xticks=x, xticklabels=labels); axes[0].legend(fontsize=8)
    for j, sea in enumerate(["DJF", "JJA"], start=1):
        axes[j].bar(x, [r["forward_validation"]["seasonal"][sea]["pearson_r"] for r in results], color="#3973ac")
        axes[j].axhline(.6, color="k", ls="--", lw=1); axes[j].set(title=sea, ylabel="Validation count r", xticks=x, xticklabels=labels)
    for ax in axes: ax.grid(axis="y", alpha=.2)
    fig.tight_layout(); fig.savefig(FIG / "vorticity_detector_forward_validation.png", dpi=180); plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resolutions", nargs="+", type=float, default=[0.25, 1.0, 1.25, 1.5])
    parser.add_argument("--force-scan", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True); FIG.mkdir(parents=True, exist_ok=True); CACHE.mkdir(parents=True, exist_ok=True)
    results = []
    for resolution in args.resolutions:
        results.append(run_resolution(resolution, args.force_scan))
        (OUT / "vorticity_threshold_detector_resolution_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    make_figure(results)
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    main()
