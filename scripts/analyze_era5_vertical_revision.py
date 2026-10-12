"""Diagnose ERA5 lower-tropospheric structure and the sigma-coordinate chain term."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import xarray as xr

from download_era5_model_level_sigma_full import selection
from analyze_native_model_level_sigma_pilot import canonical


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw"
OUT = ROOT / "data/processed"
MANIFEST = ROOT / "data/manifests/era5_vertical_revision_djf_manifest.json"
COEFFICIENTS = ROOT / "data/manifests/era5_l137_revision_coefficients.json"
SIGMA = 0.85
EARTH_R = 6_371_000.0
CORE_RADIUS_DEG = 1.5
MIN_CORE_VALID = 0.80
KAPPA = 0.2854
PRESSURE_TARGETS_HPA = (700.0, 850.0, 900.0)


def full_pressure_pa(sp: np.ndarray, levels: np.ndarray, coefficients: dict[int, dict[str, float]]) -> np.ndarray:
    result = []
    for level in levels:
        upper = coefficients[int(level) - 1]
        lower = coefficients[int(level)]
        result.append(0.5 * (upper["a_pa"] + lower["a_pa"] + (upper["b"] + lower["b"]) * sp))
    return np.stack(result)


def interpolate_at_pressure(values: np.ndarray, pressures: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Linearly interpolate level-first values to a 2-D pressure target."""
    if target.ndim == 0:
        target = np.full(pressures.shape[1:], float(target))
    below = np.sum(pressures <= target[None, ...], axis=0) - 1
    valid = (below >= 0) & (below < pressures.shape[0] - 1)
    index = np.clip(below, 0, pressures.shape[0] - 2)
    p0 = np.take_along_axis(pressures, index[None, ...], axis=0)[0]
    p1 = np.take_along_axis(pressures, (index + 1)[None, ...], axis=0)[0]
    v0 = np.take_along_axis(values, index[None, ...], axis=0)[0]
    v1 = np.take_along_axis(values, (index + 1)[None, ...], axis=0)[0]
    alpha = (target - p0) / (p1 - p0)
    interpolated = v0 + alpha * (v1 - v0)
    interpolated[~valid] = np.nan
    derivative = (v1 - v0) / (p1 - p0)
    derivative[~valid] = np.nan
    return interpolated, derivative


def horizontal_gradient(field: np.ndarray, lat: np.ndarray, lon: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return eastward and northward gradients on a regular lat/lon grid."""
    per_degree = EARTH_R * np.deg2rad(1.0)
    d_dlon = np.gradient(field, lon, axis=1, edge_order=2)
    d_dlat = np.gradient(field, lat, axis=0, edge_order=2)
    dx = per_degree * np.cos(np.deg2rad(lat))[:, None]
    return d_dlon / dx, d_dlat / per_degree


def vorticity(u: np.ndarray, v: np.ndarray, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    dv_dx, _ = horizontal_gradient(v, lat, lon)
    _, du_dy = horizontal_gradient(u, lat, lon)
    return dv_dx - du_dy


def fit_trend(annual: pd.DataFrame, response: str) -> dict[str, float | int]:
    work = annual.dropna(subset=[response])
    result: dict[str, float | int] = {"n_winters": int(len(work))}
    if len(work) < 10:
        result.update({"per_decade": np.nan, "ci_low": np.nan, "ci_high": np.nan, "p_hac": np.nan})
        return result
    x = work.season_year - work.season_year.mean()
    fit = sm.OLS(work[response], sm.add_constant(x)).fit(cov_type="HAC", cov_kwds={"maxlags": 2})
    ci = fit.conf_int().iloc[1]
    result.update({
        "per_decade": float(fit.params.iloc[1] * 10.0),
        "ci_low": float(ci.iloc[0] * 10.0),
        "ci_high": float(ci.iloc[1] * 10.0),
        "p_hac": float(fit.pvalues.iloc[1]),
    })
    return result


def diagnose_point(model: xr.Dataset, surface: xr.Dataset, row: object) -> dict[str, object]:
    timestamp = pd.Timestamp(row.time).tz_localize(None)
    bounds = {"lat": slice(row.lat - 2.0, row.lat + 2.0), "lon": slice(row.lon - 2.0, row.lon + 2.0)}
    sample = model[["u", "v", "t"]].sel(time=timestamp).sel(**bounds)
    sp_da = surface.sp.sel(time=timestamp).sel(lat=sample.lat, lon=sample.lon)
    lat = np.asarray(sample.lat, dtype=float)
    lon = np.asarray(sample.lon, dtype=float)
    sp = np.asarray(sp_da, dtype=float)
    levels = np.asarray(sample.level, dtype=int)
    pressure = full_pressure_pa(sp, levels, COEFF)
    u = np.asarray(sample.u, dtype=float)
    v = np.asarray(sample.v, dtype=float)
    temperature = np.asarray(sample.t, dtype=float)

    target = SIGMA * sp
    u_sigma, du_dp = interpolate_at_pressure(u, pressure, target)
    v_sigma, dv_dp = interpolate_at_pressure(v, pressure, target)
    t_sigma, dt_dp = interpolate_at_pressure(temperature, pressure, target)
    total = vorticity(u_sigma, v_sigma, lat, lon)
    dps_dx, dps_dy = horizontal_gradient(sp, lat, lon)
    chain = SIGMA * (dv_dp * dps_dx - du_dp * dps_dy)
    fixed = total - chain

    interpolated: dict[float, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for target_hpa in PRESSURE_TARGETS_HPA:
        target_pa = np.asarray(target_hpa * 100.0)
        up, _ = interpolate_at_pressure(u, pressure, target_pa)
        vp, _ = interpolate_at_pressure(v, pressure, target_pa)
        tp, _ = interpolate_at_pressure(temperature, pressure, target_pa)
        interpolated[target_hpa] = (up, vp, tp)

    radius = np.sqrt((lat[:, None] - row.lat) ** 2 + ((lon[None, :] - row.lon) * np.cos(np.deg2rad(row.lat))) ** 2)
    core = radius <= CORE_RADIUS_DEG
    valid = core & np.isfinite(total) & np.isfinite(chain)
    valid_fraction = float(valid.sum() / core.sum())
    result: dict[str, object] = {
        "event_id": row.event_id, "time": timestamp, "lat": row.lat, "lon": row.lon,
        "core_valid_fraction": valid_fraction,
    }
    if valid_fraction < MIN_CORE_VALID:
        for name in ("sigma_total_vorticity_s_1", "sigma_fixed_pressure_vorticity_s_1", "sigma_chain_vorticity_s_1",
                     "sigma_temperature_k", "sigma_vertical_shear_ms_per_100hpa",
                     "sigma_static_stability_k_per_100hpa", "shear_700_850_ms",
                     "theta_700_minus_850_k", "theta_850_minus_900_k",
                     "vorticity_700_s_1", "vorticity_850_s_1"):
            result[name] = np.nan
        return result

    candidate = np.where(valid, total, np.nan)
    iy, ix = np.unravel_index(np.nanargmax(candidate), candidate.shape)
    result.update({
        "max_lat": float(lat[iy]), "max_lon": float(lon[ix]),
        "surface_pressure_hpa": float(sp[iy, ix] / 100.0),
        "sigma_target_pressure_hpa": float(target[iy, ix] / 100.0),
        "sigma_total_vorticity_s_1": float(total[iy, ix]),
        "sigma_fixed_pressure_vorticity_s_1": float(fixed[iy, ix]),
        "sigma_chain_vorticity_s_1": float(chain[iy, ix]),
        "sigma_temperature_k": float(t_sigma[iy, ix]),
        "sigma_vertical_shear_ms_per_100hpa": float(np.hypot(du_dp[iy, ix], dv_dp[iy, ix]) * 10000.0),
        "sigma_static_stability_k_per_100hpa": float(
            -(100000.0 / target[iy, ix]) ** KAPPA
            * (dt_dp[iy, ix] - KAPPA * t_sigma[iy, ix] / target[iy, ix]) * 10000.0
        ),
    })
    u700, v700, t700 = interpolated[700.0]
    u850, v850, t850 = interpolated[850.0]
    u900, v900, t900 = interpolated[900.0]
    result["shear_700_850_ms"] = float(np.hypot(u700[iy, ix] - u850[iy, ix], v700[iy, ix] - v850[iy, ix]))
    theta700 = t700[iy, ix] * (1000.0 / 700.0) ** KAPPA
    theta850 = t850[iy, ix] * (1000.0 / 850.0) ** KAPPA
    theta900 = t900[iy, ix] * (1000.0 / 900.0) ** KAPPA
    result["theta_700_minus_850_k"] = float(theta700 - theta850)
    result["theta_850_minus_900_k"] = float(theta850 - theta900)
    result["vorticity_700_s_1"] = float(vorticity(u700, v700, lat, lon)[iy, ix])
    result["vorticity_850_s_1"] = float(vorticity(u850, v850, lat, lon)[iy, ix])
    return result


def build_annual(events: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for season_year, group in events.groupby("season_year"):
        threshold = group.sigma_total_vorticity_s_1.quantile(0.9)
        upper = group.loc[group.sigma_total_vorticity_s_1 >= threshold]
        rows.append({
            "season_year": int(season_year), "events": int(len(group)), "upper_events": int(len(upper)),
            "q90_sigma_total_1e5_s_1": float(threshold * 1e5),
            "upper_mean_sigma_total_1e5_s_1": float(upper.sigma_total_vorticity_s_1.mean() * 1e5),
            "upper_mean_fixed_pressure_1e5_s_1": float(upper.sigma_fixed_pressure_vorticity_s_1.mean() * 1e5),
            "upper_mean_chain_1e5_s_1": float(upper.sigma_chain_vorticity_s_1.mean() * 1e5),
            "upper_mean_sigma_vertical_shear_ms_per_100hpa": float(upper.sigma_vertical_shear_ms_per_100hpa.mean()),
            "upper_mean_sigma_static_stability_k_per_100hpa": float(upper.sigma_static_stability_k_per_100hpa.mean()),
            "upper_mean_shear_700_850_ms": float(upper.shear_700_850_ms.mean()),
            "upper_mean_theta_700_minus_850_k": float(upper.theta_700_minus_850_k.mean()),
            "upper_mean_theta_850_minus_900_k": float(upper.theta_850_minus_900_k.mean()),
        })
    return pd.DataFrame(rows).sort_values("season_year")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, help="Calendar-year smoke test; outputs receive a test suffix")
    args = parser.parse_args()
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if manifest.get("status") != "selected_scope_complete" or len(manifest.get("files", [])) != 45:
        raise RuntimeError("Vertical-revision download manifest is incomplete")
    points, _ = selection()
    files = {int(item["year"]): ROOT / item["path"] for item in manifest["files"]}
    if args.year is not None:
        points = points.loc[points.time.dt.year.eq(args.year)].copy()
        if points.empty:
            raise ValueError("Requested calendar year is outside the frozen selection")
    rows = []
    for year, group in points.groupby(points.time.dt.year):
        year = int(year)
        with xr.open_dataset(files[year]) as raw_model, xr.open_dataset(
            RAW / f"era5_surface_pressure_6hourly/era5_surface_pressure_{year}.nc"
        ) as raw_surface:
            model = canonical(raw_model).load()
            surface = canonical(raw_surface).load()
        try:
            for row in group.sort_values(["time", "event_id"]).itertuples():
                rows.append(diagnose_point(model, surface, row))
        finally:
            model.close(); surface.close()
        print(f"analysed {year}: {len(group)} track points", flush=True)
    points_out = pd.DataFrame(rows)
    metadata = pd.read_csv(OUT / "events_terrain_qc_intensity.csv.gz", parse_dates=["genesis_time"],
        usecols=["event_id", "genesis_time", "track_cluster", "duration_hours"])
    metadata["season_year"] = metadata.genesis_time.dt.year + metadata.genesis_time.dt.month.eq(12).astype(int)
    chosen = points_out.dropna(subset=["sigma_total_vorticity_s_1"]).sort_values(
        ["event_id", "sigma_total_vorticity_s_1"]).groupby("event_id", as_index=False).tail(1)
    events = chosen.merge(metadata, on="event_id", how="left", validate="one_to_one")
    annual = build_annual(events)
    trend_columns = [column for column in annual if column not in {"season_year", "events", "upper_events"}]
    trends = {column: fit_trend(annual, column) for column in trend_columns}
    closure = points_out.sigma_total_vorticity_s_1 - (
        points_out.sigma_fixed_pressure_vorticity_s_1 + points_out.sigma_chain_vorticity_s_1)
    summary = {
        "scope": "calendar-year smoke test" if args.year is not None else "frozen 899-event DJF cohort, 1980-2023 season years",
        "track_points": int(len(points_out)), "events": int(len(events)),
        "valid_point_fraction": float(points_out.sigma_total_vorticity_s_1.notna().mean()),
        "diagnostic_valid_counts": {
            name: int(points_out[name].notna().sum()) for name in (
                "sigma_vertical_shear_ms_per_100hpa", "sigma_static_stability_k_per_100hpa",
                "shear_700_850_ms", "theta_700_minus_850_k", "theta_850_minus_900_k")
        },
        "decomposition_max_abs_closure_s_1": float(closure.abs().max()),
        "annual_trends": trends,
        "interpretation_guardrail": "The chain-term decomposition diagnoses coordinate geometry; it does not alone establish a complete dynamical or thermodynamic mechanism.",
    }
    suffix = f"_test_{args.year}" if args.year is not None else ""
    points_out.to_csv(OUT / f"era5_vertical_revision_points{suffix}.csv.gz", index=False)
    events.to_csv(OUT / f"era5_vertical_revision_events{suffix}.csv.gz", index=False)
    annual.to_csv(OUT / f"era5_vertical_revision_annual{suffix}.csv", index=False)
    (OUT / f"era5_vertical_revision_summary{suffix}.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    coefficient_payload = json.loads(COEFFICIENTS.read_text(encoding="utf-8"))
    COEFF = {int(key): value for key, value in coefficient_payload["coefficients"].items()}
    main()
