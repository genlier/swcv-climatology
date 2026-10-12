"""Event-relative JJA moisture-flux composites with season-year inference.

All event composites use anomalies from a 1991--2020 month-by-UTC-hour
climatology. Inference is performed on 44 season-year means, not on individual
events. ERA5 ``vimdf`` is divergence, so positive convergence is ``-vimdf``.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import xarray as xr


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "era5_jja_moisture_flux_6hourly"
OUT = ROOT / "data" / "processed"
FIG = ROOT / "figures"
CLIM_FILE = OUT / "era5_moisture_flux_climatology_1991_2020.nc"
VARS = ["viwve", "viwvn", "vimdf"]
LAGS = np.arange(-48, 73, 6)
EARTH_RADIUS_M = 6_371_000.0
SECONDS_PER_DAY = 86_400.0
BOOTSTRAPS = 2000
SEED = 20260907


def key_for_time(time: pd.Timestamp) -> str:
    return f"M{time.month:02d}_H{time.hour:02d}"


def expected_paths() -> list[Path]:
    paths = [RAW / f"era5_jja_moisture_flux_{year}.nc" for year in range(1979, 2024)]
    missing = [path.name for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Finish moisture download first; missing {len(missing)} files")
    return paths


def build_climatology(paths: list[Path]) -> xr.Dataset:
    keys = [f"M{month:02d}_H{hour:02d}" for month in range(5, 10) for hour in (0, 6, 12, 18)]
    with xr.open_dataset(paths[0]) as sample:
        latitude = sample.latitude.values
        longitude = sample.longitude.values
    sums = {name: np.zeros((len(keys), len(latitude), len(longitude)), dtype=np.float64) for name in VARS}
    counts = np.zeros(len(keys), dtype=np.int64)
    key_index = {key: idx for idx, key in enumerate(keys)}
    for path in paths:
        year = int(path.stem.rsplit("_", 1)[1])
        if not 1991 <= year <= 2020:
            continue
        with xr.open_dataset(path) as ds:
            times = pd.DatetimeIndex(ds.valid_time.values)
            for key in keys:
                month = int(key[1:3])
                hour = int(key[5:7])
                selected = np.flatnonzero((times.month == month) & (times.hour == hour))
                idx = key_index[key]
                counts[idx] += len(selected)
                for name in VARS:
                    sums[name][idx] += ds[name].isel(valid_time=selected).sum("valid_time").values
    if np.any(counts == 0):
        raise RuntimeError("Incomplete climatology groups")
    climatology = xr.Dataset(
        {
            name: (("climatology_key", "latitude", "longitude"), sums[name] / counts[:, None, None])
            for name in VARS
        },
        coords={"climatology_key": keys, "latitude": latitude, "longitude": longitude},
        attrs={"baseline": "1991-2020", "grouping": "calendar month and UTC hour"},
    )
    climatology.to_netcdf(CLIM_FILE)
    return climatology


def load_or_build_climatology(paths: list[Path]) -> xr.Dataset:
    if CLIM_FILE.exists():
        with xr.open_dataset(CLIM_FILE) as ds:
            if ds.sizes.get("climatology_key") == 20 and set(VARS).issubset(ds.data_vars):
                return ds.load()
    return build_climatology(paths)


def anomaly_arrays(ds: xr.Dataset, climatology: xr.Dataset) -> dict[str, np.ndarray]:
    times = pd.DatetimeIndex(ds.valid_time.values)
    result = {name: ds[name].values.astype(np.float64) for name in VARS}
    for position, time in enumerate(times):
        key = key_for_time(time)
        for name in VARS:
            result[name][position] -= climatology[name].sel(climatology_key=key).values
    return result


def fixed_box_metrics(
    arrays: dict[str, np.ndarray], latitude: np.ndarray, longitude: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return area -divergence and flux-boundary convergence in kg m-2 s-1."""
    north, south, west, east = 33.0, 24.0, 98.0, 110.0
    lat_mask = (latitude >= south) & (latitude <= north)
    lon_mask = (longitude >= west) & (longitude <= east)
    weights = np.cos(np.deg2rad(latitude[lat_mask]))[:, None]
    convergence = -arrays["vimdf"][:, lat_mask][:, :, lon_mask]
    area_convergence = np.sum(convergence * weights[None, :, :], axis=(1, 2)) / (
        np.sum(weights) * np.count_nonzero(lon_mask)
    )

    lat_positions = np.flatnonzero(lat_mask)
    lon_positions = np.flatnonzero(lon_mask)
    lat_order = lat_positions[np.argsort(latitude[lat_positions])]
    lon_order = lon_positions[np.argsort(longitude[lon_positions])]
    lat_radians = np.deg2rad(latitude[lat_order])
    lon_radians = np.deg2rad(longitude[lon_order])
    west_idx = int(np.argmin(np.abs(longitude - west)))
    east_idx = int(np.argmin(np.abs(longitude - east)))
    south_idx = int(np.argmin(np.abs(latitude - south)))
    north_idx = int(np.argmin(np.abs(latitude - north)))

    west_in = np.trapezoid(arrays["viwve"][:, lat_order, west_idx], EARTH_RADIUS_M * lat_radians, axis=1)
    east_out = np.trapezoid(arrays["viwve"][:, lat_order, east_idx], EARTH_RADIUS_M * lat_radians, axis=1)
    south_x = EARTH_RADIUS_M * np.cos(np.deg2rad(south)) * lon_radians
    north_x = EARTH_RADIUS_M * np.cos(np.deg2rad(north)) * lon_radians
    south_in = np.trapezoid(arrays["viwvn"][:, south_idx, lon_order], south_x, axis=1)
    north_out = np.trapezoid(arrays["viwvn"][:, north_idx, lon_order], north_x, axis=1)
    area = EARTH_RADIUS_M**2 * np.deg2rad(east - west) * (
        np.sin(np.deg2rad(north)) - np.sin(np.deg2rad(south))
    )
    boundary_convergence = (west_in - east_out + south_in - north_out) / area
    return area_convergence, boundary_convergence


def centered_convergence(
    convergence_field: np.ndarray,
    latitude: np.ndarray,
    longitude: np.ndarray,
    center_lat: float,
    center_lon: float,
    radius_degrees: float = 3.0,
) -> float:
    lat_delta = latitude[:, None] - center_lat
    lon_delta = (longitude[None, :] - center_lon) * np.cos(np.deg2rad(center_lat))
    mask = lat_delta**2 + lon_delta**2 <= radius_degrees**2
    weights = np.cos(np.deg2rad(latitude))[:, None]
    valid = mask & np.isfinite(convergence_field)
    return float(np.sum(convergence_field[valid] * np.broadcast_to(weights, mask.shape)[valid]) / np.sum(np.broadcast_to(weights, mask.shape)[valid]))


def event_composites(paths: list[Path], climatology: xr.Dataset) -> tuple[pd.DataFrame, pd.DataFrame]:
    events = pd.read_csv(OUT / "events_season_year_terrain_qc.csv.gz", parse_dates=["genesis_time"])
    events = events.loc[events.season.eq("JJA") & events.season_year.between(1980, 2023)].copy()
    events["anchor_time"] = events.genesis_time.dt.tz_convert(None).dt.round("6h")
    event_rows = []
    time_rows = []
    for path in paths:
        year = int(path.stem.rsplit("_", 1)[1])
        year_events = events.loc[events.genesis_time.dt.year.eq(year)]
        with xr.open_dataset(path) as ds:
            times = pd.DatetimeIndex(ds.valid_time.values)
            lookup = {time: idx for idx, time in enumerate(times)}
            arrays = anomaly_arrays(ds, climatology)
            area_conv, boundary_conv = fixed_box_metrics(arrays, ds.latitude.values, ds.longitude.values)
            for time, area_value, boundary_value in zip(times, area_conv, boundary_conv):
                time_rows.append(
                    {
                        "time": time,
                        "area_convergence_anom_kg_m2_s": area_value,
                        "boundary_convergence_anom_kg_m2_s": boundary_value,
                    }
                )
            convergence = -arrays["vimdf"]
            for event in year_events.itertuples():
                rounding_offset = (event.anchor_time - event.genesis_time.tz_convert(None)).total_seconds() / 3600
                for lag in LAGS:
                    target = event.anchor_time + pd.Timedelta(hours=int(lag))
                    if target not in lookup:
                        continue
                    idx = lookup[target]
                    event_rows.append(
                        {
                            "event_id": event.event_id,
                            "season_year": event.season_year,
                            "anchor_time": event.anchor_time,
                            "rounding_offset_hours": rounding_offset,
                            "lag_hours": lag,
                            "track_cluster": event.track_cluster,
                            "eastward_escape": event.eastward_escape,
                            "area_convergence_anom_kg_m2_s": area_conv[idx],
                            "boundary_convergence_anom_kg_m2_s": boundary_conv[idx],
                            "centered_convergence_anom_kg_m2_s": centered_convergence(
                                convergence[idx], ds.latitude.values, ds.longitude.values,
                                event.genesis_lat, event.genesis_lon,
                            ),
                        }
                    )
    time_metrics = pd.DataFrame(time_rows)
    event_metrics = pd.DataFrame(event_rows)
    time_metrics.to_csv(OUT / "era5_moisture_flux_time_metrics.csv.gz", index=False)
    event_metrics.to_csv(OUT / "jja_event_moisture_lag_event_level.csv.gz", index=False)
    return time_metrics, event_metrics


def bh_adjust(pvalues: pd.Series) -> pd.Series:
    result = pd.Series(np.nan, index=pvalues.index)
    valid = pvalues.dropna().sort_values()
    if valid.empty:
        return result
    n = len(valid)
    adjusted = (valid * n / np.arange(1, n + 1)).iloc[::-1].cummin().iloc[::-1].clip(upper=1)
    result.loc[adjusted.index] = adjusted
    return result


def inference(event_metrics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metrics = [
        "area_convergence_anom_kg_m2_s",
        "boundary_convergence_anom_kg_m2_s",
        "centered_convergence_anom_kg_m2_s",
    ]
    year_lag = event_metrics.groupby(["season_year", "lag_hours"], as_index=False)[metrics].mean()
    climate = pd.read_csv(OUT / "seasonal_joint_metrics_1980_2023.csv")
    climate = climate.loc[climate.season.eq("JJA"), ["season_year", "frequency", "iobm_40_100e_c_dt_z"]]
    circulation = pd.read_csv(OUT / "jja_era5_mechanism_indices_1979_2023.csv")
    circulation = circulation[["season_year", "wnpac_z850_gpm_dt_z", "monc_cyclonic_dt_z"]]
    year_lag = year_lag.merge(climate, on="season_year", validate="many_to_one")
    year_lag = year_lag.merge(circulation, on="season_year", validate="many_to_one")
    year_lag.to_csv(OUT / "jja_event_moisture_lag_year_equal.csv", index=False)

    windows = {"pre_48_to_12h": (-48, -12), "onset_6h": (-6, 6), "post_12_to_72h": (12, 72)}
    rows = []
    for window, (lower, upper) in windows.items():
        sample = year_lag.loc[year_lag.lag_hours.between(lower, upper)].groupby("season_year", as_index=False).agg(
            {**{metric: "mean" for metric in metrics}, "iobm_40_100e_c_dt_z": "first", "wnpac_z850_gpm_dt_z": "first"}
        )
        for predictor in ["wnpac_z850_gpm_dt_z", "iobm_40_100e_c_dt_z"]:
            for metric in metrics:
                fit = sm.OLS(sample[metric], sm.add_constant(sample[[predictor]])).fit(
                    cov_type="HAC", cov_kwds={"maxlags": 3}
                )
                ci = fit.conf_int().loc[predictor]
                rows.append(
                    {
                        "window": window,
                        "response": metric,
                        "predictor": predictor,
                        "coefficient_kg_m2_s_per_sd": fit.params[predictor],
                        "coefficient_mm_day_per_sd": fit.params[predictor] * SECONDS_PER_DAY,
                        "ci95_low_mm_day_per_sd": ci.iloc[0] * SECONDS_PER_DAY,
                        "ci95_high_mm_day_per_sd": ci.iloc[1] * SECONDS_PER_DAY,
                        "p_hac": fit.pvalues[predictor],
                        "n_years": int(fit.nobs),
                    }
                )
    models = pd.DataFrame(rows)
    role = {
        "area_convergence_anom_kg_m2_s": "primary_fixed_box_divergence",
        "boundary_convergence_anom_kg_m2_s": "budget_validation_boundary_flux",
        "centered_convergence_anom_kg_m2_s": "spatial_sensitivity_genesis_centered",
    }
    models["analysis_role"] = models.response.map(role)
    models["p_fdr_bh_within_predictor_response"] = models.groupby(
        ["predictor", "response"]
    ).p_hac.transform(bh_adjust)
    models["p_fdr_bh_all_responses_within_predictor"] = models.groupby(
        "predictor"
    ).p_hac.transform(bh_adjust)
    models.to_csv(OUT / "jja_event_moisture_window_models.csv", index=False)

    joint_rows = []
    predictors = ["wnpac_z850_gpm_dt_z", "iobm_40_100e_c_dt_z"]
    for window, (lower, upper) in windows.items():
        sample = year_lag.loc[year_lag.lag_hours.between(lower, upper)].groupby(
            "season_year", as_index=False
        ).agg({**{metric: "mean" for metric in metrics}, **{name: "first" for name in predictors}})
        for metric in metrics:
            fit = sm.OLS(sample[metric], sm.add_constant(sample[predictors])).fit(
                cov_type="HAC", cov_kwds={"maxlags": 3}
            )
            for predictor in predictors:
                ci = fit.conf_int().loc[predictor]
                joint_rows.append(
                    {
                        "window": window,
                        "response": metric,
                        "predictor": predictor,
                        "coefficient_mm_day_per_sd": fit.params[predictor] * SECONDS_PER_DAY,
                        "ci95_low_mm_day_per_sd": ci.iloc[0] * SECONDS_PER_DAY,
                        "ci95_high_mm_day_per_sd": ci.iloc[1] * SECONDS_PER_DAY,
                        "p_hac": fit.pvalues[predictor],
                        "n_years": int(fit.nobs),
                        "analysis_role": role[metric],
                    }
                )
    joint_models = pd.DataFrame(joint_rows)
    joint_models["p_fdr_bh_within_predictor_response"] = joint_models.groupby(
        ["predictor", "response"]
    ).p_hac.transform(bh_adjust)
    joint_models.to_csv(OUT / "jja_event_moisture_window_joint_models.csv", index=False)

    rng = np.random.default_rng(SEED)
    composite_rows = []
    for predictor in ["wnpac_z850_gpm_dt_z", "iobm_40_100e_c_dt_z"]:
        annual_exposure = year_lag[["season_year", predictor]].drop_duplicates()
        low_cut, high_cut = annual_exposure[predictor].quantile([1 / 3, 2 / 3])
        low_years = annual_exposure.loc[annual_exposure[predictor] <= low_cut, "season_year"].to_numpy()
        high_years = annual_exposure.loc[annual_exposure[predictor] >= high_cut, "season_year"].to_numpy()
        for lag in LAGS:
            lag_data = year_lag.loc[year_lag.lag_hours.eq(lag)].set_index("season_year")
            for metric in metrics:
                low = lag_data.loc[low_years, metric].to_numpy()
                high = lag_data.loc[high_years, metric].to_numpy()
                estimate = high.mean() - low.mean()
                draws = np.empty(BOOTSTRAPS)
                for rep in range(BOOTSTRAPS):
                    draws[rep] = rng.choice(high, size=len(high), replace=True).mean() - rng.choice(
                        low, size=len(low), replace=True
                    ).mean()
                lower, upper = np.quantile(draws, [0.025, 0.975])
                composite_rows.append(
                    {
                        "predictor": predictor,
                        "lag_hours": lag,
                        "response": metric,
                        "high_minus_low_mm_day": estimate * SECONDS_PER_DAY,
                        "pointwise_ci95_low_mm_day": lower * SECONDS_PER_DAY,
                        "pointwise_ci95_high_mm_day": upper * SECONDS_PER_DAY,
                        "n_high_years": len(high),
                        "n_low_years": len(low),
                    }
                )
    composites = pd.DataFrame(composite_rows)
    composites.to_csv(OUT / "jja_event_moisture_high_low_composites.csv", index=False)
    return year_lag, models, composites


def quality_audit(time_metrics: pd.DataFrame, event_metrics: pd.DataFrame) -> pd.DataFrame:
    paired = time_metrics[["area_convergence_anom_kg_m2_s", "boundary_convergence_anom_kg_m2_s"]].dropna()
    audit = pd.DataFrame(
        [
            {
                "check": "boundary_vs_area_divergence_correlation",
                "value": paired.corr().iloc[0, 1],
                "n": len(paired),
            },
            {
                "check": "event_lag_rows",
                "value": len(event_metrics),
                "n": event_metrics.event_id.nunique(),
            },
            {
                "check": "max_abs_anchor_rounding_hours",
                "value": event_metrics.rounding_offset_hours.abs().max(),
                "n": event_metrics.event_id.nunique(),
            },
        ]
    )
    audit.to_csv(OUT / "jja_event_moisture_quality_audit.csv", index=False)
    return audit


def plot_composites(composites: pd.DataFrame) -> None:
    labels = {
        "area_convergence_anom_kg_m2_s": "Fixed-box -divergence",
        "boundary_convergence_anom_kg_m2_s": "Fixed-box boundary influx",
        "centered_convergence_anom_kg_m2_s": "Genesis-centered -divergence",
    }
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True, constrained_layout=True)
    for row_idx, predictor in enumerate(["wnpac_z850_gpm_dt_z", "iobm_40_100e_c_dt_z"]):
        for ax, (metric, label) in zip(axes[row_idx], labels.items()):
            sample = composites.loc[(composites.predictor.eq(predictor)) & (composites.response.eq(metric))]
            ax.plot(sample.lag_hours, sample.high_minus_low_mm_day, color="#c43c35", marker="o")
            ax.fill_between(
                sample.lag_hours,
                sample.pointwise_ci95_low_mm_day,
                sample.pointwise_ci95_high_mm_day,
                color="#c43c35",
                alpha=0.2,
            )
            ax.axhline(0, color="0.3", linewidth=1)
            ax.axvline(0, color="0.3", linestyle="--", linewidth=1)
            ax.set_title(label)
            ax.set_ylabel(("High-minus-low WNPAC" if row_idx == 0 else "High-minus-low IOBM") + " (mm day$^{-1}$)")
            ax.grid(alpha=0.25)
    for ax in axes[-1]:
        ax.set_xlabel("Hours relative to rounded genesis time")
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "jja_event_moisture_composites.png", dpi=180)
    plt.close(fig)


def main() -> None:
    paths = expected_paths()
    climatology = load_or_build_climatology(paths)
    time_metrics, event_metrics = event_composites(paths, climatology)
    _, models, composites = inference(event_metrics)
    audit = quality_audit(time_metrics, event_metrics)
    plot_composites(composites)
    print("Quality audit")
    print(audit.to_string(index=False))
    print("\nWindow models")
    print(models.to_string(index=False))


if __name__ == "__main__":
    main()
