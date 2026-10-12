"""Diagnose JJA circumglobal teleconnection (CGT) definitions and SWCV links."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import xarray as xr


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "era5_jja_cgt_monthly"
OUT = ROOT / "data" / "processed"
FIG = ROOT / "figures"
G = 9.80665
YEARS = np.arange(1980, 2024)
BOOTSTRAPS = 2000
SEED = 20260908


def area_mean(field: xr.DataArray, north: float, south: float, west: float, east: float) -> xr.DataArray:
    subset = field.sel(latitude=slice(north, south), longitude=slice(west, east))
    return subset.weighted(np.cos(np.deg2rad(subset.latitude))).mean(("latitude", "longitude"))


def detrend(field: xr.DataArray) -> xr.DataArray:
    x = xr.DataArray(YEARS - YEARS.mean(), dims="year", coords={"year": YEARS})
    centered = field - field.mean("year")
    slope = (centered * x).sum("year") / (x**2).sum("year")
    return centered - slope * x


def zscore(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return (values - values.mean()) / values.std(ddof=0)


def load_fields() -> xr.Dataset:
    paths = sorted(RAW.glob("era5_jja_cgt_monthly_*.nc"))
    if len(paths) != 3:
        raise FileNotFoundError(f"Expected 3 CGT chunks, found {len(paths)}")
    loaded = []
    for path in paths:
        with xr.open_dataset(path) as source:
            ds = source.load()
        time_name = "valid_time" if "valid_time" in ds.coords else "time"
        level_name = "pressure_level" if "pressure_level" in ds.coords else "level"
        if level_name in ds.dims:
            ds = ds.sel({level_name: 200}, drop=True)
        loaded.append(ds.rename({time_name: "time"}) if time_name != "time" else ds)
    monthly = xr.concat(loaded, dim="time").sortby("time")
    annual = monthly.groupby("time.year").mean("time").sel(year=YEARS)
    annual["height"] = annual.z / G
    baseline = annual.sel(year=slice(1991, 2020)).mean("year")
    anomalies = annual[["height", "u", "v"]] - baseline[["height", "u", "v"]]
    return xr.Dataset({name: detrend(anomalies[name]) for name in ["height", "u", "v"]})


def eof_indices(fields: xr.Dataset) -> tuple[pd.DataFrame, xr.Dataset, dict[str, object]]:
    belt = fields.v.sel(latitude=slice(60, 20))
    lat_weights = np.sqrt(np.cos(np.deg2rad(belt.latitude.values)))
    matrix = belt.values.reshape(len(YEARS), -1)
    weights = np.repeat(lat_weights, belt.sizes["longitude"])
    weighted = matrix * weights[None, :]
    u, singular, vt = np.linalg.svd(weighted, full_matrices=False)
    eigenvalues = singular**2 / (len(YEARS) - 1)
    explained = eigenvalues / eigenvalues.sum()
    pcs = u * singular[None, :]

    dw_raw = area_mean(fields.height, 40, 35, 60, 70).values
    dw = zscore(dw_raw)
    pc1 = zscore(pcs[:, 0])
    pc2 = zscore(pcs[:, 1])
    # EOF sign is arbitrary. Orient EOF2 so its regressed Z200 in the fixed
    # Ding--Wang box is positive; this rule does not use SWCV frequency.
    if np.corrcoef(pc2, dw)[0, 1] < 0:
        pc2 *= -1
        vt[1] *= -1

    indices = pd.DataFrame(
        {"season_year": YEARS, "cgt_dw_z200_dt_z": dw, "v200_eof1_dt_z": pc1, "v200_eof2_dt_z": pc2}
    )
    eof_fields = xr.Dataset(
        {
            "v200_eof1_weighted_loading": (("latitude", "longitude"), vt[0].reshape(belt.shape[1:])),
            "v200_eof2_weighted_loading": (("latitude", "longitude"), vt[1].reshape(belt.shape[1:])),
        },
        coords={"latitude": belt.latitude, "longitude": belt.longitude},
    )
    north_error = np.sqrt(2 / len(YEARS))
    diagnostics = {
        "analysis_years": [int(YEARS[0]), int(YEARS[-1])],
        "eof_domain": {"latitude": [20, 60], "longitude": [0, 357.5]},
        "area_weighting": "sqrt(cos(latitude))",
        "explained_variance_fraction_eof1": float(explained[0]),
        "explained_variance_fraction_eof2": float(explained[1]),
        "north_rule_relative_sampling_error": float(north_error),
        "eof1_interval": [float(eigenvalues[0] * (1 - north_error)), float(eigenvalues[0] * (1 + north_error))],
        "eof2_interval": [float(eigenvalues[1] * (1 - north_error)), float(eigenvalues[1] * (1 + north_error))],
        "eof1_eof2_north_rule_separated": bool(eigenvalues[1] * (1 + north_error) < eigenvalues[0] * (1 - north_error)),
        "correlation_dw_eof2": float(np.corrcoef(dw, pc2)[0, 1]),
        "sign_orientation": "EOF2 sign chosen for positive Z200 regression in 35-40N, 60-70E; no SWCV data used",
    }
    return indices, eof_fields, diagnostics


def regression_map(field: xr.DataArray, predictor: np.ndarray) -> xr.DataArray:
    x = xr.DataArray(zscore(predictor), dims="year", coords={"year": YEARS})
    return (field * x).mean("year") / (x**2).mean("year")


def pattern_correlation(a: xr.DataArray, b: xr.DataArray, south: float = 20, north: float = 60) -> float:
    aa = a.sel(latitude=slice(north, south)).values
    bb = b.sel(latitude=slice(north, south)).values
    lat = a.sel(latitude=slice(north, south)).latitude.values
    weights = np.cos(np.deg2rad(lat))[:, None] * np.ones((1, aa.shape[1]))
    mask = np.isfinite(aa) & np.isfinite(bb)
    w = weights[mask]
    av = aa[mask] - np.average(aa[mask], weights=w)
    bv = bb[mask] - np.average(bb[mask], weights=w)
    return float(np.sum(w * av * bv) / np.sqrt(np.sum(w * av**2) * np.sum(w * bv**2)))


def wave5_power_fraction(field: xr.DataArray) -> float:
    belt = field.sel(latitude=slice(45, 35))
    series = belt.weighted(np.cos(np.deg2rad(belt.latitude))).mean("latitude").values
    series = series - series.mean()
    power = np.abs(np.fft.rfft(series)) ** 2
    return float(power[5] / power[1:11].sum())


def bh_adjust(values: pd.Series) -> pd.Series:
    out = pd.Series(np.nan, index=values.index)
    valid = values.dropna().sort_values()
    n = len(valid)
    adjusted = (valid * n / np.arange(1, n + 1)).iloc[::-1].cummin().iloc[::-1].clip(upper=1)
    out.loc[adjusted.index] = adjusted
    return out


def fit_ols(data: pd.DataFrame, response: str, predictors: list[str], label: str) -> list[dict]:
    sample = data[[response, *predictors]].dropna()
    fit = sm.OLS(sample[response], sm.add_constant(sample[predictors])).fit(
        cov_type="HAC", cov_kwds={"maxlags": 3}
    )
    rows = []
    for predictor in predictors:
        ci = fit.conf_int().loc[predictor]
        rows.append(
            {"model": label, "family": "OLS-HAC3", "response": response, "predictor": predictor,
             "coefficient": fit.params[predictor], "ci95_low": ci.iloc[0], "ci95_high": ci.iloc[1],
             "p_hac": fit.pvalues[predictor], "n_years": int(fit.nobs), "effect_percent": np.nan,
             "effect_ci95_low_percent": np.nan, "effect_ci95_high_percent": np.nan}
        )
    return rows


def fit_poisson(data: pd.DataFrame, predictors: list[str], label: str) -> list[dict]:
    sample = data[["frequency", *predictors]].dropna()
    fit = sm.GLM(sample.frequency, sm.add_constant(sample[predictors]), family=sm.families.Poisson()).fit(
        cov_type="HAC", cov_kwds={"maxlags": 3}
    )
    rows = []
    for predictor in predictors:
        ci = fit.conf_int().loc[predictor]
        rows.append(
            {"model": label, "family": "Poisson-HAC3", "response": "frequency", "predictor": predictor,
             "coefficient": fit.params[predictor], "ci95_low": ci.iloc[0], "ci95_high": ci.iloc[1],
             "p_hac": fit.pvalues[predictor], "n_years": int(fit.nobs),
             "effect_percent": 100 * np.expm1(fit.params[predictor]),
             "effect_ci95_low_percent": 100 * np.expm1(ci.iloc[0]),
             "effect_ci95_high_percent": 100 * np.expm1(ci.iloc[1])}
        )
    return rows


def fit_models(indices: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    seasonal = pd.read_csv(OUT / "seasonal_joint_metrics_1980_2023.csv")
    climate = seasonal.loc[seasonal.season.eq("JJA"), ["season_year", "frequency", "iobm_40_100e_c_dt_z"]]
    mechanism = pd.read_csv(OUT / "jja_era5_mechanism_indices_1979_2023.csv")
    data = climate.merge(mechanism[["season_year", "monc_cyclonic_dt_z", "wnpac_z850_gpm_dt_z"]], on="season_year")
    data = data.merge(indices, on="season_year")
    rows = []
    rows += fit_ols(data, "monc_cyclonic_dt_z", ["cgt_dw_z200_dt_z"], "MONC_on_CGT_DW")
    rows += fit_ols(data, "monc_cyclonic_dt_z", ["v200_eof2_dt_z"], "MONC_on_V200_EOF2")
    rows += fit_poisson(data, ["cgt_dw_z200_dt_z"], "frequency_on_CGT_DW")
    rows += fit_poisson(data, ["v200_eof2_dt_z"], "frequency_on_V200_EOF2")
    rows += fit_poisson(data, ["wnpac_z850_gpm_dt_z", "cgt_dw_z200_dt_z"], "frequency_on_WNPAC_and_CGT_DW")
    rows += fit_poisson(data, ["wnpac_z850_gpm_dt_z", "v200_eof2_dt_z"], "frequency_on_WNPAC_and_V200_EOF2")
    models = pd.DataFrame(rows)
    models["p_fdr_bh_model_family"] = bh_adjust(models.p_hac)
    models.to_csv(OUT / "jja_cgt_mechanism_models.csv", index=False)
    data.to_csv(OUT / "jja_cgt_indices_1980_2023.csv", index=False)

    rng = np.random.default_rng(SEED)
    boot_rows = []
    for row in models.itertuples():
        predictors = {
            "MONC_on_CGT_DW": ["cgt_dw_z200_dt_z"],
            "MONC_on_V200_EOF2": ["v200_eof2_dt_z"],
            "frequency_on_CGT_DW": ["cgt_dw_z200_dt_z"],
            "frequency_on_V200_EOF2": ["v200_eof2_dt_z"],
            "frequency_on_WNPAC_and_CGT_DW": ["wnpac_z850_gpm_dt_z", "cgt_dw_z200_dt_z"],
            "frequency_on_WNPAC_and_V200_EOF2": ["wnpac_z850_gpm_dt_z", "v200_eof2_dt_z"],
        }[row.model]
        sample = data[[row.response, *predictors]].dropna().reset_index(drop=True)
        for length in (3, 5):
            draws = []
            n = len(sample)
            for _ in range(BOOTSTRAPS):
                starts = rng.integers(0, n, size=int(np.ceil(n / length)))
                idx = np.concatenate([(start + np.arange(length)) % n for start in starts])[:n]
                draw = sample.iloc[idx]
                x = sm.add_constant(draw[predictors])
                if row.family == "OLS-HAC3":
                    value = sm.OLS(draw[row.response], x).fit().params[row.predictor]
                else:
                    value = sm.GLM(draw.frequency, x, family=sm.families.Poisson()).fit().params[row.predictor]
                draws.append(value)
            low, high = np.quantile(draws, [0.025, 0.975])
            boot_rows.append(
                {"model": row.model, "response": row.response, "predictor": row.predictor,
                 "block_length_years": length, "estimate": row.coefficient,
                 "ci95_low": low, "ci95_high": high, "bootstrap_replicates": BOOTSTRAPS}
            )
    boot = pd.DataFrame(boot_rows)
    boot.to_csv(OUT / "jja_cgt_moving_block_bootstrap.csv", index=False)
    return data, models


def spatial_diagnostics(fields: xr.Dataset, data: pd.DataFrame) -> tuple[xr.Dataset, pd.DataFrame]:
    predictors = {
        "frequency": data.frequency.values,
        "monc_cyclonic": data.monc_cyclonic_dt_z.values,
        "cgt_dw": data.cgt_dw_z200_dt_z.values,
        "v200_eof2": data.v200_eof2_dt_z.values,
    }
    maps = xr.Dataset({f"z200_on_{name}": regression_map(fields.height, values) for name, values in predictors.items()})
    rows = []
    for name in predictors:
        field = maps[f"z200_on_{name}"]
        rows.append({"map": name, "wave5_power_fraction_35_45n": wave5_power_fraction(field)})
    for name in ["frequency", "monc_cyclonic", "v200_eof2"]:
        rows.append(
            {"map": f"pattern_correlation_{name}_vs_cgt_dw",
             "pattern_correlation_20_60n": pattern_correlation(maps[f"z200_on_{name}"], maps.z200_on_cgt_dw)}
        )
    rows.append(
        {"map": "pattern_correlation_frequency_vs_v200_eof2",
         "pattern_correlation_20_60n": pattern_correlation(maps.z200_on_frequency, maps.z200_on_v200_eof2)}
    )
    rows.append(
        {"map": "pattern_correlation_monc_cyclonic_vs_v200_eof2",
         "pattern_correlation_20_60n": pattern_correlation(maps.z200_on_monc_cyclonic, maps.z200_on_v200_eof2)}
    )
    diagnostics = pd.DataFrame(rows)
    diagnostics.to_csv(OUT / "jja_cgt_spatial_diagnostics.csv", index=False)
    maps.to_netcdf(OUT / "jja_cgt_z200_regression_maps.nc")
    return maps, diagnostics


def plots(maps: xr.Dataset, data: pd.DataFrame) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    panels = [
        ("z200_on_frequency", "Z200 on SWCV frequency"),
        ("z200_on_monc_cyclonic", "Z200 on MONC cyclonic strength"),
        ("z200_on_cgt_dw", "Z200 on Ding–Wang CGTI"),
        ("z200_on_v200_eof2", "Z200 on V200 EOF2"),
    ]
    values = np.concatenate([maps[name].sel(latitude=slice(70, 10)).values.ravel() for name, _ in panels])
    limit = np.nanpercentile(np.abs(values), 98)
    fig, axes = plt.subplots(2, 2, figsize=(15, 7.5), sharex=True, sharey=True, constrained_layout=True)
    for ax, (name, title) in zip(axes.flat, panels):
        field = maps[name].sel(latitude=slice(70, 10))
        mesh = ax.pcolormesh(field.longitude, field.latitude, field, cmap="RdBu_r", vmin=-limit, vmax=limit, shading="auto")
        ax.contour(field.longitude, field.latitude, field, levels=np.linspace(-limit, limit, 9), colors="0.25", linewidths=0.35)
        ax.plot([60, 70, 70, 60, 60], [35, 35, 40, 40, 35], color="black", linewidth=1.2)
        ax.plot([85, 110, 110, 85, 85], [40, 40, 50, 50, 40], color="#f4a261", linewidth=1.2)
        ax.set_title(title)
        ax.set_xlim(0, 357.5); ax.set_ylim(10, 70)
        ax.set_xticks(np.arange(0, 361, 60)); ax.set_yticks(np.arange(10, 71, 10))
        ax.grid(alpha=0.15)
    fig.colorbar(mesh, ax=axes, label="gpm per predictor SD", shrink=0.9)
    fig.savefig(FIG / "jja_cgt_z200_regression_patterns.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True, constrained_layout=True)
    axes[0].plot(data.season_year, data.cgt_dw_z200_dt_z, label="Ding–Wang CGTI", linewidth=1.6)
    axes[0].plot(data.season_year, data.v200_eof2_dt_z, label="V200 EOF2", linewidth=1.4)
    axes[0].plot(data.season_year, data.monc_cyclonic_dt_z, label="MONC cyclonic", linewidth=1.2)
    axes[0].axhline(0, color="0.3", linewidth=0.8); axes[0].legend(ncol=3); axes[0].grid(alpha=0.25)
    axes[1].plot(data.season_year, zscore(data.frequency), label="SWCV frequency", color="#c43c35")
    axes[1].plot(data.season_year, data.wnpac_z850_gpm_dt_z, label="WNPAC", color="#2a6f97")
    axes[1].axhline(0, color="0.3", linewidth=0.8); axes[1].legend(ncol=2); axes[1].grid(alpha=0.25)
    axes[1].set_xlabel("JJA season year"); axes[0].set_ylabel("Detrended SD"); axes[1].set_ylabel("SD")
    fig.savefig(FIG / "jja_cgt_indices_timeseries.png", dpi=180)
    plt.close(fig)


def main() -> None:
    fields = load_fields()
    indices, eof_fields, eof_diagnostics = eof_indices(fields)
    data, models = fit_models(indices)
    maps, spatial = spatial_diagnostics(fields, data)
    eof_fields.to_netcdf(OUT / "jja_cgt_v200_eof_loadings.nc")
    (OUT / "jja_cgt_eof_diagnostics.json").write_text(json.dumps(eof_diagnostics, indent=2), encoding="utf-8")
    plots(maps, data)
    print("EOF diagnostics")
    print(json.dumps(eof_diagnostics, indent=2))
    print("\nMechanism models")
    print(models.to_string(index=False))
    print("\nSpatial diagnostics")
    print(spatial.to_string(index=False))


if __name__ == "__main__":
    main()
