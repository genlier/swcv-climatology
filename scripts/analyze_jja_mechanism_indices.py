"""Test a parsimonious JJA IOBM–circulation–SWCV mechanism chain.

This analysis uses 44 annual JJA units. WNPAC and Mongolian-cyclone indices
follow the boxes reported by Sun et al. (2026). Moisture transport is explicitly
labelled a monthly-mean proxy because q*v formed from monthly means omits
submonthly covariance.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import xarray as xr


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "era5_jja_mechanism_monthly"
OUT = ROOT / "data" / "processed"
FIG = ROOT / "figures"
GRAVITY = 9.80665


def area_mean(field: xr.DataArray, north: float, south: float, west: float, east: float) -> float:
    subset = field.sel(latitude=slice(north, south), longitude=slice(west, east))
    weights = np.cos(np.deg2rad(subset.latitude))
    return float(subset.weighted(weights).mean(("latitude", "longitude"), skipna=True))


def detrended_z(series: pd.Series, year: pd.Series, fit_mask: pd.Series) -> pd.Series:
    """Remove the 1980--2023 trend and standardize on that analysis period."""
    fit = sm.OLS(series.loc[fit_mask], sm.add_constant(year.loc[fit_mask].astype(float))).fit()
    residual = series - (fit.params.iloc[0] + fit.params.iloc[1] * year.astype(float))
    return (residual - residual.loc[fit_mask].mean()) / residual.loc[fit_mask].std(ddof=0)


def bh_adjust(pvalues: pd.Series) -> pd.Series:
    result = pd.Series(np.nan, index=pvalues.index)
    valid = pvalues.dropna().sort_values()
    n = len(valid)
    if not n:
        return result
    adjusted = (valid * n / np.arange(1, n + 1)).iloc[::-1].cummin().iloc[::-1].clip(upper=1)
    result.loc[adjusted.index] = adjusted
    return result


def load_era5_indices() -> pd.DataFrame:
    rows: list[dict[str, float | int]] = []
    for year in range(1979, 2024):
        path = RAW / f"era5_jja_mechanism_monthly_{year}.nc"
        if not path.exists():
            raise FileNotFoundError(f"Missing {path.name}; finish the mechanism download first")
        with xr.open_dataset(path) as ds:
            jja = ds.mean("valid_time")
            z850 = jja.z.sel(pressure_level=850) / GRAVITY
            z200 = jja.z.sel(pressure_level=200) / GRAVITY
            q850 = jja.q.sel(pressure_level=850)
            v850 = jja.v.sel(pressure_level=850)
            qv850 = q850 * v850
            rows.append(
                {
                    "season_year": year,
                    # WNPAC box and MONC box from Sun et al. (2026).
                    "wnpac_z850_gpm": area_mean(z850, 25, 15, 115, 150),
                    "monc_z200_gpm": area_mean(z200, 50, 40, 85, 110),
                    # Northward warm/moist inflow immediately south/east of SW China.
                    "southwest_inflow_qv850_proxy": area_mean(qv850, 27.5, 20, 100, 115),
                    "southwest_v850_ms": area_mean(v850, 27.5, 20, 100, 115),
                    "southwest_q850_kgkg": area_mean(q850, 27.5, 20, 100, 115),
                }
            )
    frame = pd.DataFrame(rows)
    baseline = frame.season_year.between(1991, 2020)
    analysis_period = frame.season_year.between(1980, 2023)
    for column in [name for name in frame if name != "season_year"]:
        frame[f"{column}_anom"] = frame[column] - frame.loc[baseline, column].mean()
        frame[f"{column}_dt_z"] = detrended_z(
            frame[f"{column}_anom"], frame.season_year, analysis_period
        )
    # Positive MONC strength denotes a deeper/lower 200-hPa geopotential anomaly.
    frame["monc_cyclonic_dt_z"] = -frame["monc_z200_gpm_dt_z"]
    frame.to_csv(OUT / "jja_era5_mechanism_indices_1979_2023.csv", index=False)
    return frame


def ols_hac(data: pd.DataFrame, response: str, predictors: list[str], label: str) -> list[dict]:
    sample = data[[response, *predictors]].dropna()
    fit = sm.OLS(sample[response], sm.add_constant(sample[predictors])).fit(
        cov_type="HAC", cov_kwds={"maxlags": 3}
    )
    rows = []
    for predictor in predictors:
        ci = fit.conf_int().loc[predictor]
        rows.append(
            {
                "model": label,
                "family": "OLS-HAC3",
                "response": response,
                "predictor": predictor,
                "coefficient": float(fit.params[predictor]),
                "ci95_low": float(ci.iloc[0]),
                "ci95_high": float(ci.iloc[1]),
                "p_hac": float(fit.pvalues[predictor]),
                "n_years": int(fit.nobs),
                "r_squared": float(fit.rsquared),
            }
        )
    return rows


def poisson_hac(data: pd.DataFrame, predictors: list[str], label: str) -> list[dict]:
    sample = data[["frequency", *predictors]].dropna()
    fit = sm.GLM(
        sample.frequency, sm.add_constant(sample[predictors]), family=sm.families.Poisson()
    ).fit(cov_type="HAC", cov_kwds={"maxlags": 3})
    rows = []
    for predictor in predictors:
        ci = fit.conf_int().loc[predictor]
        rows.append(
            {
                "model": label,
                "family": "Poisson-HAC3",
                "response": "frequency",
                "predictor": predictor,
                "coefficient": float(fit.params[predictor]),
                "ci95_low": float(ci.iloc[0]),
                "ci95_high": float(ci.iloc[1]),
                "p_hac": float(fit.pvalues[predictor]),
                "n_years": int(fit.nobs),
                "r_squared": np.nan,
                "effect_percent": float(100 * np.expm1(fit.params[predictor])),
                "effect_ci95_low_percent": float(100 * np.expm1(ci.iloc[0])),
                "effect_ci95_high_percent": float(100 * np.expm1(ci.iloc[1])),
                "poisson_dispersion": float(np.sum(fit.resid_pearson**2) / fit.df_resid),
            }
        )
    return rows


def fit_chain(indices: pd.DataFrame) -> pd.DataFrame:
    seasonal = pd.read_csv(OUT / "seasonal_joint_metrics_1980_2023.csv")
    jja = seasonal.loc[seasonal.season.eq("JJA")].merge(indices, on="season_year", validate="one_to_one")
    rows: list[dict] = []
    rows += ols_hac(jja, "wnpac_z850_gpm_dt_z", ["iobm_40_100e_c_dt_z"], "IOBM_to_WNPAC")
    rows += ols_hac(
        jja, "southwest_inflow_qv850_proxy_dt_z", ["wnpac_z850_gpm_dt_z"], "WNPAC_to_moist_inflow_proxy"
    )
    rows += poisson_hac(jja, ["iobm_40_100e_c_dt_z"], "frequency_on_IOBM")
    rows += poisson_hac(jja, ["wnpac_z850_gpm_dt_z"], "frequency_on_WNPAC")
    rows += poisson_hac(jja, ["monc_cyclonic_dt_z"], "frequency_on_MONC")
    rows += poisson_hac(
        jja, ["iobm_40_100e_c_dt_z", "wnpac_z850_gpm_dt_z"], "frequency_on_IOBM_and_WNPAC"
    )
    rows += poisson_hac(
        jja, ["wnpac_z850_gpm_dt_z", "monc_cyclonic_dt_z"], "frequency_on_WNPAC_and_MONC"
    )
    result = pd.DataFrame(rows)
    result["p_fdr_bh_chain_family"] = bh_adjust(result.p_hac)
    result.to_csv(OUT / "jja_iobm_circulation_mechanism_models.csv", index=False)

    FIG.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    panels = [
        ("iobm_40_100e_c_dt_z", "wnpac_z850_gpm_dt_z", "IOBM (SD)", "WNPAC Z850 (SD)"),
        ("wnpac_z850_gpm_dt_z", "frequency", "WNPAC Z850 (SD)", "JJA event count"),
        ("monc_cyclonic_dt_z", "frequency", "MONC cyclonic strength (SD)", "JJA event count"),
    ]
    for ax, (x_name, y_name, x_label, y_label) in zip(axes, panels):
        x = jja[x_name]
        y = jja[y_name]
        ax.scatter(x, y, alpha=0.8)
        slope, intercept = np.polyfit(x, y, 1)
        line_x = np.linspace(x.min(), x.max(), 100)
        ax.plot(line_x, intercept + slope * line_x, color="#c43c35", linewidth=2)
        ax.set(xlabel=x_label, ylabel=y_label)
        ax.grid(alpha=0.25)
    fig.savefig(FIG / "jja_mechanism_chain.png", dpi=180)
    plt.close(fig)
    return result


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    indices = load_era5_indices()
    result = fit_chain(indices)
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
