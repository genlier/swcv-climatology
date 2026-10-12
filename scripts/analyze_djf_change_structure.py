"""Diagnose the temporal structure and source decomposition of DJF changes.

The primary endpoint is DJF frequency. Secondary endpoints are terrain-safe
annual P90 height deficit and vorticity. A one-break continuous hinge model is
searched over 1990--2013; its p-value is selection-adjusted by repeating the
complete breakpoint search under the linear null.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import theilslopes


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "processed"
FIG = ROOT / "figures"
YEARS = np.arange(1980, 2024)
BREAKS = np.arange(1990, 2014)
BOOTSTRAPS = 2000
SEED = 20260906


def bh_adjust(pvalues: pd.Series) -> pd.Series:
    result = pd.Series(np.nan, index=pvalues.index)
    valid = pvalues.dropna().sort_values()
    if valid.empty:
        return result
    n = len(valid)
    adjusted = (valid * n / np.arange(1, n + 1)).iloc[::-1].cummin().iloc[::-1].clip(upper=1)
    result.loc[adjusted.index] = adjusted
    return result


def circular_block_sample(values: np.ndarray, length: int, rng: np.random.Generator) -> np.ndarray:
    n = len(values)
    starts = rng.integers(0, n, size=int(np.ceil(n / length)))
    indices = np.concatenate([(start + np.arange(length)) % n for start in starts])[:n]
    return values[indices]


def annual_table() -> pd.DataFrame:
    events = pd.read_csv(OUT / "events_season_year_terrain_qc.csv.gz", parse_dates=["genesis_time"])
    events = events.loc[events.season.eq("DJF") & events.season_year.between(1980, 2023)].copy()
    events["genesis_month"] = events.genesis_time.dt.month
    base = pd.DataFrame({"season_year": YEARS})
    annual = events.groupby("season_year", as_index=False).agg(
        frequency=("event_id", "size"),
        q90_height_lifecycle=("max_height_deficit_gpm_700_c25", lambda x: x.quantile(0.9)),
        q90_vorticity_lifecycle=("max_vorticity_s_1_700_c25", lambda x: x.quantile(0.9)),
        median_genesis_lat=("genesis_lat", "median"),
        median_genesis_lon=("genesis_lon", "median"),
        median_max_lon=("max_lon", "median"),
        median_displacement_km=("displacement_km", "median"),
        eastward_escape_fraction=("eastward_escape", "mean"),
        mobile_eastward_fraction=("mobile_eastward", "mean"),
    )
    annual = base.merge(annual, on="season_year", how="left", validate="one_to_one")
    for month, label in [(12, "dec_frequency"), (1, "jan_frequency"), (2, "feb_frequency")]:
        counts = events.loc[events.genesis_month.eq(month)].groupby("season_year").size()
        annual[label] = annual.season_year.map(counts).fillna(0).astype(int)
    for cluster in (0, 1, 2):
        counts = events.loc[events.track_cluster.eq(cluster)].groupby("season_year").size()
        annual[f"cluster{cluster}_frequency"] = annual.season_year.map(counts).fillna(0).astype(int)
        annual[f"cluster{cluster}_fraction"] = annual[f"cluster{cluster}_frequency"] / annual.frequency
    annual["year_centered"] = annual.season_year - annual.season_year.mean()
    annual.to_csv(OUT / "djf_change_structure_annual.csv", index=False)
    return annual


def trend_row(data: pd.DataFrame, response: str, family: str, group: str) -> dict:
    sample = data[[response, "year_centered"]].dropna()
    x = sm.add_constant(sample[["year_centered"]])
    if family == "count":
        fit = sm.GLM(sample[response], x, family=sm.families.Poisson()).fit(
            cov_type="HAC", cov_kwds={"maxlags": 3}
        )
        ci = fit.conf_int().loc["year_centered"] * 10
        coefficient = fit.params["year_centered"] * 10
        return {
            "group": group,
            "response": response,
            "model": "Poisson-HAC3",
            "coefficient_per_decade": coefficient,
            "ci95_low": ci.iloc[0],
            "ci95_high": ci.iloc[1],
            "effect_percent_per_decade": 100 * np.expm1(coefficient),
            "effect_ci95_low_percent": 100 * np.expm1(ci.iloc[0]),
            "effect_ci95_high_percent": 100 * np.expm1(ci.iloc[1]),
            "p_hac": fit.pvalues["year_centered"],
            "dispersion": np.sum(fit.resid_pearson**2) / fit.df_resid,
            "n_years": int(fit.nobs),
        }
    fit = sm.OLS(sample[response], x).fit(cov_type="HAC", cov_kwds={"maxlags": 3})
    ci = fit.conf_int().loc["year_centered"] * 10
    return {
        "group": group,
        "response": response,
        "model": "OLS-HAC3",
        "coefficient_per_decade": fit.params["year_centered"] * 10,
        "ci95_low": ci.iloc[0],
        "ci95_high": ci.iloc[1],
        "p_hac": fit.pvalues["year_centered"],
        "r_squared": fit.rsquared,
        "n_years": int(fit.nobs),
    }


def decomposition_trends(annual: pd.DataFrame) -> pd.DataFrame:
    families = {
        "primary": (["frequency"], "count"),
        "month_counts": (["dec_frequency", "jan_frequency", "feb_frequency"], "count"),
        "cluster_counts": ([f"cluster{x}_frequency" for x in (0, 1, 2)], "count"),
        "cluster_fractions": ([f"cluster{x}_fraction" for x in (0, 1, 2)], "continuous"),
        "continuous_path": (
            [
                "median_genesis_lat", "median_genesis_lon", "median_max_lon",
                "median_displacement_km", "eastward_escape_fraction", "mobile_eastward_fraction",
            ],
            "continuous",
        ),
        "intensity_tail": (["q90_height_lifecycle", "q90_vorticity_lifecycle"], "continuous"),
    }
    rows = []
    for group, (responses, family) in families.items():
        for response in responses:
            rows.append(trend_row(annual, response, family, group))
    result = pd.DataFrame(rows)
    result["p_fdr_bh_within_group"] = result.groupby("group").p_hac.transform(bh_adjust)

    # Absolute OLS slopes make additive monthly/cluster contributions interpretable.
    total = sm.OLS(annual.frequency, sm.add_constant(annual.year_centered)).fit().params.year_centered * 10
    for group in ("month_counts", "cluster_counts"):
        mask = result.group.eq(group)
        for idx in result.index[mask]:
            response = result.loc[idx, "response"]
            slope = sm.OLS(annual[response], sm.add_constant(annual.year_centered)).fit().params.year_centered * 10
            result.loc[idx, "absolute_ols_change_per_decade"] = slope
            result.loc[idx, "share_of_total_ols_trend"] = slope / total
    result.to_csv(OUT / "djf_change_source_trends.csv", index=False)
    return result


def design(year: np.ndarray, breakpoint: int | None = None) -> np.ndarray:
    centered = year - year.mean()
    columns = [np.ones(len(year)), centered]
    if breakpoint is not None:
        columns.append(np.maximum(0, year - breakpoint))
    return np.column_stack(columns)


def select_break_poisson(year: np.ndarray, y: np.ndarray) -> tuple[sm.GLM, sm.GLM, int, float, float]:
    base = sm.GLM(y, design(year), family=sm.families.Poisson()).fit()
    candidates = []
    for breakpoint in BREAKS:
        fit = sm.GLM(y, design(year, breakpoint), family=sm.families.Poisson()).fit()
        candidates.append((fit.bic_llf, breakpoint, fit))
    _, breakpoint, selected = min(candidates, key=lambda item: item[0])
    improvement = base.deviance - selected.deviance
    return base, selected, breakpoint, improvement, base.bic_llf - selected.bic_llf


def select_break_ols(year: np.ndarray, y: np.ndarray) -> tuple[sm.OLS, sm.OLS, int, float, float]:
    base = sm.OLS(y, design(year)).fit()
    candidates = []
    for breakpoint in BREAKS:
        fit = sm.OLS(y, design(year, breakpoint)).fit()
        candidates.append((fit.bic, breakpoint, fit))
    _, breakpoint, selected = min(candidates, key=lambda item: item[0])
    improvement = base.ssr - selected.ssr
    return base, selected, breakpoint, improvement, base.bic - selected.bic


def breakpoint_tests(annual: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    rows = []
    for response, family in [
        ("frequency", "poisson"),
        ("q90_height_lifecycle", "ols"),
        ("q90_vorticity_lifecycle", "ols"),
    ]:
        sample = annual[["season_year", response]].dropna()
        year = sample.season_year.to_numpy(dtype=float)
        y = sample[response].to_numpy(dtype=float)
        if family == "poisson":
            base, selected, breakpoint, improvement, delta_bic = select_break_poisson(year, y)
            expected = base.predict(design(year))
            null_values = []
            for _ in range(BOOTSTRAPS):
                simulated = rng.poisson(expected)
                null_values.append(select_break_poisson(year, simulated)[3])
        else:
            base, selected, breakpoint, improvement, delta_bic = select_break_ols(year, y)
            fitted = base.predict(design(year))
            residuals = y - fitted
            null_values = []
            for _ in range(BOOTSTRAPS):
                simulated = fitted + circular_block_sample(residuals, 3, rng)
                null_values.append(select_break_ols(year, simulated)[3])
        null_values = np.asarray(null_values)
        hinge = selected.params[2]
        before = selected.params[1]
        after = before + hinge
        row = {
                "response": response,
                "family": family,
                "selected_breakpoint": int(breakpoint),
                "delta_bic_linear_minus_segmented": delta_bic,
                "improvement_statistic": improvement,
                "selection_adjusted_bootstrap_p": (np.count_nonzero(null_values >= improvement) + 1)
                / (BOOTSTRAPS + 1),
                "slope_before_break_per_decade": before * 10,
                "slope_after_break_per_decade": after * 10,
                "bootstrap_replicates": BOOTSTRAPS,
            }
        if family == "poisson":
            row["rate_change_before_percent_per_decade"] = 100 * np.expm1(before * 10)
            row["rate_change_after_percent_per_decade"] = 100 * np.expm1(after * 10)
        rows.append(row)
    result = pd.DataFrame(rows)
    result.to_csv(OUT / "djf_selected_breakpoint_tests.csv", index=False)
    return result


def robust_key_trends(annual: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(SEED + 1)
    rows = []
    x = annual.season_year.to_numpy(dtype=float)
    for response in ["frequency", "q90_height_lifecycle", "q90_vorticity_lifecycle"]:
        y = annual[response].to_numpy(dtype=float)
        valid = np.isfinite(y)
        xx, yy = x[valid], y[valid]
        slope, intercept, _, _ = theilslopes(yy, xx)
        fitted = intercept + slope * xx
        residuals = yy - fitted
        for length in (3, 5):
            values = []
            for _ in range(BOOTSTRAPS):
                simulated = fitted + circular_block_sample(residuals, length, rng)
                values.append(theilslopes(simulated, xx)[0] * 10)
            lower, upper = np.quantile(values, [0.025, 0.975])
            rows.append(
                {
                    "response": response,
                    "theil_sen_slope_per_decade": slope * 10,
                    "block_length_years": length,
                    "ci95_low": lower,
                    "ci95_high": upper,
                    "bootstrap_replicates": BOOTSTRAPS,
                }
            )
    result = pd.DataFrame(rows)
    result.to_csv(OUT / "djf_theil_sen_block_bootstrap.csv", index=False)
    return result


def plot_results(annual: pd.DataFrame, breaks: pd.DataFrame) -> None:
    from plot_djf_revision import make_figure
    make_figure(annual, breaks, pd.read_csv(OUT / 'djf_change_source_trends.csv'))


def main() -> None:
    annual = annual_table()
    trends = decomposition_trends(annual)
    breaks = breakpoint_tests(annual)
    robust = robust_key_trends(annual)
    plot_results(annual, breaks)
    print("\nTrend decomposition")
    print(trends.to_string(index=False))
    print("\nSelection-adjusted breakpoint tests")
    print(breaks.to_string(index=False))
    print("\nTheil-Sen residual-block bootstrap")
    print(robust.to_string(index=False))


if __name__ == "__main__":
    main()
