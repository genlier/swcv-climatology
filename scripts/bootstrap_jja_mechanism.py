"""Moving-block bootstrap for preselected JJA mechanism-chain coefficients."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "processed"
REPLICATES = 2000
SEED = 20260906

SPECS = [
    ("IOBM_to_WNPAC", "ols", "wnpac_z850_gpm_dt_z", ["iobm_40_100e_c_dt_z"], "iobm_40_100e_c_dt_z"),
    (
        "WNPAC_to_moist_inflow_proxy",
        "ols",
        "southwest_inflow_qv850_proxy_dt_z",
        ["wnpac_z850_gpm_dt_z"],
        "wnpac_z850_gpm_dt_z",
    ),
    ("frequency_on_IOBM", "poisson", "frequency", ["iobm_40_100e_c_dt_z"], "iobm_40_100e_c_dt_z"),
    (
        "frequency_on_WNPAC_and_MONC_WNPAC",
        "poisson",
        "frequency",
        ["wnpac_z850_gpm_dt_z", "monc_cyclonic_dt_z"],
        "wnpac_z850_gpm_dt_z",
    ),
    (
        "frequency_on_WNPAC_and_MONC_MONC",
        "poisson",
        "frequency",
        ["wnpac_z850_gpm_dt_z", "monc_cyclonic_dt_z"],
        "monc_cyclonic_dt_z",
    ),
    (
        "frequency_on_IOBM_and_WNPAC_IOBM",
        "poisson",
        "frequency",
        ["iobm_40_100e_c_dt_z", "wnpac_z850_gpm_dt_z"],
        "iobm_40_100e_c_dt_z",
    ),
    (
        "frequency_on_IOBM_and_WNPAC_WNPAC",
        "poisson",
        "frequency",
        ["iobm_40_100e_c_dt_z", "wnpac_z850_gpm_dt_z"],
        "wnpac_z850_gpm_dt_z",
    ),
]


def block_indices(n: int, length: int, rng: np.random.Generator) -> np.ndarray:
    starts = rng.integers(0, n, size=int(np.ceil(n / length)))
    return np.concatenate([(start + np.arange(length)) % n for start in starts])[:n]


def fit_coefficient(data: pd.DataFrame, family: str, response: str, predictors: list[str], target: str) -> float:
    x = sm.add_constant(data[predictors])
    if family == "ols":
        fit = sm.OLS(data[response], x).fit()
    else:
        fit = sm.GLM(data[response], x, family=sm.families.Poisson()).fit()
    return float(fit.params[target])


def main() -> None:
    indices = pd.read_csv(OUT / "jja_era5_mechanism_indices_1979_2023.csv")
    seasonal = pd.read_csv(OUT / "seasonal_joint_metrics_1980_2023.csv")
    data = seasonal.loc[seasonal.season.eq("JJA")].merge(indices, on="season_year", validate="one_to_one")
    data = data.sort_values("season_year").reset_index(drop=True)
    rng = np.random.default_rng(SEED)
    rows = []
    for label, family, response, predictors, target in SPECS:
        columns = [response, *predictors]
        sample = data[columns].dropna().reset_index(drop=True)
        estimate = fit_coefficient(sample, family, response, predictors, target)
        for length in (3, 5):
            values = []
            failures = 0
            for _ in range(REPLICATES):
                draw = sample.iloc[block_indices(len(sample), length, rng)]
                try:
                    values.append(fit_coefficient(draw, family, response, predictors, target))
                except Exception:
                    failures += 1
            values_array = np.asarray(values)
            lower, upper = np.quantile(values_array, [0.025, 0.975])
            left = (np.count_nonzero(values_array <= 0) + 1) / (len(values_array) + 1)
            right = (np.count_nonzero(values_array >= 0) + 1) / (len(values_array) + 1)
            row = {
                "result": label,
                "family": family,
                "response": response,
                "predictors": "+".join(predictors),
                "target": target,
                "estimate": estimate,
                "block_length_years": length,
                "bootstrap_successes": len(values_array),
                "bootstrap_failures": failures,
                "ci95_low": lower,
                "ci95_high": upper,
                "p_two_sided_bootstrap": min(1.0, 2 * min(left, right)),
            }
            if family == "poisson":
                row.update(
                    {
                        "effect_percent": 100 * np.expm1(estimate),
                        "effect_ci95_low_percent": 100 * np.expm1(lower),
                        "effect_ci95_high_percent": 100 * np.expm1(upper),
                    }
                )
            rows.append(row)
    result = pd.DataFrame(rows)
    result.to_csv(OUT / "jja_mechanism_moving_block_bootstrap.csv", index=False)
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
