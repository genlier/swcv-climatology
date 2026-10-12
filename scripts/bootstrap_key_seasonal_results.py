"""Moving-block bootstrap for preselected seasonal SWCV results."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "processed"
RNG_SEED = 20260905
N_BOOTSTRAP = 2000


def moving_block_indices(n: int, length: int, rng: np.random.Generator) -> np.ndarray:
    starts = rng.integers(0, n - length + 1, size=int(np.ceil(n / length)))
    return np.concatenate([np.arange(start, start + length) for start in starts])[:n]


def coefficient(data: pd.DataFrame, response: str, predictors: list[str], target: str, family: str) -> float:
    x = sm.add_constant(data[predictors], has_constant="add")
    if family == "poisson":
        fit = sm.GLM(data[response], x, family=sm.families.Poisson()).fit(maxiter=100, disp=0)
    else:
        fit = sm.OLS(data[response], x).fit()
    return float(fit.params[target])


def bootstrap_model(
    data: pd.DataFrame,
    name: str,
    response: str,
    predictors: list[str],
    target: str,
    family: str,
    scale: float,
) -> list[dict]:
    data = data[[response, *predictors]].dropna().reset_index(drop=True)
    estimate = coefficient(data, response, predictors, target, family) * scale
    rows = []
    for block_length in (3, 5):
        rng = np.random.default_rng(RNG_SEED + block_length)
        draws = []
        failures = 0
        for _ in range(N_BOOTSTRAP):
            indices = moving_block_indices(len(data), block_length, rng)
            try:
                draws.append(coefficient(data.iloc[indices], response, predictors, target, family) * scale)
            except Exception:
                failures += 1
        draw = np.asarray(draws)
        pvalue = min(1.0, 2 * min((np.count_nonzero(draw <= 0) + 1) / (len(draw) + 1), (np.count_nonzero(draw >= 0) + 1) / (len(draw) + 1)))
        rows.append(
            {
                "result": name,
                "response": response,
                "family": family,
                "predictors": "+".join(predictors),
                "target": target,
                "scale": scale,
                "estimate": estimate,
                "block_length_years": block_length,
                "bootstrap_replicates_requested": N_BOOTSTRAP,
                "bootstrap_successes": len(draw),
                "bootstrap_failures": failures,
                "ci95_low": float(np.quantile(draw, 0.025)),
                "ci95_high": float(np.quantile(draw, 0.975)),
                "p_two_sided_bootstrap": float(pvalue),
                "effect_percent": float((np.exp(estimate) - 1) * 100) if family == "poisson" else np.nan,
                "effect_ci95_low_percent": float((np.exp(np.quantile(draw, 0.025)) - 1) * 100) if family == "poisson" else np.nan,
                "effect_ci95_high_percent": float((np.exp(np.quantile(draw, 0.975)) - 1) * 100) if family == "poisson" else np.nan,
            }
        )
    return rows


def main() -> None:
    annual = pd.read_csv(OUT / "seasonal_joint_metrics_1980_2023.csv")
    annual["year_centered"] = annual.season_year - annual.season_year.mean()
    djf = annual.loc[annual.season.eq("DJF")].sort_values("season_year")
    jja = annual.loc[annual.season.eq("JJA")].sort_values("season_year")
    jobs = [
        (djf, "DJF frequency trend", "frequency", ["year_centered"], "year_centered", "poisson", 10.0),
        (djf, "DJF q90 height trend", "q90_height_lifecycle", ["year_centered"], "year_centered", "ols", 10.0),
        (djf, "DJF q90 vorticity trend", "q90_vorticity_lifecycle", ["year_centered"], "year_centered", "ols", 10.0),
        (jja, "JJA frequency IOBM100", "frequency", ["gmst_anomaly_c", "iobm_40_100e_c_dt_z"], "iobm_40_100e_c_dt_z", "poisson", 1.0),
        (jja, "JJA frequency IOBM110", "frequency", ["gmst_anomaly_c", "iobm_40_110e_c_dt_z"], "iobm_40_110e_c_dt_z", "poisson", 1.0),
        (jja, "JJA frequency independent IOBM100", "frequency", ["gmst_anomaly_c", "nino34_prev_djf_dt_z", "iobm_independent_dt_z"], "iobm_independent_dt_z", "poisson", 1.0),
        (jja, "JJA frequency independent IOBM110", "frequency", ["gmst_anomaly_c", "nino34_prev_djf_dt_z", "iobm110_independent_dt_z"], "iobm110_independent_dt_z", "poisson", 1.0),
        (jja, "JJA eastward escape IOBM100", "eastward_escape_fraction", ["gmst_anomaly_c", "iobm_40_100e_c_dt_z"], "iobm_40_100e_c_dt_z", "ols", 100.0),
    ]
    rows = []
    for job in jobs:
        rows.extend(bootstrap_model(*job))
        print(f"completed {job[1]}", flush=True)
    result = pd.DataFrame(rows)
    result.to_csv(OUT / "key_seasonal_moving_block_bootstrap.csv", index=False)
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
