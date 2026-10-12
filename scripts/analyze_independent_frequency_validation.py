"""Validate SWCV frequency against published Yearbook and CRA-40 counts.

Published integer counts are transcribed from Sun et al. (2026), Figs. 2--3.
Figure 2 values are cross-checked against the totals reported in the article.
Figure 3 marker positions were independently checked against the integer y axis;
its internal inconsistencies are retained and reported rather than corrected.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import pearsonr, spearmanr


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "processed"
FIG = ROOT / "figures"
YEARS_FIG2 = np.arange(2012, 2023)
YEARS_FIG3 = np.arange(1979, 2025)
BOOTSTRAPS = 2000
SEED = 20260909

# Fig. 2 panel a. These sums reproduce the article text exactly: Yearbook 233,
# CRA-40 271, and ERA5 256 events during 2012--2022.
FIG2_COUNTS = {
    "yearbook": [27, 22, 26, 24, 17, 21, 15, 20, 22, 27, 12],
    "cra40_fig2": [20, 26, 29, 25, 21, 29, 20, 25, 27, 28, 21],
    "era5_sun_fig2": [17, 23, 33, 21, 21, 25, 17, 27, 31, 23, 18],
}

# Fig. 3 panel a, digitized at the integer-valued black markers. The plotted
# series sums to 1175, whereas the paper reports 1184 summer events. In 2022 it
# gives 25, whereas Fig. 2 and its reported 2012--2022 total imply 21. We do not
# silently alter either published representation.
FIG3_CRA40 = [
    29, 30, 30, 24, 36, 29, 23, 24, 30, 25, 23, 26, 24, 29, 27, 18,
    26, 22, 26, 25, 18, 32, 21, 20, 26, 26, 30, 20, 22, 23, 29, 23,
    27, 20, 26, 29, 25, 21, 29, 20, 25, 27, 28, 25, 27, 30,
]


def moving_block_indices(n: int, length: int, rng: np.random.Generator) -> np.ndarray:
    starts = rng.integers(0, n, size=int(np.ceil(n / length)))
    return np.concatenate([(start + np.arange(length)) % n for start in starts])[:n]


def paired_metrics(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    reference = np.asarray(reference, dtype=float)
    candidate = np.asarray(candidate, dtype=float)
    return {
        "n_years": len(reference),
        "pearson_r": pearsonr(reference, candidate).statistic,
        "pearson_p": pearsonr(reference, candidate).pvalue,
        "spearman_rho": spearmanr(reference, candidate).statistic,
        "spearman_p": spearmanr(reference, candidate).pvalue,
        "mean_bias_candidate_minus_reference": np.mean(candidate - reference),
        "mae": np.mean(np.abs(candidate - reference)),
        "rmse": np.sqrt(np.mean((candidate - reference) ** 2)),
        "sd_ratio_candidate_over_reference": np.std(candidate, ddof=1) / np.std(reference, ddof=1),
    }


def correlation_bootstrap(
    reference: np.ndarray, candidate: np.ndarray, lengths: tuple[int, ...], label: str
) -> list[dict[str, float]]:
    rng = np.random.default_rng(SEED + len(reference))
    rows = []
    reference = np.asarray(reference, dtype=float)
    candidate = np.asarray(candidate, dtype=float)
    for length in lengths:
        draws = []
        for _ in range(BOOTSTRAPS):
            idx = moving_block_indices(len(reference), length, rng)
            if np.std(reference[idx]) > 0 and np.std(candidate[idx]) > 0:
                draws.append(np.corrcoef(reference[idx], candidate[idx])[0, 1])
        low, high = np.quantile(draws, [0.025, 0.975])
        rows.append(
            {
                "comparison": label,
                "block_length_years": length,
                "pearson_r": np.corrcoef(reference, candidate)[0, 1],
                "ci95_low": low,
                "ci95_high": high,
                "bootstrap_replicates": BOOTSTRAPS,
            }
        )
    return rows


def poisson_hac(data: pd.DataFrame, response: str, predictors: list[str], label: str) -> list[dict]:
    fit = sm.GLM(
        data[response], sm.add_constant(data[predictors]), family=sm.families.Poisson()
    ).fit(cov_type="HAC", cov_kwds={"maxlags": 3})
    rows = []
    for predictor in predictors:
        low, high = fit.conf_int().loc[predictor]
        rows.append(
            {
                "model": label,
                "response": response,
                "predictor": predictor,
                "coefficient": fit.params[predictor],
                "ci95_low": low,
                "ci95_high": high,
                "p_hac": fit.pvalues[predictor],
                "effect_percent": 100 * np.expm1(fit.params[predictor]),
                "effect_ci95_low_percent": 100 * np.expm1(low),
                "effect_ci95_high_percent": 100 * np.expm1(high),
                "n_years": int(fit.nobs),
            }
        )
    return rows


def bh_adjust(values: pd.Series) -> pd.Series:
    ordered = values.sort_values()
    n = len(ordered)
    adjusted = (ordered * n / np.arange(1, n + 1)).iloc[::-1].cummin().iloc[::-1].clip(upper=1)
    return adjusted.reindex(values.index)


def mechanism_block_bootstrap(data: pd.DataFrame, models: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    rows = []
    predictor_sets = {
        "CRA40_frequency_on_WNPAC_and_CGT_DW": ["wnpac_z850_gpm_dt_z", "cgt_dw_z200_dt_z"],
        "CRA40_frequency_on_WNPAC_and_V200_EOF2": ["wnpac_z850_gpm_dt_z", "v200_eof2_dt_z"],
    }
    for row in models.itertuples():
        predictors = predictor_sets[row.model]
        for length in (3, 5):
            draws = []
            for _ in range(BOOTSTRAPS):
                idx = moving_block_indices(len(data), length, rng)
                draw = data.iloc[idx]
                fit = sm.GLM(
                    draw.cra40_fig3, sm.add_constant(draw[predictors]), family=sm.families.Poisson()
                ).fit()
                draws.append(fit.params[row.predictor])
            low, high = np.quantile(draws, [0.025, 0.975])
            rows.append(
                {
                    "model": row.model,
                    "predictor": row.predictor,
                    "block_length_years": length,
                    "estimate": row.coefficient,
                    "ci95_low": low,
                    "ci95_high": high,
                    "bootstrap_replicates": BOOTSTRAPS,
                }
            )
    return pd.DataFrame(rows)


def trend_row(year: np.ndarray, count: np.ndarray, label: str) -> dict:
    decade = (np.asarray(year) - np.mean(year)) / 10
    fit = sm.GLM(count, sm.add_constant(decade), family=sm.families.Poisson()).fit(
        cov_type="HAC", cov_kwds={"maxlags": 3}
    )
    low, high = fit.conf_int()[1]
    return {
        "series": label,
        "coefficient_log_per_decade": fit.params[1],
        "effect_percent_per_decade": 100 * np.expm1(fit.params[1]),
        "effect_ci95_low_percent": 100 * np.expm1(low),
        "effect_ci95_high_percent": 100 * np.expm1(high),
        "p_hac": fit.pvalues[1],
        "n_years": len(year),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    assert {name: sum(values) for name, values in FIG2_COUNTS.items()} == {
        "yearbook": 233,
        "cra40_fig2": 271,
        "era5_sun_fig2": 256,
    }
    assert len(FIG3_CRA40) == len(YEARS_FIG3)

    short = pd.DataFrame({"season_year": YEARS_FIG2, **FIG2_COUNTS})
    long = pd.DataFrame({"season_year": YEARS_FIG3, "cra40_fig3": FIG3_CRA40})
    seasonal = pd.read_csv(OUT / "seasonal_joint_metrics_1980_2023.csv")
    cnn = seasonal.loc[seasonal.season.eq("JJA"), ["season_year", "frequency"]].rename(
        columns={"frequency": "cnn_era5_frequency"}
    )
    short = short.merge(cnn, on="season_year", how="left")
    long = long.merge(cnn, on="season_year", how="left")

    frozen = pd.concat(
        [
            short.melt("season_year", var_name="source", value_name="frequency").assign(
                source_figure="Sun et al. (2026) Fig. 2a or local CNN table",
                extraction="integer visual transcription; Fig. 2 published totals cross-checked",
            ),
            pd.DataFrame(
                {
                    "season_year": YEARS_FIG3,
                    "source": "cra40_fig3",
                    "frequency": FIG3_CRA40,
                    "source_figure": "Sun et al. (2026) Fig. 3a",
                    "extraction": "integer marker digitization; retained published internal inconsistencies",
                }
            ),
        ],
        ignore_index=True,
    )
    frozen.to_csv(OUT / "published_independent_swcv_counts.csv", index=False)

    metric_rows = []
    boot_rows = []
    for candidate in ["cra40_fig2", "era5_sun_fig2", "cnn_era5_frequency"]:
        row = {"reference": "yearbook", "candidate": candidate}
        row.update(paired_metrics(short.yearbook, short[candidate]))
        metric_rows.append(row)
        boot_rows += correlation_bootstrap(
            short.yearbook.values, short[candidate].values, (2, 3), f"yearbook_vs_{candidate}"
        )

    overlap = long.dropna(subset=["cnn_era5_frequency"])
    row = {"reference": "cra40_fig3", "candidate": "cnn_era5_frequency"}
    row.update(paired_metrics(overlap.cra40_fig3, overlap.cnn_era5_frequency))
    metric_rows.append(row)
    boot_rows += correlation_bootstrap(
        overlap.cra40_fig3.values,
        overlap.cnn_era5_frequency.values,
        (3, 5),
        "cra40_fig3_vs_cnn_era5_frequency",
    )
    pd.DataFrame(metric_rows).to_csv(OUT / "independent_frequency_agreement_metrics.csv", index=False)
    pd.DataFrame(boot_rows).to_csv(OUT / "independent_frequency_correlation_bootstrap.csv", index=False)

    audit_overlap = short[["season_year", "cra40_fig2"]].merge(
        long[["season_year", "cra40_fig3"]], on="season_year"
    )
    audit_overlap["fig3_minus_fig2"] = audit_overlap.cra40_fig3 - audit_overlap.cra40_fig2
    audit_rows = [
        {
            "check": "CRA40 2012-2022 sum in Fig. 2",
            "observed": int(short.cra40_fig2.sum()),
            "reported": 271,
            "difference": int(short.cra40_fig2.sum() - 271),
        },
        {
            "check": "CRA40 1979-2024 sum digitized from Fig. 3",
            "observed": int(long.cra40_fig3.sum()),
            "reported": 1184,
            "difference": int(long.cra40_fig3.sum() - 1184),
        },
        {
            "check": "CRA40 2022 Fig. 3 versus Fig. 2",
            "observed": int(long.loc[long.season_year.eq(2022), "cra40_fig3"].iloc[0]),
            "reported": int(short.loc[short.season_year.eq(2022), "cra40_fig2"].iloc[0]),
            "difference": int(audit_overlap.loc[audit_overlap.season_year.eq(2022), "fig3_minus_fig2"].iloc[0]),
        },
    ]
    pd.DataFrame(audit_rows).to_csv(OUT / "published_cra40_internal_consistency.csv", index=False)

    trend = pd.DataFrame(
        [
            trend_row(overlap.season_year.values, overlap.cra40_fig3.values, "CRA40_Fig3_1980_2023"),
            trend_row(overlap.season_year.values, overlap.cnn_era5_frequency.values, "CNN_ERA5_1980_2023"),
        ]
    )
    trend.to_csv(OUT / "independent_frequency_trend_comparison.csv", index=False)

    indices = pd.read_csv(OUT / "jja_cgt_indices_1980_2023.csv")
    mechanism = overlap[["season_year", "cra40_fig3"]].merge(indices, on="season_year")
    model_rows = []
    model_rows += poisson_hac(
        mechanism,
        "cra40_fig3",
        ["wnpac_z850_gpm_dt_z", "cgt_dw_z200_dt_z"],
        "CRA40_frequency_on_WNPAC_and_CGT_DW",
    )
    model_rows += poisson_hac(
        mechanism,
        "cra40_fig3",
        ["wnpac_z850_gpm_dt_z", "v200_eof2_dt_z"],
        "CRA40_frequency_on_WNPAC_and_V200_EOF2",
    )
    models = pd.DataFrame(model_rows)
    models["p_fdr_bh_independent_replication_family"] = bh_adjust(models.p_hac)
    models.to_csv(OUT / "independent_cra40_mechanism_models.csv", index=False)
    mechanism_block_bootstrap(mechanism, models).to_csv(
        OUT / "independent_cra40_mechanism_bootstrap.csv", index=False
    )

    from plot_independent_frequency_revision import prepare_sources, make_figure
    make_figure(*prepare_sources(frozen, seasonal))

    print("Agreement metrics")
    print(pd.DataFrame(metric_rows).to_string(index=False))
    print("\nTrend comparison")
    print(trend.to_string(index=False))
    print("\nCRA-40 mechanism replication")
    print(models.to_string(index=False))
    print("\nPublished-series consistency audit")
    print(pd.DataFrame(audit_rows).to_string(index=False))


if __name__ == "__main__":
    main()
