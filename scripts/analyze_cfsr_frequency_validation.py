"""Compare current ERA5/CNN seasonal counts with Feng et al. (2016) CFSR counts."""

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
BOOTSTRAPS = 2000
SEED = 20260910
SEASON_MAP = {"spring": "MAM", "summer": "JJA", "fall": "SON", "winter": "DJF"}


def moving_block_indices(n: int, length: int, rng: np.random.Generator) -> np.ndarray:
    starts = rng.integers(0, n, size=int(np.ceil(n / length)))
    return np.concatenate([(start + np.arange(length)) % n for start in starts])[:n]


def trend(year: pd.Series, count: pd.Series) -> dict[str, float]:
    decade = (year.to_numpy(float) - year.mean()) / 10
    fit = sm.GLM(count, sm.add_constant(decade), family=sm.families.Poisson()).fit(
        cov_type="HAC", cov_kwds={"maxlags": 3}
    )
    low, high = fit.conf_int().iloc[1]
    return {
        "effect_percent_per_decade": 100 * np.expm1(fit.params.iloc[1]),
        "effect_ci95_low_percent": 100 * np.expm1(low),
        "effect_ci95_high_percent": 100 * np.expm1(high),
        "p_hac": fit.pvalues.iloc[1],
    }


def bh_adjust(values: pd.Series) -> pd.Series:
    ordered = values.sort_values()
    n = len(ordered)
    adjusted = (ordered * n / np.arange(1, n + 1)).iloc[::-1].cummin().iloc[::-1].clip(upper=1)
    return adjusted.reindex(values.index)


def correlation_bootstrap(reference: np.ndarray, candidate: np.ndarray, label: str) -> list[dict]:
    rows = []
    for length in (3, 5):
        rng = np.random.default_rng(SEED + length + len(reference))
        draws = []
        for _ in range(BOOTSTRAPS):
            idx = moving_block_indices(len(reference), length, rng)
            if np.std(reference[idx]) and np.std(candidate[idx]):
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


def main() -> None:
    cfsr = pd.read_csv(OUT / "feng2016_cfsr_seasonal_counts_digitized.csv")
    current = pd.read_csv(OUT / "seasonal_joint_metrics_1980_2023.csv")
    long = cfsr.melt(
        id_vars=["year", "source", "extraction"],
        value_vars=list(SEASON_MAP),
        var_name="feng_season",
        value_name="cfsr_frequency",
    )
    long["season"] = long.feng_season.map(SEASON_MAP)
    overlap = long.merge(
        current[["season_year", "season", "frequency"]],
        left_on=["year", "season"],
        right_on=["season_year", "season"],
        how="inner",
    ).rename(columns={"frequency": "cnn_era5_frequency"})
    assert overlap.year.min() == 1980 and overlap.year.max() == 2010
    assert overlap.groupby("season").size().eq(31).all()
    overlap.to_csv(OUT / "cfsr_cnn_frequency_overlap_1980_2010.csv", index=False)

    metric_rows, trend_rows, boot_rows = [], [], []
    for season in ["DJF", "MAM", "JJA", "SON"]:
        subset = overlap.loc[overlap.season.eq(season)].sort_values("year")
        reference = subset.cfsr_frequency.to_numpy(float)
        candidate = subset.cnn_era5_frequency.to_numpy(float)
        pr = pearsonr(reference, candidate)
        sr = spearmanr(reference, candidate)
        metric_rows.append(
            {
                "season": season,
                "n_years": len(subset),
                "pearson_r": pr.statistic,
                "pearson_p": pr.pvalue,
                "spearman_rho": sr.statistic,
                "spearman_p": sr.pvalue,
                "mean_bias_cnn_minus_cfsr": np.mean(candidate - reference),
                "mae": np.mean(np.abs(candidate - reference)),
                "sd_ratio_cnn_over_cfsr": np.std(candidate, ddof=1) / np.std(reference, ddof=1),
            }
        )
        boot_rows.extend(correlation_bootstrap(reference, candidate, f"CFSR_vs_CNN_{season}_same_year"))
        for source, column in [("Feng2016_CFSR", "cfsr_frequency"), ("current_CNN_ERA5", "cnn_era5_frequency")]:
            row = {"season": season, "source": source, "n_years": len(subset)}
            row.update(trend(subset.year, subset[column]))
            trend_rows.append(row)

    # Feng et al. do not state in the figure caption whether the plotted winter
    # year denotes Dec(Y-1)-Feb(Y) or Dec(Y)-Feb(Y+1). Same-year matching is the
    # primary convention; ±1-year shifts are retained as a labelling sensitivity.
    winter = long.loc[long.season.eq("DJF"), ["year", "cfsr_frequency"]]
    cnn_winter = current.loc[current.season.eq("DJF"), ["season_year", "frequency"]]
    shift_rows = []
    for shift in (-1, 0, 1):
        shifted = winter.assign(cnn_season_year=winter_year(winter.year, shift)).merge(
            cnn_winter, left_on="cnn_season_year", right_on="season_year"
        )
        result = pearsonr(shifted.cfsr_frequency, shifted.frequency)
        shift_rows.append(
            {
                "cfsr_year_to_cnn_season_year_shift": shift,
                "n_years": len(shifted),
                "pearson_r": result.statistic,
                "pearson_p": result.pvalue,
            }
        )

    metrics = pd.DataFrame(metric_rows)
    trends = pd.DataFrame(trend_rows)
    trends["p_fdr_bh_within_source_four_seasons"] = trends.groupby("source")["p_hac"].transform(bh_adjust)
    metrics.to_csv(OUT / "cfsr_cnn_frequency_agreement_metrics.csv", index=False)
    pd.DataFrame(boot_rows).to_csv(OUT / "cfsr_cnn_frequency_correlation_bootstrap.csv", index=False)
    trends.to_csv(OUT / "cfsr_cnn_frequency_trends_1980_2010.csv", index=False)
    pd.DataFrame(shift_rows).to_csv(OUT / "cfsr_djf_year_label_sensitivity.csv", index=False)

    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True, constrained_layout=True)
    for ax, season in zip(axes.flat, ["DJF", "MAM", "JJA", "SON"]):
        subset = overlap.loc[overlap.season.eq(season)].sort_values("year")
        ax.plot(subset.year, subset.cfsr_frequency, "o-", ms=3, label="CFSR / Feng et al.")
        ax.plot(subset.year, subset.cnn_era5_frequency, "o-", ms=3, label="Current CNN / ERA5")
        r = metrics.loc[metrics.season.eq(season), "pearson_r"].iloc[0]
        ax.set_title(f"{season}: r = {r:.2f}")
        ax.set_ylabel("event count")
        ax.grid(alpha=0.25)
    axes[0, 0].legend()
    axes[1, 0].set_xlabel("season year")
    axes[1, 1].set_xlabel("season year")
    fig.savefig(FIG / "cfsr_cnn_seasonal_frequency_validation.png", dpi=180)
    plt.close(fig)

    print("Agreement metrics")
    print(metrics.to_string(index=False))
    print("\nTrend comparison")
    print(trends.to_string(index=False))
    print("\nDJF year-label sensitivity")
    print(pd.DataFrame(shift_rows).to_string(index=False))


def winter_year(year: pd.Series, shift: int) -> pd.Series:
    """Explicit helper keeps the plotted-year mapping visible in the output logic."""
    return year + shift


if __name__ == "__main__":
    main()
