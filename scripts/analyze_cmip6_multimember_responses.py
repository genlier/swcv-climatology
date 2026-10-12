"""Summarise three-member CMIP6 Z850 responses and make a review-ready figure."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from audit_panel_alignment import require_matplotlib_panel_alignment


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data" / "processed" / "cmip6_multimember_z850_responses.csv"
SELECTED = ROOT / "data" / "processed" / "cmip6_eddy_z850_forced_responses.csv"
OUT = ROOT / "data" / "processed"
FIG = ROOT / "figures"
METRICS = ("raw_z850_m", "zonal_z850_m", "eddy_z850_m")
EXPERIMENTS = ("hist-GHG", "historical")
METRIC_LABELS = {
    "raw_z850_m": "Regional height",
    "zonal_z850_m": "Zonal-mean height",
    "eddy_z850_m": "Eddy height",
}
COLORS = {"hist-GHG": "#D55E00", "historical": "#0072B2"}
SEED = 20260912


mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 7,
        "axes.titlesize": 7,
        "axes.labelsize": 7,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.7,
    }
)


def bootstrap_mean_ci(values: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    draws = rng.choice(values, size=(10_000, len(values)), replace=True).mean(axis=1)
    return tuple(np.quantile(draws, [0.025, 0.975]))


def validate(data: pd.DataFrame) -> None:
    required = {
        "model", "experiment", "member",
        *(f"{metric}_change" for metric in METRICS),
    }
    missing = required - set(data.columns)
    if missing:
        raise RuntimeError(f"Missing columns: {sorted(missing)}")
    counts = data.groupby(["model", "experiment"]).member.nunique()
    if len(counts) != 16 or not (counts == 3).all():
        raise RuntimeError(f"Expected 8 models x 2 experiments x 3 members; got {counts.to_dict()}")
    paired = data.groupby("model").experiment.nunique()
    if not (paired == 2).all():
        raise RuntimeError("Every model must contain both experiments")


def model_summary(data: pd.DataFrame) -> pd.DataFrame:
    long = data.melt(
        id_vars=["model", "experiment", "member"],
        value_vars=[f"{metric}_change" for metric in METRICS],
        var_name="metric", value_name="response_m",
    )
    long["metric"] = long.metric.str.removesuffix("_change")
    rows = []
    for keys, group in long.groupby(["model", "experiment", "metric"], sort=True):
        values = group.response_m.to_numpy(float)
        rows.append(
            {
                "model": keys[0], "experiment": keys[1], "metric": keys[2],
                "n_members": len(values), "ensemble_mean_m": values.mean(),
                "ensemble_median_m": np.median(values), "ensemble_sd_m": values.std(ddof=1),
                "member_min_m": values.min(), "member_max_m": values.max(),
                "positive_members": int((values > 0).sum()),
                "member_sign_agreement": max((values > 0).mean(), (values < 0).mean()),
            }
        )
    return pd.DataFrame(rows), long


def multimodel_summary(models: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    rows = []
    for keys, group in models.groupby(["experiment", "metric"], sort=True):
        values = group.ensemble_mean_m.to_numpy(float)
        low, high = bootstrap_mean_ci(values, rng)
        rows.append(
            {
                "experiment": keys[0], "metric": keys[1], "n_models": len(values),
                "multimodel_mean_m": values.mean(), "multimodel_median_m": np.median(values),
                "model_sd_m": values.std(ddof=1), "bootstrap_ci95_low_m": low,
                "bootstrap_ci95_high_m": high, "positive_models": int((values > 0).sum()),
                "model_sign_agreement": max((values > 0).mean(), (values < 0).mean()),
            }
        )
    return pd.DataFrame(rows)


def experiment_contrasts(long: pd.DataFrame) -> pd.DataFrame:
    wide = long.pivot(index=["model", "member", "metric"], columns="experiment", values="response_m")
    wide = wide.dropna(subset=list(EXPERIMENTS)).reset_index()
    wide["hist_ghg_minus_historical_m"] = wide["hist-GHG"] - wide["historical"]
    return wide


def selected_comparison(data: pd.DataFrame, models: pd.DataFrame) -> pd.DataFrame:
    old = pd.read_csv(SELECTED).rename(columns={"source": "model"})
    old = old.loc[old.experiment.isin(EXPERIMENTS)]
    old = old.melt(
        id_vars=["model", "experiment", "member"],
        value_vars=[f"{metric}_change" for metric in METRICS],
        var_name="metric", value_name="selected_response_m",
    )
    old["metric"] = old.metric.str.removesuffix("_change")
    recalculated = data.loc[data.member.str.match(r"^r1i")].melt(
        id_vars=["model", "experiment", "member"],
        value_vars=[f"{metric}_change" for metric in METRICS],
        var_name="metric", value_name="recalculated_selected_response_m",
    )
    recalculated["metric"] = recalculated.metric.str.removesuffix("_change")
    comparison = old.merge(
        recalculated, on=["model", "experiment", "member", "metric"],
        how="inner", validate="one_to_one",
    ).merge(
        models[["model", "experiment", "metric", "ensemble_mean_m", "member_min_m", "member_max_m"]],
        on=["model", "experiment", "metric"], how="inner", validate="one_to_one",
    )
    comparison["recalculation_minus_selected_m"] = (
        comparison.recalculated_selected_response_m - comparison.selected_response_m
    )
    comparison["ensemble_minus_selected_m"] = (
        comparison.ensemble_mean_m - comparison.selected_response_m
    )
    comparison["sign_flip"] = (
        np.sign(comparison.ensemble_mean_m) != np.sign(comparison.selected_response_m)
    )
    if len(comparison) != 8 * 2 * len(METRICS):
        raise RuntimeError(f"Expected 48 selected-member comparison rows; got {len(comparison)}")
    return comparison


def plot(data: pd.DataFrame, models: pd.DataFrame, summary: pd.DataFrame) -> None:
    order = sorted(data.model.unique())
    y = np.arange(len(order))
    fig, axes = plt.subplots(
        1, 2, figsize=(183 / 25.4, 92 / 25.4), constrained_layout=True,
        gridspec_kw={"width_ratios": [1.25, 1]},
    )

    ax = axes[0]
    offsets = {"hist-GHG": -0.13, "historical": 0.13}
    for experiment in EXPERIMENTS:
        subset = data.loc[data.experiment.eq(experiment)]
        means = models.loc[
            models.experiment.eq(experiment) & models.metric.eq("eddy_z850_m")
        ].set_index("model")
        for yi, model in enumerate(order):
            values = subset.loc[subset.model.eq(model), "eddy_z850_m_change"].to_numpy(float)
            yy = yi + offsets[experiment]
            ax.plot(values, np.full(len(values), yy), "o", ms=2.7, mfc="white",
                    mec=COLORS[experiment], mew=0.65, alpha=0.85, zorder=2)
            ax.plot(means.loc[model, "ensemble_mean_m"], yy, "D", ms=4,
                    color=COLORS[experiment], zorder=3)
    ax.axvline(0, color="0.45", lw=0.7, ls="--", zorder=0)
    ax.set_yticks(y, order)
    ax.invert_yaxis()
    ax.set_xlabel("1951–1980 to 1985–2014 response (m)")
    ax.set_title("Three-member eddy-height responses")
    ax.grid(axis="x", color="0.91", lw=0.5, zorder=-1)
    ax.legend(
        handles=[
            Line2D([0], [0], marker="D", lw=0, color=COLORS[e], label=e)
            for e in EXPERIMENTS
        ], loc="lower right", frameon=False, fontsize=6, borderaxespad=1.2,
    )

    ax = axes[1]
    positions = np.arange(len(METRICS))
    width = 0.18
    for i, experiment in enumerate(EXPERIMENTS):
        rows = summary.set_index(["experiment", "metric"]).loc[
            [(experiment, metric) for metric in METRICS]
        ]
        x = positions + (-0.55 if i == 0 else 0.55) * width
        centre = rows.multimodel_mean_m.to_numpy(float)
        low = rows.bootstrap_ci95_low_m.to_numpy(float)
        high = rows.bootstrap_ci95_high_m.to_numpy(float)
        ax.errorbar(
            x, centre, yerr=np.vstack([centre - low, high - centre]), fmt="D",
            color=COLORS[experiment], ms=4, elinewidth=1, capsize=2, label=experiment,
        )
        for xi, metric in zip(x, METRICS):
            points = models.loc[
                models.experiment.eq(experiment) & models.metric.eq(metric), "ensemble_mean_m"
            ].to_numpy(float)
            jitter = np.linspace(-0.035, 0.035, len(points))
            ax.plot(np.full(len(points), xi) + jitter, points, "o", ms=2,
                    mfc="white", mec=COLORS[experiment], mew=0.5, alpha=0.75)
    ax.axhline(0, color="0.45", lw=0.7, ls="--", zorder=0)
    ax.set_xticks(positions, [METRIC_LABELS[m] for m in METRICS], rotation=18, ha="right")
    ax.set_ylabel("Response (m)")
    ax.set_title("Multimodel means of ensemble means")
    ax.grid(axis="y", color="0.91", lw=0.5, zorder=-1)

    for label, ax in zip(("a", "b"), axes):
        ax.annotate(label, xy=(0, 1), xycoords="axes fraction", xytext=(-18, 5),
                    textcoords="offset points", fontsize=8, fontweight="bold", va="bottom")

    FIG.mkdir(exist_ok=True)
    base = FIG / "cmip6_multimember_z850_responses"
    require_matplotlib_panel_alignment(
        fig, json_out=str(base) + ".alignment.json",
        overlay_svg=str(base) + ".alignment.svg", axes=list(axes),
        panel_ids={axes[0]: "a", axes[1]: "b"}, row_groups=[["a", "b"]],
        require_panel_labels=True, tolerance_pt=1.5, gutter_tolerance_pt=1.5, strict=True,
    )
    fig.savefig(str(base) + ".pdf", bbox_inches="tight")
    fig.savefig(str(base) + ".svg", bbox_inches="tight")
    fig.savefig(str(base) + ".png", dpi=300, bbox_inches="tight")
    fig.savefig(str(base) + ".tiff", dpi=600, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    data = pd.read_csv(INPUT)
    validate(data)
    models, long = model_summary(data)
    summary = multimodel_summary(models)
    contrasts = experiment_contrasts(long)
    comparison = selected_comparison(data, models)
    models.to_csv(OUT / "cmip6_multimember_z850_model_summary.csv", index=False)
    summary.to_csv(OUT / "cmip6_multimember_z850_multimodel_summary.csv", index=False)
    contrasts.to_csv(OUT / "cmip6_multimember_z850_experiment_contrasts.csv", index=False)
    comparison.to_csv(OUT / "cmip6_multimember_z850_selected_comparison.csv", index=False)
    plot(data, models, summary)
    print(summary.to_string(index=False))
    print(f"selected comparisons: {len(comparison)} rows; sign flips={comparison.sign_flip.sum()}")


if __name__ == "__main__":
    main()
