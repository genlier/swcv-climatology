"""Build piControl noise distributions for Stage-6C mechanism responses."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PARTS = ROOT / "data" / "processed" / "cmip6_picontrol_indices_parts"
OUT = ROOT / "data" / "processed"
FORCED = OUT / "cmip6_monthly_mechanism_responses_1951_1980_vs_1985_2014.csv"
VARIABLES = {"tas": "gmst_k", "tos": "iobm_k", "zg": "wnpac_z850_m"}


def parse_name(path: Path) -> tuple[str, str, str]:
    # <source>_piControl_<member>_<variable>_<start>_<end>.csv.gz
    stem = path.name.removesuffix(".csv.gz")
    prefix, _, tail = stem.partition("_piControl_")
    member, variable, _, _ = tail.rsplit("_", 3)
    return prefix, member, variable


def load_jja() -> pd.DataFrame:
    frames = []
    for path in PARTS.glob("*_piControl_*.csv.gz"):
        source, member, variable = parse_name(path)
        data = pd.read_csv(path)
        jja = data.loc[data.month.isin([6, 7, 8])].copy()
        counts = jja.groupby("year").month.nunique()
        if len(counts) != 300 or not counts.eq(3).all():
            print(f"exclude incomplete piControl series: {path.name}", flush=True)
            continue
        if variable in {"tas", "tos"} and jja.value.nunique() < 2:
            print(f"exclude constant piControl temperature series: {path.name}", flush=True)
            continue
        annual = jja.groupby("year", as_index=False).value.mean()
        annual = annual.sort_values("year").reset_index(drop=True)
        annual["control_year_index"] = np.arange(1, len(annual) + 1)
        annual["source"] = source
        annual["member"] = member
        annual["variable"] = variable
        frames.append(annual)
    if not frames:
        return pd.DataFrame()
    long = pd.concat(frames, ignore_index=True)
    wide = long.pivot_table(index=["source", "member", "control_year_index"], columns="variable", values="value").reset_index()
    return wide.rename(columns=VARIABLES)


def detrend_and_windows(jja: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    drift_rows, window_rows = [], []
    for (source, member), group in jja.groupby(["source", "member"]):
        group = group.sort_values("control_year_index").copy()
        x = np.arange(len(group), dtype=float)
        detrended = {}
        for variable in VARIABLES.values():
            if variable not in group or group[variable].notna().sum() != len(group) or len(group) < 300:
                continue
            slope, intercept = np.polyfit(x, group[variable].to_numpy(), 1)
            drift_rows.append({
                "source": source,
                "member": member,
                "metric": variable,
                "n_years": len(group),
                "drift_per_century": slope * 100,
            })
            detrended[variable] = group[variable].to_numpy() - (intercept + slope * x)
        if "iobm_k" in detrended and "gmst_k" in detrended:
            detrended["iobm_minus_gmst_k"] = detrended["iobm_k"] - detrended["gmst_k"]
        n_windows = len(group) - 63
        for start in range(max(0, n_windows)):
            early = slice(start, start + 30)
            late = slice(start + 34, start + 64)
            row = {
                "source": source,
                "member": member,
                "window_start_index": int(group.control_year_index.iloc[start]),
                "window_end_index": int(group.control_year_index.iloc[start + 63]),
            }
            for variable, values in detrended.items():
                row[variable + "_change"] = float(np.mean(values[late]) - np.mean(values[early]))
            window_rows.append(row)
    return pd.DataFrame(drift_rows), pd.DataFrame(window_rows)


def compare_forced(null: pd.DataFrame) -> pd.DataFrame:
    forced = pd.read_csv(FORCED)
    mapping = {
        "gmst_k_change": "gmst_k_change",
        "iobm_k_change": "iobm_k_change",
        "iobm_minus_gmst_change_k": "iobm_minus_gmst_k_change",
        "wnpac_z850_m_change": "wnpac_z850_m_change",
    }
    rows = []
    for forced_row in forced.itertuples(index=False):
        model_null = null.loc[null.source.eq(forced_row.source)]
        if model_null.empty:
            continue
        for forced_metric, null_metric in mapping.items():
            values = model_null[null_metric].dropna().to_numpy()
            response = float(getattr(forced_row, forced_metric))
            if len(values) < 2 or not np.isfinite(response):
                continue
            noise_sd = float(np.std(values, ddof=1))
            rows.append({
                "source": forced_row.source,
                "experiment": forced_row.experiment,
                "member": forced_row.member,
                "metric": forced_metric,
                "forced_response": response,
                "picontrol_n_windows": len(values),
                "picontrol_noise_sd": noise_sd,
                "signal_to_noise": response / noise_sd if noise_sd > 0 else np.nan,
                "empirical_two_sided_p": (1 + np.sum(np.abs(values) >= abs(response))) / (len(values) + 1),
            })
    return pd.DataFrame(rows)


def summarize(comparison: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (experiment, metric), group in comparison.groupby(["experiment", "metric"]):
        values = group.signal_to_noise.dropna()
        rows.append({
            "experiment": experiment,
            "metric": metric,
            "n_models": len(values),
            "mean_signal_to_noise": values.mean(),
            "median_signal_to_noise": values.median(),
            "sign_agreement_fraction": max((values > 0).mean(), (values < 0).mean()),
            "fraction_abs_signal_to_noise_ge_2": (values.abs() >= 2).mean(),
        })
    return pd.DataFrame(rows)


def make_figure(comparison: pd.DataFrame) -> None:
    metrics = [
        ("gmst_k_change", "JJA GMST"),
        ("iobm_k_change", "JJA IOBM"),
        ("wnpac_z850_m_change", "JJA WNPAC-box Z850"),
    ]
    experiments = ["historical", "hist-GHG", "hist-aer", "hist-nat"]
    colors = {"historical": "#333333", "hist-GHG": "#c43c39", "hist-aer": "#3973ac", "hist-nat": "#61a85f"}
    sources = sorted(comparison.source.unique())
    y = np.arange(len(sources))
    offsets = dict(zip(experiments, [-.24, -.08, .08, .24]))
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4), sharey=True)
    for ax, (metric, label) in zip(axes, metrics):
        for experiment in experiments:
            q = comparison.loc[(comparison.metric == metric) & (comparison.experiment == experiment)].set_index("source")
            ax.scatter(q.reindex(sources).signal_to_noise, y + offsets[experiment], s=27,
                       color=colors[experiment], label=experiment)
        ax.axvline(0, color="0.45", lw=1)
        ax.axvline(-2, color="0.65", lw=.8, ls="--")
        ax.axvline(2, color="0.65", lw=.8, ls="--")
        ax.set_xlabel(f"{label} response / piControl noise SD")
        ax.grid(axis="x", alpha=.2)
    axes[0].set(yticks=y, yticklabels=sources)
    axes[-1].legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(ROOT / "figures" / "cmip6_mechanism_signal_to_control_noise.png", dpi=180)
    plt.close(fig)


def main() -> None:
    jja = load_jja()
    if jja.empty:
        raise SystemExit("No complete piControl indices found")
    drift, null = detrend_and_windows(jja)
    comparison = compare_forced(null)
    summary = summarize(comparison)
    jja.to_csv(OUT / "cmip6_picontrol_jja_indices.csv.gz", index=False)
    drift.to_csv(OUT / "cmip6_picontrol_drift_summary.csv", index=False)
    null.to_csv(OUT / "cmip6_picontrol_30yr_difference_null.csv.gz", index=False)
    comparison.to_csv(OUT / "cmip6_mechanism_forced_vs_picontrol.csv", index=False)
    summary.to_csv(OUT / "cmip6_mechanism_forced_vs_picontrol_summary.csv", index=False)
    make_figure(comparison)
    status = {
        "models_available": sorted(jja.source.unique()),
        "n_models": int(jja.source.nunique()),
        "control_years_per_model": jja.groupby("source").control_year_index.nunique().astype(int).to_dict(),
        "complete_models_by_metric": {
            metric: sorted(jja.loc[jja[metric].notna()].groupby("source").filter(lambda q: len(q) >= 300).source.unique())
            for metric in VARIABLES.values()
        },
        "window_definition": "late 30-year mean minus early 30-year mean, separated by 4 years",
        "drift_treatment": "linear drift removed separately from each annual JJA index",
        "overlap_caveat": "empirical windows overlap and are not independent",
    }
    (OUT / "cmip6_picontrol_noise_status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    print(json.dumps(status, indent=2))
    print(drift.to_string(index=False))
    print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()
