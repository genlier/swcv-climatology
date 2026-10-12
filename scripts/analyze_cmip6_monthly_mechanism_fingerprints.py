"""Combine Stage-6C monthly subsets into JJA forced mechanism fingerprints."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PARTS = ROOT / "data" / "processed" / "cmip6_monthly_indices_parts"
OUT = ROOT / "data" / "processed"
FIG = ROOT / "figures"
EARLY = range(1951, 1981)
LATE = range(1985, 2015)
EXPERIMENTS = ["historical", "hist-GHG", "hist-aer", "hist-nat"]


def parse_name(path: Path) -> tuple[str, str, str, str]:
    stem = path.name.removesuffix("_1950_2014.csv.gz")
    variable = stem.rsplit("_", 1)[1]
    prefix = stem.rsplit("_", 1)[0]
    member = prefix.rsplit("_", 1)[1]
    prefix = prefix.rsplit("_", 1)[0]
    experiment = next(x for x in EXPERIMENTS if prefix.endswith("_" + x))
    source = prefix[: -(len(experiment) + 1)]
    return source, experiment, member, variable


def load_jja() -> pd.DataFrame:
    frames = []
    expected = {(year, month) for year in range(1951, 2015) for month in (6, 7, 8)}
    for path in PARTS.glob("*_1950_2014.csv.gz"):
        source, experiment, member, variable = parse_name(path)
        data = pd.read_csv(path)
        data = data.loc[data.month.isin([6, 7, 8]) & data.year.between(1951, 2014)]
        available = set(map(tuple, data[["year", "month"]].drop_duplicates().to_numpy()))
        if available != expected:
            print(f"exclude incomplete target JJA series: {path.name}", flush=True)
            continue
        if variable in {"tas", "tos"} and data.value.nunique() < 2:
            print(f"exclude constant temperature series: {path.name}", flush=True)
            continue
        annual = data.groupby("year", as_index=False).value.mean()
        annual["source"] = source; annual["experiment"] = experiment
        annual["member"] = member; annual["variable"] = variable
        frames.append(annual)
    long = pd.concat(frames, ignore_index=True)
    wide = long.pivot_table(index=["source", "experiment", "member", "year"], columns="variable", values="value").reset_index()
    return wide.rename(columns={"tas": "gmst_k", "tos": "iobm_k", "zg": "wnpac_z850_m"})


def responses(jja: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for key, group in jja.groupby(["source", "experiment", "member"]):
        early = group.loc[group.year.isin(EARLY)]
        late = group.loc[group.year.isin(LATE)]
        row = dict(zip(["source", "experiment", "member"], key))
        for variable in ["gmst_k", "iobm_k", "wnpac_z850_m"]:
            early_values = early[variable].dropna()
            late_values = late[variable].dropna()
            complete = len(early_values) == 30 and len(late_values) == 30
            row[f"{variable}_early"] = early_values.mean() if complete else np.nan
            row[f"{variable}_late"] = late_values.mean() if complete else np.nan
            row[f"{variable}_change"] = (
                late_values.mean() - early_values.mean() if complete else np.nan
            )
        row["iobm_minus_gmst_change_k"] = row["iobm_k_change"] - row["gmst_k_change"]
        row["wnpac_m_per_k_gmst"] = row["wnpac_z850_m_change"] / row["gmst_k_change"] if abs(row["gmst_k_change"]) > .05 else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def summary_table(response: pd.DataFrame) -> pd.DataFrame:
    metrics = ["gmst_k_change", "iobm_k_change", "iobm_minus_gmst_change_k", "wnpac_z850_m_change", "wnpac_m_per_k_gmst"]
    rows = []
    for experiment, group in response.groupby("experiment"):
        for metric in metrics:
            values = group[metric].dropna()
            rows.append({"experiment": experiment, "metric": metric, "n_models": len(values),
                         "multi_model_mean": values.mean(), "median": values.median(),
                         "sign_agreement_fraction": max((values > 0).mean(), (values < 0).mean()) if len(values) else np.nan,
                         "positive_fraction": (values > 0).mean() if len(values) else np.nan})
    return pd.DataFrame(rows)


def make_figure(response: pd.DataFrame) -> None:
    metrics = [("gmst_k_change", "JJA GMST change (K)"), ("iobm_k_change", "JJA IOBM change (K)"),
               ("wnpac_z850_m_change", "JJA WNPAC Z850 change (m)")]
    sources = sorted(response.source.unique())
    colors = {"historical": "#333333", "hist-GHG": "#c43c39", "hist-aer": "#3973ac", "hist-nat": "#61a85f"}
    fig, axes = plt.subplots(1, 3, figsize=(14, max(4, .35 * len(sources) + 1)), sharey=True)
    offsets = dict(zip(EXPERIMENTS, [-.24, -.08, .08, .24]))
    y = np.arange(len(sources))
    for ax, (metric, label) in zip(axes, metrics):
        for experiment in EXPERIMENTS:
            q = response.loc[response.experiment == experiment].set_index("source")
            ax.scatter(q.reindex(sources)[metric], y + offsets[experiment], s=25, color=colors[experiment], label=experiment)
        ax.axvline(0, color="0.5", lw=1); ax.set_xlabel(label); ax.grid(axis="x", alpha=.2)
    axes[0].set(yticks=y, yticklabels=sources); axes[-1].legend(fontsize=8, loc="best")
    fig.tight_layout(); fig.savefig(FIG / "cmip6_monthly_mechanism_fingerprints.png", dpi=180); plt.close(fig)


def main() -> None:
    jja = load_jja(); response = responses(jja); summary = summary_table(response)
    jja.to_csv(OUT / "cmip6_monthly_jja_mechanism_indices.csv.gz", index=False)
    response.to_csv(OUT / "cmip6_monthly_mechanism_responses_1951_1980_vs_1985_2014.csv", index=False)
    summary.to_csv(OUT / "cmip6_monthly_mechanism_multimodel_summary.csv", index=False)
    make_figure(response)
    coverage = {
        metric: response.loc[response[metric].notna()].groupby("experiment").source.nunique().astype(int).to_dict()
        for metric in ["gmst_k_change", "iobm_k_change", "wnpac_z850_m_change"]
    }
    status = {"models_available": sorted(response.source.unique()), "n_models": int(response.source.nunique()),
              "experiments": sorted(response.experiment.unique()), "early_period": "1951-1980",
              "late_period": "1985-2014", "complete_models_by_metric_experiment": coverage,
              "scope": "monthly mechanism fingerprints, not direct SWCV attribution"}
    (OUT / "cmip6_monthly_mechanism_analysis_status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    print(json.dumps(status, indent=2), flush=True); print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
