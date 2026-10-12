"""Analyze CMIP6/DAMIP wind-based WNPAC fingerprints and piControl noise."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PARTS = ROOT / "data" / "processed" / "cmip6_wnpac_wind_parts"
OUT = ROOT / "data" / "processed"
DESIGN = ROOT / "data" / "manifests" / "cmip6_monthly_ua_selected_datasets.csv"
EXPERIMENTS = ["historical", "hist-GHG", "hist-aer", "hist-nat"]
METRICS = ["north_u850_ms", "south_u850_ms", "wnpac_u850_ms"]


def file_for(source: str, experiment: str, member: str) -> Path | None:
    matches = sorted(PARTS.glob(f"{source}_{experiment}_{member}_wnpac_u850_*.csv.gz"))
    return matches[0] if len(matches) == 1 else None


def available_models(design: pd.DataFrame) -> list[str]:
    complete = []
    for source, group in design.groupby("source"):
        control_rows = group.loc[group.experiment.eq("piControl")]
        if len(control_rows) != 1:
            continue
        row = control_rows.iloc[0]
        path = file_for(row.source, row.experiment, row.member)
        if path is None:
            continue
        try:
            jja(path, 300, False)
            complete.append(source)
        except RuntimeError:
            pass
    return sorted(complete)


def jja(path: Path, expected_years: int, transient: bool) -> pd.DataFrame:
    data = pd.read_csv(path)
    data = data.loc[data.month.isin([6, 7, 8])].copy()
    if transient:
        data = data.loc[data.year.between(1951, 2014)]
    counts = data.groupby("year").month.nunique()
    if len(counts) != expected_years or not counts.eq(3).all():
        raise RuntimeError(f"Incomplete JJA data in {path.name}: {len(counts)} years, counts={counts.value_counts().to_dict()}")
    return data.groupby("year", as_index=False)[METRICS].mean().sort_values("year")


def forced(design: pd.DataFrame, models: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    rows, annual, exclusions = [], [], []
    q = design.loc[design.source.isin(models) & design.experiment.isin(EXPERIMENTS)]
    for item in q.itertuples(index=False):
        path = file_for(item.source, item.experiment, item.member)
        if path is None:
            exclusions.append({"source": item.source, "experiment": item.experiment,
                               "member": item.member, "reason": "missing compact series"})
            continue
        try:
            series = jja(path, 64, True)
        except RuntimeError as exc:
            exclusions.append({"source": item.source, "experiment": item.experiment,
                               "member": item.member, "reason": str(exc)})
            continue
        tagged = series.copy()
        tagged.insert(0, "member", item.member)
        tagged.insert(0, "experiment", item.experiment)
        tagged.insert(0, "source", item.source)
        annual.append(tagged)
        early = series.loc[series.year.between(1951, 1980)]
        late = series.loc[series.year.between(1985, 2014)]
        row = {"source": item.source, "experiment": item.experiment, "member": item.member}
        for metric in METRICS:
            row[f"{metric}_early"] = float(early[metric].mean())
            row[f"{metric}_late"] = float(late[metric].mean())
            row[f"{metric}_change"] = row[f"{metric}_late"] - row[f"{metric}_early"]
        rows.append(row)
    return pd.DataFrame(rows), pd.concat(annual, ignore_index=True), exclusions


def control(design: pd.DataFrame, models: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows, drift, annual_all = [], [], []
    q = design.loc[design.source.isin(models) & design.experiment.eq("piControl")]
    for item in q.itertuples(index=False):
        series = jja(file_for(item.source, item.experiment, item.member), 300, False).reset_index(drop=True)
        series["control_year_index"] = np.arange(1, 301)
        tagged = series.copy()
        tagged.insert(0, "member", item.member)
        tagged.insert(0, "source", item.source)
        annual_all.append(tagged)
        x = np.arange(300, dtype=float)
        detrended = {}
        for metric in METRICS:
            slope, intercept = np.polyfit(x, series[metric].to_numpy(), 1)
            detrended[metric] = series[metric].to_numpy() - (intercept + slope * x)
            drift.append({"source": item.source, "member": item.member, "metric": metric,
                          "n_years": 300, "drift_per_century": slope * 100})
        for start in range(237):
            early, late = slice(start, start + 30), slice(start + 34, start + 64)
            row = {"source": item.source, "member": item.member,
                   "window_start_index": start + 1, "window_end_index": start + 64}
            for metric in METRICS:
                row[f"{metric}_change"] = float(np.mean(detrended[metric][late]) - np.mean(detrended[metric][early]))
            rows.append(row)
    return pd.DataFrame(rows), pd.DataFrame(drift), pd.concat(annual_all, ignore_index=True)


def compare(responses: pd.DataFrame, null: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for item in responses.itertuples(index=False):
        q = null.loc[null.source.eq(item.source)]
        for metric in METRICS:
            values = q[f"{metric}_change"].to_numpy()
            response = float(getattr(item, f"{metric}_change"))
            noise = float(np.std(values, ddof=1))
            rows.append({"source": item.source, "experiment": item.experiment, "member": item.member,
                         "metric": metric, "forced_response": response, "picontrol_n_windows": len(values),
                         "picontrol_noise_sd": noise, "signal_to_noise": response / noise,
                         "empirical_two_sided_p": (1 + np.sum(np.abs(values) >= abs(response))) / (len(values) + 1)})
    return pd.DataFrame(rows)


def summary(comparison: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (experiment, metric), group in comparison.groupby(["experiment", "metric"], sort=False):
        response, sn = group.forced_response, group.signal_to_noise
        rows.append({"experiment": experiment, "metric": metric, "n_models": len(group),
                     "mean_forced_response": response.mean(), "median_forced_response": response.median(),
                     "positive_models": int((response > 0).sum()), "negative_models": int((response < 0).sum()),
                     "median_signal_to_noise": sn.median(),
                     "fraction_abs_signal_to_noise_ge_2": (sn.abs() >= 2).mean()})
    return pd.DataFrame(rows)


def eddy_comparison(wind: pd.DataFrame, wind_control: pd.DataFrame) -> pd.DataFrame:
    eddy_file = OUT / "cmip6_eddy_z850_forced_vs_control.csv"
    if not eddy_file.exists():
        return pd.DataFrame()
    eddy = pd.read_csv(eddy_file)
    eddy = eddy.loc[eddy.metric.eq("eddy_z850_m"), ["source", "experiment", "forced_response", "signal_to_noise"]]
    eddy = eddy.rename(columns={"forced_response": "eddy_z850_m_change", "signal_to_noise": "eddy_z850_signal_to_noise"})
    winds = wind_control.loc[wind_control.metric.eq("wnpac_u850_ms"),
                             ["source", "experiment", "forced_response", "signal_to_noise"]]
    winds = winds.rename(columns={"forced_response": "wnpac_u850_ms_change", "signal_to_noise": "wind_signal_to_noise"})
    return winds.merge(eddy, on=["source", "experiment"], how="inner", validate="one_to_one")


def make_figure(comparison: pd.DataFrame) -> None:
    from plot_cmip6_wind_revision import make_figure as plot_frozen_responses
    plot_frozen_responses(comparison)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+")
    args = parser.parse_args()
    design = pd.read_csv(DESIGN)
    models = available_models(design)
    if args.models:
        requested = set(args.models)
        models = [model for model in models if model in requested]
    if not models:
        raise SystemExit("No complete five-experiment models")
    responses, forced_annual, exclusions = forced(design, models)
    null, drift, control_annual = control(design, models)
    comparison = compare(responses, null)
    summaries = summary(comparison)
    cross = eddy_comparison(responses, comparison)
    responses.to_csv(OUT / "cmip6_wnpac_wind_forced_responses.csv", index=False)
    forced_annual.to_csv(OUT / "cmip6_wnpac_wind_forced_jja.csv.gz", index=False)
    control_annual.to_csv(OUT / "cmip6_wnpac_wind_picontrol_jja.csv.gz", index=False)
    null.to_csv(OUT / "cmip6_wnpac_wind_picontrol_null.csv.gz", index=False)
    drift.to_csv(OUT / "cmip6_wnpac_wind_picontrol_drift.csv", index=False)
    comparison.to_csv(OUT / "cmip6_wnpac_wind_forced_vs_control.csv", index=False)
    summaries.to_csv(OUT / "cmip6_wnpac_wind_multimodel_summary.csv", index=False)
    cross.to_csv(OUT / "cmip6_wnpac_wind_vs_eddy_z850.csv", index=False)
    pd.DataFrame(exclusions, columns=["source", "experiment", "member", "reason"]).to_csv(
        OUT / "cmip6_wnpac_wind_exclusions.csv", index=False
    )
    make_figure(comparison)
    status = {"models_complete": models, "n_models": len(models), "forced_groups": len(responses),
              "control_years_per_model": 300, "null_windows_per_model": 237,
              "n_models_by_experiment": responses.groupby("experiment").source.nunique().to_dict(),
              "forced_exclusions": exclusions,
              "definition": "U850 north box (22.5-32.5N,110-140E) minus south box (5-15N,90-130E)",
              "sign": "positive denotes WNPAC-like anticyclonic shear",
              "periods": "JJA 1985-2014 minus 1951-1980"}
    (OUT / "cmip6_wnpac_wind_status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    print(json.dumps(status, indent=2), flush=True)
    print(summaries.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
