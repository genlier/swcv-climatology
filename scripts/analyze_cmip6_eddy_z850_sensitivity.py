"""Stage 6D: separate thermodynamic height rise from the WNPAC Z850 response.

The dynamic proxy is the JJA WNPAC-box mean Z850 (15--25N, 115--150E)
minus the simultaneous 15--25N zonal-mean Z850. Forced changes are compared
with identically constructed, linearly detrended 300-year piControl windows.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "processed"
REGIONAL = OUT / "cmip6_monthly_indices_parts"
CONTROL_REGIONAL = OUT / "cmip6_picontrol_indices_parts"
ZONAL = OUT / "cmip6_z850_zonal_parts"
FORCED_BASE = OUT / "cmip6_monthly_mechanism_responses_1951_1980_vs_1985_2014.csv"
EXPERIMENTS = ["historical", "hist-GHG", "hist-aer", "hist-nat"]
METRICS = ["raw_z850_m", "zonal_z850_m", "eddy_z850_m"]


def one(pattern: Path) -> Path:
    matches = sorted(pattern.parent.glob(pattern.name))
    if len(matches) != 1:
        raise RuntimeError(f"Expected one file for {pattern}, found {len(matches)}")
    return matches[0]


def read_pair(regional: Path, zonal: Path, expected_jja_months: int) -> pd.DataFrame:
    a = pd.read_csv(regional).rename(columns={"value": "raw_z850_m"})
    b = pd.read_csv(zonal).rename(columns={"value": "zonal_z850_m"})
    merged = a.merge(b, on=["year", "month"], how="inner", validate="one_to_one")
    merged = merged.loc[merged.month.isin([6, 7, 8])].copy()
    if len(merged) != expected_jja_months:
        raise RuntimeError(
            f"JJA key mismatch for {regional.name} and {zonal.name}: "
            f"got {len(merged)}, expected {expected_jja_months}"
        )
    merged["eddy_z850_m"] = merged.raw_z850_m - merged.zonal_z850_m
    return merged


def complete_jja(monthly: pd.DataFrame, expected_years: int) -> pd.DataFrame:
    counts = monthly.groupby("year").month.nunique()
    if not counts.eq(3).all() or len(counts) != expected_years:
        raise RuntimeError(f"Incomplete JJA years: n={len(counts)}, counts={counts.value_counts().to_dict()}")
    return monthly.groupby("year", as_index=False)[METRICS].mean().sort_values("year")


def forced_responses(design: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    response_rows, annual_frames = [], []
    for row in design.itertuples(index=False):
        regional_matches = sorted(REGIONAL.glob(f"{row.source}_{row.experiment}_{row.member}_zg_1950_2014.csv.gz"))
        zonal_matches = sorted(ZONAL.glob(f"{row.source}_{row.experiment}_{row.member}_z850_zonal15_25n_1950_2014.csv.gz"))
        if len(regional_matches) != 1 or len(zonal_matches) != 1:
            print(f"exclude incomplete forced Z850 group: {row.source} {row.experiment} {row.member}", flush=True)
            continue
        try:
            monthly = read_pair(regional_matches[0], zonal_matches[0], 65 * 3)
        except RuntimeError as exc:
            print(f"exclude incomplete forced Z850 group: {row.source} {row.experiment} {row.member}: {exc}", flush=True)
            continue
        monthly = monthly.loc[monthly.year.between(1951, 2014)]
        annual = complete_jja(monthly, 64)
        annual.insert(0, "member", row.member)
        annual.insert(0, "experiment", row.experiment)
        annual.insert(0, "source", row.source)
        annual_frames.append(annual)
        early = annual.loc[annual.year.between(1951, 1980), METRICS]
        late = annual.loc[annual.year.between(1985, 2014), METRICS]
        if len(early) != 30 or len(late) != 30:
            raise RuntimeError(f"Bad forced periods for {row.source} {row.experiment}")
        result = {"source": row.source, "experiment": row.experiment, "member": row.member}
        for metric in METRICS:
            result[f"{metric}_early"] = float(early[metric].mean())
            result[f"{metric}_late"] = float(late[metric].mean())
            result[f"{metric}_change"] = result[f"{metric}_late"] - result[f"{metric}_early"]
        raw = result["raw_z850_m_change"]
        zonal_change = result["zonal_z850_m_change"]
        result["zonal_fraction_of_raw_change"] = zonal_change / raw if abs(raw) > 1e-12 else np.nan
        response_rows.append(result)
    return pd.DataFrame(response_rows), pd.concat(annual_frames, ignore_index=True)


def control_null(design: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    null_rows, drift_rows, annual_frames = [], [], []
    for source, member in design[["source", "member"]].drop_duplicates().itertuples(index=False):
        regional_matches = sorted(CONTROL_REGIONAL.glob(f"{source}_piControl_{member}_zg_*.csv.gz"))
        zonal_matches = sorted(ZONAL.glob(f"{source}_piControl_{member}_z850_zonal15_25n_*.csv.gz"))
        if len(regional_matches) != 1 or len(zonal_matches) != 1:
            print(f"exclude incomplete piControl Z850 group: {source} {member}", flush=True)
            continue
        try:
            annual = complete_jja(read_pair(regional_matches[0], zonal_matches[0], 300 * 3), 300).reset_index(drop=True)
        except RuntimeError as exc:
            print(f"exclude incomplete piControl Z850 group: {source} {member}: {exc}", flush=True)
            continue
        annual["control_year_index"] = np.arange(1, 301)
        annual.insert(0, "member", member)
        annual.insert(0, "source", source)
        annual_frames.append(annual)
        x = np.arange(300, dtype=float)
        detrended = {}
        for metric in METRICS:
            slope, intercept = np.polyfit(x, annual[metric].to_numpy(), 1)
            detrended[metric] = annual[metric].to_numpy() - (intercept + slope * x)
            drift_rows.append({
                "source": source, "member": member, "metric": metric,
                "n_years": 300, "drift_per_century": slope * 100,
            })
        for start in range(300 - 63):
            early, late = slice(start, start + 30), slice(start + 34, start + 64)
            item = {
                "source": source, "member": member,
                "window_start_index": start + 1, "window_end_index": start + 64,
            }
            for metric in METRICS:
                item[f"{metric}_change"] = float(
                    np.mean(detrended[metric][late]) - np.mean(detrended[metric][early])
                )
            null_rows.append(item)
    return pd.DataFrame(null_rows), pd.DataFrame(drift_rows), pd.concat(annual_frames, ignore_index=True)


def compare(responses: pd.DataFrame, null: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for forced in responses.itertuples(index=False):
        model_null = null.loc[null.source.eq(forced.source)]
        if model_null.empty:
            continue
        for metric in METRICS:
            values = model_null[f"{metric}_change"].to_numpy()
            response = float(getattr(forced, f"{metric}_change"))
            if len(values) < 2 or not np.isfinite(response):
                continue
            noise = float(np.std(values, ddof=1))
            rows.append({
                "source": forced.source, "experiment": forced.experiment, "member": forced.member,
                "metric": metric, "forced_response": response,
                "picontrol_n_windows": len(values), "picontrol_noise_sd": noise,
                "signal_to_noise": response / noise,
                "empirical_two_sided_p": (1 + np.sum(np.abs(values) >= abs(response))) / (len(values) + 1),
            })
    return pd.DataFrame(rows)


def summarize(comparison: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (experiment, metric), group in comparison.groupby(["experiment", "metric"], sort=False):
        values = group.signal_to_noise
        responses = group.forced_response
        rows.append({
            "experiment": experiment, "metric": metric, "n_models": len(group),
            "mean_forced_response": responses.mean(), "median_forced_response": responses.median(),
            "positive_models": int((responses > 0).sum()), "negative_models": int((responses < 0).sum()),
            "median_signal_to_noise": values.median(),
            "sign_agreement_fraction": max((values > 0).mean(), (values < 0).mean()),
            "fraction_abs_signal_to_noise_ge_2": (values.abs() >= 2).mean(),
        })
    return pd.DataFrame(rows)


def make_figure(comparison: pd.DataFrame) -> None:
    from plot_cmip6_height_revision import make_figure as plot_paired_responses
    plot_paired_responses(comparison)


def main() -> None:
    design = pd.read_csv(FORCED_BASE)[["source", "experiment", "member"]]
    responses, forced_annual = forced_responses(design)
    null, drift, control_annual = control_null(design)
    comparison = compare(responses, null)
    summary = summarize(comparison)
    responses.to_csv(OUT / "cmip6_eddy_z850_forced_responses.csv", index=False)
    forced_annual.to_csv(OUT / "cmip6_eddy_z850_forced_jja.csv.gz", index=False)
    control_annual.to_csv(OUT / "cmip6_eddy_z850_picontrol_jja.csv.gz", index=False)
    null.to_csv(OUT / "cmip6_eddy_z850_picontrol_null.csv.gz", index=False)
    drift.to_csv(OUT / "cmip6_eddy_z850_picontrol_drift.csv", index=False)
    comparison.to_csv(OUT / "cmip6_eddy_z850_forced_vs_control.csv", index=False)
    summary.to_csv(OUT / "cmip6_eddy_z850_multimodel_summary.csv", index=False)
    make_figure(comparison)
    status = {
        "metric": "WNPAC-box Z850 (15-25N, 115-150E) minus 15-25N zonal-mean Z850",
        "models": sorted(responses.source.unique()), "n_models": int(responses.source.nunique()),
        "forced_groups": len(responses), "control_years_per_model": 300,
        "forced_groups_by_experiment": responses.groupby("experiment").source.nunique().astype(int).to_dict(),
        "comparison_groups_by_experiment": comparison.groupby("experiment").source.nunique().astype(int).to_dict(),
        "control_models_used": sorted(null.source.unique()),
        "null_windows_per_model": 237,
        "periods": "JJA 1985-2014 minus JJA 1951-1980",
        "control_treatment": "linear detrending of annual JJA index, then overlapping matched 30-year windows with a 4-year gap",
    }
    (OUT / "cmip6_eddy_z850_status.json").write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(status, ensure_ascii=False, indent=2))
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
