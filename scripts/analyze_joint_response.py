"""Joint frequency-intensity-path diagnostics for the 1979-2023 SWCV record."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import linregress, pearsonr


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
FIGURES = ROOT / "figures"


def zscore(series: pd.Series) -> pd.Series:
    return (series - series.mean()) / series.std(ddof=0)


def hac_table(annual: pd.DataFrame, responses: list[str]) -> pd.DataFrame:
    rows = []
    for response in responses:
        for specification, predictors in (
            ("gmst_only", ["gmst_anomaly_c"]),
            ("gmst_plus_year", ["gmst_anomaly_c", "year_centered"]),
        ):
            fit = sm.OLS(annual[response], sm.add_constant(annual[predictors])).fit(
                cov_type="HAC", cov_kwds={"maxlags": 3}
            )
            rows.append(
                {
                    "response": response,
                    "specification": specification,
                    "gmst_coefficient_per_c": fit.params["gmst_anomaly_c"],
                    "gmst_se_hac": fit.bse["gmst_anomaly_c"],
                    "gmst_p_hac": fit.pvalues["gmst_anomaly_c"],
                    "r_squared": fit.rsquared,
                    "n_years": int(fit.nobs),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    events = pd.read_csv(PROCESSED / "events.csv", parse_dates=["genesis_time", "lysis_time"])
    events = events.loc[events.year >= 1979].copy()
    intensity = pd.read_csv(PROCESSED / "era5_event_intensity.csv")
    events = events.merge(intensity, on="event_id", validate="one_to_one")

    # A two-field standardized intensity avoids treating gpm and s-1 as
    # commensurable. It is a transparent screening index, not a new physical unit.
    events["height_intensity_z"] = zscore(events["max_height_deficit_gpm"])
    events["vorticity_intensity_z"] = zscore(events["max_vorticity_s_1"])
    events["dynamic_intensity_z"] = (
        events["height_intensity_z"] + events["vorticity_intensity_z"]
    ) / 2
    strong_threshold = float(events["dynamic_intensity_z"].quantile(0.90))
    events["strong_event"] = events["dynamic_intensity_z"] >= strong_threshold
    events["mobile_eastward"] = events["track_cluster"].eq(2)
    events["strong_mobile_event"] = events["strong_event"] & events["mobile_eastward"]

    grouped = events.groupby("year")
    annual = grouped.agg(
        frequency=("event_id", "size"),
        median_height_deficit_gpm=("max_height_deficit_gpm", "median"),
        q90_height_deficit_gpm=("max_height_deficit_gpm", lambda x: x.quantile(0.9)),
        median_max_vorticity_s_1=("max_vorticity_s_1", "median"),
        q90_max_vorticity_s_1=("max_vorticity_s_1", lambda x: x.quantile(0.9)),
        median_dynamic_intensity_z=("dynamic_intensity_z", "median"),
        q90_dynamic_intensity_z=("dynamic_intensity_z", lambda x: x.quantile(0.9)),
        strong_event_count=("strong_event", "sum"),
        strong_event_fraction=("strong_event", "mean"),
        median_displacement_km=("displacement_km", "median"),
        mobile_eastward_count=("mobile_eastward", "sum"),
        mobile_eastward_fraction=("mobile_eastward", "mean"),
        strong_mobile_count=("strong_mobile_event", "sum"),
        eastward_escape_fraction=("max_lon", lambda x: (x >= 110).mean()),
    ).reset_index()
    annual["strong_given_mobile_fraction"] = annual["strong_mobile_count"] / annual["mobile_eastward_count"]

    gmst = pd.read_csv(ROOT / "data" / "raw" / "gistemp_v4_global.csv", skiprows=1, na_values="***")
    gmst = gmst.rename(columns={"Year": "year", "J-D": "gmst_anomaly_c"})
    annual = annual.merge(gmst[["year", "gmst_anomaly_c"]], on="year", validate="one_to_one")
    annual["year_centered"] = annual.year - annual.year.mean()

    annual["frequency_z"] = zscore(annual.frequency)
    annual["intensity_z"] = zscore(annual.median_dynamic_intensity_z)
    annual["path_z"] = (
        zscore(annual.median_displacement_km)
        + zscore(annual.mobile_eastward_fraction)
        + zscore(annual.eastward_escape_fraction)
    ) / 3
    annual["joint_fip_index"] = (
        annual["frequency_z"] + annual["intensity_z"] + annual["path_z"]
    ) / 3

    responses = [
        "median_height_deficit_gpm", "q90_height_deficit_gpm",
        "median_max_vorticity_s_1", "q90_max_vorticity_s_1",
        "median_dynamic_intensity_z", "q90_dynamic_intensity_z",
        "strong_event_fraction", "mobile_eastward_fraction",
        "strong_given_mobile_fraction", "joint_fip_index",
    ]
    models = hac_table(annual, responses)
    annual.to_csv(PROCESSED / "annual_joint_fip_metrics.csv", index=False)
    models.to_csv(PROCESSED / "joint_warming_associations_hac.csv", index=False)
    events.to_csv(PROCESSED / "events_with_era5_intensity.csv.gz", index=False)

    cluster = events.groupby("track_cluster").agg(
        n=("event_id", "size"),
        median_height_deficit_gpm=("max_height_deficit_gpm", "median"),
        median_max_vorticity_s_1=("max_vorticity_s_1", "median"),
        median_dynamic_intensity_z=("dynamic_intensity_z", "median"),
        strong_event_fraction=("strong_event", "mean"),
        mean_duration_h=("duration_hours", "mean"),
        mean_displacement_km=("displacement_km", "mean"),
    )
    cluster.to_csv(PROCESSED / "track_cluster_intensity_summary.csv")

    summary = {
        "period": "1979-2023",
        "events": int(len(events)),
        "intensity_coverage": 1.0,
        "strong_threshold_dynamic_z": strong_threshold,
        "metrics": {},
    }
    for column in [
        "median_height_deficit_gpm", "q90_height_deficit_gpm",
        "median_max_vorticity_s_1", "q90_max_vorticity_s_1",
        "strong_event_fraction", "mobile_eastward_fraction", "joint_fip_index",
    ]:
        trend = linregress(annual.year, annual[column])
        correlation = pearsonr(annual[column], annual.gmst_anomaly_c)
        summary["metrics"][column] = {
            "mean": float(annual[column].mean()),
            "trend_per_decade": float(trend.slope * 10),
            "trend_p": float(trend.pvalue),
            "gmst_r": float(correlation.statistic),
            "gmst_p": float(correlation.pvalue),
        }
    (PROCESSED / "joint_summary_statistics.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    FIGURES.mkdir(exist_ok=True)
    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    panels = [
        ("median_height_deficit_gpm", "Median max height deficit (gpm)"),
        ("median_max_vorticity_s_1", "Median max vorticity (s$^{-1}$)"),
        ("strong_event_fraction", "Strong-event fraction"),
    ]
    for ax, (column, label) in zip(axes, panels):
        ax.plot(annual.year, annual[column], color="0.65", lw=1)
        ax.plot(annual.year, annual[column].rolling(11, center=True).mean(), color="#c43c39", lw=2)
        trend = linregress(annual.year, annual[column])
        ax.plot(annual.year, trend.intercept + trend.slope * annual.year, "--", color="#2b6cb0")
        ax.set_ylabel(label); ax.grid(alpha=0.2)
    axes[-1].set_xlabel("Year")
    fig.tight_layout(); fig.savefig(FIGURES / "annual_era5_intensity.png", dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(annual.year, annual.frequency_z, label="Frequency", lw=1.5)
    ax.plot(annual.year, annual.intensity_z, label="Intensity", lw=1.5)
    ax.plot(annual.year, annual.path_z, label="Path mobility", lw=1.5)
    ax.plot(annual.year, annual.joint_fip_index, label="Joint F-I-P", color="black", lw=2.3)
    ax.axhline(0, color="0.5", lw=0.8)
    ax.set(xlabel="Year", ylabel="Standardized annual index")
    ax.grid(alpha=0.2); ax.legend(ncol=4); fig.tight_layout()
    fig.savefig(FIGURES / "joint_frequency_intensity_path.png", dpi=180); plt.close(fig)

    print(f"events={len(events):,}; years={annual.year.min()}-{annual.year.max()}")
    print(cluster.to_string())
    print(models.to_string(index=False))


if __name__ == "__main__":
    main()
