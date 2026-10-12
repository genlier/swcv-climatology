"""Frequency--intensity--path sensitivity to the 700-hPa terrain mask.

The filtered event count is a conservative sensitivity bound, not a new SWCV
climatology: filtering is applied after the published tracks were identified.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import linregress, pearsonr


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "processed"
FIG = ROOT / "figures"


def zscore(series: pd.Series) -> pd.Series:
    return (series - series.mean()) / series.std(ddof=0)


def diagnostics(year: pd.Series, value: pd.Series, gmst: pd.Series) -> dict[str, float]:
    valid = pd.DataFrame({"year": year, "value": value, "gmst": gmst}).dropna()
    trend = linregress(valid.year, valid.value)
    corr = pearsonr(valid.gmst, valid.value)
    return {
        "mean": float(valid.value.mean()),
        "trend_per_decade": float(trend.slope * 10),
        "trend_p": float(trend.pvalue),
        "gmst_r": float(corr.statistic),
        "gmst_p": float(corr.pvalue),
        "n_years": int(len(valid)),
    }


def hac_rows(annual: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    rows = []
    for metric in metrics:
        sample = annual[["year_centered", "gmst_anomaly_c", metric]].dropna()
        for specification, predictors in (
            ("gmst_only", ["gmst_anomaly_c"]),
            ("gmst_plus_year", ["gmst_anomaly_c", "year_centered"]),
        ):
            fit = sm.OLS(sample[metric], sm.add_constant(sample[predictors])).fit(
                cov_type="HAC", cov_kwds={"maxlags": 3}
            )
            rows.append(
                {
                    "response": metric,
                    "specification": specification,
                    "gmst_coefficient_per_c": fit.params["gmst_anomaly_c"],
                    "gmst_se_hac": fit.bse["gmst_anomaly_c"],
                    "gmst_p_hac": fit.pvalues["gmst_anomaly_c"],
                    "r_squared": fit.rsquared,
                    "n_years": int(fit.nobs),
                }
            )
    return pd.DataFrame(rows)


def season(month: pd.Series) -> pd.Series:
    return pd.Series(
        np.select(
            [month.isin([12, 1, 2]), month.isin([3, 4, 5]), month.isin([6, 7, 8])],
            ["DJF", "MAM", "JJA"],
            default="SON",
        ),
        index=month.index,
    )


def main() -> None:
    events = pd.read_csv(OUT / "events_terrain_qc_intensity.csv.gz", parse_dates=["genesis_time"])
    events = events.loc[events.year.between(1979, 2023)].copy()
    events["intensity_valid_700_c25"] = (
        events.max_height_deficit_gpm_700_c25.notna()
        & events.max_vorticity_s_1_700_c25.notna()
    )
    events["terrain_safe_700_c25"] = (
        events.intensity_valid_700_c25
        & events.center_within_25hpa_700_fraction.eq(0)
    )
    events["mobile_eastward"] = events.track_cluster.eq(2)
    events["eastward_escape"] = events.max_lon.ge(110)
    events["season"] = season(events.genesis_time.dt.month)

    valid = events.loc[events.intensity_valid_700_c25].copy()
    valid["height_z"] = zscore(valid.max_height_deficit_gpm_700_c25)
    valid["vorticity_z"] = zscore(valid.max_vorticity_s_1_700_c25)
    valid["dynamic_intensity_z_700_c25"] = (valid.height_z + valid.vorticity_z) / 2
    events = events.merge(
        valid[["event_id", "dynamic_intensity_z_700_c25"]], on="event_id", how="left",
        validate="one_to_one",
    )

    strict = events.loc[events.terrain_safe_700_c25].copy()
    annual_all = events.groupby("year").agg(
        original_frequency=("event_id", "size"),
        center_safe_frequency=("center_within_25hpa_700_fraction", lambda x: x.eq(0).sum()),
    )
    annual_strict = strict.groupby("year").agg(
        strict_frequency=("event_id", "size"),
        median_height_deficit_700_c25=("max_height_deficit_gpm_700_c25", "median"),
        q90_height_deficit_700_c25=("max_height_deficit_gpm_700_c25", lambda x: x.quantile(0.9)),
        median_vorticity_700_c25=("max_vorticity_s_1_700_c25", "median"),
        q90_vorticity_700_c25=("max_vorticity_s_1_700_c25", lambda x: x.quantile(0.9)),
        median_dynamic_intensity_z_700_c25=("dynamic_intensity_z_700_c25", "median"),
        median_displacement_km=("displacement_km", "median"),
        mobile_eastward_fraction=("mobile_eastward", "mean"),
        eastward_escape_fraction=("eastward_escape", "mean"),
    )
    annual = annual_all.join(annual_strict).reset_index()
    annual["center_safe_retention"] = annual.center_safe_frequency / annual.original_frequency
    annual["strict_retention"] = annual.strict_frequency / annual.original_frequency

    gmst = pd.read_csv(ROOT / "data" / "raw" / "gistemp_v4_global.csv", skiprows=1, na_values="***")
    gmst = gmst.rename(columns={"Year": "year", "J-D": "gmst_anomaly_c"})
    annual = annual.merge(gmst[["year", "gmst_anomaly_c"]], on="year", validate="one_to_one")
    annual["year_centered"] = annual.year - annual.year.mean()
    annual["frequency_z"] = zscore(annual.strict_frequency)
    annual["intensity_z"] = zscore(annual.median_dynamic_intensity_z_700_c25)
    annual["path_z"] = (
        zscore(annual.median_displacement_km)
        + zscore(annual.mobile_eastward_fraction)
        + zscore(annual.eastward_escape_fraction)
    ) / 3
    annual["strict_joint_fip_index"] = (
        annual.frequency_z + annual.intensity_z + annual.path_z
    ) / 3
    annual.to_csv(OUT / "annual_terrain_qc_joint_sensitivity.csv", index=False)

    metrics = [
        "original_frequency", "center_safe_frequency", "strict_frequency",
        "center_safe_retention", "strict_retention",
        "median_height_deficit_700_c25", "q90_height_deficit_700_c25",
        "median_vorticity_700_c25", "q90_vorticity_700_c25",
        "mobile_eastward_fraction", "eastward_escape_fraction",
        "strict_joint_fip_index",
    ]
    hac_rows(annual, metrics).to_csv(OUT / "terrain_qc_sensitivity_hac.csv", index=False)

    seasonal = events.groupby(["year", "season"]).agg(
        original_frequency=("event_id", "size"),
        center_safe_frequency=("center_within_25hpa_700_fraction", lambda x: x.eq(0).sum()),
        strict_frequency=("terrain_safe_700_c25", "sum"),
    ).reset_index()
    seasonal["center_safe_retention"] = seasonal.center_safe_frequency / seasonal.original_frequency
    seasonal["strict_retention"] = seasonal.strict_frequency / seasonal.original_frequency
    seasonal = seasonal.merge(gmst[["year", "gmst_anomaly_c"]], on="year", validate="many_to_one")
    seasonal["year_centered"] = seasonal.year - seasonal.year.mean()
    seasonal.to_csv(OUT / "seasonal_terrain_qc_frequency_sensitivity.csv", index=False)
    seasonal_hac = []
    for name, sample in seasonal.groupby("season"):
        table = hac_rows(
            sample,
            ["original_frequency", "center_safe_frequency", "strict_frequency", "strict_retention"],
        )
        table.insert(0, "season", name)
        seasonal_hac.append(table)
    pd.concat(seasonal_hac, ignore_index=True).to_csv(
        OUT / "seasonal_terrain_qc_sensitivity_hac.csv", index=False
    )

    cluster = events.groupby("track_cluster").agg(
        events=("event_id", "size"),
        center_safe_fraction=("center_within_25hpa_700_fraction", lambda x: x.eq(0).mean()),
        intensity_valid_fraction=("intensity_valid_700_c25", "mean"),
        strict_retention=("terrain_safe_700_c25", "mean"),
    ).reset_index()
    cluster.to_csv(OUT / "terrain_qc_cluster_retention.csv", index=False)

    summary = {
        "period": "1979-2023",
        "interpretation": (
            "Filtered counts are conservative post-identification sensitivity bounds, "
            "not replacement SWCV frequencies."
        ),
        "events": int(len(events)),
        "center_safe_events": int(events.center_within_25hpa_700_fraction.eq(0).sum()),
        "intensity_valid_events": int(events.intensity_valid_700_c25.sum()),
        "strict_events": int(events.terrain_safe_700_c25.sum()),
        "center_safe_fraction": float(events.center_within_25hpa_700_fraction.eq(0).mean()),
        "intensity_valid_fraction": float(events.intensity_valid_700_c25.mean()),
        "strict_fraction": float(events.terrain_safe_700_c25.mean()),
        "metrics": {metric: diagnostics(annual.year, annual[metric], annual.gmst_anomaly_c) for metric in metrics},
        "seasonal_frequency": {},
        "cluster_retention": cluster.to_dict(orient="records"),
    }
    for name, sample in seasonal.groupby("season"):
        summary["seasonal_frequency"][name] = {
            metric: diagnostics(sample.year, sample[metric], sample.gmst_anomaly_c)
            for metric in ["original_frequency", "center_safe_frequency", "strict_frequency", "strict_retention"]
        }
    (OUT / "terrain_qc_sensitivity_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    FIG.mkdir(exist_ok=True)
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    panels = [
        ("strict_retention", "Strict retained fraction"),
        ("median_height_deficit_700_c25", "Median terrain-QC deficit (gpm)"),
        ("mobile_eastward_fraction", "Strict-sample mobile fraction"),
    ]
    for ax, (metric, ylabel) in zip(axes, panels):
        ax.plot(annual.year, annual[metric], color="0.65", lw=1)
        ax.plot(annual.year, annual[metric].rolling(11, center=True).mean(), color="#c43c39", lw=2)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.2)
    axes[-1].set_xlabel("Year")
    fig.tight_layout()
    fig.savefig(FIG / "terrain_qc_joint_sensitivity.png", dpi=180)
    plt.close(fig)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
