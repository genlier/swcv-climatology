"""Path--intensity associations with fixed early-life windows and year-clustered inference."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "processed"
FIG = ROOT / "figures"
METRICS = {
    "height": "height_deficit_gpm_700_c25",
    "vorticity": "max_vorticity_s_1_700_c25",
}


def bh_adjust(values: pd.Series) -> pd.Series:
    result = pd.Series(np.nan, index=values.index)
    valid = values.dropna().sort_values()
    if valid.empty:
        return result
    n = len(valid)
    adjusted = (valid * n / np.arange(1, n + 1)).iloc[::-1].cummin().iloc[::-1].clip(upper=1)
    result.loc[adjusted.index] = adjusted
    return result


def zscore(series: pd.Series) -> pd.Series:
    return (series - series.mean()) / series.std(ddof=0)


def point_summaries(events: pd.DataFrame) -> pd.DataFrame:
    points = pd.read_csv(OUT / "terrain_qc_intensity_points.csv.gz", parse_dates=["time"])
    points = points.merge(events[["event_id", "genesis_time"]], on="event_id", validate="many_to_one")
    points["lag_hours"] = (points.time - points.genesis_time).dt.total_seconds() / 3600
    points = points.loc[points.lag_hours.ge(0)].sort_values(["event_id", "lag_hours"])
    rows = []
    for event_id, sample in points.groupby("event_id", sort=False):
        row = {"event_id": event_id}
        for label, column in METRICS.items():
            valid = sample.dropna(subset=[column]).sort_values("lag_hours")
            early = valid.loc[valid.lag_hours.le(6)]
            row[f"first_{label}"] = early[column].iloc[0] if len(early) else np.nan
            row[f"first_{label}_lag_h"] = early.lag_hours.iloc[0] if len(early) else np.nan
            for hours, minimum in ((12, 2), (24, 3)):
                window = valid.loc[valid.lag_hours.le(hours)]
                row[f"n_{label}_{hours}h"] = len(window)
                row[f"peak_{label}_{hours}h"] = window[column].max() if len(window) >= minimum else np.nan
                if len(window) >= minimum and window.lag_hours.nunique() >= 2:
                    row[f"slope_{label}_{hours}h_per6h"] = (
                        np.polyfit(window.lag_hours, window[column], 1)[0] * 6
                    )
                else:
                    row[f"slope_{label}_{hours}h_per6h"] = np.nan
            pair = valid.iloc[:2]
            consecutive = (
                len(pair) == 2
                and pair.lag_hours.iloc[0] <= 6
                and np.isclose(pair.lag_hours.iloc[1] - pair.lag_hours.iloc[0], 6)
            )
            row[f"growth_{label}_first_pair"] = (
                pair[column].iloc[1] - pair[column].iloc[0] if consecutive else np.nan
            )
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT / "path_intensity_fixed_window_event_metrics.csv.gz", index=False)
    return summary


def event_table() -> pd.DataFrame:
    events = pd.read_csv(
        OUT / "events_season_year_terrain_qc.csv.gz", parse_dates=["genesis_time", "lysis_time"]
    )
    events["eastward_reach_deg"] = events.max_lon - events.genesis_lon
    events["zonal_endpoint_deg"] = events.lysis_lon - events.genesis_lon
    events["meridional_endpoint_deg"] = events.lysis_lat - events.genesis_lat
    summaries = point_summaries(events)
    events = events.merge(summaries, on="event_id", how="left", validate="one_to_one")
    moisture = pd.read_csv(OUT / "jja_event_moisture_lag_event_level.csv.gz")
    moisture = moisture.loc[moisture.lag_hours.between(-48, -12)].groupby("event_id", as_index=False).agg(
        pregen_area_convergence_kg_m2_s=("area_convergence_anom_kg_m2_s", "mean")
    )
    events = events.merge(moisture, on="event_id", how="left", validate="one_to_one")
    events.to_csv(OUT / "path_intensity_fixed_window_analysis_table.csv.gz", index=False)
    return events


def fit_models(events: pd.DataFrame) -> pd.DataFrame:
    endpoint_specs = []
    for metric in METRICS:
        endpoint_specs.extend(
            [
                (metric, "first_0_to_6h", f"first_{metric}", None),
                (metric, "growth_first_consecutive_pair", f"growth_{metric}_first_pair", f"first_{metric}"),
                (metric, "peak_0_to_12h", f"peak_{metric}_12h", f"first_{metric}"),
                (metric, "peak_0_to_24h", f"peak_{metric}_24h", f"first_{metric}"),
                (metric, "lifecycle_peak", f"max_{'height_deficit_gpm' if metric == 'height' else 'vorticity_s_1'}_700_c25", None),
            ]
        )
    rows = []
    for metric, endpoint, response, initial in endpoint_specs:
        controls = ["genesis_lat", "genesis_lon"]
        if endpoint == "lifecycle_peak":
            controls.append("log_sampled_points")
        elif "12h" in endpoint:
            controls.append(f"n_{metric}_12h")
        elif "24h" in endpoint:
            controls.append(f"n_{metric}_24h")
        if initial:
            controls.append(initial)
        columns = [response, "eastward_reach_deg", "meridional_endpoint_deg", "season_year", "season", *controls]
        sample = events[columns].dropna().copy()
        for column in [response, "eastward_reach_deg", "meridional_endpoint_deg", *controls]:
            sample[f"z_{column}"] = zscore(sample[column])
        formula = f"z_{response} ~ z_eastward_reach_deg + z_meridional_endpoint_deg + C(season)"
        formula += " + " + " + ".join(f"z_{name}" for name in controls)
        fit = smf.ols(formula, sample).fit(
            cov_type="cluster", cov_kwds={"groups": sample.season_year, "use_correction": True}
        )
        for term in ["z_eastward_reach_deg", "z_meridional_endpoint_deg"]:
            ci = fit.conf_int().loc[term]
            rows.append(
                {
                    "model_family": "continuous_path",
                    "metric": metric,
                    "endpoint": endpoint,
                    "term": term,
                    "coefficient_sd": fit.params[term],
                    "ci95_low_sd": ci.iloc[0],
                    "ci95_high_sd": ci.iloc[1],
                    "p_year_cluster": fit.pvalues[term],
                    "n_events": int(fit.nobs),
                    "n_years": sample.season_year.nunique(),
                    "formula": formula,
                }
            )

        cluster_formula = f"z_{response} ~ C(track_cluster, Treatment(reference=1)) + C(season)"
        cluster_formula += " + " + " + ".join(f"z_{name}" for name in controls)
        cluster_fit = smf.ols(cluster_formula, sample.join(events.loc[sample.index, ["track_cluster"]])).fit(
            cov_type="cluster", cov_kwds={"groups": sample.season_year, "use_correction": True}
        )
        for cluster in (0, 2):
            term = f"C(track_cluster, Treatment(reference=1))[T.{cluster}]"
            ci = cluster_fit.conf_int().loc[term]
            rows.append(
                {
                    "model_family": "track_cluster",
                    "metric": metric,
                    "endpoint": endpoint,
                    "term": f"cluster{cluster}_vs_cluster1",
                    "coefficient_sd": cluster_fit.params[term],
                    "ci95_low_sd": ci.iloc[0],
                    "ci95_high_sd": ci.iloc[1],
                    "p_year_cluster": cluster_fit.pvalues[term],
                    "n_events": int(cluster_fit.nobs),
                    "n_years": sample.season_year.nunique(),
                    "formula": cluster_formula,
                }
            )
    result = pd.DataFrame(rows)
    result["p_fdr_bh_within_metric_term"] = result.groupby(
        ["model_family", "metric", "term"]
    ).p_year_cluster.transform(bh_adjust)
    result.to_csv(OUT / "path_intensity_fixed_window_models.csv", index=False)
    return result


def jja_moisture_sensitivity(events: pd.DataFrame) -> pd.DataFrame:
    rows = []
    jja = events.loc[events.season.eq("JJA")].copy()
    for metric in METRICS:
        for endpoint, response, n_control in (
            ("growth_first_consecutive_pair", f"growth_{metric}_first_pair", None),
            ("peak_0_to_12h", f"peak_{metric}_12h", f"n_{metric}_12h"),
            ("peak_0_to_24h", f"peak_{metric}_24h", f"n_{metric}_24h"),
        ):
            controls = [f"first_{metric}", "genesis_lat", "genesis_lon"]
            if n_control:
                controls.append(n_control)
            columns = [response, "pregen_area_convergence_kg_m2_s", "eastward_reach_deg", "season_year", *controls]
            sample = jja[columns].dropna().copy()
            for column in [response, "pregen_area_convergence_kg_m2_s", "eastward_reach_deg", *controls]:
                sample[f"z_{column}"] = zscore(sample[column])
            formula = f"z_{response} ~ z_pregen_area_convergence_kg_m2_s + z_eastward_reach_deg"
            formula += " + " + " + ".join(f"z_{name}" for name in controls)
            fit = smf.ols(formula, sample).fit(
                cov_type="cluster", cov_kwds={"groups": sample.season_year, "use_correction": True}
            )
            for term in ["z_pregen_area_convergence_kg_m2_s", "z_eastward_reach_deg"]:
                ci = fit.conf_int().loc[term]
                rows.append(
                    {
                        "metric": metric,
                        "endpoint": endpoint,
                        "term": term,
                        "coefficient_sd": fit.params[term],
                        "ci95_low_sd": ci.iloc[0],
                        "ci95_high_sd": ci.iloc[1],
                        "p_year_cluster": fit.pvalues[term],
                        "n_events": int(fit.nobs),
                        "n_years": sample.season_year.nunique(),
                        "formula": formula,
                    }
                )
    result = pd.DataFrame(rows)
    result["p_fdr_bh_within_metric_term"] = result.groupby(
        ["metric", "term"]
    ).p_year_cluster.transform(bh_adjust)
    result.to_csv(OUT / "jja_path_intensity_moisture_sensitivity.csv", index=False)
    return result


def plot_results(models: pd.DataFrame) -> None:
    subset = models.loc[
        (models.model_family.eq("continuous_path")) & models.term.eq("z_eastward_reach_deg")
    ].copy()
    endpoints = ["first_0_to_6h", "growth_first_consecutive_pair", "peak_0_to_12h", "peak_0_to_24h", "lifecycle_peak"]
    labels = ["First (0–6 h)", "First-pair growth", "Peak 0–12 h", "Peak 0–24 h", "Lifecycle peak"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, constrained_layout=True)
    for ax, metric in zip(axes, ["height", "vorticity"]):
        sample = subset.loc[subset.metric.eq(metric)].set_index("endpoint").loc[endpoints]
        y = np.arange(len(endpoints))
        ax.errorbar(
            sample.coefficient_sd,
            y,
            xerr=[sample.coefficient_sd - sample.ci95_low_sd, sample.ci95_high_sd - sample.coefficient_sd],
            fmt="o",
            color="#2a6f97",
            capsize=3,
        )
        ax.axvline(0, color="0.3", linewidth=1)
        ax.set_yticks(y, labels)
        ax.set_title(f"{metric.title()} vs eastward reach")
        ax.set_xlabel("Standardized coefficient (year-clustered 95% CI)")
        ax.grid(alpha=0.25)
        ax.invert_yaxis()
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "path_intensity_fixed_window_coefficients.png", dpi=180)
    plt.close(fig)


def main() -> None:
    events = event_table()
    models = fit_models(events)
    moisture = jja_moisture_sensitivity(events)
    plot_results(models)
    print("Continuous eastward-reach models")
    print(models.loc[(models.model_family == "continuous_path") & (models.term == "z_eastward_reach_deg")].to_string(index=False))
    print("\nJJA moisture sensitivity")
    print(moisture.to_string(index=False))


if __name__ == "__main__":
    main()
