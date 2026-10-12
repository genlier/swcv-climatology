"""Build a reproducible frequency/persistence/path climatology for SWCV tracks.

Dynamic vortex intensity is deliberately not inferred from trajectories.  It
requires collocated 700-hPa ERA5 fields and is handled by a separate stage.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import linregress, pearsonr
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "swcv_zenodo_15073815"
OUT = ROOT / "data" / "processed"
FIG = ROOT / "figures"


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))


def read_tracks() -> tuple[pd.DataFrame, pd.DataFrame]:
    points, events = [], []
    for path in sorted(RAW.glob("[12][0-9][0-9][0-9].txt")):
        frame = pd.read_csv(path, sep=r"\s+", names=["step", "lat", "lon", "time"], dtype={"time": str})
        frame["time"] = pd.to_datetime(frame["time"], format="%Y%m%d%H", utc=True)
        frame["new"] = frame["step"].eq(0)
        frame["event_no"] = frame["new"].cumsum() - 1
        frame["event_id"] = path.stem + "_" + frame["event_no"].astype(str).str.zfill(3)
        for event_id, group in frame.groupby("event_id", sort=False):
            group = group.sort_values("time").copy()
            segment = haversine_km(
                group["lat"].to_numpy()[:-1], group["lon"].to_numpy()[:-1],
                group["lat"].to_numpy()[1:], group["lon"].to_numpy()[1:]
            ) if len(group) > 1 else np.array([])
            elapsed = (group["time"].iloc[-1] - group["time"].iloc[0]).total_seconds() / 3600
            duration = elapsed + 1.0
            displacement = float(haversine_km(group["lat"].iloc[0], group["lon"].iloc[0], group["lat"].iloc[-1], group["lon"].iloc[-1]))
            path_length = float(segment.sum())
            synoptic = group.iloc[::6]
            if synoptic.index[-1] != group.index[-1]:
                synoptic = pd.concat([synoptic, group.iloc[[-1]]])
            synoptic_segment = haversine_km(
                synoptic["lat"].to_numpy()[:-1], synoptic["lon"].to_numpy()[:-1],
                synoptic["lat"].to_numpy()[1:], synoptic["lon"].to_numpy()[1:]
            ) if len(synoptic) > 1 else np.array([])
            events.append({
                "event_id": event_id,
                "year": int(path.stem),
                "genesis_time": group["time"].iloc[0],
                "lysis_time": group["time"].iloc[-1],
                "genesis_lat": group["lat"].iloc[0],
                "genesis_lon": group["lon"].iloc[0],
                "lysis_lat": group["lat"].iloc[-1],
                "lysis_lon": group["lon"].iloc[-1],
                "observed_hours": len(group),
                "duration_hours": duration,
                "path_length_km": path_length,
                "synoptic_path_km": float(synoptic_segment.sum()),
                "displacement_km": displacement,
                "mean_translation_kmh": path_length / elapsed if elapsed > 0 else 0.0,
                "max_lon": group["lon"].max(),
                "max_lat": group["lat"].max(),
            })
            group["segment_km"] = np.r_[0.0, segment]
            points.append(group.drop(columns=["new", "event_no"]))
    return pd.concat(points, ignore_index=True), pd.DataFrame(events)


def add_track_clusters(events: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    # Endpoint/displacement clustering is more robust than hourly accumulated
    # length because the published nearest-neighbour tracker permits 400-km jumps.
    features = ["genesis_lat", "genesis_lon", "lysis_lat", "lysis_lon", "duration_hours", "displacement_km"]
    x = StandardScaler().fit_transform(events[features])
    scores = {}
    for k in range(2, 7):
        labels = KMeans(n_clusters=k, n_init=30, random_state=42).fit_predict(x)
        scores[k] = silhouette_score(x, labels)
    best_k = max(scores, key=scores.get)
    labels = KMeans(n_clusters=best_k, n_init=50, random_state=42).fit_predict(x)
    events = events.copy()
    events["track_cluster"] = labels
    summary = events.groupby("track_cluster")[features + ["displacement_km"]].mean().round(2)
    order = summary["lysis_lon"].sort_values().index
    rename = {old: new for new, old in enumerate(order)}
    events["track_cluster"] = events["track_cluster"].map(rename)
    return events, {"best_k": int(best_k), "silhouette": scores}


def read_gistemp() -> pd.DataFrame:
    path = ROOT / "data" / "raw" / "gistemp_v4_global.csv"
    data = pd.read_csv(path, skiprows=1, na_values="***")
    data = data.rename(columns={"Year": "year", "J-D": "gmst_anomaly_c"})
    return data[["year", "gmst_anomaly_c"]]


def annual_metrics(events: pd.DataFrame) -> pd.DataFrame:
    grouped = events.groupby("year")
    annual = grouped.agg(
        frequency=("event_id", "size"),
        vortex_hours=("observed_hours", "sum"),
        median_duration_h=("duration_hours", "median"),
        mean_duration_h=("duration_hours", "mean"),
        median_path_km=("synoptic_path_km", "median"),
        mean_path_km=("synoptic_path_km", "mean"),
        median_displacement_km=("displacement_km", "median"),
        eastward_escape_fraction=("max_lon", lambda x: float((x >= 110).mean())),
    ).reset_index()
    for cluster in sorted(events["track_cluster"].unique()):
        counts = events.loc[events["track_cluster"].eq(cluster)].groupby("year").size()
        annual[f"cluster_{cluster}_frequency"] = annual["year"].map(counts).fillna(0).astype(int)
    month = events["genesis_time"].dt.month
    season = pd.Series(np.select(
        [month.isin([12, 1, 2]), month.isin([3, 4, 5]), month.isin([6, 7, 8])],
        ["DJF", "MAM", "JJA"], default="SON"), index=events.index)
    for name in ["DJF", "MAM", "JJA", "SON"]:
        counts = events.loc[season.eq(name)].groupby("year").size()
        annual[f"{name}_frequency"] = annual["year"].map(counts).fillna(0).astype(int)
    return annual.merge(read_gistemp(), on="year", how="left")


def hac_models(annual: pd.DataFrame, start=1979) -> pd.DataFrame:
    data = annual.loc[annual["year"] >= start].dropna(subset=["gmst_anomaly_c"]).copy()
    responses = ["frequency", "vortex_hours", "median_duration_h", "median_path_km", "eastward_escape_fraction",
                 "DJF_frequency", "MAM_frequency", "JJA_frequency", "SON_frequency"]
    rows = []
    for response in responses:
        for spec, predictors in [("gmst_only", ["gmst_anomaly_c"]), ("gmst_plus_year", ["gmst_anomaly_c", "year"] )]:
            x = data[predictors].copy()
            if "year" in x:
                x["year"] = x["year"] - x["year"].mean()
            fit = sm.OLS(data[response], sm.add_constant(x)).fit(cov_type="HAC", cov_kwds={"maxlags": 3})
            rows.append({
                "period": f"{start}-{int(data.year.max())}", "response": response, "specification": spec,
                "gmst_coefficient_per_c": fit.params["gmst_anomaly_c"],
                "gmst_se_hac": fit.bse["gmst_anomaly_c"], "gmst_p_hac": fit.pvalues["gmst_anomaly_c"],
                "r_squared": fit.rsquared, "n_years": int(fit.nobs),
            })
    return pd.DataFrame(rows)


def residual_block_trend_band(
    years: np.ndarray,
    values: np.ndarray,
    block_length: int = 5,
    replicates: int = 2000,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Pointwise 95% band for a linear trend from a residual moving-block bootstrap."""
    x = np.asarray(years, dtype=float)
    y = np.asarray(values, dtype=float)
    base = linregress(x, y)
    fitted = base.intercept + base.slope * x
    residuals = y - fitted
    rng = np.random.default_rng(seed)
    n = len(x)
    starts = np.arange(n - block_length + 1)
    predictions = np.empty((replicates, n), dtype=float)
    n_blocks = math.ceil(n / block_length)
    for i in range(replicates):
        sampled_starts = rng.choice(starts, size=n_blocks, replace=True)
        sampled = np.concatenate([residuals[s:s + block_length] for s in sampled_starts])[:n]
        fit = linregress(x, fitted + sampled)
        predictions[i] = fit.intercept + fit.slope * x
    return np.quantile(predictions, 0.025, axis=0), np.quantile(predictions, 0.975, axis=0)


def figures(points: pd.DataFrame, events: pd.DataFrame, annual: pd.DataFrame) -> None:
    FIG.mkdir(exist_ok=True)
    primary = annual.loc[annual.year >= 1979]
    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    series = [("frequency", "Events yr$^{-1}$"), ("median_duration_h", "Median duration (h)"), ("median_path_km", "Median path length (km)")]
    for ax, (column, ylabel) in zip(axes, series):
        ax.plot(annual.year, annual[column], color="0.65", lw=1)
        ax.plot(annual.year, annual[column].rolling(11, center=True).mean(), color="#c43c39", lw=2)
        trend = linregress(primary.year, primary[column])
        low, high = residual_block_trend_band(
            primary.year.to_numpy(), primary[column].to_numpy(), seed=42
        )
        ax.fill_between(primary.year, low, high, color="#2b6cb0", alpha=.16, linewidth=0)
        ax.plot(primary.year, trend.intercept + trend.slope * primary.year, "--", color="#2b6cb0")
        ax.set_ylabel(ylabel); ax.grid(alpha=.2)
    axes[-1].set_xlabel("Year")
    fig.tight_layout(); fig.savefig(FIG / "annual_frequency_persistence_path.png", dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 7))
    sample_ids = events.groupby("track_cluster", group_keys=False).apply(lambda x: x.sample(min(len(x), 250), random_state=42), include_groups=False).index
    sample_events = events.loc[sample_ids]
    colors = plt.cm.tab10(np.linspace(0, 1, events.track_cluster.nunique()))
    for cluster, color in zip(sorted(events.track_cluster.unique()), colors):
        ids = set(sample_events.loc[sample_events.track_cluster.eq(cluster), "event_id"])
        for _, track in points.loc[points.event_id.isin(ids)].groupby("event_id"):
            ax.plot(track.lon, track.lat, color=color, alpha=.08, lw=.7)
        sub = events.loc[events.track_cluster.eq(cluster)]
        ax.scatter(sub.genesis_lon.mean(), sub.genesis_lat.mean(), color=color, s=70, marker="o", label=f"Cluster {cluster} (n={len(sub)})")
        ax.scatter(sub.lysis_lon.mean(), sub.lysis_lat.mean(), color=color, s=80, marker="x")
    ax.set(xlim=(95, 125), ylim=(20, 40), xlabel="Longitude (°E)", ylabel="Latitude (°N)")
    ax.grid(alpha=.2); ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(FIG / "track_clusters.png", dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(primary.gmst_anomaly_c, primary.frequency, color="#2b6cb0")
    fit = linregress(primary.gmst_anomaly_c, primary.frequency)
    xx = np.linspace(primary.gmst_anomaly_c.min(), primary.gmst_anomaly_c.max())
    ax.plot(xx, fit.intercept + fit.slope * xx, color="#c43c39")
    ax.set(xlabel="GISTEMP global anomaly (°C)", ylabel="Annual SWCV frequency")
    ax.grid(alpha=.2); fig.tight_layout(); fig.savefig(FIG / "frequency_vs_gmst.png", dpi=180); plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    points, events = read_tracks()
    events, cluster_meta = add_track_clusters(events)
    annual = annual_metrics(events)
    models = hac_models(annual)
    points.to_csv(OUT / "track_points.csv.gz", index=False)
    events.to_csv(OUT / "events.csv", index=False)
    annual.to_csv(OUT / "annual_metrics.csv", index=False)
    models.to_csv(OUT / "warming_associations_hac.csv", index=False)
    decade = annual.assign(decade=(annual.year // 10) * 10).groupby("decade").agg(
        years=("year", "size"), frequency=("frequency", "mean"), vortex_hours=("vortex_hours", "mean"),
        median_duration_h=("median_duration_h", "mean"), median_path_km=("median_path_km", "mean"),
        eastward_escape_fraction=("eastward_escape_fraction", "mean"),
        DJF_frequency=("DJF_frequency", "mean"), MAM_frequency=("MAM_frequency", "mean"),
        JJA_frequency=("JJA_frequency", "mean"), SON_frequency=("SON_frequency", "mean"),
    )
    decade.to_csv(OUT / "decadal_metrics.csv")
    cluster_summary = events.groupby("track_cluster").agg(
        n=("event_id", "size"), genesis_lat=("genesis_lat", "mean"), genesis_lon=("genesis_lon", "mean"),
        lysis_lat=("lysis_lat", "mean"), lysis_lon=("lysis_lon", "mean"), duration_h=("duration_hours", "mean"),
        synoptic_path_km=("synoptic_path_km", "mean"), displacement_km=("displacement_km", "mean"),
    ).round(2)
    cluster_summary.to_csv(OUT / "track_cluster_summary.csv")
    (OUT / "cluster_diagnostics.json").write_text(json.dumps(cluster_meta, indent=2), encoding="utf-8")
    primary = annual.loc[annual.year >= 1979]
    summary = {"n_points": len(points), "n_events": len(events), "primary_period": "1979-2023"}
    for column in ["frequency", "vortex_hours", "median_duration_h", "median_path_km", "eastward_escape_fraction"]:
        trend = linregress(primary.year, primary[column])
        summary[column] = {"mean": float(primary[column].mean()), "trend_per_decade": float(trend.slope * 10), "trend_p": float(trend.pvalue)}
    summary["frequency_gmst_pearson"] = dict(zip(["r", "p"], map(float, pearsonr(primary.frequency, primary.gmst_anomaly_c))))
    (OUT / "summary_statistics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    figures(points, events, annual)
    print(f"points={len(points):,} events={len(events):,} years={annual.year.min()}-{annual.year.max()}")
    print(cluster_summary.to_string())
    print(models.to_string(index=False))


if __name__ == "__main__":
    main()
