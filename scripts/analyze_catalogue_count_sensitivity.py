"""Test headline trends under unknown exclusion of 134 catalogue events."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import linregress


ROOT = Path(__file__).resolve().parents[1]
EVENTS = ROOT / "data" / "processed" / "events.csv"
OUT = ROOT / "data" / "processed"
N_REMOVE = 134
REPLICATES = 2000
SEED = 20260912


def annual_counts(events: pd.DataFrame, metric: str) -> pd.DataFrame:
    if metric == "annual_all":
        years = np.arange(1979, 2024)
        selected = events.loc[events.year.between(1979, 2023)]
        key = "year"
    elif metric == "djf_cluster1":
        years = np.arange(1980, 2024)
        selected = events.loc[
            events.season.eq("DJF")
            & events.season_year.between(1980, 2023)
            & events.track_cluster.eq(1)
        ]
        key = "season_year"
    else:
        raise ValueError(metric)
    counts = selected.groupby(key).size().reindex(years, fill_value=0)
    return pd.DataFrame({"year": years, "count": counts.to_numpy()})


def trend(events: pd.DataFrame, metric: str) -> float:
    table = annual_counts(events, metric)
    if metric == "annual_all":
        return float(linregress(table.year, table["count"]).slope * 10)
    centered = table.year - table.year.mean()
    fit = sm.GLM(table["count"], sm.add_constant(centered), family=sm.families.Poisson()).fit()
    return float(100 * np.expm1(fit.params.iloc[1] * 10))


def adversarial(events: pd.DataFrame, metric: str, side: str) -> float:
    if metric == "annual_all":
        eligible = events.loc[events.year.between(1979, 2023)].sort_values("year")
    else:
        eligible = events.loc[
            events.season.eq("DJF")
            & events.season_year.between(1980, 2023)
            & events.track_cluster.eq(1)
        ].sort_values("season_year")
    if side == "late":
        eligible = eligible.iloc[::-1]
    remove = set(eligible.head(N_REMOVE).event_id)
    return trend(events.loc[~events.event_id.isin(remove)], metric)


def main() -> None:
    events = pd.read_csv(EVENTS, parse_dates=['genesis_time'])
    genesis = pd.to_datetime(events.genesis_time, utc=True)
    events['season_year'] = genesis.dt.year + genesis.dt.month.eq(12).astype(int)
    events['season'] = np.where(genesis.dt.month.isin([12, 1, 2]), 'DJF', 'other')
    assert len(events) == 9513 and events.event_id.nunique() == 9513
    assert annual_counts(events, 'annual_all')['count'].sum() == 5221
    rng = np.random.default_rng(SEED)
    event_ids = events.event_id.to_numpy()
    rows = []
    for replicate in range(REPLICATES):
        remove = set(rng.choice(event_ids, size=N_REMOVE, replace=False))
        retained = events.loc[~events.event_id.isin(remove)]
        for metric in ("annual_all", "djf_cluster1"):
            rows.append({"replicate": replicate, "metric": metric, "trend": trend(retained, metric)})
    simulations = pd.DataFrame(rows)
    simulations.to_csv(OUT / "catalogue_134_corrected_random_exclusion_sensitivity.csv.gz", index=False)

    summary_rows = []
    for metric in ("annual_all", "djf_cluster1"):
        values = simulations.loc[simulations.metric.eq(metric), "trend"]
        summary_rows.append(
            {
                "metric": metric,
                "unit": "events per decade" if metric == "annual_all" else "percent per decade",
                "full_sample": trend(events, metric),
                "random_delete_mean": values.mean(),
                "random_delete_q025": values.quantile(0.025),
                "random_delete_q975": values.quantile(0.975),
                "adversarial_delete_earliest": adversarial(events, metric, "early"),
                "adversarial_delete_latest": adversarial(events, metric, "late"),
                "removed_events": N_REMOVE,
                "replicates": REPLICATES,
            }
        )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUT / "catalogue_134_corrected_exclusion_sensitivity_summary.csv", index=False)
    (OUT / "catalogue_134_corrected_exclusion_sensitivity_status.json").write_text(
        json.dumps(summary.to_dict(orient="records"), indent=2), encoding="utf-8"
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
