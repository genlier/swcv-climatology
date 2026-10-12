"""Season-year SWCV responses with parsimonious internal-variability controls.

Inference is performed on season-year aggregates (1980--2023), not thousands
of events, because GMST and climate indices vary at the year/season level.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "processed"
INDICES = ROOT / "data" / "indices"
FIG = ROOT / "figures"
SEASONS = ["DJF", "MAM", "JJA", "SON"]
SEASON_ORDER = {name: number for number, name in enumerate(SEASONS)}
RESPONSES = [
    "frequency", "median_height_lifecycle", "q90_height_lifecycle",
    "median_vorticity_lifecycle", "q90_vorticity_lifecycle",
    "mobile_eastward_fraction", "eastward_escape_fraction", "median_displacement_km",
]


def specifications_for_season(name: str) -> list[tuple[str, list[str]]]:
    specs = [
        ("trend_only", ["year_centered"]),
        ("gmst_only", ["gmst_anomaly_c"]),
        ("gmst_nino_same", ["gmst_anomaly_c", "nino34_c_dt_z"]),
        ("gmst_nino_lag1", ["gmst_anomaly_c", "nino34_c_lag1_dt_z"]),
        ("gmst_pdo_same", ["gmst_anomaly_c", "pdo_index_dt_z"]),
    ]
    if name in {"MAM", "JJA"}:
        specs.append(("gmst_iobm_same", ["gmst_anomaly_c", "iobm_40_100e_c_dt_z"]))
        specs.append(("gmst_iobm110_same", ["gmst_anomaly_c", "iobm_40_110e_c_dt_z"]))
    if name == "SON":
        specs.append(("gmst_dmi_same", ["gmst_anomaly_c", "dmi_c_dt_z"]))
    if name == "JJA":
        specs.extend(
            [
                ("gmst_prev_djf_nino", ["gmst_anomaly_c", "nino34_prev_djf_dt_z"]),
                ("gmst_nino_iobm_chain", ["gmst_anomaly_c", "nino34_prev_djf_dt_z", "iobm_independent_dt_z"]),
                ("gmst_nino_iobm110_chain", ["gmst_anomaly_c", "nino34_prev_djf_dt_z", "iobm110_independent_dt_z"]),
            ]
        )
    return specs


def add_season(frame: pd.DataFrame, date_column: str) -> pd.DataFrame:
    frame = frame.copy()
    date = frame[date_column]
    month = date.dt.month
    frame["season"] = np.select(
        [month.isin([12, 1, 2]), month.isin([3, 4, 5]), month.isin([6, 7, 8])],
        ["DJF", "MAM", "JJA"], default="SON",
    )
    frame["season_year"] = date.dt.year + month.eq(12).astype(int)
    frame["season_ord"] = frame.season.map(SEASON_ORDER)
    frame["season_index"] = frame.season_year * 4 + frame.season_ord
    return frame


def detrended_z(series: pd.Series, year: pd.Series) -> pd.Series:
    valid = series.notna() & year.notna()
    result = pd.Series(np.nan, index=series.index)
    x = sm.add_constant(year.loc[valid].astype(float))
    residual = sm.OLS(series.loc[valid], x).fit().resid
    result.loc[valid] = (residual - residual.mean()) / residual.std(ddof=0)
    return result


def bh_adjust(pvalues: pd.Series) -> pd.Series:
    result = pd.Series(np.nan, index=pvalues.index)
    valid = pvalues.dropna().sort_values()
    if valid.empty:
        return result
    n = len(valid)
    adjusted = (valid * n / np.arange(1, n + 1)).iloc[::-1].cummin().iloc[::-1].clip(upper=1)
    result.loc[adjusted.index] = adjusted
    return result


def fit_hac(sample: pd.DataFrame, response: str, predictors: list[str], specification: str) -> dict:
    columns = [response, *predictors]
    data = sample[columns].dropna()
    x = sm.add_constant(data[predictors])
    fit = sm.OLS(data[response], x).fit(cov_type="HAC", cov_kwds={"maxlags": 3})
    target = "year_centered" if specification == "trend_only" else "gmst_anomaly_c"
    ci = fit.conf_int().loc[target]
    standardized_x = (data[predictors] - data[predictors].mean()) / data[predictors].std(ddof=0)
    condition_number = float(np.linalg.cond(sm.add_constant(standardized_x).to_numpy()))
    row = {
        "response": response,
        "specification": specification,
        "target": target,
        "coefficient": float(fit.params[target] * (10 if target == "year_centered" else 1)),
        "se_hac": float(fit.bse[target] * (10 if target == "year_centered" else 1)),
        "p_hac": float(fit.pvalues[target]),
        "ci95_low": float(ci.iloc[0] * (10 if target == "year_centered" else 1)),
        "ci95_high": float(ci.iloc[1] * (10 if target == "year_centered" else 1)),
        "r_squared": float(fit.rsquared),
        "n_season_years": int(fit.nobs),
        "condition_number_standardized": condition_number,
    }
    for predictor in predictors:
        if predictor != target:
            row[f"coef_{predictor}"] = float(fit.params[predictor])
            row[f"p_{predictor}"] = float(fit.pvalues[predictor])
    return row


def climate_covariates() -> pd.DataFrame:
    climate = pd.read_csv(INDICES / "climate_indices_monthly.csv", parse_dates=["date"])
    iobm = pd.read_csv(INDICES / "iobm_ersstv5_monthly.csv", parse_dates=["date"])
    climate = climate.merge(
        iobm[["date", "iobm_40_100e_c", "iobm_40_110e_c"]], on="date", how="left",
        validate="one_to_one",
    )
    climate = add_season(climate, "date")
    variables = ["nino34_c", "pdo_index", "dmi_c", "iobm_40_100e_c", "iobm_40_110e_c"]
    seasonal = climate.groupby(["season_year", "season", "season_ord", "season_index"], as_index=False)[variables].mean()
    seasonal = seasonal.loc[seasonal.season_year.between(1980, 2023)].copy()
    seasonal = seasonal.sort_values("season_index")
    lagged = seasonal[["season_index", *variables]].copy()
    lagged["season_index"] += 1
    lagged = lagged.rename(columns={name: f"{name}_lag1" for name in variables})
    seasonal = seasonal.merge(lagged, on="season_index", how="left", validate="one_to_one")
    winter = seasonal.loc[seasonal.season.eq("DJF"), ["season_year", "nino34_c"]].rename(
        columns={"nino34_c": "nino34_prev_djf_c"}
    )
    seasonal = seasonal.merge(winter, on="season_year", how="left", validate="many_to_one")
    for variable in variables:
        seasonal[f"{variable}_dt_z"] = seasonal.groupby("season", group_keys=False).apply(
            lambda x: detrended_z(x[variable], x.season_year), include_groups=False
        )
        seasonal[f"{variable}_lag1_dt_z"] = seasonal.groupby("season", group_keys=False).apply(
            lambda x: detrended_z(x[f"{variable}_lag1"], x.season_year), include_groups=False
        )
    seasonal["nino34_prev_djf_dt_z"] = seasonal.groupby("season", group_keys=False).apply(
        lambda x: detrended_z(x.nino34_prev_djf_c, x.season_year), include_groups=False
    )
    jja = seasonal.season.eq("JJA")
    x = sm.add_constant(seasonal.loc[jja, "nino34_prev_djf_dt_z"])
    seasonal.loc[jja, "iobm_independent_dt_z"] = sm.OLS(
        seasonal.loc[jja, "iobm_40_100e_c_dt_z"], x, missing="drop"
    ).fit().resid
    seasonal.loc[jja, "iobm110_independent_dt_z"] = sm.OLS(
        seasonal.loc[jja, "iobm_40_110e_c_dt_z"], x, missing="drop"
    ).fit().resid
    seasonal.to_csv(OUT / "seasonal_climate_covariates_1980_2023.csv", index=False)
    return seasonal


def event_table() -> pd.DataFrame:
    events = pd.read_csv(OUT / "events_terrain_qc_intensity.csv.gz", parse_dates=["genesis_time"])
    points = pd.read_csv(OUT / "terrain_qc_intensity_points.csv.gz", parse_dates=["time"])
    first = points.sort_values(["event_id", "time"]).drop_duplicates("event_id").rename(
        columns={
            "height_deficit_gpm_700_c25": "first_height_700_c25",
            "max_vorticity_s_1_700_c25": "first_vorticity_700_c25",
        }
    )
    events = events.merge(
        first[["event_id", "first_height_700_c25", "first_vorticity_700_c25"]],
        on="event_id", how="left", validate="one_to_one",
    )
    events = add_season(events, "genesis_time")
    events = events.loc[events.season_year.between(1980, 2023)].copy()
    events["mobile_eastward"] = events.track_cluster.eq(2)
    events["eastward_escape"] = events.max_lon.ge(110)
    events["log_sampled_points"] = np.log1p(events.sampled_points)
    events.to_csv(OUT / "events_season_year_terrain_qc.csv.gz", index=False)
    return events


def aggregate(events: pd.DataFrame, climate: pd.DataFrame) -> pd.DataFrame:
    annual = events.groupby(["season_year", "season"], as_index=False).agg(
        frequency=("event_id", "size"),
        median_height_lifecycle=("max_height_deficit_gpm_700_c25", "median"),
        q90_height_lifecycle=("max_height_deficit_gpm_700_c25", lambda x: x.quantile(0.9)),
        median_vorticity_lifecycle=("max_vorticity_s_1_700_c25", "median"),
        q90_vorticity_lifecycle=("max_vorticity_s_1_700_c25", lambda x: x.quantile(0.9)),
        median_height_first=("first_height_700_c25", "median"),
        median_vorticity_first=("first_vorticity_700_c25", "median"),
        first_point_coverage=("first_height_700_c25", lambda x: x.notna().mean()),
        mobile_eastward_fraction=("mobile_eastward", "mean"),
        eastward_escape_fraction=("eastward_escape", "mean"),
        median_displacement_km=("displacement_km", "median"),
    )
    gmst = pd.read_csv(ROOT / "data" / "raw" / "gistemp_v4_global.csv", skiprows=1, na_values="***")
    gmst = gmst.rename(columns={"Year": "season_year", "J-D": "gmst_anomaly_c"})
    annual = annual.merge(gmst[["season_year", "gmst_anomaly_c"]], on="season_year", validate="many_to_one")
    annual = annual.merge(climate, on=["season_year", "season"], validate="one_to_one")
    annual["year_centered"] = annual.season_year - annual.season_year.mean()
    annual.to_csv(OUT / "seasonal_joint_metrics_1980_2023.csv", index=False)
    return annual


def fit_models(annual: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name in SEASONS:
        sample = annual.loc[annual.season.eq(name)].copy()
        for response in RESPONSES:
            for specification, predictors in specifications_for_season(name):
                row = fit_hac(sample, response, predictors, specification)
                row["season"] = name
                rows.append(row)
    result = pd.DataFrame(rows)
    result["p_fdr_bh_within_specification"] = result.groupby("specification").p_hac.transform(bh_adjust)
    result.to_csv(OUT / "seasonal_internal_variability_hac.csv", index=False)
    internal_rows = []
    for row in result.itertuples():
        for predictor in [
            "nino34_c_dt_z", "nino34_c_lag1_dt_z", "pdo_index_dt_z",
            "iobm_40_100e_c_dt_z", "nino34_prev_djf_dt_z",
            "iobm_40_110e_c_dt_z", "iobm_independent_dt_z",
            "iobm110_independent_dt_z", "dmi_c_dt_z",
        ]:
            coefficient = getattr(row, f"coef_{predictor}", np.nan)
            pvalue = getattr(row, f"p_{predictor}", np.nan)
            if pd.notna(coefficient):
                internal_rows.append(
                    {
                        "season": row.season, "response": row.response,
                        "specification": row.specification, "predictor": predictor,
                        "coefficient_per_sd": coefficient, "p_hac": pvalue,
                        "n_season_years": row.n_season_years,
                    }
                )
    internal = pd.DataFrame(internal_rows)
    internal["p_fdr_bh_within_specification"] = internal.groupby("specification").p_hac.transform(bh_adjust)
    internal.to_csv(OUT / "seasonal_internal_predictor_effects.csv", index=False)
    return result


def fit_count_models(annual: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name in SEASONS:
        sample = annual.loc[annual.season.eq(name)].copy()
        for specification, predictors in specifications_for_season(name):
            data = sample[["frequency", *predictors]].dropna()
            x = sm.add_constant(data[predictors])
            poisson = sm.GLM(data.frequency, x, family=sm.families.Poisson()).fit()
            dispersion = float(poisson.pearson_chi2 / poisson.df_resid)
            if dispersion > 1.5:
                fit = sm.NegativeBinomial(data.frequency, x).fit(
                    disp=0, cov_type="HAC", cov_kwds={"maxlags": 3}
                )
                family = "negative_binomial"
            else:
                fit = sm.GLM(data.frequency, x, family=sm.families.Poisson()).fit(
                    cov_type="HAC", cov_kwds={"maxlags": 3}
                )
                family = "poisson"
            target = "year_centered" if specification == "trend_only" else "gmst_anomaly_c"
            scale = 10 if target == "year_centered" else 1
            beta = float(fit.params[target] * scale)
            ci = fit.conf_int().loc[target] * scale
            row = {
                    "season": name, "specification": specification, "family": family,
                    "poisson_dispersion": dispersion, "target": target,
                    "log_rate_coefficient": beta,
                    "rate_change_percent": float((np.exp(beta) - 1) * 100),
                    "rate_change_ci95_low_percent": float((np.exp(ci.iloc[0]) - 1) * 100),
                    "rate_change_ci95_high_percent": float((np.exp(ci.iloc[1]) - 1) * 100),
                    "p_hac": float(fit.pvalues[target]), "n_season_years": int(fit.nobs),
                    "alpha": float(fit.params.get("alpha", np.nan)),
                }
            for predictor in predictors:
                if predictor != target:
                    row[f"log_rate_{predictor}"] = float(fit.params[predictor])
                    row[f"rate_change_{predictor}_percent"] = float((np.exp(fit.params[predictor]) - 1) * 100)
                    row[f"p_{predictor}"] = float(fit.pvalues[predictor])
            rows.append(row)
    result = pd.DataFrame(rows)
    result["p_fdr_bh_within_specification"] = result.groupby("specification").p_hac.transform(bh_adjust)
    result.to_csv(OUT / "seasonal_frequency_count_models.csv", index=False)
    internal_rows = []
    for row in result.itertuples():
        for predictor in [
            "nino34_c_dt_z", "nino34_c_lag1_dt_z", "pdo_index_dt_z",
            "iobm_40_100e_c_dt_z", "nino34_prev_djf_dt_z",
            "iobm_40_110e_c_dt_z", "iobm_independent_dt_z",
            "iobm110_independent_dt_z", "dmi_c_dt_z",
        ]:
            coefficient = getattr(row, f"log_rate_{predictor}", np.nan)
            pvalue = getattr(row, f"p_{predictor}", np.nan)
            if pd.notna(coefficient):
                internal_rows.append(
                    {
                        "season": row.season, "specification": row.specification,
                        "family": row.family, "predictor": predictor,
                        "log_rate_coefficient_per_sd": coefficient,
                        "rate_change_percent_per_sd": float((np.exp(coefficient) - 1) * 100),
                        "p_hac": pvalue, "n_season_years": row.n_season_years,
                    }
                )
    internal = pd.DataFrame(internal_rows)
    internal["p_fdr_bh_within_specification"] = internal.groupby("specification").p_hac.transform(bh_adjust)
    internal.to_csv(OUT / "seasonal_frequency_internal_effects.csv", index=False)
    return result


def djf_endpoint_sensitivity(annual: pd.DataFrame) -> pd.DataFrame:
    djf = annual.loc[annual.season.eq("DJF")].copy()
    rows = []
    periods = [(1980, 2018), (1980, 2020), (1980, 2023), (2001, 2023)]
    responses = [
        "frequency", "median_height_lifecycle", "q90_height_lifecycle",
        "median_vorticity_lifecycle", "q90_vorticity_lifecycle",
    ]
    for start, end in periods:
        sample = djf.loc[djf.season_year.between(start, end)].copy()
        sample["year_centered"] = sample.season_year - sample.season_year.mean()
        for response in responses:
            for specification, predictors in (
                ("trend_only", ["year_centered"]),
                ("gmst_only", ["gmst_anomaly_c"]),
            ):
                row = fit_hac(sample, response, predictors, specification)
                row.update({"start_year": start, "end_year": end})
                rows.append(row)
    result = pd.DataFrame(rows)
    result.to_csv(OUT / "djf_endpoint_sensitivity.csv", index=False)
    return result


def stratified_exploration(events: pd.DataFrame, climate: pd.DataFrame) -> pd.DataFrame:
    valid = events.dropna(subset=["max_height_deficit_gpm_700_c25", "max_vorticity_s_1_700_c25"]).copy()
    counts = valid.groupby(["season", "track_cluster"]).agg(
        events=("event_id", "size"), years=("season_year", "nunique")
    ).reset_index()
    years_ge5 = valid.groupby(["season", "track_cluster", "season_year"]).size().ge(5).groupby(level=[0, 1]).sum()
    counts["years_ge5_events"] = [int(years_ge5.get((r.season, r.track_cluster), 0)) for r in counts.itertuples()]
    rows = []
    for cell in counts.itertuples():
        sample = valid.loc[(valid.season == cell.season) & (valid.track_cluster == cell.track_cluster)]
        for quantile, label, eligible in (
            (0.5, "q50", cell.events >= 100 and cell.years >= 30),
            (0.9, "q90", cell.events >= 200 and cell.years >= 35 and cell.years_ge5_events >= 20),
        ):
            if not eligible:
                rows.append({
                    "season": cell.season, "track_cluster": cell.track_cluster,
                    "quantile": label, "eligible": False, "events": cell.events,
                    "years": cell.years, "years_ge5_events": cell.years_ge5_events,
                })
                continue
            annual_cell = sample.groupby("season_year").max_height_deficit_gpm_700_c25.quantile(quantile).rename("response").reset_index()
            annual_cell = annual_cell.merge(
                climate[["season_year", "season", "nino34_c_dt_z"]].loc[lambda x: x.season.eq(cell.season)],
                on="season_year", validate="one_to_one",
            )
            gmst = pd.read_csv(ROOT / "data" / "raw" / "gistemp_v4_global.csv", skiprows=1, na_values="***").rename(
                columns={"Year": "season_year", "J-D": "gmst_anomaly_c"}
            )
            annual_cell = annual_cell.merge(gmst[["season_year", "gmst_anomaly_c"]], on="season_year")
            annual_cell["year_centered"] = annual_cell.season_year - annual_cell.season_year.mean()
            for specification, predictors in (
                ("trend_only", ["year_centered"]),
                ("gmst_only", ["gmst_anomaly_c"]),
                ("gmst_nino_same", ["gmst_anomaly_c", "nino34_c_dt_z"]),
            ):
                row = fit_hac(annual_cell, "response", predictors, specification)
                row.update({
                    "season": cell.season, "track_cluster": cell.track_cluster,
                    "quantile": label, "eligible": True, "events": cell.events,
                    "years": cell.years, "years_ge5_events": cell.years_ge5_events,
                })
                rows.append(row)
    result = pd.DataFrame(rows)
    mask = result.eligible.fillna(False)
    result.loc[mask, "p_fdr_bh"] = result.loc[mask].groupby(["quantile", "specification"]).p_hac.transform(bh_adjust)
    result.to_csv(OUT / "season_track_quantile_exploration.csv", index=False)
    return result


def duration_opportunity_audit(events: pd.DataFrame) -> pd.DataFrame:
    usable = events.dropna(subset=["max_height_deficit_gpm_700_c25", "max_vorticity_s_1_700_c25"]).copy()
    usable["height_z"] = (usable.max_height_deficit_gpm_700_c25 - usable.max_height_deficit_gpm_700_c25.mean()) / usable.max_height_deficit_gpm_700_c25.std(ddof=0)
    usable["vorticity_z"] = (usable.max_vorticity_s_1_700_c25 - usable.max_vorticity_s_1_700_c25.mean()) / usable.max_vorticity_s_1_700_c25.std(ddof=0)
    rows = []
    for response in ["height_z", "vorticity_z"]:
        for specification, formula in (
            ("season_adjusted", f"{response} ~ C(track_cluster, Treatment(reference=1)) + C(season)"),
            ("opportunity_adjusted", f"{response} ~ C(track_cluster, Treatment(reference=1)) + C(season) + log_sampled_points + genesis_lat + genesis_lon"),
        ):
            fit = smf.ols(formula, data=usable).fit(
                cov_type="cluster", cov_kwds={"groups": usable.season_year, "use_correction": True}
            )
            for term in [
                "C(track_cluster, Treatment(reference=1))[T.0]",
                "C(track_cluster, Treatment(reference=1))[T.2]",
            ]:
                if term in fit.params:
                    rows.append({
                        "response": response, "specification": specification,
                        "contrast_vs_cluster1": term, "coefficient_sd": fit.params[term],
                        "se_year_cluster": fit.bse[term], "p_year_cluster": fit.pvalues[term],
                        "n_events": int(fit.nobs), "n_year_clusters": usable.season_year.nunique(),
                    })
    result = pd.DataFrame(rows)
    result.to_csv(OUT / "path_intensity_opportunity_audit.csv", index=False)
    return result


def first_point_sensitivity_models(annual: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name in SEASONS:
        sample = annual.loc[annual.season.eq(name)].copy()
        for response in ["median_height_first", "median_vorticity_first", "first_point_coverage"]:
            for specification, predictors in (
                ("trend_only", ["year_centered"]),
                ("gmst_only", ["gmst_anomaly_c"]),
                ("gmst_nino_same", ["gmst_anomaly_c", "nino34_c_dt_z"]),
            ):
                row = fit_hac(sample, response, predictors, specification)
                row["season"] = name
                rows.append(row)
    result = pd.DataFrame(rows)
    result["p_fdr_bh_within_specification"] = result.groupby("specification").p_hac.transform(bh_adjust)
    result.to_csv(OUT / "seasonal_first_point_sensitivity_hac.csv", index=False)
    return result


def plots(annual: pd.DataFrame) -> None:
    FIG.mkdir(exist_ok=True)
    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    for name in SEASONS:
        sample = annual.loc[annual.season.eq(name)]
        axes[0].plot(sample.season_year, sample.frequency, label=name)
        axes[1].plot(sample.season_year, sample.median_height_lifecycle, label=name)
        axes[2].plot(sample.season_year, sample.mobile_eastward_fraction, label=name)
    for ax, ylabel in zip(axes, ["Frequency", "Median terrain-QC deficit (gpm)", "Mobile-eastward fraction"]):
        ax.set_ylabel(ylabel); ax.grid(alpha=0.2)
    axes[0].legend(ncol=4)
    axes[-1].set_xlabel("Season year (December belongs to following DJF)")
    fig.tight_layout(); fig.savefig(FIG / "seasonal_internal_variability_inputs.png", dpi=180); plt.close(fig)

    djf = annual.loc[annual.season.eq("DJF")]
    jja = annual.loc[annual.season.eq("JJA")]
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    for ax, response, ylabel in (
        (axes[0, 0], "frequency", "DJF event count"),
        (axes[0, 1], "q90_height_lifecycle", "DJF Q90 deficit (gpm)"),
    ):
        ax.plot(djf.season_year, djf[response], color="0.55", lw=1, marker="o", ms=3)
        slope, intercept = np.polyfit(djf.season_year, djf[response], 1)
        ax.plot(djf.season_year, intercept + slope * djf.season_year, color="#c43c39", lw=2)
        ax.set(xlabel="DJF season year", ylabel=ylabel); ax.grid(alpha=0.2)
    for ax, response, ylabel in (
        (axes[1, 0], "frequency", "JJA event count"),
        (axes[1, 1], "eastward_escape_fraction", "JJA eastward-escape fraction"),
    ):
        x = jja.iobm_40_100e_c_dt_z
        y = jja[response]
        ax.scatter(x, y, s=28, color="#2b6cb0", alpha=0.8)
        slope, intercept = np.polyfit(x, y, 1)
        grid = np.linspace(x.min(), x.max(), 100)
        ax.plot(grid, intercept + slope * grid, color="#c43c39", lw=2)
        ax.set(xlabel="Detrended JJA IOBM (SD)", ylabel=ylabel); ax.grid(alpha=0.2)
    fig.tight_layout(); fig.savefig(FIG / "seasonal_key_signals.png", dpi=180); plt.close(fig)


def main() -> None:
    climate = climate_covariates()
    events = event_table()
    annual = aggregate(events, climate)
    models = fit_models(annual)
    count_models = fit_count_models(annual)
    stratified = stratified_exploration(events, climate)
    opportunity = duration_opportunity_audit(events)
    first_point_models = first_point_sensitivity_models(annual)
    endpoint = djf_endpoint_sensitivity(annual)
    plots(annual)
    summary = {
        "period": "season-years 1980-2023",
        "independent_time_units_per_season": 44,
        "events": int(len(events)),
        "warning": "Observational associations only; GMST and time are not causally identifiable in one historical realization.",
        "confirmatory_endpoints": RESPONSES,
        "minimum_cell_rules": {"q50": "n>=100 and years>=30", "q90": "n>=200, years>=35, years_with_5_events>=20"},
        "eligible_stratified_tests": int(stratified.eligible.fillna(False).sum()),
        "model_rows": int(len(models)),
        "count_model_rows": int(len(count_models)),
        "duration_opportunity_audit": opportunity.to_dict(orient="records"),
        "first_point_sensitivity_rows": int(len(first_point_models)),
        "djf_endpoint_sensitivity_rows": int(len(endpoint)),
    }
    (OUT / "seasonal_internal_variability_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
