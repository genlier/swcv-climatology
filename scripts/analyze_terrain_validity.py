"""Audit 700/850-hPa SWCV intensity against ERA5 surface pressure.

Pressure-level fields are masked wherever the pressure surface is underground
or lies within a configurable pressure clearance of the surface.  A metric is
accepted only when at least 80% of its core and environmental-ring cells pass
the mask.  Results for 0, 25 and 50 hPa clearance are retained for sensitivity.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from scipy.stats import linregress, pearsonr


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"
FIG = ROOT / "figures"
G = 9.80665
EARTH_R = 6_371_000.0
CLEARANCES = (0, 25, 50)
MIN_VALID_FRACTION = 0.80


def normalize(ds: xr.Dataset, level: int | None = None) -> xr.Dataset:
    rename = {}
    for old, new in (("valid_time", "time"), ("pressure_level", "level")):
        if old in ds.coords and new not in ds.coords:
            rename[old] = new
    ds = ds.rename(rename)
    if level is not None and "level" in ds.dims:
        ds = ds.sel(level=level, drop=True)
    return ds.sortby("latitude").sortby("longitude")


def relative_vorticity(u: xr.DataArray, v: xr.DataArray) -> xr.DataArray:
    dlat = float(u.latitude.diff("latitude").median())
    dlon = float(u.longitude.diff("longitude").median())
    dy_per_degree = EARTH_R * np.deg2rad(1.0)
    dx_per_degree = EARTH_R * np.cos(np.deg2rad(u.latitude)) * np.deg2rad(1.0)
    return v.differentiate("longitude") / dx_per_degree - u.differentiate("latitude") / dy_per_degree


def pressure_level_metrics(
    height: xr.DataArray,
    vorticity: xr.DataArray,
    surface_pressure_hpa: xr.DataArray,
    lat: float,
    lon: float,
    level: int,
) -> dict[str, float]:
    zbox = height.sel(latitude=slice(lat - 3, lat + 3), longitude=slice(lon - 3, lon + 3))
    vbox = vorticity.sel(latitude=slice(lat - 1.5, lat + 1.5), longitude=slice(lon - 1.5, lon + 1.5))
    pz = surface_pressure_hpa.sel(latitude=zbox.latitude, longitude=zbox.longitude)
    pv = surface_pressure_hpa.sel(latitude=vbox.latitude, longitude=vbox.longitude)
    rz = np.sqrt((zbox.latitude - lat) ** 2 + ((zbox.longitude - lon) * np.cos(np.deg2rad(lat))) ** 2)
    rv = np.sqrt((vbox.latitude - lat) ** 2 + ((vbox.longitude - lon) * np.cos(np.deg2rad(lat))) ** 2)
    core = rz <= 1.0
    ring = (rz >= 2.0) & (rz <= 3.0)
    vort_core = rv <= 1.5
    result: dict[str, float] = {}
    for clearance in CLEARANCES:
        valid_z = pz >= level + clearance
        valid_v = pv >= level + clearance
        core_fraction = float(valid_z.where(core).mean(skipna=True))
        ring_fraction = float(valid_z.where(ring).mean(skipna=True))
        vort_fraction = float(valid_v.where(vort_core).mean(skipna=True))
        suffix = f"{level}_c{clearance}"
        result[f"core_valid_fraction_{suffix}"] = core_fraction
        result[f"ring_valid_fraction_{suffix}"] = ring_fraction
        result[f"vort_valid_fraction_{suffix}"] = vort_fraction
        spatially_valid = min(core_fraction, ring_fraction, vort_fraction) >= MIN_VALID_FRACTION
        if spatially_valid:
            masked_z = zbox.where(valid_z)
            masked_v = vbox.where(valid_v)
            result[f"height_deficit_gpm_{suffix}"] = float(
                masked_z.where(ring).mean(skipna=True) - masked_z.where(core).min(skipna=True)
            )
            result[f"max_vorticity_s_1_{suffix}"] = float(masked_v.where(vort_core).max(skipna=True))
        else:
            result[f"height_deficit_gpm_{suffix}"] = np.nan
            result[f"max_vorticity_s_1_{suffix}"] = np.nan
    return result


def collocate() -> pd.DataFrame:
    points = pd.read_csv(OUT / "track_points.csv.gz", parse_dates=["time"])
    points = points.loc[(points.time.dt.year >= 1979) & points.time.dt.hour.isin([0, 6, 12, 18])]
    rows = []
    for year, sample in points.groupby(points.time.dt.year):
        paths = {
            700: RAW / "era5_700hpa_6hourly" / f"era5_700hpa_{year}.nc",
            850: RAW / "era5_850hpa_6hourly" / f"era5_850hpa_{year}.nc",
            "sp": RAW / "era5_surface_pressure_6hourly" / f"era5_surface_pressure_{year}.nc",
        }
        missing = [str(path) for path in paths.values() if not path.exists()]
        if missing:
            raise FileNotFoundError("Missing terrain-audit inputs: " + ", ".join(missing))
        with xr.open_dataset(paths[700]) as raw700, xr.open_dataset(paths[850]) as raw850, xr.open_dataset(paths["sp"]) as rawsp:
            ds700 = normalize(raw700, 700)
            ds850 = normalize(raw850, 850)
            dssp = normalize(rawsp)
            fields = {
                700: (ds700["z"] / G, relative_vorticity(ds700["u"], ds700["v"])),
                850: (ds850["z"] / G, relative_vorticity(ds850["u"], ds850["v"])),
            }
            for row in sample.itertuples():
                time = row.time.to_datetime64()
                sp = dssp["sp"].sel(time=time, method="nearest") / 100.0
                center_ps = float(sp.sel(latitude=row.lat, longitude=row.lon, method="nearest"))
                result = {
                    "event_id": row.event_id,
                    "time": row.time,
                    "lat": row.lat,
                    "lon": row.lon,
                    "surface_pressure_hpa": center_ps,
                    "center_below_700": center_ps < 700,
                    "center_within_25hpa_700": center_ps < 725,
                    "center_within_50hpa_700": center_ps < 750,
                    "center_below_850": center_ps < 850,
                }
                for level, (height, vorticity) in fields.items():
                    result.update(
                        pressure_level_metrics(
                            height.sel(time=time, method="nearest"),
                            vorticity.sel(time=time, method="nearest"),
                            sp,
                            row.lat,
                            row.lon,
                            level,
                        )
                    )
                rows.append(result)
        print(f"collocated {year}: {len(sample)} points", flush=True)
    result = pd.DataFrame(rows)
    result.to_csv(OUT / "terrain_qc_intensity_points.csv.gz", index=False)
    return result


def summarize(points: pd.DataFrame) -> None:
    events = pd.read_csv(OUT / "events.csv")
    events = events.loc[events.year >= 1979].copy()
    aggregations = {
        "sampled_points": ("time", "size"),
        "min_surface_pressure_hpa": ("surface_pressure_hpa", "min"),
        "center_below_700_fraction": ("center_below_700", "mean"),
        "center_within_25hpa_700_fraction": ("center_within_25hpa_700", "mean"),
        "center_within_50hpa_700_fraction": ("center_within_50hpa_700", "mean"),
    }
    for level in (700, 850):
        for clearance in CLEARANCES:
            suffix = f"{level}_c{clearance}"
            aggregations[f"valid_point_fraction_{suffix}"] = (
                f"height_deficit_gpm_{suffix}", lambda x: x.notna().mean()
            )
            aggregations[f"max_height_deficit_gpm_{suffix}"] = (f"height_deficit_gpm_{suffix}", "max")
            aggregations[f"max_vorticity_s_1_{suffix}"] = (f"max_vorticity_s_1_{suffix}", "max")
    event_qc = points.groupby("event_id").agg(**aggregations).reset_index()
    event_qc = events.merge(event_qc, on="event_id", how="left", validate="one_to_one")
    event_qc.to_csv(OUT / "events_terrain_qc_intensity.csv.gz", index=False)

    annual = event_qc.groupby("year").agg(
        events=("event_id", "size"),
        event_valid_700_c25_fraction=("max_height_deficit_gpm_700_c25", lambda x: x.notna().mean()),
        median_height_deficit_700_c25=("max_height_deficit_gpm_700_c25", "median"),
        q90_height_deficit_700_c25=("max_height_deficit_gpm_700_c25", lambda x: x.quantile(0.9)),
        median_vorticity_700_c25=("max_vorticity_s_1_700_c25", "median"),
        q90_vorticity_700_c25=("max_vorticity_s_1_700_c25", lambda x: x.quantile(0.9)),
        event_valid_850_c25_fraction=("max_height_deficit_gpm_850_c25", lambda x: x.notna().mean()),
        median_height_deficit_850_c25=("max_height_deficit_gpm_850_c25", "median"),
        median_vorticity_850_c25=("max_vorticity_s_1_850_c25", "median"),
    ).reset_index()
    gmst = pd.read_csv(RAW / "gistemp_v4_global.csv", skiprows=1, na_values="***")
    gmst = gmst.rename(columns={"Year": "year", "J-D": "gmst_anomaly_c"})
    annual = annual.merge(gmst[["year", "gmst_anomaly_c"]], on="year", validate="one_to_one")
    annual.to_csv(OUT / "annual_terrain_qc_intensity.csv", index=False)

    cluster = event_qc.groupby("track_cluster").agg(
        events=("event_id", "size"),
        center_below_700_event_fraction=("center_below_700_fraction", lambda x: (x > 0).mean()),
        center_near_700_event_fraction=("center_within_25hpa_700_fraction", lambda x: (x > 0).mean()),
        valid_700_c25_event_fraction=("max_height_deficit_gpm_700_c25", lambda x: x.notna().mean()),
        valid_850_c25_event_fraction=("max_height_deficit_gpm_850_c25", lambda x: x.notna().mean()),
        median_height_deficit_700_c25=("max_height_deficit_gpm_700_c25", "median"),
        median_vorticity_700_c25=("max_vorticity_s_1_700_c25", "median"),
        median_height_deficit_850_c25=("max_height_deficit_gpm_850_c25", "median"),
        median_vorticity_850_c25=("max_vorticity_s_1_850_c25", "median"),
    )
    cluster.to_csv(OUT / "terrain_qc_by_track_cluster.csv")

    metrics = [
        "event_valid_700_c25_fraction", "median_height_deficit_700_c25",
        "q90_height_deficit_700_c25", "median_vorticity_700_c25",
        "q90_vorticity_700_c25", "event_valid_850_c25_fraction",
    ]
    summary = {
        "period": "1979-2023",
        "minimum_spatial_valid_fraction": MIN_VALID_FRACTION,
        "point_count": int(len(points)),
        "event_count": int(len(event_qc)),
        "point_below_700_fraction": float(points.center_below_700.mean()),
        "point_within_25hpa_700_fraction": float(points.center_within_25hpa_700.mean()),
        "point_within_50hpa_700_fraction": float(points.center_within_50hpa_700.mean()),
        "metrics": {},
    }
    for metric in metrics:
        valid = annual[["year", "gmst_anomaly_c", metric]].dropna()
        trend = linregress(valid.year, valid[metric])
        corr = pearsonr(valid.gmst_anomaly_c, valid[metric])
        summary["metrics"][metric] = {
            "mean": float(valid[metric].mean()),
            "trend_per_decade": float(trend.slope * 10),
            "trend_p": float(trend.pvalue),
            "gmst_r": float(corr.statistic),
            "gmst_p": float(corr.pvalue),
        }
    (OUT / "terrain_qc_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    FIG.mkdir(exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    for ax, metric, ylabel in (
        (axes[0], "median_height_deficit_700_c25", "Terrain-QC median height deficit (gpm)"),
        (axes[1], "event_valid_700_c25_fraction", "Valid-event fraction"),
    ):
        ax.plot(annual.year, annual[metric], color="0.6", lw=1)
        ax.plot(annual.year, annual[metric].rolling(11, center=True).mean(), color="#c43c39", lw=2)
        valid = annual[["year", metric]].dropna()
        trend = linregress(valid.year, valid[metric])
        ax.plot(valid.year, trend.intercept + trend.slope * valid.year, "--", color="#2b6cb0")
        ax.set_ylabel(ylabel); ax.grid(alpha=0.2)
    axes[-1].set_xlabel("Year")
    fig.tight_layout(); fig.savefig(FIG / "terrain_qc_700hpa_intensity.png", dpi=180); plt.close(fig)

    print(json.dumps(summary, indent=2))
    print(cluster.to_string())


def main() -> None:
    points = collocate()
    summarize(points)


if __name__ == "__main__":
    main()
