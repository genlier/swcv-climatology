"""Collocate 700-hPa ERA5 height/vorticity with the published SWCV tracks."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


ROOT = Path(__file__).resolve().parents[1]
ERA5 = ROOT / "data" / "raw" / "era5_700hpa_6hourly"
OUT = ROOT / "data" / "processed"
G = 9.80665
EARTH_R = 6_371_000.0


def normalize(ds: xr.Dataset) -> xr.Dataset:
    rename = {}
    for old, new in [("valid_time", "time"), ("pressure_level", "level")]:
        if old in ds.coords and new not in ds.coords:
            rename[old] = new
    ds = ds.rename(rename)
    if "level" in ds.dims:
        ds = ds.sel(level=700, drop=True)
    return ds.sortby("latitude").sortby("longitude")


def relative_vorticity(u: xr.DataArray, v: xr.DataArray) -> xr.DataArray:
    lat_rad = np.deg2rad(u.latitude)
    dy = EARTH_R * np.deg2rad(float(u.latitude.diff("latitude").median()))
    dx = EARTH_R * np.cos(lat_rad) * np.deg2rad(float(u.longitude.diff("longitude").median()))
    return v.differentiate("longitude") / (dx / float(u.longitude.diff("longitude").median())) - u.differentiate("latitude") / (dy / float(u.latitude.diff("latitude").median()))


def main() -> None:
    points = pd.read_csv(OUT / "track_points.csv.gz", parse_dates=["time"])
    points = points.loc[(points.time.dt.year >= 1979) & points.time.dt.hour.isin([0, 6, 12, 18])].copy()
    rows = []
    for year, sample in points.groupby(points.time.dt.year):
        path = ERA5 / f"era5_700hpa_{year}.nc"
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}; run download_era5_700hpa.py first")
        with xr.open_dataset(path) as raw:
            ds = normalize(raw)
            zeta = relative_vorticity(ds["u"], ds["v"])
            height = ds["z"] / G
            for row in sample.itertuples():
                field_z = height.sel(time=row.time.to_datetime64(), method="nearest")
                field_v = zeta.sel(time=row.time.to_datetime64(), method="nearest")
                outer_z = field_z.sel(latitude=slice(row.lat - 3, row.lat + 3), longitude=slice(row.lon - 3, row.lon + 3))
                outer_v = field_v.sel(latitude=slice(row.lat - 1.5, row.lat + 1.5), longitude=slice(row.lon - 1.5, row.lon + 1.5))
                radius_z = np.sqrt((outer_z.latitude - row.lat) ** 2 + ((outer_z.longitude - row.lon) * np.cos(np.deg2rad(row.lat))) ** 2)
                radius_v = np.sqrt((outer_v.latitude - row.lat) ** 2 + ((outer_v.longitude - row.lon) * np.cos(np.deg2rad(row.lat))) ** 2)
                inner_z = outer_z.where(radius_z <= 1.0)
                ring_z = outer_z.where((radius_z >= 2.0) & (radius_z <= 3.0))
                inner_v = outer_v.where(radius_v <= 1.5)
                rows.append({
                    "event_id": row.event_id, "time": row.time, "lat": row.lat, "lon": row.lon,
                    "height_deficit_gpm": float(ring_z.mean() - inner_z.min()),
                    "max_relative_vorticity_s-1": float(inner_v.max()),
                })
    intensity = pd.DataFrame(rows)
    intensity.to_csv(OUT / "era5_intensity_points_6hourly.csv.gz", index=False)
    event = intensity.groupby("event_id").agg(
        max_height_deficit_gpm=("height_deficit_gpm", "max"),
        max_vorticity_s_1=("max_relative_vorticity_s-1", "max"),
        mean_vorticity_s_1=("max_relative_vorticity_s-1", "mean"),
    )
    event.to_csv(OUT / "era5_event_intensity.csv")


if __name__ == "__main__":
    main()
