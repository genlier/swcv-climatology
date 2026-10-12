"""Extract tas/tos indices for the first 300 common piControl years from Zarr."""

from __future__ import annotations

import argparse
from pathlib import Path

import gcsfs
import numpy as np
import pandas as pd
import xarray as xr


ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "data" / "manifests" / "cmip6_cloud_monthly_target_catalog.csv"
DATASETS = ROOT / "data" / "manifests" / "cmip6_monthly_selected_datasets.csv"
SPANS = ROOT / "data" / "processed" / "cmip6_picontrol_common_spans.csv"
PARTS = ROOT / "data" / "processed" / "cmip6_picontrol_indices_parts"


def coordinate_names(field: xr.DataArray, variable: str) -> tuple[str, str]:
    if variable == "tos" and "latitude" in field.coords and field.latitude.ndim == 2:
        return "latitude", "longitude"
    if "nav_lat" in field.coords:
        return "nav_lat", "nav_lon"
    if "lat" in field.coords:
        return "lat", "lon"
    return "latitude", "longitude"


def area_mean(field: xr.DataArray, variable: str) -> xr.DataArray:
    lat_name, lon_name = coordinate_names(field, variable)
    lat, lon = field[lat_name], field[lon_name] % 360
    if variable == "tos":
        mask = (lat >= -20) & (lat <= 20) & (lon >= 40) & (lon <= 100)
        field = field.where(mask)
        weights = np.cos(np.deg2rad(lat)).where(mask, 0).fillna(0)
    else:
        weights = np.cos(np.deg2rad(lat)).fillna(0)
    dims = [dim for dim in field.dims if dim != "time"]
    return field.weighted(weights).mean(dims, skipna=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+")
    parser.add_argument("--variables", nargs="+", choices=["tas", "tos"], default=["tas", "tos"])
    parser.add_argument("--use-first-300-position", action="store_true",
                        help="Use the first 3600 stored months and label years 1..300 (for cloud/file span mismatch).")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    catalog = pd.read_csv(CATALOG, low_memory=False)
    selected = pd.read_csv(DATASETS)
    selected = selected.loc[selected.experiment.eq("piControl") & selected.variable.isin(args.variables)]
    spans = pd.read_csv(SPANS)
    if args.models:
        selected = selected.loc[selected.source.isin(args.models)]
        spans = spans.loc[spans.source.isin(args.models)]
    PARTS.mkdir(parents=True, exist_ok=True)
    fs = gcsfs.GCSFileSystem(token="anon")
    failures = []
    for index, row in enumerate(selected.itertuples(index=False), 1):
        span = spans.loc[(spans.source == row.source) & (spans.member == row.member)].iloc[0]
        start, end = int(span.recommended_first_300_start), int(span.recommended_first_300_end)
        output_start, output_end = ((1, 300) if args.use_first_300_position else (start, end))
        output = PARTS / f"{row.source}_piControl_{row.member}_{row.variable}_{output_start}_{output_end}.csv.gz"
        if output.exists() and not args.force:
            print(f"skip {index}/{len(selected)} {output.name}", flush=True)
            continue
        table = "Amon" if row.variable == "tas" else "Omon"
        q = catalog.loc[
            (catalog.source_id == row.source)
            & (catalog.experiment_id == "piControl")
            & (catalog.member_id == row.member)
            & (catalog.table_id == table)
            & (catalog.variable_id == row.variable)
        ].copy()
        if q.empty:
            failures.append({"source": row.source, "variable": row.variable, "error": "no cloud zstore"})
            continue
        q["grid_rank"] = q.grid_label.map({"gr1": 0, "gr": 1, "gn": 2}).fillna(9)
        store = q.sort_values(["grid_rank", "version"], ascending=[True, False]).iloc[0].zstore
        print(f"extract {index}/{len(selected)} {row.source} {row.variable} {start}-{end}", flush=True)
        try:
            with xr.open_zarr(fs.get_mapper(store), consolidated=True) as ds:
                field = ds[row.variable]
                if args.use_first_300_position:
                    field = field.isel(time=slice(0, 3600))
                else:
                    field = field.where((field.time.dt.year >= start) & (field.time.dt.year <= end), drop=True)
                mean = area_mean(field, row.variable).compute()
                if args.use_first_300_position:
                    years = np.repeat(np.arange(1, 301), 12)
                    months = np.tile(np.arange(1, 13), 300)
                else:
                    years = mean.time.dt.year.values.astype(int)
                    months = mean.time.dt.month.values.astype(int)
                result = pd.DataFrame({
                    "year": years,
                    "month": months,
                    "value": mean.values.astype(float),
                })
                result.to_csv(output, index=False)
        except Exception as exc:
            failures.append({"source": row.source, "variable": row.variable, "error": str(exc)})
            print(f"FAILED: {exc}", flush=True)
    pd.DataFrame(failures).to_csv(ROOT / "data" / "processed" / "cmip6_picontrol_zarr_failures.csv", index=False)
    print(f"groups={len(selected)} failures={len(failures)}", flush=True)


if __name__ == "__main__":
    main()
