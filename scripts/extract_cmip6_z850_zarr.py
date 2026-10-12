"""Cloud-Zarr fallback producing regional and zonal-mean Z850 together."""

from __future__ import annotations

import argparse
from pathlib import Path

import gcsfs
import numpy as np
import pandas as pd
import xarray as xr


ROOT = Path(__file__).resolve().parents[1]
SELECTED = ROOT / "data" / "manifests" / "cmip6_monthly_selected_datasets.csv"
SPANS = ROOT / "data" / "processed" / "cmip6_picontrol_common_spans.csv"
REGIONAL = ROOT / "data" / "processed" / "cmip6_monthly_indices_parts"
REGIONAL_CONTROL = ROOT / "data" / "processed" / "cmip6_picontrol_indices_parts"
ZONAL = ROOT / "data" / "processed" / "cmip6_z850_zonal_parts"
OUT = ROOT / "data" / "processed"


def area_mean(field: xr.DataArray, west: float, east: float) -> xr.DataArray:
    q = field.where((field.lon >= west) & (field.lon <= east), drop=True)
    weights = np.cos(np.deg2rad(q.lat)).where(q.notnull()).fillna(0)
    return q.weighted(weights).mean([d for d in q.dims if d != "time"], skipna=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--experiments", nargs="+", required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    selected = pd.read_csv(SELECTED)
    selected = selected.loc[selected.variable.eq("zg") & selected.source.isin(args.models)
                            & selected.experiment.isin(args.experiments)]
    spans = pd.read_csv(SPANS)
    fs = gcsfs.GCSFileSystem(token="anon")
    failures = []
    for row in selected.itertuples(index=False):
        if row.experiment == "piControl":
            span = spans.loc[(spans.source == row.source) & (spans.member == row.member)].iloc[0]
            start, end = int(span.recommended_first_300_start), int(span.recommended_first_300_end)
        else:
            start, end = 1950, 2014
        regional_dir = REGIONAL_CONTROL if row.experiment == "piControl" else REGIONAL
        regional_path = regional_dir / f"{row.source}_{row.experiment}_{row.member}_zg_{start}_{end}.csv.gz"
        zonal_path = ZONAL / f"{row.source}_{row.experiment}_{row.member}_z850_zonal15_25n_{start}_{end}.csv.gz"
        if regional_path.exists() and zonal_path.exists() and not args.force:
            print(f"skip {row.source} {row.experiment}", flush=True)
            continue
        store = "cmip6/" + str(row.instance_id).replace(".", "/")
        try:
            if not fs.exists(store):
                raise RuntimeError(f"cloud store missing: gs://{store}")
            print(f"extract {row.source} {row.experiment} gs://{store}", flush=True)
            with xr.open_zarr(fs.get_mapper(store), consolidated=True) as ds:
                field = ds.zg.sel(plev=85000, method="nearest")
                field = field.where(
                    (field.time.dt.year >= start) & (field.time.dt.year <= end)
                    & (field.lat >= 15) & (field.lat <= 25), drop=True
                )
                means = xr.Dataset({
                    "regional": area_mean(field, 115, 150),
                    "zonal": area_mean(field, -180, 360),
                }).compute()
                result = pd.DataFrame({
                    "year": means.time.dt.year.values.astype(int),
                    "month": means.time.dt.month.values.astype(int),
                    "regional": means.regional.values.astype(float),
                    "zonal": means.zonal.values.astype(float),
                }).drop_duplicates(["year", "month"]).sort_values(["year", "month"])
                result[["year", "month", "regional"]].rename(
                    columns={"regional": "value"}).to_csv(regional_path, index=False)
                result[["year", "month", "zonal"]].rename(
                    columns={"zonal": "value"}).to_csv(zonal_path, index=False)
        except Exception as exc:
            failures.append({"source": row.source, "experiment": row.experiment,
                             "member": row.member, "error": str(exc)})
            print(f"FAILED: {exc}", flush=True)
    pd.DataFrame(failures, columns=["source", "experiment", "member", "error"]).to_csv(
        OUT / "cmip6_z850_zarr_failures.csv", index=False
    )
    print(f"groups={len(selected)} failures={len(failures)}", flush=True)


if __name__ == "__main__":
    main()
