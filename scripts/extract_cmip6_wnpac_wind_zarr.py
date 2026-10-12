"""Cloud-Zarr fallback for wind-index groups without an ESGF NCSS endpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import gcsfs
import numpy as np
import pandas as pd
import xarray as xr


ROOT = Path(__file__).resolve().parents[1]
SELECTED = ROOT / "data" / "manifests" / "cmip6_monthly_ua_selected_datasets.csv"
PARTS = ROOT / "data" / "processed" / "cmip6_wnpac_wind_parts"
SPANS = ROOT / "data" / "processed" / "cmip6_picontrol_common_spans.csv"
BASE = "cmip6/CMIP6"


def area_mean(field: xr.DataArray, south: float, north: float, west: float, east: float) -> xr.DataArray:
    q = field.where((field.lat >= south) & (field.lat <= north)
                    & (field.lon >= west) & (field.lon <= east), drop=True)
    weights = np.cos(np.deg2rad(q.lat)).where(q.notnull()).fillna(0)
    return q.weighted(weights).mean([dim for dim in q.dims if dim != "time"], skipna=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--experiments", nargs="+", required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    selected = pd.read_csv(SELECTED)
    selected = selected.loc[selected.source.isin(args.models) & selected.experiment.isin(args.experiments)]
    spans = pd.read_csv(SPANS)
    fs = gcsfs.GCSFileSystem(token="anon")
    failures = []
    for row in selected.itertuples(index=False):
        if row.experiment == "piControl":
            span = spans.loc[(spans.source == row.source) & (spans.member == row.member)]
            if span.empty:
                failures.append({"source": row.source, "experiment": row.experiment,
                                 "error": "no frozen piControl span"})
                continue
            start, end = int(span.iloc[0].recommended_first_300_start), int(span.iloc[0].recommended_first_300_end)
        else:
            start, end = 1950, 2014
        output = PARTS / f"{row.source}_{row.experiment}_{row.member}_wnpac_u850_{start}_{end}.csv.gz"
        if output.exists() and not args.force:
            print(f"skip {output.name}", flush=True)
            continue
        activity = "CMIP" if row.experiment in {"historical", "piControl"} else "DAMIP"
        institution = str(row.instance_id).split(".")[2]
        prefix = f"{BASE}/{activity}/{institution}/{row.source}/{row.experiment}/{row.member}/Amon/ua/{row.grid}"
        try:
            versions = fs.ls(prefix)
            if not versions:
                raise RuntimeError("no cloud Zarr version")
            store = sorted(versions)[-1]
            print(f"extract {row.source} {row.experiment} gs://{store}", flush=True)
            with xr.open_zarr(fs.get_mapper(store), consolidated=True) as ds:
                field = ds.ua.sel(plev=85000, method="nearest")
                field = field.where((field.time.dt.year >= start) & (field.time.dt.year <= end)
                                    & (field.lat >= 5) & (field.lat <= 32.5)
                                    & (field.lon >= 90) & (field.lon <= 140), drop=True)
                # Compute both boxes in one Dask graph so shared source chunks are
                # downloaded only once.  Computing them sequentially roughly
                # doubles cloud I/O for stores with coarse spatial chunks.
                means = xr.Dataset({
                    "north": area_mean(field, 22.5, 32.5, 110, 140),
                    "south": area_mean(field, 5, 15, 90, 130),
                }).compute()
                north = means["north"]
                south = means["south"]
                data = pd.DataFrame({"year": north.time.dt.year.values.astype(int),
                                     "month": north.time.dt.month.values.astype(int),
                                     "north_u850_ms": north.values.astype(float),
                                     "south_u850_ms": south.values.astype(float),
                                     "wnpac_u850_ms": (north - south).values.astype(float)})
                data.drop_duplicates(["year", "month"]).sort_values(["year", "month"]).to_csv(output, index=False)
        except Exception as exc:
            failures.append({"source": row.source, "experiment": row.experiment, "error": str(exc)})
            print(f"FAILED: {exc}", flush=True)
    pd.DataFrame(failures, columns=["source", "experiment", "error"]).to_csv(
        ROOT / "data" / "processed" / "cmip6_wnpac_wind_zarr_failures.csv", index=False
    )
    print(f"groups={len(selected)} failures={len(failures)}", flush=True)


if __name__ == "__main__":
    main()
