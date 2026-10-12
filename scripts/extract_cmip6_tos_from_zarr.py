"""Extract 1950--2014 monthly Indian Ocean SST means from cloud Zarr stores.

This is a fallback for datasets whose ESGF THREDDS/NCSS endpoint is unavailable.
It deliberately prefers a regular regridded ocean product (gr1/gr) so the regional
area weighting is transparent and consistent across models.
"""

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
PARTS = ROOT / "data" / "processed" / "cmip6_monthly_indices_parts"
EXPERIMENTS = ["historical", "hist-GHG", "hist-aer", "hist-nat"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+")
    parser.add_argument("--experiments", nargs="+", choices=EXPERIMENTS)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    catalog = pd.read_csv(CATALOG, low_memory=False)
    selected = pd.read_csv(DATASETS)
    selected = selected.loc[selected.variable.eq("tos") & selected.experiment.isin(args.experiments or EXPERIMENTS)]
    if args.models:
        selected = selected.loc[selected.source.isin(args.models)]

    PARTS.mkdir(parents=True, exist_ok=True)
    fs = gcsfs.GCSFileSystem(token="anon")
    failures: list[dict[str, str]] = []
    for i, row in enumerate(selected.itertuples(index=False), 1):
        output = PARTS / f"{row.source}_{row.experiment}_{row.member}_tos_1950_2014.csv.gz"
        if output.exists() and not args.force:
            print(f"skip {i}/{len(selected)} {output.name}", flush=True)
            continue

        q = catalog.loc[
            (catalog.source_id == row.source)
            & (catalog.experiment_id == row.experiment)
            & (catalog.member_id == row.member)
            & (catalog.table_id == "Omon")
            & (catalog.variable_id == "tos")
        ].copy()
        if q.empty:
            failures.append({"source": row.source, "experiment": row.experiment, "error": "no cloud zstore"})
            continue
        q["grid_rank"] = q.grid_label.map({"gr1": 0, "gr": 1, "gn": 2}).fillna(9)
        store = q.sort_values(["grid_rank", "version"], ascending=[True, False]).iloc[0].zstore
        print(f"extract {i}/{len(selected)} {row.source} {row.experiment} {store}", flush=True)
        try:
            with xr.open_zarr(fs.get_mapper(store), consolidated=True) as ds:
                field = ds.tos.where((ds.time.dt.year >= 1950) & (ds.time.dt.year <= 2014), drop=True)
                lat_name = next(name for name in ("lat", "latitude", "nav_lat") if name in field.coords)
                lon_name = next(name for name in ("lon", "longitude", "nav_lon") if name in field.coords)
                lat = field[lat_name]
                lon = field[lon_name] % 360
                mask = (lat >= -20) & (lat <= 20) & (lon >= 40) & (lon <= 100)
                field = field.where(mask)
                weights = np.cos(np.deg2rad(lat)).where(mask, 0).fillna(0)
                dims = [dim for dim in field.dims if dim != "time"]
                mean = field.weighted(weights).mean(dims, skipna=True).compute()
                data = pd.DataFrame(
                    {
                        "year": mean.time.dt.year.values.astype(int),
                        "month": mean.time.dt.month.values.astype(int),
                        "value": mean.values.astype(float),
                    }
                )
                data.to_csv(output, index=False)
        except Exception as exc:
            failures.append({"source": row.source, "experiment": row.experiment, "error": str(exc)})
            print(f"FAILED: {exc}", flush=True)

    pd.DataFrame(failures).to_csv(ROOT / "data" / "processed" / "cmip6_tos_zarr_failures.csv", index=False)
    print(f"complete={len(selected)-len(failures)} failures={len(failures)}", flush=True)


if __name__ == "__main__":
    main()
