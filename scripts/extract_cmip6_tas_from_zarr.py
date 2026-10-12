"""Extract 1950--2014 global monthly tas from cloud Zarr stores."""

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
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    catalog = pd.read_csv(CATALOG, low_memory=False)
    selected = pd.read_csv(DATASETS)
    selected = selected.loc[selected.variable.eq("tas") & selected.experiment.isin(EXPERIMENTS)]
    if args.models:
        selected = selected.loc[selected.source.isin(args.models)]
    fs = gcsfs.GCSFileSystem(token="anon")
    failures = []
    for i, row in enumerate(selected.itertuples(index=False), 1):
        output = PARTS / f"{row.source}_{row.experiment}_{row.member}_tas_1950_2014.csv.gz"
        if output.exists() and not args.force:
            print(f"skip {i}/{len(selected)} {output.name}", flush=True); continue
        q = catalog.loc[(catalog.source_id == row.source) & (catalog.experiment_id == row.experiment)
                        & (catalog.member_id == row.member) & (catalog.table_id == "Amon")
                        & (catalog.variable_id == "tas")].copy()
        if q.empty:
            failures.append({"source": row.source, "experiment": row.experiment, "error": "no cloud zstore"}); continue
        q["grid_rank"] = q.grid_label.map({"gn": 0, "gr": 1, "gr1": 2, "gr2": 3}).fillna(9)
        store = q.sort_values(["grid_rank", "version"], ascending=[True, False]).iloc[0].zstore
        print(f"extract {i}/{len(selected)} {row.source} {row.experiment} {store}", flush=True)
        try:
            with xr.open_zarr(fs.get_mapper(store), consolidated=True) as ds:
                field = ds.tas.where((ds.time.dt.year >= 1950) & (ds.time.dt.year <= 2014), drop=True)
                lat_name = "lat" if "lat" in field.coords else "latitude"
                weights = np.cos(np.deg2rad(field[lat_name]))
                dims = [d for d in field.dims if d != "time"]
                mean = field.weighted(weights).mean(dims, skipna=True).compute()
                data = pd.DataFrame({"year": mean.time.dt.year.values.astype(int),
                                     "month": mean.time.dt.month.values.astype(int),
                                     "value": mean.values.astype(float)})
                data.to_csv(output, index=False)
        except Exception as exc:
            failures.append({"source": row.source, "experiment": row.experiment, "error": str(exc)})
            print(f"FAILED: {exc}", flush=True)
    pd.DataFrame(failures).to_csv(ROOT / "data" / "processed" / "cmip6_tas_zarr_failures.csv", index=False)
    print(f"complete={len(selected)-len(failures)} failures={len(failures)}", flush=True)


if __name__ == "__main__":
    main()
