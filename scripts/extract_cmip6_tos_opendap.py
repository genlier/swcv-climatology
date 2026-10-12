"""OPeNDAP fallback for 1950--2014 Indian Ocean SST means.

Supports regular and curvilinear ocean grids.  For 2-D coordinates it first
derives the smallest i/j rectangle enclosing the fixed Indian Ocean mask, so
the remote server does not transmit the global field.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import time

import numpy as np
import pandas as pd
import xarray as xr


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data" / "manifests" / "cmip6_monthly_esgf_file_manifest.csv"
PARTS = ROOT / "data" / "processed" / "cmip6_monthly_indices_parts"
OUT = ROOT / "data" / "processed"


def file_years(filename: str) -> tuple[int, int] | None:
    match = re.search(r"_(\d{4})\d{2}-(\d{4})\d{2}\.nc$", str(filename))
    return (int(match.group(1)), int(match.group(2))) if match else None


def url_for(row) -> str:
    if isinstance(row.opendap_url, str) and row.opendap_url:
        return row.opendap_url.removesuffix(".html")
    if isinstance(row.ncss_url, str) and "/ncss/grid/" in row.ncss_url:
        return row.ncss_url.replace("/ncss/grid/", "/dodsC/")
    raise ValueError("no OPeNDAP endpoint")


def bounding_indexers(mask: xr.DataArray) -> dict[str, slice]:
    result = {}
    for dim in mask.dims:
        other = [d for d in mask.dims if d != dim]
        keep = mask.any(other).values if other else mask.values
        indices = np.flatnonzero(keep)
        if not len(indices):
            raise RuntimeError("Indian Ocean mask is empty")
        result[dim] = slice(int(indices.min()), int(indices.max()) + 1)
    return result


def read_piece(row, first: int, last: int) -> pd.DataFrame:
    error = None
    for attempt in range(3):
        try:
            coder = xr.coders.CFDatetimeCoder(use_cftime=True)
            with xr.open_dataset(url_for(row), decode_times=coder) as ds:
                lat_name = next(n for n in ("lat", "latitude", "nav_lat") if n in ds.coords)
                lon_name = next(n for n in ("lon", "longitude", "nav_lon") if n in ds.coords)
                lat, lon = ds[lat_name].load(), (ds[lon_name].load() % 360)
                full_mask = (lat >= -20) & (lat <= 20) & (lon >= 40) & (lon <= 100)
                years = ds.time.dt.year.values.astype(int)
                time_indices = np.flatnonzero((years >= first) & (years <= last))
                if not len(time_indices):
                    raise RuntimeError(f"No time samples in {first}-{last}")
                indexers = bounding_indexers(full_mask)
                indexers["time"] = slice(int(time_indices.min()), int(time_indices.max()) + 1)
                # Positional slicing avoids a remote-backend failure observed when
                # xarray.where(drop=True) was applied to an OPeNDAP time axis.
                field = ds.tos.isel(indexers)
                lat, lon = field[lat_name], field[lon_name] % 360
                mask = (lat >= -20) & (lat <= 20) & (lon >= 40) & (lon <= 100)
                weights = np.cos(np.deg2rad(lat)).where(mask, 0).fillna(0)
                mean = field.where(mask).weighted(weights).mean(
                    [d for d in field.dims if d != "time"], skipna=True
                ).load()
                values = mean.values.astype(float)
                if not np.isfinite(values).all() or np.nanstd(values) < 1e-6:
                    raise RuntimeError("Extracted SST series is non-finite or constant")
                return pd.DataFrame({"year": mean.time.dt.year.values.astype(int),
                                     "month": mean.time.dt.month.values.astype(int),
                                     "value": values})
        except Exception as exc:
            error = exc
            if attempt < 2:
                time.sleep(2 ** attempt)
    raise RuntimeError(f"OPeNDAP failed after 3 attempts: {error}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--experiments", nargs="+", default=["historical", "hist-GHG", "hist-aer", "hist-nat"])
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    files = pd.read_csv(MANIFEST)
    files = files.loc[files.variable.eq("tos") & files.source.isin(args.models)
                      & files.experiment.isin(args.experiments)]
    failures = []
    groups = files.groupby(["source", "experiment", "member"], sort=True)
    for i, ((source, experiment, member), group) in enumerate(groups, 1):
        output = PARTS / f"{source}_{experiment}_{member}_tos_1950_2014.csv.gz"
        if output.exists() and not args.force:
            print(f"skip {i}/{len(groups)} {output.name}", flush=True)
            continue
        print(f"extract {i}/{len(groups)} {source} {experiment}", flush=True)
        pieces = []
        try:
            for row in group.sort_values("filename").itertuples(index=False):
                years = file_years(row.filename)
                if years is None or years[1] < 1950 or years[0] > 2014:
                    continue
                first, last = max(1950, years[0]), min(2014, years[1])
                pieces.append(read_piece(row, first, last))
                print(f"  {Path(row.filename).name} {first}-{last}", flush=True)
            if not pieces:
                raise RuntimeError("no files overlap 1950-2014")
            (pd.concat(pieces, ignore_index=True).drop_duplicates(["year", "month"])
             .sort_values(["year", "month"])).to_csv(output, index=False)
        except Exception as exc:
            failures.append({"source": source, "experiment": experiment,
                             "member": member, "error": str(exc)})
            print(f"FAILED: {exc}", flush=True)
    run_label = "_".join(args.models)
    failure_dir = OUT / "cmip6_tos_opendap_failures"
    failure_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(failures, columns=["source", "experiment", "member", "error"]).to_csv(
        failure_dir / f"{run_label}.csv", index=False
    )
    print(f"groups={len(groups)} failures={len(failures)}", flush=True)


if __name__ == "__main__":
    main()
