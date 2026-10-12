"""Extract a compact monthly tas or tos index from local CMIP6 NetCDF files.

This is the HTTPS-file fallback when neither NCSS nor OPeNDAP is available.
It uses the same cosine-latitude weighting and domains as the main extractor:
global tas (GMST proxy) and 20S--20N, 40--100E tos (IOBM proxy).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


def bounding_indexers(mask: xr.DataArray) -> dict[str, slice]:
    result = {}
    for dim in mask.dims:
        other = [name for name in mask.dims if name != dim]
        keep = mask.any(other).values if other else mask.values
        indices = np.flatnonzero(keep)
        if not len(indices):
            raise RuntimeError("Requested spatial mask is empty")
        result[dim] = slice(int(indices.min()), int(indices.max()) + 1)
    return result


def extract(path: Path, variable: str) -> pd.DataFrame:
    coder = xr.coders.CFDatetimeCoder(use_cftime=True)
    with xr.open_dataset(path, decode_times=coder) as ds:
        field = ds[variable]
        lat_name = next(name for name in ("lat", "latitude", "nav_lat") if name in ds.coords)
        lat = ds[lat_name]
        if variable == "tos":
            lon_name = next(name for name in ("lon", "longitude", "nav_lon") if name in ds.coords)
            lon = ds[lon_name] % 360
            full_mask = (lat >= -20) & (lat <= 20) & (lon >= 40) & (lon <= 100)
            field = field.isel(bounding_indexers(full_mask))
            lat = field[lat_name]
            lon = field[lon_name] % 360
            mask = (lat >= -20) & (lat <= 20) & (lon >= 40) & (lon <= 100)
            field = field.where(mask)
        weights = np.cos(np.deg2rad(lat)).where(field.notnull()).fillna(0)
        mean = field.weighted(weights).mean([d for d in field.dims if d != "time"], skipna=True).load()
        return pd.DataFrame({
            "year": mean.time.dt.year.values.astype(int),
            "month": mean.time.dt.month.values.astype(int),
            "value": mean.values.astype(float),
        })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variable", choices=["tas", "tos"], required=True)
    parser.add_argument("--inputs", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    frames = [extract(path, args.variable) for path in args.inputs]
    result = (pd.concat(frames, ignore_index=True)
              .loc[lambda q: q.year.between(1950, 2014)]
              .drop_duplicates(["year", "month"])
              .sort_values(["year", "month"]))
    expected = {(year, month) for year in range(1951, 2015) for month in (6, 7, 8)}
    available = set(map(tuple, result[["year", "month"]].to_numpy()))
    missing_jja = sorted(expected - available)
    if missing_jja:
        raise RuntimeError(f"Missing target JJA months: {missing_jja[:12]} (n={len(missing_jja)})")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(f"wrote {args.output} rows={len(result)} target_jja=complete", flush=True)


if __name__ == "__main__":
    main()
