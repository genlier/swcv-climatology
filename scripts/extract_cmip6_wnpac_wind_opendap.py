"""OPeNDAP fallback for WNPAC wind groups whose NCSS endpoint is broken.

The ORNL THREDDS NCSS service returns HTTP 500 for some CESM2 netCDF-4
files, while its OPeNDAP endpoint supports constrained reads correctly.
This script reads only 850 hPa and the union of the two frozen index boxes.
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
MANIFEST = ROOT / "data" / "manifests" / "cmip6_monthly_ua_esgf_file_manifest.csv"
SPANS = ROOT / "data" / "processed" / "cmip6_picontrol_common_spans.csv"
PARTS = ROOT / "data" / "processed" / "cmip6_wnpac_wind_parts"
OUT = ROOT / "data" / "processed"


def file_years(filename: str) -> tuple[int, int] | None:
    match = re.search(r"_(\d{4})\d{2}-(\d{4})\d{2}\.nc$", str(filename))
    return (int(match.group(1)), int(match.group(2))) if match else None


def dap_url(ncss_url: str) -> str:
    if not isinstance(ncss_url, str) or "/ncss/grid/" not in ncss_url:
        raise ValueError("no convertible THREDDS NCSS URL")
    return ncss_url.replace("/ncss/grid/", "/dodsC/")


def area_mean(field: xr.DataArray, south: float, north: float,
              west: float, east: float) -> xr.DataArray:
    q = field.where(
        (field.lat >= south) & (field.lat <= north)
        & (field.lon >= west) & (field.lon <= east), drop=True
    )
    weights = np.cos(np.deg2rad(q.lat)).where(q.notnull()).fillna(0)
    return q.weighted(weights).mean([d for d in q.dims if d != "time"], skipna=True)


def read_piece(row, first: int, last: int) -> pd.DataFrame:
    error = None
    for attempt in range(3):
        try:
            coder = xr.coders.CFDatetimeCoder(use_cftime=True)
            with xr.open_dataset(dap_url(row.ncss_url), decode_times=coder) as ds:
                field = ds.ua.sel(plev=85000, method="nearest")
                field = field.where(
                    (field.time.dt.year >= first) & (field.time.dt.year <= last)
                    & (field.lat >= 5) & (field.lat <= 32.5)
                    & (field.lon >= 90) & (field.lon <= 140), drop=True
                )
                means = xr.Dataset({
                    "north": area_mean(field, 22.5, 32.5, 110, 140),
                    "south": area_mean(field, 5, 15, 90, 130),
                }).load()
                north, south = means.north, means.south
                return pd.DataFrame({
                    "year": north.time.dt.year.values.astype(int),
                    "month": north.time.dt.month.values.astype(int),
                    "north_u850_ms": north.values.astype(float),
                    "south_u850_ms": south.values.astype(float),
                    "wnpac_u850_ms": (north - south).values.astype(float),
                })
        except Exception as exc:
            error = exc
            if attempt < 2:
                time.sleep(2 ** attempt)
    raise RuntimeError(f"OPeNDAP failed after 3 attempts: {error}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--experiments", nargs="+", required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    files = pd.read_csv(MANIFEST)
    files = files.loc[files.source.isin(args.models) & files.experiment.isin(args.experiments)]
    spans = pd.read_csv(SPANS)
    failures = []
    groups = files.groupby(["source", "experiment", "member"], sort=True)
    for i, ((source, experiment, member), group) in enumerate(groups, 1):
        if experiment == "piControl":
            span = spans.loc[(spans.source == source) & (spans.member == member)].iloc[0]
            start, end = int(span.recommended_first_300_start), int(span.recommended_first_300_end)
        else:
            start, end = 1950, 2014
        output = PARTS / f"{source}_{experiment}_{member}_wnpac_u850_{start}_{end}.csv.gz"
        if output.exists() and not args.force:
            print(f"skip {i}/{len(groups)} {output.name}", flush=True)
            continue
        print(f"extract {i}/{len(groups)} {source} {experiment} {start}-{end}", flush=True)
        pieces = []
        try:
            for row in group.sort_values("filename").itertuples(index=False):
                years = file_years(row.filename)
                if years is None or years[1] < start or years[0] > end:
                    continue
                first, last = max(start, years[0]), min(end, years[1])
                pieces.append(read_piece(row, first, last))
                print(f"  {Path(row.filename).name} {first}-{last}", flush=True)
            if not pieces:
                raise RuntimeError("no manifest files overlap requested years")
            result = (pd.concat(pieces, ignore_index=True)
                      .drop_duplicates(["year", "month"])
                      .sort_values(["year", "month"]))
            result.to_csv(output, index=False)
        except Exception as exc:
            failures.append({"source": source, "experiment": experiment,
                             "member": member, "error": str(exc)})
            print(f"FAILED: {exc}", flush=True)
    pd.DataFrame(failures, columns=["source", "experiment", "member", "error"]).to_csv(
        OUT / "cmip6_wnpac_wind_opendap_failures.csv", index=False
    )
    print(f"groups={len(groups)} failures={len(failures)}", flush=True)


if __name__ == "__main__":
    main()
