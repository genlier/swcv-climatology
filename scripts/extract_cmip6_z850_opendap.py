"""OPeNDAP fallback producing both regional and zonal-mean Z850 indices."""

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
SPANS = ROOT / "data" / "processed" / "cmip6_picontrol_common_spans.csv"
REGIONAL = ROOT / "data" / "processed" / "cmip6_monthly_indices_parts"
REGIONAL_CONTROL = ROOT / "data" / "processed" / "cmip6_picontrol_indices_parts"
ZONAL = ROOT / "data" / "processed" / "cmip6_z850_zonal_parts"
OUT = ROOT / "data" / "processed"


def file_years(filename: str) -> tuple[int, int] | None:
    match = re.search(r"_(\d{4})\d{2}-(\d{4})\d{2}\.nc$", str(filename))
    return (int(match.group(1)), int(match.group(2))) if match else None


def url_for(row) -> str:
    if isinstance(row.ncss_url, str) and "/ncss/grid/" in row.ncss_url:
        return row.ncss_url.replace("/ncss/grid/", "/dodsC/")
    if isinstance(row.opendap_url, str) and row.opendap_url:
        return row.opendap_url.removesuffix(".html")
    raise ValueError("no OPeNDAP endpoint")


def area_mean(field: xr.DataArray, west: float, east: float) -> xr.DataArray:
    q = field.where((field.lon >= west) & (field.lon <= east), drop=True)
    weights = np.cos(np.deg2rad(q.lat)).where(q.notnull()).fillna(0)
    return q.weighted(weights).mean([d for d in q.dims if d != "time"], skipna=True)


def read_piece(row, first: int, last: int) -> pd.DataFrame:
    error = None
    for attempt in range(3):
        try:
            coder = xr.coders.CFDatetimeCoder(use_cftime=True)
            with xr.open_dataset(url_for(row), decode_times=coder) as ds:
                field = ds.zg.sel(plev=85000, method="nearest")
                field = field.where(
                    (field.time.dt.year >= first) & (field.time.dt.year <= last)
                    & (field.lat >= 15) & (field.lat <= 25), drop=True
                )
                means = xr.Dataset({
                    "regional": area_mean(field, 115, 150),
                    "zonal": area_mean(field, -180, 360),
                }).load()
                return pd.DataFrame({
                    "year": means.time.dt.year.values.astype(int),
                    "month": means.time.dt.month.values.astype(int),
                    "regional": means.regional.values.astype(float),
                    "zonal": means.zonal.values.astype(float),
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
    files = files.loc[files.source.isin(args.models) & files.experiment.isin(args.experiments)
                      & files.variable.eq("zg")]
    spans = pd.read_csv(SPANS)
    failures = []
    groups = files.groupby(["source", "experiment", "member"], sort=True)
    for i, ((source, experiment, member), group) in enumerate(groups, 1):
        if experiment == "piControl":
            span = spans.loc[(spans.source == source) & (spans.member == member)].iloc[0]
            start, end = int(span.recommended_first_300_start), int(span.recommended_first_300_end)
        else:
            start, end = 1950, 2014
        regional_dir = REGIONAL_CONTROL if experiment == "piControl" else REGIONAL
        regional_path = regional_dir / f"{source}_{experiment}_{member}_zg_{start}_{end}.csv.gz"
        zonal_path = ZONAL / f"{source}_{experiment}_{member}_z850_zonal15_25n_{start}_{end}.csv.gz"
        if regional_path.exists() and zonal_path.exists() and not args.force:
            print(f"skip {i}/{len(groups)} {source} {experiment}", flush=True)
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
                raise RuntimeError("no files overlap requested period")
            result = (pd.concat(pieces, ignore_index=True).drop_duplicates(["year", "month"])
                      .sort_values(["year", "month"]))
            result[["year", "month", "regional"]].rename(
                columns={"regional": "value"}).to_csv(regional_path, index=False)
            result[["year", "month", "zonal"]].rename(
                columns={"zonal": "value"}).to_csv(zonal_path, index=False)
        except Exception as exc:
            failures.append({"source": source, "experiment": experiment,
                             "member": member, "error": str(exc)})
            print(f"FAILED: {exc}", flush=True)
    pd.DataFrame(failures, columns=["source", "experiment", "member", "error"]).to_csv(
        OUT / "cmip6_z850_opendap_failures.csv", index=False
    )
    print(f"groups={len(groups)} failures={len(failures)}", flush=True)


if __name__ == "__main__":
    main()
