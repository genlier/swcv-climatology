"""Extract two-period JJA Z850 responses for validated CMIP6 member pairs."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import time

import numpy as np
import pandas as pd
import requests
import xarray as xr


ROOT = Path(__file__).resolve().parents[1]
FILES = ROOT / "data" / "manifests" / "cmip6_multimember_zg_opendap_files.csv"
SUBSETS = ROOT / "data" / "raw" / "cmip6_multimember_z850_subsets"
PARTS = ROOT / "data" / "processed" / "cmip6_multimember_z850_parts"
ANNUAL_OUT = ROOT / "data" / "processed" / "cmip6_multimember_z850_jja_annual.csv.gz"
RESPONSE_OUT = ROOT / "data" / "processed" / "cmip6_multimember_z850_responses.csv"
FAILURE_OUT = ROOT / "data" / "processed" / "cmip6_multimember_z850_failures.csv"
PERIODS = ((1951, 1980), (1985, 2014))
YEARS = np.concatenate([np.arange(first, last + 1) for first, last in PERIODS])
METRICS = ["raw_z850_m", "zonal_z850_m", "eddy_z850_m"]


def height_factor(units: str) -> float:
    normalised = units.lower().replace("**", "").replace("^", "").replace(" ", "")
    if normalised in {"m", "meter", "metre"}:
        return 1.0
    if normalised in {"m2s-2", "m2/s2"}:
        return 1.0 / 9.80665
    raise RuntimeError(f"Unsupported zg units: {units!r}")


def valid_part(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        data = pd.read_csv(path)
        return (
            len(data) == len(YEARS)
            and data.year.tolist() == YEARS.tolist()
            and data[METRICS].notna().all().all()
        )
    except Exception:
        return False


def download_subset(url: str, first: int, last: int, target: Path) -> None:
    if target.exists() and target.stat().st_size > 0:
        return
    ncss = url.replace("/dodsC/", "/ncss/grid/")
    query = {
        "var": "zg", "time_start": f"{first}-01-01T00:00:00Z",
        "time_end": f"{last}-12-30T23:59:59Z", "accept": "netcdf4",
        "addLatLon": "true", "north": 25, "south": 15,
        "west": -180, "east": 180, "vertCoord": 85000,
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(".nc.part")
    for attempt in range(4):
        try:
            with requests.get(ncss, params=query, timeout=(30, 300), stream=True) as response:
                response.raise_for_status()
                with partial.open("wb") as handle:
                    for chunk in response.iter_content(1024 * 1024):
                        if chunk:
                            handle.write(chunk)
            partial.replace(target)
            return
        except Exception:
            partial.unlink(missing_ok=True)
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)


def annual_from_subset(path: Path, model: str, experiment: str, member: str) -> pd.DataFrame:
    coder = xr.coders.CFDatetimeCoder(use_cftime=True)
    with xr.open_dataset(path, decode_times=coder) as ds:
        expected = {"source_id": model, "experiment_id": experiment, "variant_label": member}
        mismatches = {
            key: (value, ds.attrs.get(key))
            for key, value in expected.items()
            if ds.attrs.get(key) not in (None, value)
        }
        if mismatches:
            raise RuntimeError(f"NCSS identity mismatch in {path.name}: {mismatches}")
        rename = {}
        for old, new in (("latitude", "lat"), ("longitude", "lon"), ("valid_time", "time")):
            if old in ds.coords or old in ds.dims:
                rename[old] = new
        ds = ds.rename(rename)
        field = ds.zg
        if "plev" in field.dims:
            field = field.sel(plev=85000.0, method="nearest")
        field = field * height_factor(ds.zg.attrs.get("units", ""))
        months = field.time.dt.month.values.astype(int)
        field = field.isel(time=np.flatnonzero(np.isin(months, [6, 7, 8]))).load()
        lon_mod = field.lon % 360.0
        regional = field.where((lon_mod >= 115.0) & (lon_mod <= 150.0), drop=True)
        weights = np.cos(np.deg2rad(field.lat))
        raw = regional.weighted(weights).mean(("lat", "lon"))
        zonal = field.weighted(weights).mean(("lat", "lon"))
        monthly = pd.DataFrame(
            {
                "year": field.time.dt.year.values.astype(int),
                "month": field.time.dt.month.values.astype(int),
                "raw_z850_m": raw.values.astype(float),
                "zonal_z850_m": zonal.values.astype(float),
            }
        )
    monthly["eddy_z850_m"] = monthly.raw_z850_m - monthly.zonal_z850_m
    return monthly.groupby("year", as_index=False)[METRICS].mean()


def extract(group: pd.DataFrame) -> Path:
    row = group.iloc[0]
    part = PARTS / f"{row.model}_{row.experiment}_{row.member}_z850_jja.csv"
    if valid_part(part):
        print(f"reuse {part.name}", flush=True)
        return part
    pieces = []
    for source in group.sort_values("filename").itertuples():
        for period_first, period_last in PERIODS:
            first = max(period_first, int(source.file_start_year))
            last = min(period_last, int(source.file_end_year))
            if first > last:
                continue
            target = (
                SUBSETS / row.model / row.experiment / row.member /
                f"{Path(source.filename).stem}__z850_15_25n_{first}_{last}.nc"
            )
            download_subset(source.opendap_url, first, last, target)
            pieces.append(annual_from_subset(target, row.model, row.experiment, row.member))
    if not pieces:
        raise RuntimeError("No source files overlap the response periods")
    annual = pd.concat(pieces, ignore_index=True).drop_duplicates("year").sort_values("year")
    if len(annual) != len(YEARS) or annual.year.tolist() != YEARS.tolist():
        missing = sorted(set(YEARS) - set(annual.year))
        raise RuntimeError(f"Incomplete response years: {missing}")
    annual.insert(0, "dataset_id", row.dataset_id)
    annual.insert(0, "member", row.member)
    annual.insert(0, "experiment", row.experiment)
    annual.insert(0, "model", row.model)
    annual.to_csv(part, index=False)
    print(f"wrote {part.name}", flush=True)
    return part


def responses(annual: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, group in annual.groupby(["model", "experiment", "member"], sort=True):
        early = group.loc[group.year.between(1951, 1980)]
        late = group.loc[group.year.between(1985, 2014)]
        if len(early) != 30 or len(late) != 30:
            raise RuntimeError(f"Bad response periods for {keys}")
        row = {"model": keys[0], "experiment": keys[1], "member": keys[2]}
        for metric in METRICS:
            row[f"{metric}_early"] = float(early[metric].mean())
            row[f"{metric}_late"] = float(late[metric].mean())
            row[f"{metric}_change"] = row[f"{metric}_late"] - row[f"{metric}_early"]
        raw_change = row["raw_z850_m_change"]
        row["zonal_fraction_of_raw_change"] = (
            row["zonal_z850_m_change"] / raw_change if abs(raw_change) > 1e-12 else np.nan
        )
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+")
    parser.add_argument("--max-workers", type=int, default=4)
    args = parser.parse_args()
    files = pd.read_csv(FILES)
    if args.models:
        files = files.loc[files.model.isin(args.models)]
    groups = list(files.groupby(["model", "member", "experiment"], sort=True))
    if not groups:
        raise RuntimeError("No resolved NCSS datasets selected")
    PARTS.mkdir(parents=True, exist_ok=True)
    parts, failures = [], []
    with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
        futures = {pool.submit(extract, group): key for key, group in groups}
        for future in as_completed(futures):
            model, member, experiment = futures[future]
            try:
                parts.append(future.result())
            except Exception as exc:
                failures.append(
                    {"model": model, "experiment": experiment, "member": member,
                     "error": f"{type(exc).__name__}: {exc}"}
                )
                print(f"failed {model} {member} {experiment}: {exc}", flush=True)
    pd.DataFrame(failures, columns=["model", "experiment", "member", "error"]).to_csv(
        FAILURE_OUT, index=False
    )
    if failures:
        raise SystemExit(2)
    annual = pd.concat([pd.read_csv(path) for path in parts], ignore_index=True)
    annual.to_csv(ANNUAL_OUT, index=False, compression={"method": "gzip", "mtime": 0})
    response = responses(annual)
    response.to_csv(RESPONSE_OUT, index=False)
    print(f"complete: {len(response)} datasets, {annual.model.nunique()} models", flush=True)


if __name__ == "__main__":
    main()
