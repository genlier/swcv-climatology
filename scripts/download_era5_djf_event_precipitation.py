"""Download ERA5 hourly precipitation for local-basin DJF vortex event days."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import tempfile
import time
import zipfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import cdsapi
import pandas as pd
import xarray as xr


ROOT = Path(__file__).resolve().parents[1]
EVENTS = ROOT / "data" / "processed" / "events_season_year_terrain_qc.csv.gz"
TRACK_POINTS = ROOT / "data" / "processed" / "track_points.csv.gz"
OUT = ROOT / "data" / "raw" / "era5_djf_event_precipitation"
MANIFEST = ROOT / "data" / "manifests" / "era5_djf_event_precipitation_manifest.json"
AREA = [38, 95, 20, 120]
GRID = [0.5, 0.5]
VARIABLES = {"tp", "sf", "t2m"}
TIMES = [f"{hour:02d}:00" for hour in range(24)]


def event_dates(selection: str) -> tuple[dict[tuple[int, int], list[str]], dict[str, int]]:
    events = pd.read_csv(EVENTS, parse_dates=["genesis_time", "lysis_time"])
    eligible = events.loc[
        events.season.eq("DJF")
        & events.season_year.between(1980, 2023)
        & events.track_cluster.isin([1, 2])
    ].copy()
    if selection == "cluster1":
        selected = eligible.loc[eligible.track_cluster.eq(1)].copy()
    else:
        # Cluster 2 is always downloaded over the expanded domain.  For
        # cluster 1, only events whose 500-km circle can touch the legacy
        # 20--38 N, 95--120 E boundary need an expanded-domain supplement.
        points = pd.read_csv(TRACK_POINTS, usecols=["event_id", "lat", "lon"])
        points = points.loc[points.event_id.isin(eligible.event_id)].copy()
        dlat = 500.0 / 111.2
        dlon = 500.0 / (111.2 * points.lat.map(lambda x: math.cos(math.radians(x))))
        points["legacy_safe_500km"] = (
            points.lat.sub(dlat).ge(20.0)
            & points.lat.add(dlat).le(38.0)
            & points.lon.sub(dlon).ge(95.0)
            & points.lon.add(dlon).le(120.0)
        )
        safe = points.groupby("event_id").legacy_safe_500km.all()
        unsafe_c1 = set(safe.index[~safe])
        selected = eligible.loc[
            eligible.track_cluster.eq(2) | eligible.event_id.isin(unsafe_c1)
        ].copy()
    grouped: dict[tuple[int, int], set[str]] = defaultdict(set)
    for row in selected.itertuples():
        start = pd.Timestamp(row.genesis_time).floor("D")
        end = (pd.Timestamp(row.lysis_time) + pd.Timedelta(hours=24)).floor("D")
        for day in pd.date_range(start, end, freq="D"):
            grouped[(day.year, day.month)].add(f"{day.day:02d}")
    requests = {key: sorted(values) for key, values in grouped.items()}
    summary = {
        "events": int(len(selected)),
        "cluster1_events": int(selected.track_cluster.eq(1).sum()),
        "cluster2_events": int(selected.track_cluster.eq(2).sum()),
        "unique_dates": int(sum(len(values) for values in requests.values())),
        "year_month_requests": int(len(requests)),
    }
    return requests, summary


def canonical(ds: xr.Dataset) -> xr.Dataset:
    rename = {}
    for old, new in (("valid_time", "time"), ("latitude", "lat"), ("longitude", "lon")):
        if old in ds.coords or old in ds.dims:
            rename[old] = new
    return ds.rename(rename)


def normalise_payload(partial: Path, target: Path) -> None:
    """Promote a CDS NetCDF or merge a ZIP split by GRIB step type."""
    if zipfile.is_zipfile(partial):
        with tempfile.TemporaryDirectory(dir=OUT, prefix="precip_unpack_") as temp_name:
            temp = Path(temp_name)
            with zipfile.ZipFile(partial) as archive:
                archive.extractall(temp)
            datasets = []
            for member in sorted(temp.rglob("*.nc")):
                with xr.open_dataset(member) as source:
                    datasets.append(canonical(source).load())
            if not datasets:
                raise RuntimeError("CDS ZIP contains no NetCDF members")
            merged = xr.merge(datasets, compat="override", join="exact")
            merged_path = temp / "merged.nc"
            merged.to_netcdf(merged_path, engine="netcdf4")
            merged.close()
            merged_path.replace(target)
        partial.unlink()
    else:
        partial.replace(target)


def validate(path: Path, year: int, month: int, days: list[str]) -> tuple[bool, str]:
    if not path.exists() or path.stat().st_size == 0:
        return False, "missing or empty"
    try:
        with xr.open_dataset(path) as source:
            ds = canonical(source)
            if not VARIABLES.issubset(ds.data_vars):
                return False, f"missing variables: {sorted(VARIABLES - set(ds.data_vars))}"
            actual_dates = set(pd.to_datetime(ds.time.values).strftime("%Y-%m-%d"))
            expected_dates = {f"{year:04d}-{month:02d}-{day}" for day in days}
            if actual_dates != expected_dates:
                return False, "unexpected date coverage"
            if ds.sizes["time"] != len(days) * 24:
                return False, f"expected {len(days) * 24} hours, found {ds.sizes['time']}"
            if any(bool(ds[name].isnull().all()) for name in VARIABLES):
                return False, "one or more variables are entirely missing"
        return True, "ok"
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_month(year: int, month: int, days: list[str]) -> dict[str, object]:
    target = OUT / f"era5_djf_event_precip_{year}{month:02d}.nc"
    partial = target.with_suffix(".nc.part")
    valid, _ = validate(target, year, month, days)
    if not valid:
        # Earlier CDS jobs may have left a complete ZIP payload under .part.
        # Salvage it before submitting a duplicate request.
        if partial.exists() and partial.stat().st_size > 0:
            try:
                normalise_payload(partial, target)
            except Exception as exc:
                print(f"discard unusable partial {target.name}: {exc}", flush=True)
                partial.unlink(missing_ok=True)
        valid, _ = validate(target, year, month, days)
        if not valid:
            print(f"request {year}-{month:02d}: {len(days)} event-window days", flush=True)
            request = {
                "product_type": ["reanalysis"],
                "variable": ["2m_temperature", "snowfall", "total_precipitation"],
                "year": [str(year)],
                "month": [f"{month:02d}"],
                "day": days,
                "time": TIMES,
                "data_format": "netcdf",
                "download_format": "unarchived",
                "area": AREA,
                "grid": GRID,
            }
            for attempt in range(1, 5):
                try:
                    cdsapi.Client().retrieve(
                        "reanalysis-era5-single-levels", request, str(partial)
                    )
                    normalise_payload(partial, target)
                    break
                except Exception as exc:
                    # A connection can fail after the payload arrives; salvage it
                    # before submitting another CDS job.
                    if partial.exists() and partial.stat().st_size > 0:
                        try:
                            normalise_payload(partial, target)
                        except Exception:
                            partial.unlink(missing_ok=True)
                        else:
                            recovered, _ = validate(target, year, month, days)
                            if recovered:
                                print(f"recovered payload after {type(exc).__name__}", flush=True)
                                break
                    if attempt == 4:
                        raise
                    delay = 5 * attempt
                    print(
                        f"retry {attempt}/3 for {year}-{month:02d} after "
                        f"{type(exc).__name__} ({delay}s)",
                        flush=True,
                    )
                    time.sleep(delay)
        valid, reason = validate(target, year, month, days)
        if not valid:
            raise RuntimeError(f"validation failed: {reason}")
        print(f"verified {target.name} ({target.stat().st_size} bytes)", flush=True)
    return {
        "year": year,
        "month": month,
        "days": days,
        "path": str(target.relative_to(ROOT)).replace("\\", "/"),
        "bytes": target.stat().st_size,
        "sha256": sha256(target),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--selection", choices=["cluster1", "track-relative"], default="cluster1"
    )
    return parser.parse_args()


def main() -> None:
    global OUT, MANIFEST, AREA
    args = parse_args()
    if args.selection == "track-relative":
        OUT = ROOT / "data" / "raw" / "era5_djf_event_precipitation_extended"
        MANIFEST = ROOT / "data" / "manifests" / "era5_djf_event_precipitation_extended_manifest.json"
        AREA = [42, 94, 20, 125]
    OUT.mkdir(parents=True, exist_ok=True)
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    requests, selection = event_dates(args.selection)
    entries, failures = [], []
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = {
            pool.submit(download_month, year, month, days): (year, month)
            for (year, month), days in requests.items()
        }
        for future in as_completed(futures):
            year, month = futures[future]
            try:
                entries.append(future.result())
            except Exception as exc:
                failures.append({"year": year, "month": month, "error": f"{type(exc).__name__}: {exc}"})
                print(f"failed {year}-{month:02d}: {exc}", flush=True)
    entries.sort(key=lambda row: (int(row["year"]), int(row["month"])))
    failures.sort(key=lambda row: (int(row["year"]), int(row["month"])))
    manifest = {
        "dataset": "reanalysis-era5-single-levels",
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "selection": (
            "DJF 1980-2023 cluster-1 events"
            if args.selection == "cluster1"
            else "DJF 1980-2023 cluster-2 events plus cluster-1 events unsafe in the legacy 500-km domain"
        ) + ", genesis day through floor(lysis + 24 h)",
        "selection_counts": selection,
        "area_nwse": AREA,
        "grid_degrees": GRID,
        "times_utc": TIMES,
        "variables": ["2m_temperature", "snowfall", "total_precipitation"],
        "files": entries,
        "failures": failures,
    }
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {MANIFEST.relative_to(ROOT)}: {len(entries)} files, {len(failures)} failures")
    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
