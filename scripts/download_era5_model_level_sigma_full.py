"""Download ERA5 native model-level winds for all DJF 1980--2023 events."""

from __future__ import annotations

import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import cdsapi
import numpy as np
import pandas as pd
import xarray as xr


ROOT = Path(__file__).resolve().parents[1]
EVENTS = ROOT / "data/processed/events_terrain_qc_intensity.csv.gz"
POINTS = ROOT / "data/processed/track_points.csv.gz"
OUT = ROOT / "data/raw/era5_model_level_sigma_djf"
MANIFEST = ROOT / "data/manifests/era5_model_level_sigma_djf_manifest.json"
LEVELS = [110, 111, 112, 113, 114, 115]
AREA = [38.5, 95.5, 20.5, 120.0]
GRID = [0.5, 0.5]
TIMES = ["00:00", "06:00", "12:00", "18:00"]
HALF_LEVEL_COEFFICIENTS = {
    109: {"a_pa": 8880.453125, "b": 0.680643},
    110: {"a_pa": 8163.375000, "b": 0.704669},
    111: {"a_pa": 7470.343750, "b": 0.727739},
    112: {"a_pa": 6804.421875, "b": 0.749797},
    113: {"a_pa": 6168.531250, "b": 0.770798},
    114: {"a_pa": 5564.382813, "b": 0.790717},
    115: {"a_pa": 4993.796875, "b": 0.809536},
}
COEFFICIENT_SOURCE = "https://confluence.ecmwf.int/display/UDOC/L137+model+level+definitions"
REQUEST_SOURCE = "https://cds.climate.copernicus.eu/datasets/reanalysis-era5-complete?tab=d_download"


def selection() -> tuple[pd.DataFrame, dict[int, list[str]]]:
    events = pd.read_csv(EVENTS, parse_dates=["genesis_time"])
    events["season_year"] = events.year + events.genesis_time.dt.month.eq(12).astype(int)
    event_ids = set(
        events.loc[
            events.genesis_time.dt.month.isin([12, 1, 2])
            & events.season_year.between(1980, 2023),
            "event_id",
        ]
    )
    points = pd.read_csv(POINTS, parse_dates=["time"])
    points = points.loc[
        points.event_id.isin(event_ids) & points.time.dt.hour.isin([0, 6, 12, 18])
    ].copy()
    dates: dict[int, list[str]] = {}
    for year, group in points.groupby(points.time.dt.year):
        dates[int(year)] = sorted(group.time.dt.strftime("%Y-%m-%d").unique().tolist())
    if len(event_ids) != 899 or len(points) != 2044 or sum(len(value) for value in dates.values()) != 1126:
        raise RuntimeError(
            "Frozen full selection changed: expected 899 events, 2044 points and 1126 year-specific dates"
        )
    return points, dates


def canonical(ds: xr.Dataset) -> xr.Dataset:
    rename = {}
    for old, new in (
        ("valid_time", "time"),
        ("latitude", "lat"),
        ("longitude", "lon"),
        ("hybrid", "level"),
        ("model_level", "level"),
    ):
        if (old in ds.coords or old in ds.dims) and new not in ds.coords and new not in ds.dims:
            rename[old] = new
    return ds.rename(rename)


def level_name(ds: xr.Dataset) -> str:
    for name in ("level", "hybrid", "model_level"):
        if name in ds.coords or name in ds.dims:
            return name
    candidates = [name for name, size in ds.sizes.items() if size == len(LEVELS)]
    if len(candidates) == 1:
        return candidates[0]
    raise KeyError(f"Cannot identify model-level coordinate in {dict(ds.sizes)}")


def validate(path: Path, dates: list[str]) -> tuple[bool, str]:
    if not path.exists() or path.stat().st_size == 0:
        return False, "missing or empty"
    try:
        with xr.open_dataset(path) as source:
            ds = canonical(source)
            if not {"u", "v"}.issubset(ds.data_vars):
                return False, f"missing u/v: {list(ds.data_vars)}"
            lev = level_name(ds)
            actual_levels = set(np.asarray(ds[lev]).astype(int).tolist())
            if actual_levels != set(LEVELS):
                return False, f"unexpected levels: {sorted(actual_levels)}"
            actual_dates = set(pd.to_datetime(ds.time.values).strftime("%Y-%m-%d"))
            if actual_dates != set(dates):
                return False, "unexpected date coverage"
            if ds.sizes["time"] != len(dates) * 4:
                return False, f"unexpected time count: {ds.sizes['time']}"
            if any(bool(ds[name].isnull().all()) for name in ("u", "v")):
                return False, "u or v entirely missing"
            if float(ds.lat.max()) < AREA[0] or float(ds.lat.min()) > AREA[2]:
                return False, "latitude coverage incomplete"
            if float(ds.lon.min()) > AREA[1] or float(ds.lon.max()) < AREA[3]:
                return False, "longitude coverage incomplete"
        return True, "ok"
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def retrieve_year(year: int, dates: list[str]) -> dict[str, object]:
    target = OUT / f"era5_model_level_sigma_djf_dates_{year}.nc"
    partial = target.with_suffix(".nc.part")
    valid, _ = validate(target, dates)
    if not valid and partial.exists():
        recovered, _ = validate(partial, dates)
        if recovered:
            partial.replace(target)
            valid = True
        else:
            partial.unlink()
    if not valid:
        request = {
            "date": "/".join(dates),
            "levelist": "/".join(str(value) for value in LEVELS),
            "levtype": "ml",
            "param": "131/132",
            "stream": "oper",
            "time": "00/to/18/by/6",
            "type": "an",
            "area": "/".join(str(value) for value in AREA),
            "grid": "/".join(str(value) for value in GRID),
            "format": "netcdf",
        }
        print(f"request {year}: {len(dates)} dates", flush=True)
        for attempt in range(1, 4):
            try:
                cdsapi.Client().retrieve("reanalysis-era5-complete", request, str(partial))
                break
            except Exception as exc:
                if attempt == 3:
                    raise
                partial.unlink(missing_ok=True)
                delay = 10 * attempt
                print(f"retry {year} after {type(exc).__name__} ({delay}s)", flush=True)
                time.sleep(delay)
        valid, reason = validate(partial, dates)
        if not valid:
            raise RuntimeError(f"{year} payload failed validation: {reason}")
        partial.replace(target)
    valid, reason = validate(target, dates)
    if not valid:
        raise RuntimeError(f"{year} stored file failed validation: {reason}")
    print(f"verified {year}: {target.stat().st_size:,} bytes", flush=True)
    return {
        "year": year,
        "dates": dates,
        "path": str(target.relative_to(ROOT)).replace("\\", "/"),
        "bytes": target.stat().st_size,
        "sha256": sha256(target),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    points, dates_by_year = selection()
    entries: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = {
            pool.submit(retrieve_year, year, dates): year
            for year, dates in dates_by_year.items()
        }
        for future in as_completed(futures):
            year = futures[future]
            try:
                entries.append(future.result())
            except Exception as exc:
                failures.append({"year": year, "error": f"{type(exc).__name__}: {exc}"})
                print(f"failed {year}: {exc}", flush=True)
    entries.sort(key=lambda row: int(row["year"]))
    failures.sort(key=lambda row: int(row["year"]))
    manifest = {
        "dataset": "reanalysis-era5-complete",
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "selection": "all 6-hourly track points for DJF season-years 1980-2023",
        "selection_counts": {
            "events": int(points.event_id.nunique()),
            "track_points": int(len(points)),
            "year_specific_unique_dates": int(sum(len(value) for value in dates_by_year.values())),
            "calendar_year_files": int(len(dates_by_year)),
        },
        "area_nwse": AREA,
        "grid_degrees": GRID,
        "times_utc": TIMES,
        "model_levels": LEVELS,
        "variables": {"131": "u_component_of_wind", "132": "v_component_of_wind"},
        "surface_pressure_source": "data/raw/era5_surface_pressure_6hourly annual files",
        "observed_surface_pressure_patch_range_hpa": [561.1693, 1032.4299],
        "half_level_coefficients": HALF_LEVEL_COEFFICIENTS,
        "coefficient_source": COEFFICIENT_SOURCE,
        "request_documentation": REQUEST_SOURCE,
        "files": entries,
        "failures": failures,
    }
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {MANIFEST.relative_to(ROOT)}: {len(entries)} files, {len(failures)} failures", flush=True)
    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
