"""Download a resumable native-model-level ERA5 sigma=0.85 pilot.

The pilot deliberately requests only the dates containing six-hourly track
points in DJF 2011. Surface pressure is reused from the already validated ERA5
single-level archive; only native model-level u and v are downloaded here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import cdsapi
import numpy as np
import pandas as pd
import xarray as xr


ROOT = Path(__file__).resolve().parents[1]
EVENTS = ROOT / "data/processed/events_terrain_qc_intensity.csv.gz"
POINTS = ROOT / "data/processed/track_points.csv.gz"
OUT = ROOT / "data/raw/era5_model_level_sigma_pilot"
MANIFEST = ROOT / "data/manifests/era5_model_level_sigma_pilot_manifest.json"
SEASON_YEAR = 2011
LEVELS = [111, 112, 113, 114, 115]
AREA = [35.5, 101.0, 27.5, 111.5]
GRID = [0.5, 0.5]
TIMES = ["00:00", "06:00", "12:00", "18:00"]
COEFFICIENT_SOURCE = "https://confluence.ecmwf.int/display/UDOC/L137+model+level+definitions"
REQUEST_SOURCE = "https://cds.climate.copernicus.eu/datasets/reanalysis-era5-complete?tab=d_download"

# Half-level coefficients needed to compute full-level pressures for levels
# 111--115. Units of a are Pa; b is dimensionless. Values are copied verbatim
# from the ECMWF L137 definition table cited above.
HALF_LEVEL_COEFFICIENTS = {
    110: {"a_pa": 8163.375000, "b": 0.704669},
    111: {"a_pa": 7470.343750, "b": 0.727739},
    112: {"a_pa": 6804.421875, "b": 0.749797},
    113: {"a_pa": 6168.531250, "b": 0.770798},
    114: {"a_pa": 5564.382813, "b": 0.790717},
    115: {"a_pa": 4993.796875, "b": 0.809536},
}


def pilot_points() -> pd.DataFrame:
    events = pd.read_csv(EVENTS, parse_dates=["genesis_time"])
    events["season_year"] = events.year + events.genesis_time.dt.month.eq(12).astype(int)
    event_ids = set(
        events.loc[
            events.genesis_time.dt.month.isin([12, 1, 2])
            & events.season_year.eq(SEASON_YEAR),
            "event_id",
        ]
    )
    points = pd.read_csv(POINTS, parse_dates=["time"])
    points = points.loc[
        points.event_id.isin(event_ids) & points.time.dt.hour.isin([0, 6, 12, 18])
    ].copy()
    if len(event_ids) != 19 or len(points) != 46:
        raise RuntimeError(
            f"Frozen pilot selection changed: expected 19 events/46 points, "
            f"found {len(event_ids)} events/{len(points)} points"
        )
    return points.sort_values(["time", "event_id"])


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


def validate(path: Path, expected_dates: list[str]) -> tuple[bool, str]:
    if not path.exists() or path.stat().st_size == 0:
        return False, "missing or empty"
    try:
        with xr.open_dataset(path) as source:
            ds = canonical(source)
            if not {"u", "v"}.issubset(ds.data_vars):
                return False, f"missing u/v: variables={list(ds.data_vars)}"
            lev = level_name(ds)
            actual_levels = set(np.asarray(ds[lev]).astype(int).tolist())
            if actual_levels != set(LEVELS):
                return False, f"unexpected levels: {sorted(actual_levels)}"
            actual_dates = set(pd.to_datetime(ds.time.values).strftime("%Y-%m-%d"))
            if actual_dates != set(expected_dates):
                return False, f"unexpected dates: {sorted(actual_dates)}"
            if ds.sizes["time"] != len(expected_dates) * 4:
                return False, f"unexpected time count: {ds.sizes['time']}"
            if any(bool(ds[name].isnull().all()) for name in ("u", "v")):
                return False, "u or v is entirely missing"
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


def download(mode: str, dates: list[str]) -> dict[str, object]:
    target = OUT / f"era5_model_level_sigma_{mode}_DJF{SEASON_YEAR}.nc"
    partial = target.with_suffix(".nc.part")
    valid, _ = validate(target, dates)
    if not valid:
        partial.unlink(missing_ok=True)
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
        print(
            f"request {mode}: {len(dates)} dates, levels {LEVELS[0]}-{LEVELS[-1]}, "
            f"area {AREA}",
            flush=True,
        )
        for attempt in range(1, 4):
            try:
                cdsapi.Client().retrieve("reanalysis-era5-complete", request, str(partial))
                break
            except Exception as exc:
                if attempt == 3:
                    raise
                partial.unlink(missing_ok=True)
                delay = 5 * attempt
                print(f"retry after {type(exc).__name__} ({delay}s)", flush=True)
                time.sleep(delay)
        ok, reason = validate(partial, dates)
        if not ok:
            raise RuntimeError(f"downloaded payload failed validation: {reason}")
        partial.replace(target)
    ok, reason = validate(target, dates)
    if not ok:
        raise RuntimeError(f"stored payload failed validation: {reason}")
    print(f"verified {target.name} ({target.stat().st_size:,} bytes)", flush=True)
    return {
        "mode": mode,
        "dates": dates,
        "path": str(target.relative_to(ROOT)).replace("\\", "/"),
        "bytes": target.stat().st_size,
        "sha256": sha256(target),
        "validated": True,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", action="store_true", help="request only the first pilot date")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    points = pilot_points()
    all_dates = sorted(points.time.dt.strftime("%Y-%m-%d").unique().tolist())
    mode = "preflight" if args.preflight else "pilot"
    dates = all_dates[:1] if args.preflight else all_dates
    entry = download(mode, dates)
    previous = json.loads(MANIFEST.read_text(encoding="utf-8")) if MANIFEST.exists() else {}
    files = {row["mode"]: row for row in previous.get("files", [])}
    files[mode] = entry
    manifest = {
        "dataset": "reanalysis-era5-complete",
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "season_year": SEASON_YEAR,
        "selection": "all 6-hourly track points from the 19-event DJF 2011 representative winter",
        "selection_counts": {
            "events": int(points.event_id.nunique()),
            "track_points": int(len(points)),
            "unique_dates": int(len(all_dates)),
        },
        "area_nwse": AREA,
        "grid_degrees": GRID,
        "times_utc": TIMES,
        "model_levels": LEVELS,
        "variables": {"131": "u_component_of_wind", "132": "v_component_of_wind"},
        "surface_pressure_source": "existing data/raw/era5_surface_pressure_6hourly annual files",
        "half_level_coefficients": HALF_LEVEL_COEFFICIENTS,
        "coefficient_source": COEFFICIENT_SOURCE,
        "request_documentation": REQUEST_SOURCE,
        "files": [files[key] for key in sorted(files)],
        "failures": [],
    }
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {MANIFEST.relative_to(ROOT)}", flush=True)


if __name__ == "__main__":
    main()
