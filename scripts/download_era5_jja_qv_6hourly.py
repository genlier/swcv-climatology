"""Download same-level 6-hourly ERA5 q and v for the JJA transient audit."""

from __future__ import annotations

import hashlib
import json
import tempfile
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import cdsapi
import pandas as pd
import xarray as xr


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/raw/era5_jja_qv_6hourly"
MANIFEST = ROOT / "data/manifests/era5_jja_qv_6hourly_manifest.json"
YEARS = tuple(range(1980, 2024))
# CDS rejects the previous four-year requests at its current per-request cost
# ceiling. Annual requests remain comfortably below that limit and preserve
# restartability without changing the requested scientific sample.
CHUNKS = tuple((year,) for year in YEARS)
LEVELS = {700, 850}
VARIABLES = {"q", "v"}
AREA = [27.5, 100.0, 20.0, 115.0]
GRID = [0.5, 0.5]
TIMES = ["00:00", "06:00", "12:00", "18:00"]
EXPECTED_PER_YEAR = 92 * len(TIMES)


def canonical(ds: xr.Dataset) -> xr.Dataset:
    rename = {}
    for old, new in (("valid_time", "time"), ("latitude", "lat"), ("longitude", "lon"), ("pressure_level", "level")):
        if old in ds.coords or old in ds.dims:
            rename[old] = new
    return ds.rename(rename)


def normalise_payload(partial: Path, target: Path) -> None:
    if not zipfile.is_zipfile(partial):
        partial.replace(target)
        return
    with tempfile.TemporaryDirectory(dir=OUT, prefix="qv_unpack_") as temp_name:
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
        merged.to_netcdf(target, engine="netcdf4")
        merged.close()
    partial.unlink()


def validate(path: Path, years: tuple[int, ...]) -> tuple[bool, str]:
    if not path.exists() or path.stat().st_size == 0:
        return False, "missing or empty"
    try:
        with xr.open_dataset(path) as source:
            ds = canonical(source)
            if not VARIABLES.issubset(ds.data_vars):
                return False, f"missing variables {VARIABLES - set(ds.data_vars)}"
            if set(ds.level.values.astype(int)) != LEVELS:
                return False, "unexpected pressure levels"
            if ds.sizes.get("time") != EXPECTED_PER_YEAR * len(years):
                return False, f"unexpected time count {ds.sizes.get('time')}"
            index = pd.DatetimeIndex(ds.time.values)
            if set(index.year) != set(years) or set(index.month) != {6, 7, 8}:
                return False, "unexpected date coverage"
            if ds.sizes.get("lat") != 16 or ds.sizes.get("lon") != 31:
                return False, "unexpected grid"
            if any(bool(ds[name].isnull().all()) for name in VARIABLES):
                return False, "one or more variables are entirely missing"
        return True, "ok"
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def split_chunk(chunk_path: Path, years: tuple[int, ...]) -> list[dict[str, object]]:
    rows = []
    with xr.open_dataset(chunk_path) as source:
        ds = canonical(source)
        for year in years:
            target = OUT / f"era5_jja_qv_6hourly_{year}.nc"
            valid, _ = validate(target, (year,))
            if not valid:
                temporary = target.with_suffix(".nc.part")
                subset = ds.sel(time=ds.time.dt.year == year).load()
                encoding = {name: {"zlib": True, "complevel": 4, "shuffle": True} for name in VARIABLES}
                subset.to_netcdf(temporary, engine="netcdf4", encoding=encoding)
                valid, reason = validate(temporary, (year,))
                if not valid:
                    raise RuntimeError(f"Split {year} validation failed: {reason}")
                temporary.replace(target)
            rows.append({
                "year": year,
                "path": target.relative_to(ROOT).as_posix(),
                "bytes": target.stat().st_size,
                "sha256": sha256(target),
            })
    return rows


def download_chunk(years: tuple[int, ...]) -> list[dict[str, object]]:
    annual_paths = [OUT / f"era5_jja_qv_6hourly_{year}.nc" for year in years]
    if all(validate(path, (year,))[0] for path, year in zip(annual_paths, years)):
        print(f"skip completed {years[0]}-{years[-1]}", flush=True)
        return split_chunk(annual_paths[0], (years[0],)) if len(years) == 1 else [
            {"year": year, "path": path.relative_to(ROOT).as_posix(), "bytes": path.stat().st_size, "sha256": sha256(path)}
            for path, year in zip(annual_paths, years)
        ]
    chunk = OUT / f"chunk_{years[0]}_{years[-1]}.nc"
    valid, _ = validate(chunk, years)
    if not valid:
        partial = chunk.with_suffix(".nc.part")
        print(f"request {years[0]}-{years[-1]}", flush=True)
        request = {
            "product_type": ["reanalysis"],
            "variable": ["specific_humidity", "v_component_of_wind"],
            "pressure_level": ["700", "850"],
            "year": [str(year) for year in years],
            "month": ["06", "07", "08"],
            "day": [f"{day:02d}" for day in range(1, 32)],
            "time": TIMES,
            "data_format": "netcdf",
            "download_format": "unarchived",
            "area": AREA,
            "grid": GRID,
        }
        for attempt in range(1, 5):
            try:
                cdsapi.Client().retrieve("reanalysis-era5-pressure-levels", request, str(partial))
                normalise_payload(partial, chunk)
                break
            except Exception:
                if attempt == 4:
                    raise
                time.sleep(5 * attempt)
        valid, reason = validate(chunk, years)
        if not valid:
            raise RuntimeError(f"Chunk validation failed: {reason}")
        print(f"verified {chunk.name} ({chunk.stat().st_size} bytes)", flush=True)
    return split_chunk(chunk, years)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    rows, failures = [], []
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = {pool.submit(download_chunk, years): years for years in CHUNKS}
        for future in as_completed(futures):
            years = futures[future]
            try:
                rows.extend(future.result())
            except Exception as exc:
                failures.append({"years": list(years), "error": f"{type(exc).__name__}: {exc}"})
                print(f"failed {years[0]}-{years[-1]}: {exc}", flush=True)
    rows.sort(key=lambda row: int(row["year"]))
    manifest = {
        "dataset": "reanalysis-era5-pressure-levels",
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "period": "JJA 1980--2023",
        "times_utc": TIMES,
        "pressure_levels_hpa": sorted(LEVELS),
        "area_nwse": AREA,
        "grid_degrees": GRID,
        "variables": {"q": "specific humidity; kg kg-1", "v": "northward wind; m s-1"},
        "files": rows,
        "failures": failures,
    }
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {MANIFEST.relative_to(ROOT)}: {len(rows)} files, {len(failures)} failures")
    if failures or len(rows) != len(YEARS):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
