"""Download resumable ERA5 monthly pressure-level fields for DJF mechanism tests."""

from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import cdsapi
import xarray as xr


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "raw" / "era5_djf_mechanism_monthly"
MANIFEST = ROOT / "data" / "manifests" / "era5_djf_mechanism_manifest.json"
YEARS = range(1978, 2024)
LEVELS = {200, 500, 700, 850}
VARIABLES = {"z", "t", "u", "v"}
AREA = [55, 60, 10, 140]
GRID = [0.5, 0.5]


def months_for_year(year: int) -> list[str]:
    if year == 1978:
        return ["12"]
    if year == 2023:
        return ["01", "02"]
    return ["01", "02", "12"]


def coordinate_name(ds: xr.Dataset, candidates: tuple[str, ...]) -> str:
    for name in candidates:
        if name in ds.coords or name in ds.dims:
            return name
    raise KeyError(f"None of {candidates} found in {list(ds.coords)}")


def validate(path: Path, year: int) -> tuple[bool, str]:
    if not path.exists() or path.stat().st_size == 0:
        return False, "missing or empty"
    try:
        with xr.open_dataset(path) as ds:
            time_name = coordinate_name(ds, ("valid_time", "time"))
            level_name = coordinate_name(ds, ("pressure_level", "level"))
            lat_name = coordinate_name(ds, ("latitude", "lat"))
            lon_name = coordinate_name(ds, ("longitude", "lon"))
            months = set(ds[time_name].dt.month.values.astype(int).tolist())
            years = set(ds[time_name].dt.year.values.astype(int).tolist())
            expected_months = {int(value) for value in months_for_year(year)}
            if years != {year} or months != expected_months or ds.sizes[time_name] != len(expected_months):
                return False, f"unexpected time coverage: years={years}, months={months}"
            if set(ds[level_name].values.astype(int).tolist()) != LEVELS:
                return False, "unexpected pressure levels"
            if not VARIABLES.issubset(ds.data_vars):
                return False, f"missing variables: {sorted(VARIABLES - set(ds.data_vars))}"
            if float(ds[lat_name].max()) < AREA[0] or float(ds[lat_name].min()) > AREA[2]:
                return False, "latitude coverage incomplete"
            if float(ds[lon_name].min()) > AREA[1] or float(ds[lon_name].max()) < AREA[3]:
                return False, "longitude coverage incomplete"
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


def download_year(year: int) -> dict[str, object]:
    target = OUT / f"era5_djf_mechanism_monthly_{year}.nc"
    partial = target.with_suffix(".nc.part")
    valid, _ = validate(target, year)
    if not valid:
        if partial.exists():
            partial.unlink()
        print(f"request {year}: {','.join(months_for_year(year))}", flush=True)
        cdsapi.Client().retrieve(
            "reanalysis-era5-pressure-levels-monthly-means",
            {
                "product_type": ["monthly_averaged_reanalysis"],
                "variable": [
                    "geopotential",
                    "temperature",
                    "u_component_of_wind",
                    "v_component_of_wind",
                ],
                "pressure_level": [str(value) for value in sorted(LEVELS)],
                "year": [str(year)],
                "month": months_for_year(year),
                "time": ["00:00"],
                "data_format": "netcdf",
                "download_format": "unarchived",
                "area": AREA,
                "grid": GRID,
            },
            str(partial),
        )
        valid, reason = validate(partial, year)
        if not valid:
            raise RuntimeError(f"Downloaded file failed validation: {reason}")
        partial.replace(target)
        print(f"verified {target.name} ({target.stat().st_size} bytes)", flush=True)
    else:
        print(f"skip verified {target.name}", flush=True)
    return {
        "year": year,
        "months": months_for_year(year),
        "path": str(target.relative_to(ROOT)).replace("\\", "/"),
        "bytes": target.stat().st_size,
        "sha256": sha256(target),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    entries: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(download_year, year): year for year in YEARS}
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
        "dataset": "reanalysis-era5-pressure-levels-monthly-means",
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "period": "1978-12 through 2023-02 (DJF months only)",
        "djf_year_definition": "January year; DJF 1979 is 1978-12 plus 1979-01 and 1979-02",
        "area_nwse": AREA,
        "grid_degrees": GRID,
        "pressure_levels_hpa": sorted(LEVELS),
        "variables": ["geopotential", "temperature", "u_component_of_wind", "v_component_of_wind"],
        "files": entries,
        "failures": failures,
    }
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {MANIFEST.relative_to(ROOT)}: {len(entries)} files, {len(failures)} failures")
    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
