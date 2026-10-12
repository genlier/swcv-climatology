"""Download 6-hourly vertically integrated ERA5 moisture flux for JJA events.

May--September are retained so every June--August genesis has a complete
-48 h to +72 h window. The 0.5-degree grid was explicitly tested through CDS
before use. Annual files are resumable, validated, and checksum-manifested.
"""

from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import cdsapi
import xarray as xr


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "raw" / "era5_jja_moisture_flux_6hourly"
MANIFEST = ROOT / "data" / "indices" / "era5_jja_moisture_flux_manifest.json"
YEARS = range(1979, 2024)
YEAR_CHUNKS = [
    tuple(range(1979, 1983)),
    tuple(range(1983, 1988)),
    tuple(range(1988, 1993)),
    tuple(range(1993, 1998)),
    tuple(range(1998, 2003)),
    tuple(range(2003, 2008)),
    tuple(range(2008, 2013)),
    tuple(range(2013, 2018)),
    tuple(range(2018, 2023)),
    (2023,),
]
EXPECTED_VARS = {"viwve", "viwvn", "vimdf"}
EXPECTED_TIMES = 153 * 4  # May--September, four synoptic times per day.


def valid_file(path: Path, year: int) -> tuple[bool, str]:
    if not path.exists() or path.stat().st_size == 0:
        return False, "missing or empty"
    try:
        with xr.open_dataset(path) as ds:
            time_name = "valid_time" if "valid_time" in ds.dims else "time"
            if ds.sizes.get(time_name) != EXPECTED_TIMES:
                return False, f"unexpected time count {ds.sizes.get(time_name)}"
            if set(ds[time_name].dt.year.values.astype(int)) != {year}:
                return False, "unexpected year"
            if set(ds[time_name].dt.month.values.astype(int)) != {5, 6, 7, 8, 9}:
                return False, "unexpected months"
            if ds.sizes.get("latitude") != 41 or ds.sizes.get("longitude") != 61:
                return False, "unexpected 0.5-degree grid"
            if not EXPECTED_VARS.issubset(ds.data_vars):
                return False, f"missing variables {EXPECTED_VARS - set(ds.data_vars)}"
            for name in EXPECTED_VARS:
                if bool(ds[name].isnull().all()):
                    return False, f"{name} is entirely missing"
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
    target = OUT / f"era5_jja_moisture_flux_{year}.nc"
    partial = target.with_suffix(target.suffix + ".part")
    valid, _ = valid_file(target, year)
    if valid:
        print(f"skip {target.name}", flush=True)
    else:
        if target.exists():
            target.unlink()
        if partial.exists():
            partial.unlink()
        print(f"request {year}", flush=True)
        cdsapi.Client().retrieve(
            "reanalysis-era5-single-levels",
            {
                "product_type": ["reanalysis"],
                "variable": [
                    "vertical_integral_of_eastward_water_vapour_flux",
                    "vertical_integral_of_northward_water_vapour_flux",
                    "vertical_integral_of_divergence_of_moisture_flux",
                ],
                "year": [str(year)],
                "month": ["05", "06", "07", "08", "09"],
                "day": [f"{day:02d}" for day in range(1, 32)],
                "time": ["00:00", "06:00", "12:00", "18:00"],
                "data_format": "netcdf",
                "download_format": "unarchived",
                "area": [38, 92, 18, 122],
                "grid": ["0.5", "0.5"],
            },
            str(partial),
        )
        valid, reason = valid_file(partial, year)
        if not valid:
            raise RuntimeError(f"Downloaded {year} file failed validation: {reason}")
        partial.replace(target)
        print(f"verified {target.name} ({target.stat().st_size} bytes)", flush=True)
    return {
        "year": year,
        "path": str(target.relative_to(ROOT)).replace("\\", "/"),
        "bytes": target.stat().st_size,
        "sha256": sha256(target),
    }


def valid_chunk(path: Path, years: tuple[int, ...]) -> tuple[bool, str]:
    if not path.exists() or path.stat().st_size == 0:
        return False, "missing or empty"
    try:
        with xr.open_dataset(path) as ds:
            time_name = "valid_time" if "valid_time" in ds.dims else "time"
            if ds.sizes.get(time_name) != EXPECTED_TIMES * len(years):
                return False, "unexpected time count"
            if set(ds[time_name].dt.year.values.astype(int)) != set(years):
                return False, "unexpected years"
            if ds.sizes.get("latitude") != 41 or ds.sizes.get("longitude") != 61:
                return False, "unexpected grid"
            if not EXPECTED_VARS.issubset(ds.data_vars):
                return False, "missing variables"
        return True, "ok"
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def split_chunk(chunk: Path, years: tuple[int, ...]) -> list[dict[str, object]]:
    entries = []
    with xr.open_dataset(chunk) as ds:
        time_name = "valid_time" if "valid_time" in ds.dims else "time"
        for year in years:
            target = OUT / f"era5_jja_moisture_flux_{year}.nc"
            valid, _ = valid_file(target, year)
            if not valid:
                partial = target.with_suffix(target.suffix + ".part")
                if partial.exists():
                    partial.unlink()
                subset = ds.sel({time_name: ds[time_name].dt.year == year}).load()
                encoding = {
                    name: {"zlib": True, "complevel": 4, "shuffle": True}
                    for name in EXPECTED_VARS
                }
                subset.to_netcdf(partial, engine="netcdf4", encoding=encoding)
                valid, reason = valid_file(partial, year)
                if not valid:
                    raise RuntimeError(f"Split {year} file failed validation: {reason}")
                partial.replace(target)
                print(f"split and verified {target.name} ({target.stat().st_size} bytes)", flush=True)
            entries.append(
                {
                    "year": year,
                    "path": str(target.relative_to(ROOT)).replace("\\", "/"),
                    "bytes": target.stat().st_size,
                    "sha256": sha256(target),
                }
            )
    return entries


def download_chunk(years: tuple[int, ...]) -> list[dict[str, object]]:
    if all(valid_file(OUT / f"era5_jja_moisture_flux_{year}.nc", year)[0] for year in years):
        print(f"skip completed chunk {years[0]}-{years[-1]}", flush=True)
        return [download_year(year) for year in years]
    chunk = OUT / f"chunk_{years[0]}_{years[-1]}.nc"
    partial = chunk.with_suffix(chunk.suffix + ".part")
    valid, _ = valid_chunk(chunk, years)
    if not valid:
        if chunk.exists():
            chunk.unlink()
        if partial.exists():
            partial.unlink()
        print(f"request chunk {years[0]}-{years[-1]}", flush=True)
        cdsapi.Client().retrieve(
            "reanalysis-era5-single-levels",
            {
                "product_type": ["reanalysis"],
                "variable": [
                    "vertical_integral_of_eastward_water_vapour_flux",
                    "vertical_integral_of_northward_water_vapour_flux",
                    "vertical_integral_of_divergence_of_moisture_flux",
                ],
                "year": [str(year) for year in years],
                "month": ["05", "06", "07", "08", "09"],
                "day": [f"{day:02d}" for day in range(1, 32)],
                "time": ["00:00", "06:00", "12:00", "18:00"],
                "data_format": "netcdf",
                "download_format": "unarchived",
                "area": [38, 92, 18, 122],
                "grid": ["0.5", "0.5"],
            },
            str(partial),
        )
        valid, reason = valid_chunk(partial, years)
        if not valid:
            raise RuntimeError(f"Downloaded chunk {years[0]}-{years[-1]} failed validation: {reason}")
        partial.replace(chunk)
        print(f"verified chunk {chunk.name} ({chunk.stat().st_size} bytes)", flush=True)
    return split_chunk(chunk, years)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    entries: list[dict[str, object]] = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(download_chunk, years): years for years in YEAR_CHUNKS}
        for future in as_completed(futures):
            years = futures[future]
            try:
                entries.extend(future.result())
            except Exception as exc:
                raise RuntimeError(f"Moisture-flux download failed for {years[0]}-{years[-1]}") from exc
    entries.sort(key=lambda item: int(item["year"]))
    manifest = {
        "dataset": "reanalysis-era5-single-levels",
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "period_each_year": "May 1 through September 30, 1979--2023",
        "times_utc": ["00:00", "06:00", "12:00", "18:00"],
        "area_nwse": [38, 92, 18, 122],
        "grid_degrees": [0.5, 0.5],
        "variables": {
            "viwve": "total-column vertically integrated eastward water-vapour flux; kg m-1 s-1",
            "viwvn": "total-column vertically integrated northward water-vapour flux; kg m-1 s-1",
            "vimdf": "total-column vertically integrated moisture-divergence flux; kg m-2 s-1",
        },
        "files": entries,
    }
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {MANIFEST.relative_to(ROOT)} with {len(entries)} verified files")


if __name__ == "__main__":
    main()
