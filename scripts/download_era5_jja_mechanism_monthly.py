"""Download minimal ERA5 monthly JJA fields for SWCV mechanism diagnostics.

The compact domain covers the published MONC and WNPAC boxes plus the inflow
sector immediately south/east of Southwest China. Files are split by year,
validated before promotion from ``.part``, and accompanied by a SHA-256
manifest so the acquisition is resumable and auditable.
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
OUT = ROOT / "data" / "raw" / "era5_jja_mechanism_monthly"
MANIFEST = ROOT / "data" / "indices" / "era5_jja_mechanism_manifest.json"
YEARS = range(1979, 2024)
LEVELS = {200, 850}
VARIABLES = {"z", "v", "q"}


def coordinate_name(ds: xr.Dataset, candidates: tuple[str, ...]) -> str:
    for name in candidates:
        if name in ds.coords or name in ds.dims:
            return name
    raise KeyError(f"None of {candidates} found in {list(ds.coords)}")


def valid_year_file(path: Path, year: int) -> tuple[bool, str]:
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
            levels = set(ds[level_name].values.astype(int).tolist())
            if ds.sizes[time_name] != 3 or months != {6, 7, 8} or years != {year}:
                return False, "unexpected time coverage"
            if levels != LEVELS:
                return False, f"unexpected pressure levels: {sorted(levels)}"
            if not VARIABLES.issubset(ds.data_vars):
                return False, f"missing variables: {sorted(VARIABLES - set(ds.data_vars))}"
            if ds[lat_name].min() > 15 or ds[lat_name].max() < 50:
                return False, "latitude coverage incomplete"
            if ds[lon_name].min() > 85 or ds[lon_name].max() < 150:
                return False, "longitude coverage incomplete"
            for name in VARIABLES:
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
    """Download and validate one year; safe to run in a worker thread."""
    target = OUT / f"era5_jja_mechanism_monthly_{year}.nc"
    partial = target.with_suffix(target.suffix + ".part")
    valid, _ = valid_year_file(target, year)
    if valid:
        print(f"skip {target.name}", flush=True)
    else:
        if target.exists():
            target.unlink()
        if partial.exists():
            partial.unlink()
        print(f"request {year}", flush=True)
        client = cdsapi.Client()
        client.retrieve(
            "reanalysis-era5-pressure-levels-monthly-means",
            {
                "product_type": ["monthly_averaged_reanalysis"],
                "variable": [
                    "geopotential",
                    "specific_humidity",
                    "v_component_of_wind",
                ],
                "pressure_level": ["200", "850"],
                "year": [str(year)],
                "month": ["06", "07", "08"],
                "time": ["00:00"],
                "data_format": "netcdf",
                "download_format": "unarchived",
                # North, west, south, east.
                "area": [50, 85, 15, 150],
            },
            str(partial),
        )
        valid, reason = valid_year_file(partial, year)
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


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    entries: list[dict[str, object]] = []
    # Four concurrent requests are accepted reliably by CDS for this account;
    # larger bursts may be rejected by the service queue.
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(download_year, year): year for year in YEARS}
        for future in as_completed(futures):
            year = futures[future]
            try:
                entries.append(future.result())
            except Exception as exc:
                raise RuntimeError(f"ERA5 mechanism download failed for {year}") from exc
    entries.sort(key=lambda item: int(item["year"]))

    manifest = {
        "dataset": "reanalysis-era5-pressure-levels-monthly-means",
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "period": "1979-06 through 2023-08 (JJA only)",
        "area_nwse": [50, 85, 15, 150],
        "pressure_levels_hpa": sorted(LEVELS),
        "variables": [
            "geopotential",
            "specific_humidity",
            "v_component_of_wind",
        ],
        "files": entries,
    }
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {MANIFEST.relative_to(ROOT)} with {len(entries)} verified files")


if __name__ == "__main__":
    main()
