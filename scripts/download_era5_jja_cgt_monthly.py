"""Download coarse-grid global ERA5 JJA 200-hPa fields for CGT diagnostics."""

from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import cdsapi
import xarray as xr


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "raw" / "era5_jja_cgt_monthly"
MANIFEST = ROOT / "data" / "indices" / "era5_jja_cgt_manifest.json"
CHUNKS = [tuple(range(1979, 1994)), tuple(range(1994, 2009)), tuple(range(2009, 2024))]
VARIABLES = {"z", "u", "v"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def valid_chunk(path: Path, years: tuple[int, ...]) -> tuple[bool, str]:
    if not path.exists() or path.stat().st_size == 0:
        return False, "missing or empty"
    try:
        with xr.open_dataset(path) as ds:
            time_name = "valid_time" if "valid_time" in ds.coords else "time"
            level_name = "pressure_level" if "pressure_level" in ds.coords else "level"
            found_years = set(ds[time_name].dt.year.values.astype(int).tolist())
            months = set(ds[time_name].dt.month.values.astype(int).tolist())
            if found_years != set(years) or months != {6, 7, 8}:
                return False, f"unexpected time coverage: years={sorted(found_years)}, months={sorted(months)}"
            if ds.sizes[time_name] != 3 * len(years):
                return False, f"unexpected time size {ds.sizes[time_name]}"
            if set(ds[level_name].values.astype(int).tolist()) != {200}:
                return False, "unexpected pressure level"
            if not VARIABLES.issubset(ds.data_vars):
                return False, f"missing variables {sorted(VARIABLES - set(ds.data_vars))}"
            if ds.sizes.get("latitude") != 73 or ds.sizes.get("longitude") != 144:
                return False, f"unexpected grid {ds.sizes.get('latitude')}x{ds.sizes.get('longitude')}"
            if any(bool(ds[name].isnull().all()) for name in VARIABLES):
                return False, "entirely missing variable"
        return True, "ok"
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def download_chunk(years: tuple[int, ...]) -> dict[str, object]:
    first, last = years[0], years[-1]
    target = OUT / f"era5_jja_cgt_monthly_{first}_{last}.nc"
    partial = target.with_suffix(".nc.part")
    valid, _ = valid_chunk(target, years)
    if not valid:
        if target.exists():
            target.unlink()
        if partial.exists():
            partial.unlink()
        print(f"request {first}-{last}", flush=True)
        cdsapi.Client().retrieve(
            "reanalysis-era5-pressure-levels-monthly-means",
            {
                "product_type": ["monthly_averaged_reanalysis"],
                "variable": ["geopotential", "u_component_of_wind", "v_component_of_wind"],
                "pressure_level": ["200"],
                "year": [str(year) for year in years],
                "month": ["06", "07", "08"],
                "time": ["00:00"],
                "grid": ["2.5", "2.5"],
                "data_format": "netcdf",
                "download_format": "unarchived",
            },
            str(partial),
        )
        valid, reason = valid_chunk(partial, years)
        if not valid:
            raise RuntimeError(f"Downloaded {first}-{last} failed validation: {reason}")
        partial.replace(target)
        print(f"verified {target.name} ({target.stat().st_size} bytes)", flush=True)
    else:
        print(f"skip {target.name}", flush=True)
    return {
        "years": [first, last],
        "path": str(target.relative_to(ROOT)).replace("\\", "/"),
        "bytes": target.stat().st_size,
        "sha256": sha256(target),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    entries = []
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = {pool.submit(download_chunk, years): years for years in CHUNKS}
        for future in as_completed(futures):
            years = futures[future]
            try:
                entries.append(future.result())
            except Exception as exc:
                raise RuntimeError(f"CGT download failed for {years[0]}-{years[-1]}") from exc
    entries.sort(key=lambda item: item["years"][0])
    manifest = {
        "dataset": "reanalysis-era5-pressure-levels-monthly-means",
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "period": "1979-06 through 2023-08 (JJA only)",
        "grid_degrees": [2.5, 2.5],
        "pressure_level_hpa": 200,
        "variables": ["geopotential", "u_component_of_wind", "v_component_of_wind"],
        "files": entries,
    }
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"wrote {MANIFEST.relative_to(ROOT)} with {len(entries)} verified chunks")


if __name__ == "__main__":
    main()
