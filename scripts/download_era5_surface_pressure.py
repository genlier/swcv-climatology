"""Download 6-hourly ERA5 surface pressure for terrain validity checks."""

from __future__ import annotations

from pathlib import Path

import cdsapi
import xarray as xr


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "raw" / "era5_surface_pressure_6hourly"


def valid_year_file(path: Path, year: int) -> bool:
    if not path.exists():
        return False
    expected_times = 1464 if year % 4 == 0 else 1460
    try:
        with xr.open_dataset(path) as ds:
            time_name = "valid_time" if "valid_time" in ds.dims else "time"
            return (
                ds.sizes.get(time_name) == expected_times
                and ds.sizes.get("latitude") == 81
                and ds.sizes.get("longitude") == 121
                and "sp" in ds.data_vars
                and not bool(ds["sp"].isnull().any())
            )
    except Exception:
        return False


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    client = cdsapi.Client()
    for year in range(1979, 2024):
        target = OUT / f"era5_surface_pressure_{year}.nc"
        partial = OUT / f"era5_surface_pressure_{year}.nc.part"
        if valid_year_file(target, year):
            print(f"skip {target.name}", flush=True)
            continue
        if partial.exists():
            partial.unlink()
        client.retrieve(
            "reanalysis-era5-single-levels",
            {
                "product_type": ["reanalysis"],
                "variable": ["surface_pressure"],
                "year": [str(year)],
                "month": [f"{month:02d}" for month in range(1, 13)],
                "day": [f"{day:02d}" for day in range(1, 32)],
                "time": ["00:00", "06:00", "12:00", "18:00"],
                "data_format": "netcdf",
                "download_format": "unarchived",
                "area": [40, 95, 20, 125],
            },
            str(partial),
        )
        if not valid_year_file(partial, year):
            raise RuntimeError(f"Downloaded file failed validation: {partial}")
        partial.replace(target)
        print(f"verified {target.name} ({target.stat().st_size} bytes)", flush=True)


if __name__ == "__main__":
    main()
