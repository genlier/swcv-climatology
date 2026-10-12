"""Download monthly ERSSTv5 and compute reproducible Indian Ocean basin indices."""

from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import xarray as xr


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "raw" / "ersstv5_monthly"
INDEX_OUT = ROOT / "data" / "indices"
BASE = "https://www.ncei.noaa.gov/pub/data/cmb/ersst/v5/netcdf"
MONTHS = pd.date_range("1978-12-01", "2023-12-01", freq="MS")


def valid(path: Path) -> bool:
    if not path.exists() or path.stat().st_size < 50_000:
        return False
    try:
        with xr.open_dataset(path) as ds:
            return "sst" in ds and ds.sizes.get("time") == 1 and ds.sizes.get("lat", 0) > 80
    except Exception:
        return False


def fetch(date: pd.Timestamp) -> tuple[pd.Timestamp, Path]:
    name = f"ersst.v5.{date:%Y%m}.nc"
    target = OUT / name
    if valid(target):
        return date, target
    partial = target.with_suffix(".nc.part")
    url = f"{BASE}/{name}"
    for attempt in range(1, 6):
        try:
            response = requests.get(
                url,
                timeout=90,
                headers={"User-Agent": "swcv-climatology/1.0 (academic research)"},
            )
            response.raise_for_status()
            partial.write_bytes(response.content)
            if not valid(partial):
                raise RuntimeError(f"NetCDF validation failed: {name}")
            partial.replace(target)
            return date, target
        except Exception:
            if partial.exists():
                partial.unlink()
            if attempt == 5:
                raise
    raise AssertionError("unreachable")


def regional_mean(ds: xr.Dataset, lon_max: float) -> float:
    sst = ds.sst.squeeze(drop=True).sel(lat=slice(-20, 20), lon=slice(40, lon_max))
    weights = np.cos(np.deg2rad(sst.lat))
    return float(sst.weighted(weights).mean(("lat", "lon"), skipna=True))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    INDEX_OUT.mkdir(parents=True, exist_ok=True)
    completed: dict[pd.Timestamp, Path] = {}
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(fetch, date): date for date in MONTHS}
        for number, future in enumerate(as_completed(futures), start=1):
            date, path = future.result()
            completed[date] = path
            if number % 50 == 0 or number == len(MONTHS):
                print(f"verified {number}/{len(MONTHS)} monthly ERSSTv5 files", flush=True)

    rows = []
    for date in MONTHS:
        path = completed[date]
        with xr.open_dataset(path) as ds:
            rows.append(
                {
                    "date": date,
                    "io_sst_40_100e_c": regional_mean(ds, 100),
                    "io_sst_40_110e_c": regional_mean(ds, 110),
                }
            )
    frame = pd.DataFrame(rows)
    for source, target in (
        ("io_sst_40_100e_c", "iobm_40_100e_c"),
        ("io_sst_40_110e_c", "iobm_40_110e_c"),
    ):
        baseline = frame.loc[frame.date.dt.year.between(1991, 2020)].groupby(frame.date.dt.month)[source].mean()
        frame[target] = frame[source] - frame.date.dt.month.map(baseline)
    frame.to_csv(INDEX_OUT / "iobm_ersstv5_monthly.csv", index=False)

    hashes = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(completed.values())
    }
    manifest = {
        "source": BASE,
        "dataset": "NOAA ERSSTv5 monthly NetCDF",
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "period": [MONTHS.min().date().isoformat(), MONTHS.max().date().isoformat()],
        "monthly_files": len(completed),
        "baseline": "1991-2020 monthly climatology",
        "regions": {
            "iobm_40_100e_c": "20S-20N, 40E-100E",
            "iobm_40_110e_c": "20S-20N, 40E-110E sensitivity",
        },
        "sha256": hashes,
    }
    (INDEX_OUT / "iobm_ersstv5_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(frame.tail().to_string(index=False))


if __name__ == "__main__":
    main()
