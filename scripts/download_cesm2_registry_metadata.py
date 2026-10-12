"""Materialize a tiny CESM2 NetCDF subset for the CMIP6 metadata audit.

The scientific fields in this file are not used by the analysis.  Its purpose
is to preserve the authoritative global/grid attributes from the frozen Pangeo
CMIP6 Zarr catalogue so CESM2 can pass the same local-NetCDF metadata checks as
the other 14 models in the ensemble.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import gcsfs
import xarray as xr


ROOT = Path(__file__).resolve().parents[1]
STORE = (
    "gs://cmip6/CMIP6/CMIP/NCAR/CESM2/historical/r1i1p1f1/"
    "Amon/zg/gn/v20190308/"
)
OUT = (
    ROOT
    / "data/raw/cmip6_monthly_subsets/CESM2/historical/r1i1p1f1/zg"
    / "CESM2_historical_r1i1p1f1_Amon_zg_gn_registry.nc"
)
MANIFEST = ROOT / "data/manifests/cesm2_registry_metadata.json"
EXPECTED = {
    "source_id": "CESM2",
    "institution_id": "NCAR",
    "variant_label": "r1i1p1f1",
    "grid_label": "gn",
    "nominal_resolution": "100 km",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate(path: Path) -> dict[str, object]:
    with xr.open_dataset(path, decode_times=False) as ds:
        mismatches = {
            key: {"expected": expected, "observed": ds.attrs.get(key)}
            for key, expected in EXPECTED.items()
            if ds.attrs.get(key) != expected
        }
        if mismatches:
            raise RuntimeError(f"CESM2 metadata mismatch: {mismatches}")
        if not ds.attrs.get("grid") or not ds.attrs.get("tracking_id"):
            raise RuntimeError("CESM2 subset lacks grid or tracking_id")
        if "zg" not in ds or ds.sizes.get("time") != 1:
            raise RuntimeError("CESM2 registry subset has unexpected contents")
        return {
            "dimensions": dict(ds.sizes),
            "global_attributes": {
                key: ds.attrs.get(key)
                for key in (*EXPECTED, "grid", "tracking_id", "version")
            },
            "zg_attributes": dict(ds["zg"].attrs),
        }


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    if not OUT.exists():
        fs = gcsfs.GCSFileSystem(token="anon")
        with xr.open_zarr(fs.get_mapper(STORE), consolidated=True) as ds:
            subset = ds[["zg"]].isel(time=slice(0, 1), lat=slice(0, 2), lon=slice(0, 2))
            subset = subset.load()
        temporary = OUT.with_suffix(".nc.part")
        subset.to_netcdf(temporary, engine="netcdf4")
        temporary.replace(OUT)

    audit = validate(OUT)
    record = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "local metadata verification only; not a scientific analysis input",
        "source_store": STORE,
        "output_file": OUT.relative_to(ROOT).as_posix(),
        "size_bytes": OUT.stat().st_size,
        "sha256": sha256(OUT),
        "validation": audit,
    }
    MANIFEST.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Validated CESM2 registry subset: {OUT}")
    print(f"Wrote manifest: {MANIFEST}")


if __name__ == "__main__":
    main()
