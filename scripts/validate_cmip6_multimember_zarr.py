"""Validate time, level and identity metadata for paired CMIP6 zg Zarr stores."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import re

import gcsfs
import numpy as np
import pandas as pd
import xarray as xr


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data" / "manifests" / "cmip6_multimember_zg_datasets.csv"
OUT = ROOT / "data" / "processed" / "cmip6_multimember_zarr_validation.csv"
REQUIRED_MONTHS = {(year, month) for year in range(1951, 2015) for month in range(1, 13)}


def member_sort_key(member: str) -> tuple[int, int, int, int, str]:
    match = re.fullmatch(r"r(\d+)i(\d+)p(\d+)f(\d+)", str(member))
    return (*map(int, match.groups()), str(member)) if match else (10**9, 0, 0, 0, str(member))


def validate(row: dict[str, object]) -> dict[str, object]:
    result = dict(row)
    result.update({"valid": False, "error": ""})
    url = str(row["access_url"])
    if not url.startswith("gs://"):
        result["error"] = "no cloud Zarr access URL"
        return result
    try:
        fs = gcsfs.GCSFileSystem(token="anon")
        store = url.removeprefix("gs://")
        if not fs.exists(store):
            raise RuntimeError("cloud store does not exist")
        with xr.open_zarr(fs.get_mapper(store), consolidated=True) as ds:
            time_values = ds.time.values
            years = ds.time.dt.year.values.astype(int)
            month_values = ds.time.dt.month.values.astype(int)
            months = set(zip(years, month_values))
            plev = np.asarray(ds.plev.values, dtype=float)
            checks = {
                "source_id": row["model"],
                "experiment_id": row["experiment"],
                "variant_label": row["member"],
                "grid_label": row["grid"],
            }
            mismatches = {
                key: {"expected": value, "observed": ds.attrs.get(key)}
                for key, value in checks.items()
                if ds.attrs.get(key) != value
            }
            result.update(
                {
                    "time_start": str(time_values[0]),
                    "time_end": str(time_values[-1]),
                    "required_months_present": len(REQUIRED_MONTHS & months),
                    "required_months_expected": len(REQUIRED_MONTHS),
                    "has_850_hpa": bool(np.isclose(plev, 85000.0).any()),
                    "lat_size": int(ds.sizes.get("lat", 0)),
                    "lon_size": int(ds.sizes.get("lon", 0)),
                    "identity_mismatches": str(mismatches) if mismatches else "",
                }
            )
            result["valid"] = bool(
                not mismatches
                and result["required_months_present"] == result["required_months_expected"]
                and result["has_850_hpa"]
                and result["lat_size"] > 0
                and result["lon_size"] > 0
            )
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+")
    parser.add_argument("--members-per-model", type=int, default=3)
    parser.add_argument("--max-workers", type=int, default=6)
    args = parser.parse_args()

    table = pd.read_csv(INPUT)
    if args.models:
        table = table.loc[table.model.isin(args.models)]
    else:
        # The default task is explicitly a multi-member audit. Single-member
        # catalogue rows remain in the manifest for provenance but cannot answer
        # the ensemble-size question and may not have cloud-Zarr access.
        member_experiments = table.groupby(["model", "member"]).experiment.nunique()
        complete_members = member_experiments[member_experiments.ge(2)].reset_index()
        eligible = complete_members.groupby("model").member.nunique()
        eligible_models = eligible[eligible.ge(args.members_per_model)].index
        table = table.loc[table.model.isin(eligible_models)]
    selected = []
    for model, group in table.groupby("model", sort=True):
        members = sorted(group.member.unique(), key=member_sort_key)[: args.members_per_model]
        selected.append(group.loc[group.member.isin(members)])
    work = pd.concat(selected, ignore_index=True) if selected else table.iloc[0:0]
    if work.empty:
        raise RuntimeError("No models satisfy the requested paired-member threshold")

    results = []
    with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
        futures = {pool.submit(validate, row): row for row in work.to_dict("records")}
        for future in as_completed(futures):
            row = future.result()
            results.append(row)
            print(
                f"{row['model']} {row['member']} {row['experiment']}: "
                f"{'valid' if row['valid'] else row['error'] or 'invalid metadata'}",
                flush=True,
            )
            pd.DataFrame(results).sort_values(["model", "member", "experiment"]).to_csv(OUT, index=False)

    result = pd.DataFrame(results)
    paired = result.groupby(["model", "member"]).valid.all().groupby("model").sum().astype(int)
    print("Valid paired members by model:")
    print(paired.to_string())
    if not bool(result.valid.all()):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
