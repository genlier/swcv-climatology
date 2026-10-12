"""Build the exact-member CMIP6 Amon/ua manifest for the 15-model proxy set."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import pandas as pd
import requests


ROOT = Path(__file__).resolve().parents[1]
MANIFESTS = ROOT / "data" / "manifests"
API = "https://esgf-node.llnl.gov/esg-search/search"
# Never stitch incomplete members.  CESM2 hist-aer ua has no single member
# spanning 1950--2014 and is excluded downstream for that metric only.
MEMBER_OVERRIDES = {}


def endpoint(urls, service: str) -> str | None:
    for item in urls or []:
        parts = item.split("|")
        if len(parts) >= 3 and parts[-1] == service:
            return parts[0]
    return None


def get(params: dict) -> list[dict]:
    error = None
    for attempt in range(5):
        try:
            response = requests.get(API, params=params, timeout=120)
            response.raise_for_status()
            return response.json().get("response", {}).get("docs", [])
        except Exception as exc:
            error = exc
            if attempt < 4:
                time.sleep(2 ** attempt)
    raise RuntimeError(str(error))


def select_dataset(row: dict) -> dict:
    original_member = row["member"]
    row = dict(row)
    row["member"] = MEMBER_OVERRIDES.get((row["source"], row["experiment"]), original_member)
    row["member_override_from"] = original_member if row["member"] != original_member else None
    params = {
        "format": "application/solr+json", "type": "Dataset", "project": "CMIP6",
        "source_id": row["source"], "experiment_id": row["experiment"],
        "member_id": row["member"], "table_id": "Amon", "variable_id": "ua",
        "latest": "true", "limit": 200,
    }
    try:
        docs = get(params)
    except Exception as exc:
        return {**row, "available": False, "error": str(exc)}
    candidates = []
    for doc in docs:
        grid = (doc.get("grid_label") or [None])[0]
        candidates.append({
            **row, "available": True, "grid": grid, "version": str(doc.get("version")),
            "instance_id": doc.get("instance_id"), "dataset_data_node": doc.get("data_node"),
            "same_grid_as_zg": grid == row["zg_grid"],
        })
    if not candidates:
        return {**row, "available": False, "error": "no exact-member Amon/ua dataset"}
    frame = pd.DataFrame(candidates)
    frame["node_rank"] = frame.dataset_data_node.fillna("").map(
        lambda x: 0 if "ornl" in x else 1 if "llnl" in x else 2 if "dkrz" in x else 3
    )
    return frame.sort_values(["same_grid_as_zg", "version", "node_rank"], ascending=[False, False, True]).iloc[0].drop("node_rank").to_dict()


def fetch_files(row: dict) -> list[dict]:
    if not row.get("available"):
        return []
    params = {
        "format": "application/solr+json", "type": "File", "project": "CMIP6",
        "source_id": row["source"], "experiment_id": row["experiment"],
        "member_id": row["member"], "table_id": "Amon", "variable_id": "ua",
        "grid_label": row["grid"], "version": row["version"], "limit": 1000,
    }
    docs = get(params)
    records = []
    for doc in docs:
        http = endpoint(doc.get("url"), "HTTPServer")
        opendap = endpoint(doc.get("url"), "OPENDAP")
        ncss = http.replace("/thredds/fileServer/", "/thredds/ncss/grid/") if http and "/thredds/fileServer/" in http else None
        records.append({
            **row, "filename": doc.get("title"), "file_size_bytes": doc.get("size"),
            "checksum": doc.get("checksum"), "checksum_type": doc.get("checksum_type"),
            "http_url": http, "opendap_url": opendap, "ncss_url": ncss,
            "file_data_node": doc.get("data_node"),
        })
    return records


def main() -> None:
    selected = pd.read_csv(MANIFESTS / "cmip6_monthly_selected_datasets.csv")
    design = selected.loc[selected.variable.eq("zg"), ["source", "experiment", "member", "grid"]].drop_duplicates()
    design = design.rename(columns={"grid": "zg_grid"})
    datasets = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(select_dataset, row._asdict()) for row in design.itertuples(index=False)]
        for i, future in enumerate(as_completed(futures), 1):
            datasets.append(future.result())
            if i % 15 == 0:
                print(f"dataset audit {i}/{len(futures)}", flush=True)
    datasets = pd.DataFrame(datasets).sort_values(["source", "experiment"])
    datasets.to_csv(MANIFESTS / "cmip6_monthly_ua_selected_datasets.csv", index=False)
    files, failures = [], []
    available = datasets.loc[datasets.available.eq(True)]
    with ThreadPoolExecutor(max_workers=8) as pool:
        future_map = {pool.submit(fetch_files, row._asdict()): row for row in available.itertuples(index=False)}
        for i, future in enumerate(as_completed(future_map), 1):
            row = future_map[future]
            try:
                files.extend(future.result())
            except Exception as exc:
                failures.append({"source": row.source, "experiment": row.experiment, "member": row.member, "error": str(exc)})
            if i % 15 == 0 or i == len(future_map):
                print(f"file query {i}/{len(future_map)}", flush=True)
    files = pd.DataFrame(files)
    if not files.empty:
        node_rank = files.file_data_node.fillna("").map(
            lambda x: 0 if "ornl" in x else 1 if "llnl" in x else 2 if "dkrz" in x else 3 if "nci" in x else 4
        )
        files["endpoint_rank"] = files.ncss_url.isna().astype(int) * 100 + node_rank * 10 + (~files.ncss_url.fillna("").str.startswith("https://")).astype(int)
        files = files.sort_values("endpoint_rank").drop_duplicates(
            ["source", "experiment", "member", "grid", "version", "filename"], keep="first"
        ).drop(columns="endpoint_rank").sort_values(["source", "experiment", "filename"])
    files.to_csv(MANIFESTS / "cmip6_monthly_ua_esgf_file_manifest.csv", index=False)
    status = {
        "generated_utc": datetime.now(timezone.utc).isoformat(), "api": API,
        "target_groups": len(design), "selected_member_groups_available": int(datasets.available.sum()),
        "missing_groups": int((~datasets.available).sum()), "file_query_failures": len(failures),
        "file_rows": len(files), "files_with_ncss": int(files.ncss_url.notna().sum()) if not files.empty else 0,
        "definition_planned": "U850(22.5-32.5N,110-140E) minus U850(5-15N,90-130E); positive is WNPAC-like anticyclonic shear",
        "member_overrides": {f"{k[0]} {k[1]}": v for k, v in MEMBER_OVERRIDES.items()},
    }
    (MANIFESTS / "cmip6_monthly_ua_manifest_status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    pd.DataFrame(failures).to_csv(MANIFESTS / "cmip6_monthly_ua_manifest_failures.csv", index=False)
    print(json.dumps(status, indent=2), flush=True)
    if (~datasets.available).any():
        print(datasets.loc[~datasets.available, ["source", "experiment", "member", "error"]].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
