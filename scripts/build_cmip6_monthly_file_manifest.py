"""Build a resumable ESGF file/NCSS manifest for Stage-6C monthly indices."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import re
from pathlib import Path
import time

import pandas as pd
import requests


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "processed"
MANIFEST = ROOT / "data" / "manifests"
API = "https://esgf-node.llnl.gov/esg-search/search"
EXPERIMENTS = ["historical", "hist-GHG", "hist-aer", "hist-nat", "piControl"]
NEEDS = {("Amon", "zg"), ("Amon", "tas"), ("Omon", "tos")}


def member_key(member: str) -> tuple:
    match = re.match(r"r(\d+)i(\d+)p(\d+)f(\d+)", member)
    return tuple(map(int, match.groups())) if match else (9999, 9999, 9999, 9999)


def common_members(frame: pd.DataFrame) -> list[str]:
    return sorted([member for member, group in frame.groupby("member")
                   if set(zip(group.table, group.variable)) >= NEEDS], key=member_key)


def endpoint(urls, service: str) -> str | None:
    for item in urls or []:
        parts = item.split("|")
        if len(parts) >= 3 and parts[-1] == service:
            return parts[0]
    return None


def fetch_files(row: dict) -> list[dict]:
    params = {
        "format": "application/solr+json", "type": "File", "project": "CMIP6",
        "source_id": row["source"], "experiment_id": row["experiment"],
        "member_id": row["member"], "table_id": row["table"], "variable_id": row["variable"],
        "grid_label": row["grid"], "version": str(row["version"]), "limit": 1000,
    }
    error = None
    for attempt in range(3):
        try:
            response = requests.get(API, params=params, timeout=120)
            response.raise_for_status()
            docs = response.json().get("response", {}).get("docs", [])
            records = []
            for doc in docs:
                http = endpoint(doc.get("url"), "HTTPServer")
                opendap = endpoint(doc.get("url"), "OPENDAP")
                ncss = None
                if http and "/thredds/fileServer/" in http:
                    ncss = http.replace("/thredds/fileServer/", "/thredds/ncss/grid/")
                records.append({**row, "filename": doc.get("title"), "file_size_bytes": doc.get("size"),
                                "checksum": doc.get("checksum"), "checksum_type": doc.get("checksum_type"),
                                "http_url": http, "opendap_url": opendap, "ncss_url": ncss,
                                "file_data_node": doc.get("data_node")})
            return records
        except Exception as exc:
            error = str(exc)
            if attempt < 2:
                time.sleep(2 ** attempt)
    return [{**row, "filename": None, "query_error": error}]


def main() -> None:
    inventory = pd.read_csv(OUT / "cmip6_damip_esgf_dataset_inventory.csv")
    eligible = pd.read_csv(OUT / "cmip6_damip_monthly_proxy_intersection.csv")
    models = eligible.loc[eligible.has_all_transient_plus_picontrol, "source"].tolist()
    selected = []
    for model in models:
        for experiment in EXPERIMENTS:
            q = inventory.loc[(inventory.source == model) & (inventory.experiment == experiment)
                              & inventory.apply(lambda x: (x.table, x.variable) in NEEDS, axis=1)]
            members = common_members(q)
            if not members:
                continue
            member = members[0]
            for table, variable in sorted(NEEDS):
                v = q.loc[(q.member == member) & (q.table == table) & (q.variable == variable)].copy()
                if v.empty:
                    continue
                v["node_rank"] = (~v.data_node.astype(str).str.contains("ornl|llnl|dias|nci|dkrz", case=False)).astype(int)
                item = v.sort_values(["node_rank", "version"], ascending=[True, False]).iloc[0]
                selected.append({"source": model, "experiment": experiment, "member": member,
                                 "table": table, "variable": variable, "grid": item.grid,
                                 "version": str(item.version), "instance_id": item.instance_id,
                                 "dataset_data_node": item.data_node})
    datasets = pd.DataFrame(selected)
    datasets.to_csv(MANIFEST / "cmip6_monthly_selected_datasets.csv", index=False)
    records = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(fetch_files, row._asdict()) for row in datasets.itertuples(index=False)]
        for i, future in enumerate(as_completed(futures), 1):
            records.extend(future.result())
            if i % 20 == 0 or i == len(futures):
                print(f"queried {i}/{len(futures)} datasets", flush=True)
    files = pd.DataFrame(records)
    if "filename" in files:
        files = files.sort_values(["source", "experiment", "variable", "filename", "ncss_url"], na_position="last")
        # Search may return replicas. Prefer an HTTPS THREDDS/NCSS endpoint per logical file.
        node_rank = files.file_data_node.fillna("").map(
            lambda x: 0 if "ornl" in x else 1 if "llnl" in x else 2 if "dkrz" in x else 3 if "dias" in x else 4
        )
        files["endpoint_rank"] = (files.ncss_url.isna().astype(int) * 100
                                  + node_rank * 10
                                  + (~files.ncss_url.fillna("").str.startswith("https://")).astype(int))
        files = files.sort_values("endpoint_rank").drop_duplicates(
            ["source", "experiment", "member", "table", "variable", "filename"], keep="first")
        files = files.drop(columns="endpoint_rank")
    files.to_csv(MANIFEST / "cmip6_monthly_esgf_file_manifest.csv", index=False)
    summary = {
        "generated_utc": datetime.now(timezone.utc).isoformat(), "api": API,
        "n_selected_datasets": len(datasets), "n_file_rows": len(files),
        "n_files_with_ncss": int(files.ncss_url.notna().sum()) if "ncss_url" in files else 0,
        "n_dataset_queries_without_files": int(files.filename.isna().sum()) if "filename" in files else len(files),
        "member_selection": "lowest realization common to zg, tas, and tos within each model/experiment",
    }
    (MANIFEST / "cmip6_monthly_esgf_file_manifest_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
