"""Audit and restore the CESM2 hist-aer ua selection after coverage checks.

r1i1p1f1 is missing 1950--1999 and r3i1p1f1 is missing 2000--2014, so no
member may be stitched.  Keep the design member r1 and document exclusion
of this model from the hist-aer wind response only.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import requests


ROOT = Path(__file__).resolve().parents[1]
MANIFESTS = ROOT / "data" / "manifests"
API = "https://esgf-node.llnl.gov/esg-search/search"


def search(kind: str, member: str = "r1i1p1f1") -> list[dict]:
    query = {
        "format": "application/solr+json", "type": kind, "project": "CMIP6",
        "source_id": "CESM2", "experiment_id": "hist-aer",
        "member_id": member, "table_id": "Amon", "variable_id": "ua",
        "latest": "true", "limit": 1000,
    }
    response = requests.get(API, params=query, timeout=120)
    response.raise_for_status()
    return response.json()["response"]["docs"]


def endpoint(urls, service: str) -> str | None:
    for item in urls or []:
        parts = item.split("|")
        if len(parts) >= 3 and parts[-1] == service:
            return parts[0]
    return None


def main() -> None:
    datasets = search("Dataset", "r1i1p1f1")
    if not datasets:
        raise RuntimeError("CESM2 hist-aer r1 Amon/ua dataset not found")
    doc = sorted(datasets, key=lambda d: str(d.get("version")))[-1]
    grid = (doc.get("grid_label") or [None])[0]
    selected_path = MANIFESTS / "cmip6_monthly_ua_selected_datasets.csv"
    selected = pd.read_csv(selected_path)
    mask = selected.source.eq("CESM2") & selected.experiment.eq("hist-aer")
    selected.loc[mask, ["member", "available", "grid", "version", "instance_id",
                        "dataset_data_node", "same_grid_as_zg"]] = [
        "r1i1p1f1", True, grid, int(doc.get("version")), doc.get("instance_id"),
        doc.get("data_node"), grid == selected.loc[mask, "zg_grid"].iloc[0]
    ]
    selected["member_override_from"] = selected.get("member_override_from", pd.Series(index=selected.index, dtype=object))
    selected.loc[mask, "member_override_from"] = None
    selected.to_csv(selected_path, index=False)

    files_path = MANIFESTS / "cmip6_monthly_ua_esgf_file_manifest.csv"
    files = pd.read_csv(files_path)
    files = files.loc[~(files.source.eq("CESM2") & files.experiment.eq("hist-aer"))]
    template = selected.loc[mask].iloc[0].to_dict()
    records = []
    for item in search("File", "r1i1p1f1"):
        http = endpoint(item.get("url"), "HTTPServer")
        opendap = endpoint(item.get("url"), "OPENDAP")
        ncss = http.replace("/thredds/fileServer/", "/thredds/ncss/grid/") if http else None
        records.append({**template, "filename": item.get("title"),
                        "file_size_bytes": item.get("size"), "checksum": item.get("checksum"),
                        "checksum_type": item.get("checksum_type"), "http_url": http,
                        "opendap_url": opendap, "ncss_url": ncss,
                        "file_data_node": item.get("data_node")})
    replacement = pd.DataFrame(records).drop_duplicates("filename").sort_values("filename")
    pd.concat([files, replacement], ignore_index=True).sort_values(
        ["source", "experiment", "filename"]
    ).to_csv(files_path, index=False)
    exception = {
        "source": "CESM2", "experiment": "hist-aer", "variable": "ua",
        "preferred_member": "r1i1p1f1", "selected_member": "r1i1p1f1",
        "action": "exclude CESM2 from hist-aer wind response only",
        "reason": "r1 lacks 1950-1999 and r3 lacks 2000-2014; member stitching is prohibited",
    }
    pd.DataFrame([exception]).to_csv(MANIFESTS / "cmip6_monthly_ua_member_exceptions.csv", index=False)
    print(json.dumps(exception, indent=2), flush=True)
    print(replacement[["filename", "file_data_node"]].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
