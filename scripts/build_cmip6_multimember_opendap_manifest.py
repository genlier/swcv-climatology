"""Resolve server-subsettable ESGF files for validated multi-member zg datasets."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import re
import time

import pandas as pd
import requests


ROOT = Path(__file__).resolve().parents[1]
VALIDATION = ROOT / "data" / "processed" / "cmip6_multimember_zarr_validation.csv"
OUT = ROOT / "data" / "manifests" / "cmip6_multimember_zg_opendap_files.csv"
API = "https://esgf-node.llnl.gov/esg-search/search"
REQUIRED_YEARS = set(range(1951, 1981)) | set(range(1985, 2015))


def years(filename: str) -> tuple[int | None, int | None]:
    match = re.search(r"_(\d{4})\d{2}-(\d{4})\d{2}\.nc$", filename)
    return (int(match.group(1)), int(match.group(2))) if match else (None, None)


def endpoint(doc: dict[str, object]) -> str:
    candidates = []
    for item in doc.get("url", []):
        pieces = str(item).split("|")
        if len(pieces) >= 3 and pieces[-1] == "OPENDAP":
            url = pieces[0].removesuffix(".html")
            if url.startswith("http://"):
                url = "https://" + url.removeprefix("http://")
            candidates.append(url)
    candidates.sort(key=lambda url: ("esgf-node.ornl.gov" not in url, not url.startswith("https://"), url))
    return candidates[0] if candidates else ""


def resolve(row: dict[str, object]) -> list[dict[str, object]]:
    params = {
        "type": "File", "project": "CMIP6", "source_id": row["model"],
        "experiment_id": row["experiment"], "member_id": row["member"],
        "variable_id": "zg", "table_id": "Amon", "grid_label": row["grid"],
        "version": str(row["version"]), "format": "application/solr+json", "limit": 500,
    }
    docs: list[dict[str, object]] = []
    for allow_alternate_version in (False, True):
        query = dict(params)
        if allow_alternate_version:
            query.pop("version")
        for attempt in range(4):
            try:
                response = requests.get(API, params=query, timeout=120)
                response.raise_for_status()
                break
            except requests.RequestException:
                if attempt == 3:
                    raise
                time.sleep(2 ** attempt)
        docs = [doc for doc in response.json()["response"]["docs"] if endpoint(doc)]
        if docs:
            break
    by_filename: dict[str, list[tuple[str, str, str]]] = {}
    for doc in docs:
        filename = str(doc.get("title", ""))
        url = endpoint(doc)
        version = str(doc.get("version", row["version"]))
        dataset_id = str(doc.get("instance_id", row["dataset_id"]))
        if filename and url:
            by_filename.setdefault(filename, []).append((url, version, dataset_id))
    rows = []
    for filename, sources in by_filename.items():
        start, end = years(filename)
        chosen, resolved_version, dataset_id = sorted(
            set(sources), key=lambda source: (-int(source[1]), "esgf-node.ornl.gov" not in source[0], source[0])
        )[0]
        rows.append(
            {
                "model": row["model"], "experiment": row["experiment"],
                "member": row["member"], "grid": row["grid"],
                "validation_version": row["version"], "resolved_version": resolved_version,
                "dataset_id": dataset_id, "filename": filename,
                "file_start_year": start, "file_end_year": end, "opendap_url": chosen,
            }
        )
    if not rows:
        raise RuntimeError("no OPeNDAP files resolved")
    return rows


def main() -> None:
    validation = pd.read_csv(VALIDATION)
    validation = validation.loc[validation.valid.astype(str).str.lower().eq("true")]
    work = validation.to_dict("records")
    rows, failures = [], []
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(resolve, row): row for row in work}
        for future in as_completed(futures):
            row = futures[future]
            try:
                found = future.result()
                rows.extend(found)
                print(f"{row['model']} {row['member']} {row['experiment']}: {len(found)} files", flush=True)
            except Exception as exc:
                failures.append({**row, "error": f"{type(exc).__name__}: {exc}"})
                print(f"failed {row['model']} {row['member']} {row['experiment']}: {exc}", flush=True)
    failure_out = OUT.with_name(OUT.stem + "_failures.csv")
    if failures:
        pd.DataFrame(failures).to_csv(failure_out, index=False)
        raise SystemExit(2)
    result = pd.DataFrame(rows).sort_values(["model", "member", "experiment", "filename"])
    incomplete = []
    for keys, group in result.groupby(["model", "member", "experiment"], sort=True):
        covered: set[int] = set()
        for source in group.itertuples():
            if pd.notna(source.file_start_year) and pd.notna(source.file_end_year):
                covered.update(range(int(source.file_start_year), int(source.file_end_year) + 1))
        missing = sorted(REQUIRED_YEARS - covered)
        if missing:
            incomplete.append({"model": keys[0], "member": keys[1], "experiment": keys[2],
                               "error": f"missing response years: {missing}"})
    if len(result.groupby(["model", "member", "experiment"])) != len(work):
        raise RuntimeError("Resolved dataset-group count does not match validated input")
    if incomplete:
        pd.DataFrame(incomplete).to_csv(failure_out, index=False)
        raise RuntimeError("One or more OPeNDAP groups do not cover both response periods")
    result.to_csv(OUT, index=False)
    failure_out.unlink(missing_ok=True)
    print(f"resolved {len(result)} files for {len(work)} datasets", flush=True)


if __name__ == "__main__":
    main()
