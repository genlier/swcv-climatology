"""Download and validate monthly climate indices used as variability controls."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "indices"
SOURCES = {
    "nino34_ersstv5": {
        "url": "https://www.cpc.ncep.noaa.gov/data/indices/ersst5.nino.mth.91-20.ascii",
        "column": "nino34_c",
        "format": "cpc_nino",
        "description": "NOAA CPC Nino 3.4 SST anomaly from ERSSTv5",
    },
    "pdo_ensemble": {
        "url": "https://psl.noaa.gov/pdo/data/pdo.timeseries.sstens.csv",
        "column": "pdo_index",
        "description": "NOAA PSL ensemble PDO from COBE, ERSSTv5, and HadISST1.1",
    },
    "dmi_hadisst": {
        "url": "https://psl.noaa.gov/data/timeseries/month/data/dmi.had.long.csv",
        "column": "dmi_c",
        "description": "NOAA PSL Dipole Mode Index from HadISST1.1",
    },
}


def session() -> requests.Session:
    retry = Retry(
        total=5,
        backoff_factor=2,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
    )
    result = requests.Session()
    result.mount("https://", HTTPAdapter(max_retries=retry))
    result.headers["User-Agent"] = "swcv-climatology/1.0 (academic research)"
    return result


def parse_csv(text: str, column: str, format_name: str = "csv") -> pd.DataFrame:
    if format_name == "cpc_nino":
        names = [
            "year", "month", "nino12", "nino12_anom", "nino3", "nino3_anom",
            "nino4", "nino4_anom", "nino34", column,
        ]
        source = pd.read_csv(StringIO(text), sep=r"\s+", skiprows=1, header=None, names=names)
        frame = pd.DataFrame(
            {
                "date": pd.to_datetime(dict(year=source.year, month=source.month, day=1)),
                column: pd.to_numeric(source[column], errors="coerce"),
            }
        )
        return frame.dropna(subset=[column]).sort_values("date").drop_duplicates("date")
    frame = pd.read_csv(StringIO(text), skipinitialspace=True)
    if frame.shape[1] != 2:
        raise ValueError(f"Expected two columns, found {frame.columns.tolist()}")
    frame.columns = ["date", column]
    frame["date"] = pd.to_datetime(frame.date, errors="raise")
    frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame.loc[frame[column] <= -99, column] = pd.NA
    frame = frame.dropna(subset=[column]).sort_values("date").drop_duplicates("date")
    return frame


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    client = session()
    manifest = {
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "sources": {},
    }
    combined = None
    for name, spec in SOURCES.items():
        response = client.get(spec["url"], timeout=60)
        response.raise_for_status()
        text = response.text
        frame = parse_csv(text, spec["column"], spec.get("format", "csv"))
        audit = frame.loc[frame.date.between("1979-01-01", "2023-12-31")]
        expected = pd.date_range("1979-01-01", "2023-12-01", freq="MS")
        missing = expected.difference(audit.date)
        if len(audit) != len(expected) or len(missing):
            raise RuntimeError(f"{name} lacks required months: {missing.tolist()[:12]}")
        path = OUT / f"{name}_monthly.csv"
        frame.to_csv(path, index=False)
        combined = frame if combined is None else combined.merge(frame, on="date", how="outer")
        manifest["sources"][name] = {
            **spec,
            "sha256_source_text": hashlib.sha256(response.content).hexdigest(),
            "rows": int(len(frame)),
            "first_date": frame.date.min().date().isoformat(),
            "last_date": frame.date.max().date().isoformat(),
            "required_period_complete": True,
            "local_file": str(path.relative_to(ROOT)),
        }
        print(f"verified {name}: {len(frame)} months, {frame.date.min():%Y-%m}--{frame.date.max():%Y-%m}")

    assert combined is not None
    combined.sort_values("date").to_csv(OUT / "climate_indices_monthly.csv", index=False)
    (OUT / "climate_indices_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
