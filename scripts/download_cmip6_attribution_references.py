"""Download the two official open-access CMIP6/DAMIP design papers."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import requests


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "refs" / "literature" / "cmip6_attribution"
FILES = {
    "eyring_etal_2016_cmip6_design.pdf": "https://gmd.copernicus.org/articles/9/1937/2016/gmd-9-1937-2016.pdf",
    "gillett_etal_2016_damip.pdf": "https://gmd.copernicus.org/articles/9/3685/2016/gmd-9-3685-2016.pdf",
}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    records = []
    for name, url in FILES.items():
        response = requests.get(url, timeout=120)
        response.raise_for_status()
        payload = response.content
        if not payload.startswith(b"%PDF"):
            raise RuntimeError(f"Expected PDF but received {response.headers.get('content-type')} from {url}")
        path = OUT / name
        path.write_bytes(payload)
        records.append(
            {
                "url": url,
                "local_path": str(path.relative_to(ROOT)).replace("\\", "/"),
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
                "content_type": response.headers.get("content-type"),
                "downloaded_utc": datetime.now(timezone.utc).isoformat(),
                "license": "CC BY 3.0",
                "full_text_status": "open_pdf",
                "download_status": "downloaded",
            }
        )
    (OUT / "download_manifest.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(records, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
