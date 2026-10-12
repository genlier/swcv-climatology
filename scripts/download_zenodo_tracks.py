"""Download and checksum the public SWCV event tracks from Zenodo."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import requests


ROOT = Path(__file__).resolve().parents[1]
RECORD_PATH = ROOT / "data" / "zenodo_record_15073815.json"
OUT_DIR = ROOT / "data" / "raw" / "swcv_zenodo_15073815"


def md5sum(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    record = json.loads(RECORD_PATH.read_text(encoding="utf-8-sig"))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    manifest = []
    for item in sorted(record["files"], key=lambda value: value["key"]):
        target = OUT_DIR / item["key"]
        expected = item["checksum"].split(":", 1)[1]
        if not target.exists() or md5sum(target) != expected:
            response = session.get(item["links"]["self"], timeout=120)
            response.raise_for_status()
            target.write_bytes(response.content)
        actual = md5sum(target)
        if actual != expected:
            raise RuntimeError(f"Checksum mismatch for {target.name}: {actual} != {expected}")
        manifest.append(
            {
                "file": item["key"],
                "bytes": target.stat().st_size,
                "md5": actual,
                "source": item["links"]["self"],
            }
        )
        print(f"verified {target.name} ({target.stat().st_size} bytes)")
    (OUT_DIR / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
