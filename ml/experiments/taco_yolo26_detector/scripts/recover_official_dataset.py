#!/usr/bin/env python
"""Recovers the complete official TACO images from the archived Zenodo release.

Run 001 downloaded images one Flickr URL at a time and lost 898 of 1500 to HTTP
429 rate limiting. The archived release is a single reproducible file with a
published checksum, which is both faster and something a reader can verify years
from now — individual Flickr URLs are not.

Nothing is overwritten: the archive lands in data/downloads/, extracts to
data/official_complete/, and the partial Run 001 copy in data/raw/ is left alone
as evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import shutil
import sys
import time
import urllib.request
import zipfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
DOWNLOADS = ROOT / "data" / "downloads"
COMPLETE = ROOT / "data" / "official_complete"
REPORTS = ROOT / "reports" / "dataset_recovery"

ZENODO_RECORD = "3354286"
ARCHIVE_URL = f"https://zenodo.org/api/records/{ZENODO_RECORD}/files/TACO.zip/content"
ARCHIVE_NAME = "TACO.zip"
EXPECTED_MD5 = "5c674548402b142d5a27a1f7b6a653f3"


def digest(path: pathlib.Path, algorithm: str) -> str:
    h = hashlib.new(algorithm)
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(target: pathlib.Path, timeout: float = 120.0) -> dict:
    """Downloads with resume support, so an interrupted transfer is not restarted."""
    target.parent.mkdir(parents=True, exist_ok=True)
    existing = target.stat().st_size if target.exists() else 0
    headers = {"User-Agent": "BinSight-dataset-recovery/1.0"}
    if existing:
        headers["Range"] = f"bytes={existing}-"
        print(f"Resuming from {existing / 1e6:.1f} MB")

    started = time.time()
    request = urllib.request.Request(ARCHIVE_URL, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            mode = "ab" if response.status == 206 else "wb"
            if mode == "wb":
                existing = 0
            total = int(response.headers.get("Content-Length", 0)) + existing
            done = existing
            with open(target, mode) as handle:
                while chunk := response.read(1 << 20):
                    handle.write(chunk)
                    done += len(chunk)
                    if total and done % (50 << 20) < (1 << 20):
                        print(f"  {done / 1e6:.0f}/{total / 1e6:.0f} MB", flush=True)
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}",
                "bytes_on_disk": target.stat().st_size if target.exists() else 0}

    return {"ok": True, "elapsed_seconds": round(time.time() - started, 1),
            "bytes": target.stat().st_size}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-download", action="store_true")
    args = parser.parse_args()

    REPORTS.mkdir(parents=True, exist_ok=True)
    archive = DOWNLOADS / ARCHIVE_NAME
    report: dict = {
        "source": f"https://zenodo.org/records/{ZENODO_RECORD}",
        "doi": "10.5281/zenodo.3354286",
        "archive_url": ARCHIVE_URL,
        "archive_name": ARCHIVE_NAME,
        "published_md5": EXPECTED_MD5,
        "rationale": ("Archived release instead of per-image Flickr requests: "
                      "reproducible, checksummed, and immune to the HTTP 429 rate "
                      "limiting that cost Run 001 898 of 1500 images."),
    }

    if not args.skip_download or not archive.exists():
        print(f"Downloading {ARCHIVE_URL}")
        result = download(archive)
        report["download"] = result
        if not result["ok"]:
            report["verified"] = False
            (REPORTS / "archive_download_report.json").write_text(json.dumps(report, indent=2) + "\n")
            print("ERROR: download failed:", result["error"])
            return 1

    report["archive_bytes"] = archive.stat().st_size
    report["archive_mb"] = round(archive.stat().st_size / 1e6, 1)

    print("Verifying checksums...")
    md5 = digest(archive, "md5")
    report["calculated_md5"] = md5
    report["md5_matches_published"] = md5 == EXPECTED_MD5
    report["calculated_sha256"] = digest(archive, "sha256")

    if not report["md5_matches_published"]:
        report["verified"] = False
        (REPORTS / "archive_download_report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(f"ERROR: MD5 mismatch. published={EXPECTED_MD5} calculated={md5}")
        return 1
    print(f"MD5 OK: {md5}")

    print(f"Extracting to {COMPLETE} ...")
    COMPLETE.mkdir(parents=True, exist_ok=True)
    try:
        with zipfile.ZipFile(archive) as zf:
            members = zf.namelist()
            zf.extractall(COMPLETE)
        report["extraction"] = {"ok": True, "entries": len(members)}
    except Exception as exc:  # noqa: BLE001
        report["extraction"] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        (REPORTS / "archive_download_report.json").write_text(json.dumps(report, indent=2) + "\n")
        print("ERROR: extraction failed:", exc)
        return 1

    for junk in COMPLETE.rglob("__MACOSX"):
        if junk.is_dir():
            shutil.rmtree(junk, ignore_errors=True)

    images = [p for p in COMPLETE.rglob("*")
              if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
              and not p.name.startswith(".")]
    report["extracted_image_files"] = len(images)
    report["extracted_bytes"] = sum(p.stat().st_size for p in images)
    report["verified"] = True

    (REPORTS / "archive_download_report.json").write_text(json.dumps(report, indent=2) + "\n")
    (REPORTS / "archive_download_report.md").write_text("\n".join([
        "# Official TACO archive recovery", "",
        f"- **Source:** <{report['source']}> (DOI `{report['doi']}`)",
        f"- **File:** `{ARCHIVE_NAME}`, {report['archive_mb']} MB",
        f"- **Published MD5:** `{EXPECTED_MD5}`",
        f"- **Calculated MD5:** `{md5}` — **{'match' if report['md5_matches_published'] else 'MISMATCH'}**",
        f"- **SHA-256:** `{report['calculated_sha256']}`",
        f"- **Extraction:** {'OK' if report['extraction']['ok'] else 'FAILED'}, "
        f"{report['extraction'].get('entries')} entries",
        f"- **Image files extracted:** {report['extracted_image_files']}",
        f"- **Extracted image bytes:** {report['extracted_bytes'] / 1e6:.0f} MB", "",
        "## Why an archive rather than Flickr", "",
        report["rationale"], "",
        "The partial Run 001 download in `data/raw/` is left untouched as evidence "
        "for that run.", ""]) + "\n")

    print(f"Extracted {len(images)} image files to {COMPLETE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
