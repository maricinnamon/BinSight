#!/usr/bin/env python
"""Downloads the TACO images referenced by the official annotations.

The official `download.py` writes into the cloned repository and gives no
machine-readable account of what failed. TACO's images are hosted on Flickr and
the dataset dates from 2023, so some URLs are expected to be gone — a run that
silently skips them would leave the dataset quietly incomplete.

This downloader therefore retries transient failures, never invents a
replacement image, and records every outcome in reports/download_report.{json,md}.
"""
from __future__ import annotations

import argparse
import collections
import concurrent.futures
import json
import pathlib
import sys
import time
import urllib.error
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[1]
ANNOTATIONS = ROOT / "data" / "source" / "TACO" / "data" / "annotations.json"
RAW = ROOT / "data" / "raw"
REPORTS = ROOT / "reports"

USER_AGENT = "BinSight-dataset-fetch/1.0 (+research use; official TACO URLs)"
# Flickr rate-limits aggressive clients with HTTP 429. The first run used 10
# workers and got 429 on 898 of 1500 images — those URLs are alive, we simply
# asked too fast. Back off hard and honour Retry-After.
RETRY_DELAYS = (2.0, 10.0, 30.0, 60.0, 120.0)


def fetch(record: dict, timeout: float) -> dict:
    """Downloads one image. Returns an outcome record; never raises."""
    target = RAW / record["file_name"]
    url = record["flickr_url"]

    if target.exists() and target.stat().st_size > 0:
        return {"id": record["id"], "file_name": record["file_name"],
                "status": "already_present", "bytes": target.stat().st_size}

    target.parent.mkdir(parents=True, exist_ok=True)
    last_error = ""

    for attempt, delay in enumerate((0.0,) + RETRY_DELAYS):
        if delay:
            time.sleep(delay)
        try:
            request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(request, timeout=timeout) as response:
                payload = response.read()
            if not payload:
                last_error = "empty response body"
                continue
            target.write_bytes(payload)
            return {"id": record["id"], "file_name": record["file_name"],
                    "status": "downloaded", "bytes": len(payload),
                    "attempts": attempt + 1}
        except urllib.error.HTTPError as exc:
            last_error = f"HTTP {exc.code}"
            if exc.code == 429:
                # Rate limited, not gone. Wait what the server asks, or longer.
                retry_after = exc.headers.get("Retry-After") if exc.headers else None
                try:
                    time.sleep(min(float(retry_after), 180.0) if retry_after else 30.0)
                except (TypeError, ValueError):
                    time.sleep(30.0)
                continue
            # Other 4xx will not become available by retrying.
            if 400 <= exc.code < 500:
                break
        except Exception as exc:  # noqa: BLE001 - record whatever went wrong
            last_error = f"{type(exc).__name__}: {exc}"

    return {"id": record["id"], "file_name": record["file_name"],
            "status": "failed", "error": last_error, "url": url}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--limit", type=int, default=None, help="debug only")
    args = parser.parse_args()

    if not ANNOTATIONS.exists():
        print(f"ERROR: {ANNOTATIONS} not found — clone the TACO repository first.")
        return 1

    data = json.loads(ANNOTATIONS.read_text())
    images = data["images"]
    if args.limit:
        images = images[: args.limit]

    annotated = {a["image_id"] for a in data["annotations"]}
    RAW.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)

    print(f"Images referenced by annotations.json: {len(images)}")
    print(f"Images carrying at least one annotation: {len(annotated)}")
    print(f"Downloading into {RAW} with {args.workers} workers...")

    started = time.time()
    results: list[dict] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(fetch, record, args.timeout): record for record in images}
        for done, future in enumerate(concurrent.futures.as_completed(futures), 1):
            results.append(future.result())
            if done % 100 == 0:
                ok = sum(1 for r in results if r["status"] != "failed")
                print(f"  {done}/{len(images)}  ok={ok}  failed={done - ok}", flush=True)

    elapsed = time.time() - started
    by_status = collections.Counter(r["status"] for r in results)
    failed = [r for r in results if r["status"] == "failed"]
    total_bytes = sum(r.get("bytes", 0) for r in results)
    errors = collections.Counter(r.get("error", "") for r in failed)

    # What is actually on disk now, regardless of this run.
    present = [r for r in results if (RAW / r["file_name"]).exists()]
    missing_ids = {r["id"] for r in results} - {r["id"] for r in present}

    report = {
        "source_repository": "https://github.com/pedropro/TACO",
        "annotation_file": "data/annotations.json",
        "downloaded_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "elapsed_seconds": round(elapsed, 1),
        "images_referenced": len(images),
        "images_with_annotations": len(annotated),
        "downloaded": by_status.get("downloaded", 0),
        "already_present": by_status.get("already_present", 0),
        "failed": len(failed),
        "present_on_disk": len(present),
        "missing_locally": len(missing_ids),
        "total_bytes": total_bytes,
        "total_mb": round(total_bytes / 1e6, 1),
        "error_breakdown": dict(errors),
        "failures": sorted(failed, key=lambda r: r["file_name"]),
    }
    (REPORTS / "download_report.json").write_text(json.dumps(report, indent=2) + "\n")

    lines = [
        "# TACO image download report",
        "",
        f"- **Source:** <{report['source_repository']}> · `{report['annotation_file']}`",
        f"- **Downloaded at:** {report['downloaded_at']}",
        f"- **Elapsed:** {report['elapsed_seconds']} s",
        "",
        "| | |",
        "|---|---:|",
        f"| Images referenced by annotations | {report['images_referenced']} |",
        f"| Images with at least one annotation | {report['images_with_annotations']} |",
        f"| Downloaded this run | {report['downloaded']} |",
        f"| Already present | {report['already_present']} |",
        f"| **Failed** | **{report['failed']}** |",
        f"| Present on disk | {report['present_on_disk']} |",
        f"| Missing locally | {report['missing_locally']} |",
        f"| Total size | {report['total_mb']} MB |",
        "",
    ]
    if errors:
        lines += ["## Failure breakdown", "", "| Error | Count |", "|---|---:|"]
        lines += [f"| `{k or 'unknown'}` | {v} |" for k, v in errors.most_common()]
        lines += ["",
                  "Failed images are listed in `download_report.json`. They are recorded",
                  "rather than replaced: no substitute imagery was introduced.", ""]
    else:
        lines += ["Every referenced image was retrieved.", ""]

    (REPORTS / "download_report.md").write_text("\n".join(lines))

    print(f"\ndownloaded={report['downloaded']} already={report['already_present']} "
          f"failed={report['failed']} total={report['total_mb']} MB in {elapsed:.0f}s")
    print(f"Reports -> {REPORTS/'download_report.json'}")

    # A handful of dead Flickr URLs is expected and not fatal; losing most of
    # the dataset is.
    if report["present_on_disk"] < 0.9 * report["images_referenced"]:
        print("ERROR: more than 10% of images are missing — dataset is not usable.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
