#!/usr/bin/env python3
"""
preprocess.py — normalise and filter source datasets for the perturbation pipeline.

Produces a clean CSV per dataset with:
  - Language codes normalised to ISO 639-3
  - Unsupported languages removed
  - A 'pipeline_lang' column added (EN, DE, ES, …)
  - Rows sorted by ID ascending

Usage:
    python preprocess.py --dataset multiclaim [--input PATH] [--output PATH]
    python preprocess.py --dataset posts      [--input PATH] [--output PATH]
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

# ── path setup ────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent
DATA_DIR   = SCRIPT_DIR.parent / "data"

sys.path.insert(0, str(SCRIPT_DIR))
from utils import iso1_to_iso3, iso3_to_pipeline, SUPPORTED_ISO3

# ── per-dataset defaults ──────────────────────────────────────────────────────
DEFAULTS: dict[str, dict[str, Path]] = {
    "multiclaim": {
        "input":  DATA_DIR / "raw"       / "MultiClaim.csv",
        "output": DATA_DIR / "processed" / "preprocessed_multiclaim.csv",
    },
    "posts": {
        "input":  DATA_DIR / "raw"       / "posts.csv",
        "output": DATA_DIR / "processed" / "preprocessed_posts.csv",
    },
}


# ── dataset-specific processors ───────────────────────────────────────────────

def preprocess_multiclaim(src: Path, dst: Path) -> None:
    """
    MultiClaim.csv already uses ISO 639-3 Language codes.
    We just filter to supported languages, add pipeline_lang, and sort by NID.

    Input columns:  Claim, ClusterID, Translation, Language, NID, Timstamp, URL
    Output columns: same + pipeline_lang
    """
    with open(src, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    kept:    list[dict] = []
    dropped: list[dict] = []

    for row in rows:
        iso3 = row["Language"].strip()
        pipeline_lang = iso3_to_pipeline(iso3)
        if pipeline_lang is None:
            dropped.append(row)
            continue
        row["pipeline_lang"] = pipeline_lang
        kept.append(row)

    kept.sort(key=lambda r: int(r["NID"]))

    _write_csv(dst, kept)

    # ── report ──
    print(f"MultiClaim")
    print(f"  Input : {len(rows):,} rows")
    print(f"  Kept  : {len(kept):,} rows")
    print(f"  Dropped (unsupported lang): {len(dropped):,} rows")
    _report_langs(kept,    "Language",  "  Kept langs  :")
    _report_langs(dropped, "Language",  "  Dropped langs:")


def preprocess_posts(src: Path, dst: Path) -> None:
    """
    posts.csv uses ISO 639-1 / BCP-47 codes in post_detected_language.
    We normalise to ISO 639-3, filter to supported languages, and sort by post_id.

    Input columns:  post_id, post_body, post_body_en, post_detected_language,
                    post_detected_language_iso, instances, ocr, verdicts, text_v1
    Output columns: same + language_iso3, pipeline_lang
    """
    with open(src, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    kept:    list[dict] = []
    dropped: list[dict] = []

    for row in rows:
        iso1 = row["post_detected_language"].strip()
        iso3 = iso1_to_iso3(iso1)
        if iso3 is None or iso3 not in SUPPORTED_ISO3:
            dropped.append(row)
            continue
        pipeline_lang = iso3_to_pipeline(iso3)
        if pipeline_lang is None:          # shouldn't happen but be safe
            dropped.append(row)
            continue
        row["language_iso3"]  = iso3
        row["pipeline_lang"]  = pipeline_lang
        kept.append(row)

    kept.sort(key=lambda r: int(r["post_id"]))

    _write_csv(dst, kept)

    # ── report ──
    print(f"posts")
    print(f"  Input : {len(rows):,} rows")
    print(f"  Kept  : {len(kept):,} rows")
    print(f"  Dropped (unsupported lang): {len(dropped):,} rows")
    _report_langs(kept,    "language_iso3",        "  Kept langs  :")
    _report_langs(dropped, "post_detected_language","  Dropped langs:")


# ── helpers ───────────────────────────────────────────────────────────────────

def _write_csv(dst: Path, rows: list[dict]) -> None:
    if not rows:
        print("  WARNING: no rows to write.")
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(dst, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Output: {dst}")


def _report_langs(rows: list[dict], col: str, label: str) -> None:
    counts = Counter(r[col] for r in rows)
    top    = counts.most_common(10)
    parts  = ", ".join(f"{lang}={n:,}" for lang, n in top)
    extra  = f" (+{len(counts)-10} more)" if len(counts) > 10 else ""
    print(f"{label} {parts}{extra}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Normalise and filter source datasets for the perturbation pipeline."
    )
    parser.add_argument(
        "--dataset", required=True, choices=["multiclaim", "posts"],
        help="Which dataset to preprocess.",
    )
    parser.add_argument(
        "--input", type=Path, default=None,
        help="Path to source CSV (default: data/raw/<dataset>.csv).",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Path for preprocessed output (default: data/processed/preprocessed_<dataset>.csv).",
    )
    args = parser.parse_args()

    defaults = DEFAULTS[args.dataset]
    src = args.input  or defaults["input"]
    dst = args.output or defaults["output"]

    if not src.exists():
        sys.exit(f"Input file not found: {src}")

    print(f"Input : {src}")
    print(f"Output: {dst}\n")

    if args.dataset == "multiclaim":
        preprocess_multiclaim(src, dst)
    else:
        preprocess_posts(src, dst)


if __name__ == "__main__":
    main()
