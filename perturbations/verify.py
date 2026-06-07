#!/usr/bin/env python3
"""
verify.py — generic verifier for perturbed claim datasets.

Reads a perturbed CSV (output of perturb_multiclaim.py or perturb_posts.py),
calls a local LM Studio instance to judge each perturbation row, and writes
a verified CSV with five additional columns:

    perturbation_applied  (bool) — was the transformation visibly applied?
    meaning_preserved     (bool) — is the core factual meaning unchanged?
    verified              (bool) — perturbation_applied AND meaning matches expectation
    verify_error          (str)  — non-empty if the call failed

The 'verified' flag is always re-derived from our own logic — we do not trust
the model's self-assessed 'verified' field (it can be logically inconsistent).

Usage:
    # Verify a MultiClaim perturbed file
    python verify.py --input data/processed/perturbed_multiclaim.csv

    # Custom output path
    python verify.py --input perturbed_posts.csv --output verified_posts.csv

    # Dry run: print first 5 rows, write nothing
    python verify.py --input perturbed_multiclaim.csv --dry-run

    # Parallel verification (each row is an independent LLM call)
    python verify.py --input perturbed_multiclaim.csv --workers 4

Resume:
    If the output file already exists, completed (id, perturbation_name) pairs
    are read from it on startup and skipped. No separate checkpoint file needed.

Auto-detection:
    The script detects which dataset is being verified from the column headers:
      - 'NID'     column → MultiClaim  (id_col=NID,     text_col=Claim)
      - 'post_id' column → posts       (id_col=post_id, text_col=post_body)
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

try:
    from openai import OpenAI
except ImportError:
    sys.exit("openai package not found. Run: pip install openai --break-system-packages")

# ── Import shared metadata from the existing verifier ────────────────────────
SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))

from verify_perturbations import (  # type: ignore[import]
    PERTURBATION_META,
    SYSTEM_PROMPT,
    LM_STUDIO_BASE_URL,
    LM_STUDIO_API_KEY,
    MODEL_NAME,
)

# ── Configuration ─────────────────────────────────────────────────────────────

DATA_DIR      = SCRIPT_DIR.parent / "data"
REQUEST_DELAY = 0.3   # seconds between calls (lower than perturber — verifier calls are cheaper)
DRY_RUN_ROWS  = 5


# ── Column auto-detection ─────────────────────────────────────────────────────

def detect_columns(fieldnames: list[str]) -> tuple[str, str]:
    """Return (id_col, text_col) by inspecting the CSV header."""
    if "NID" in fieldnames:
        return "NID", "Claim"
    if "post_id" in fieldnames:
        return "post_id", "post_body"
    raise ValueError(
        f"Cannot detect dataset from columns: {fieldnames}\n"
        f"Expected 'NID' (MultiClaim) or 'post_id' (posts)."
    )


# ── Prompt builder ────────────────────────────────────────────────────────────

def build_prompt(row: dict, text_col: str) -> str:
    """Build the user prompt for a single verification row."""
    name = row["perturbation_name"]
    meta = PERTURBATION_META.get(name)
    if meta is None:
        raise ValueError(f"Unknown perturbation type: {name!r}")

    expect_label = (
        "PRESERVED — the core factual meaning should be unchanged"
        if meta["expect_preserved"]
        else "INTENTIONALLY ALTERED — the meaning is expected to have shifted"
    )

    original  = row.get("original_text") or row.get(text_col) or ""
    perturbed = row.get("perturbed_text", "")

    return (
        f"## Perturbation type: {name}\n\n"
        f"### Verification criteria:\n{meta['description']}\n\n"
        f"### Expected outcome for 'meaning_preserved': {expect_label}\n\n"
        f"---\n\n"
        f"### Original text:\n{original}\n\n"
        f"---\n\n"
        f"### Perturbed text:\n{perturbed}"
    )


# ── LM Studio call ────────────────────────────────────────────────────────────

def call_lm_studio(client: OpenAI, user_prompt: str) -> dict:
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_prompt},
        ],
    )
    msg = response.choices[0].message
    raw = msg.content or getattr(msg, "reasoning_content", "") or ""
    return json.loads(raw.strip())


# ── Per-row verification ──────────────────────────────────────────────────────

def verify_row(row: dict, text_col: str, client: OpenAI, delay: float) -> dict:
    """Verify one row. Returns a result dict with the five verification columns."""
    result: dict = {
        "perturbation_applied": None,
        "meaning_preserved":    None,
        "verified":             None,
        "verify_error":         "",
    }

    name = row["perturbation_name"]

    # Skip rows that had a perturbation error (nothing to verify)
    if row.get("error"):
        result["verify_error"] = f"skipped — perturbation error: {row['error']}"
        return result

    # Skip rows where the perturbation produced no change (clearly not applied)
    if row.get("changed") in ("False", "false", "0", "") and not row.get("perturbed_text"):
        result["perturbation_applied"] = False
        result["meaning_preserved"]    = True
        result["verified"]             = False
        return result

    try:
        user_prompt = build_prompt(row, text_col)
        verdict     = call_lm_studio(client, user_prompt)

        for key in ("perturbation_applied", "meaning_preserved", "verified"):
            if key not in verdict:
                raise KeyError(f"Model response missing key: {key!r}")

        # Re-derive 'verified' from ground truth — don't trust the model's own flag
        meta             = PERTURBATION_META[name]
        expected_pres    = meta["expect_preserved"]
        recomputed       = (
            bool(verdict["perturbation_applied"]) and
            (bool(verdict["meaning_preserved"]) == expected_pres)
        )
        verdict["verified"] = recomputed

        result.update(verdict)

        if delay > 0:
            time.sleep(delay)

    except json.JSONDecodeError as exc:
        result["verify_error"] = f"JSON parse error: {exc}"
    except KeyError as exc:
        result["verify_error"] = f"Missing key: {exc}"
    except Exception as exc:
        result["verify_error"] = str(exc)

    return result


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify perturbed claim datasets via a local LM Studio instance."
    )
    parser.add_argument(
        "--input", type=Path, required=True,
        help="Perturbed CSV file (output of perturb_multiclaim.py or perturb_posts.py).",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Output path. Default: data/processed/verified_<input_stem>.csv",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Process at most N rows (useful for staging).",
    )
    parser.add_argument(
        "--workers", type=int, default=1,
        help="Number of parallel LLM calls. Each row is independent. Default 1.",
    )
    parser.add_argument(
        "--delay", type=float, default=REQUEST_DELAY,
        help=f"Seconds between LLM calls per worker (default {REQUEST_DELAY}).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help=f"Process first {DRY_RUN_ROWS} rows, print results, write nothing.",
    )
    args = parser.parse_args()

    if not args.input.exists():
        sys.exit(f"Input file not found: {args.input}")

    # Default output path
    output: Path = args.output or (
        DATA_DIR / "processed" / f"verified_{args.input.stem.removeprefix('perturbed_')}.csv"
    )

    print(f"Input  : {args.input}")
    print(f"Output : {output}")
    print(f"Workers: {args.workers}")
    if args.limit:
        print(f"Limit  : {args.limit} rows")
    print()

    # ── load input ────────────────────────────────────────────────────────────
    with open(args.input, newline="", encoding="utf-8-sig") as f:
        reader    = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        all_rows  = list(reader)

    if args.dry_run:
        all_rows = all_rows[:DRY_RUN_ROWS]
    elif args.limit:
        all_rows = all_rows[:args.limit]

    # ── detect dataset ────────────────────────────────────────────────────────
    try:
        id_col, text_col = detect_columns(fieldnames)
    except ValueError as exc:
        sys.exit(str(exc))
    print(f"Detected: id_col={id_col!r}, text_col={text_col!r}")

    # ── output schema ─────────────────────────────────────────────────────────
    verify_cols = [
        "perturbation_applied", "meaning_preserved",
        "verified", "verify_error",
    ]
    out_fieldnames = fieldnames + verify_cols

    # ── resume: load already-done (id, perturbation_name) pairs ──────────────
    done_pairs: set[tuple[str, str]] = set()
    output_exists = (not args.dry_run) and output.exists() and output.stat().st_size > 0
    if output_exists:
        print(f"Resuming — scanning {output.name} for completed rows …", end=" ", flush=True)
        with open(output, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                done_pairs.add((row[id_col], row["perturbation_name"]))
        print(f"{len(done_pairs):,} done.")

    todo = [
        r for r in all_rows
        if (r[id_col], r["perturbation_name"]) not in done_pairs
    ]

    print(f"Total rows  : {len(all_rows):,}")
    print(f"Already done: {len(done_pairs):,}")
    print(f"To verify   : {len(todo):,}")
    if args.dry_run:
        print(f"[DRY RUN] printing first {DRY_RUN_ROWS} rows — nothing will be written.\n")

    if not todo:
        print("Nothing to do.")
        return

    # ── open output ───────────────────────────────────────────────────────────
    out_file = None
    writer   = None
    if not args.dry_run:
        output.parent.mkdir(parents=True, exist_ok=True)
        mode     = "a" if output_exists else "w"
        out_file = open(output, mode, newline="", encoding="utf-8")
        writer   = csv.DictWriter(out_file, fieldnames=out_fieldnames, extrasaction="ignore")
        if mode == "w":
            writer.writeheader()

    # ── processing loop ───────────────────────────────────────────────────────
    client     = OpenAI(base_url=LM_STUDIO_BASE_URL, api_key=LM_STUDIO_API_KEY)
    write_lock = threading.Lock()
    processed  = 0
    n_pass     = 0
    n_fail     = 0
    n_error    = 0
    total      = len(todo)

    def _task(row: dict) -> tuple[dict, dict]:
        verdict = verify_row(row, text_col, client, args.delay)
        return row, verdict

    try:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            future_to_row = {executor.submit(_task, row): row for row in todo}
            for future in as_completed(future_to_row):
                exc = future.exception()
                if exc:
                    row = future_to_row[future]
                    print(f"  FATAL {row[id_col]}/{row['perturbation_name']}: {exc}")
                    n_error += 1
                    continue

                row, verdict = future.result()
                out_row = {**row, **verdict}

                if args.dry_run:
                    status = "✓" if verdict.get("verified") else "✗"
                    print(
                        f"  {status} {row[id_col]} | {row['perturbation_name']}\n"
                        f"    applied={verdict['perturbation_applied']} "
                        f"preserved={verdict['meaning_preserved']} "
                        f"verified={verdict['verified']}\n"
                        + (f"\n    ERROR: {verdict['verify_error']}" if verdict["verify_error"] else "")
                    )
                else:
                    with write_lock:
                        writer.writerow(out_row)
                        out_file.flush()

                processed += 1
                if verdict.get("verify_error"):
                    n_error += 1
                elif verdict.get("verified"):
                    n_pass += 1
                else:
                    n_fail += 1

                pct = 100 * processed // total
                print(f"  [{processed:6d}/{total}] {pct:3d}%  "
                      f"id={row[id_col]}  {row['perturbation_name']}  "
                      f"{'✓' if verdict.get('verified') else '✗'}")

    finally:
        if out_file:
            out_file.close()

    # ── summary ───────────────────────────────────────────────────────────────
    print("\n" + "─" * 60)
    print(f"Processed  : {processed:,}")
    print(f"  ✓ Pass   : {n_pass:,}  ({100*n_pass//max(processed,1)}%)")
    print(f"  ✗ Fail   : {n_fail:,}")
    print(f"  ⚠ Error  : {n_error:,}")
    if not args.dry_run:
        print(f"Output     : {output}")
    print("─" * 60)


if __name__ == "__main__":
    main()
