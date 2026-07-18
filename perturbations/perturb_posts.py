#!/usr/bin/env python3
"""
perturb_posts.py — apply all perturbations to the posts dataset.

Reads the preprocessed posts CSV (one row per post) and writes one output row
per (post × perturbation_name), preserving all original columns.

Usage:
    # Full run (resumes automatically if output already exists)
    python perturb_posts.py

    # Custom paths
    python perturb_posts.py --input data/processed/preprocessed_posts.csv \\
                            --output data/processed/perturbed_posts.csv

    # Test on first 20 posts, 4 parallel workers, no delay
    python perturb_posts.py --limit 20 --workers 4 --delay 0

    # Dry run: process first 2 posts, print results, write nothing
    python perturb_posts.py --dry-run

Output schema (all original columns preserved, these appended):
    perturbation_name, family, perturbed_text, changed, error

Resume:
    On startup the script reads existing output to collect completed post IDs.
    Posts already fully processed are skipped. All perturbation rows for a post
    are written atomically before moving to the next post.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
DATA_DIR   = SCRIPT_DIR.parent / "data"

sys.path.insert(0, str(SCRIPT_DIR))
from utils import run_perturber

DEFAULT_INPUT  = DATA_DIR / "processed" / "preprocessed_posts.csv"
DEFAULT_OUTPUT = DATA_DIR / "processed" / "perturbed_posts.csv"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Perturb the posts dataset."
    )
    parser.add_argument("--input",   type=Path, default=DEFAULT_INPUT,
                        help=f"Preprocessed input CSV (default: {DEFAULT_INPUT})")
    parser.add_argument("--output",  type=Path, default=DEFAULT_OUTPUT,
                        help=f"Perturbed output CSV (default: {DEFAULT_OUTPUT})")
    parser.add_argument("--limit",           type=int,   default=None,
                        help="Process at most N input rows (for testing).")
    parser.add_argument("--sample-per-lang", type=int,   default=None,
                        metavar="N",
                        help="Take at most N rows per language — generates a small "
                             "balanced test set (e.g. --sample-per-lang 10).")
    parser.add_argument("--workers",         type=int,   default=1,
                        help="Thread-pool size. Default 1 (sequential). "
                             "Note: with a single-GPU local LLM, >1 mainly helps "
                             "parallelise back-translation and HTTP overhead.")
    parser.add_argument("--delay",           type=float, default=0.5,
                        help="Seconds between LLM calls per post (default 0.5).")
    parser.add_argument("--dry-run",         action="store_true",
                        help="Process first 2 posts, print output, write nothing.")
    args = parser.parse_args()

    if not args.input.exists():
        sys.exit(f"Input not found: {args.input}\n"
                 f"Run preprocess.py --dataset posts first.")

    print(f"Dataset : posts")
    print(f"Input   : {args.input}")
    print(f"Output  : {args.output}")
    print(f"Workers : {args.workers}")
    print(f"Delay   : {args.delay}s per LLM call")
    if args.limit:
        print(f"Limit   : {args.limit} rows")
    if args.sample_per_lang:
        print(f"Sample  : {args.sample_per_lang} rows/language (test-set mode)")
    print()

    run_perturber(
        input_file      = args.input,
        output_file     = args.output,
        id_col          = "post_id",
        text_col        = "post_body",
        limit           = args.limit,
        workers         = args.workers,
        delay           = args.delay,
        dry_run         = args.dry_run,
        sample_per_lang = args.sample_per_lang,
    )


if __name__ == "__main__":
    main()
