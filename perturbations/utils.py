"""
utils.py — shared utilities for the perturbation pipeline.

Provides:
  - Language code normalisation (ISO 639-1 → ISO 639-3 → pipeline code)
  - perturb_one()          apply a single named perturbation with error handling
  - get_perturbation_names() list of applicable perturbation names for a lang
  - is_llm_perturbation()  whether a perturbation hits the local LLM
  - get_family()           derive the family letter from a perturbation name
  - run_perturber()        shared main loop used by both dataset scripts
"""
from __future__ import annotations

import csv
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# ── ISO 639-1 / variant → ISO 639-3 ──────────────────────────────────────────
# posts.csv uses ISO 639-1 (and some BCP-47 variants).
# We normalise everything to ISO 639-3 so downstream code only needs one map.

ISO1_TO_ISO3: dict[str, str | None] = {
    # Core supported languages
    "en": "eng", "de": "deu", "es": "spa", "fr": "fra",
    "pl": "pol", "hr": "hrv", "bs": "bos", "sr": "srp",
    "hi": "hin", "ar": "ara", "ml": "mal",
    "zh": "zho", "zh-CN": "zho", "zh-TW": "zho",
    # Script / regional variants → canonical ISO 639-3
    "sr-Latn": "srp",   # Serbian (Latin script)
    "hi-Latn": "hin",   # Romanised Hindi — still treat as hin
    "pt-PT":   "por",   # Portuguese (Portugal) → por (unsupported, will be filtered)
    # Unsupported languages — normalised so the filter can drop them cleanly
    "pt": "por", "tr": "tur", "it": "ita", "mk": "mkd",
    "nl": "nld", "bn": "ben", "ms": "msa", "th": "tha",
    "ru": "rus", "id": "ind", "ur": "urd", "ro": "ron",
    "fil": "fil", "ko": "kor", "el": "ell", "sk": "slk",
    "cs": "ces", "hu": "hun", "fi": "fin", "da": "dan",
    "uk": "ukr", "bg": "bul", "bg-Latn": "bul",
    "no": "nor", "sv": "swe", "fa": "fas", "he": "heb",
    "si": "sin", "my": "mya", "km": "khm", "sw": "swa",
    "am": "amh", "ne": "nep", "pa": "pan", "te": "tel",
    "ta": "tam", "kn": "kan", "gu": "guj", "ca": "cat",
    "az": "aze", "ja": "jpn", "vi": "vie", "lv": "lav",
    "lt": "lit", "et": "est", "sl": "slv", "sq": "sqi",
    "gl": "glg", "ga": "gle", "lb": "ltz", "mt": "mlt",
    "ku": "kur", "ug": "uig", "mn": "mon", "kk": "kaz",
    "uz": "uzb", "tk": "tuk", "af": "afr", "cy": "wel",
    "is": "isl", "ka": "kat", "hy": "hye",
    "te-Latn": "tel", "bn-Latn": "ben", "mr-Latn": "mar",
    "zh-Latn": "zho", "kn-Latn": "kan", "ml-Latn": "mal",
    "ta-Latn": "tam", "ar-Latn": "ara",
    "pa-Arab": "pan", "gu-Latn": "guj", "ru-Latn": "rus",
    "el-Latn": "ell", "bg-Latn": "bul", "uk-Latn": "ukr",
    "fa-AF": "fas",
    "und": None,  # undetermined — will be filtered
}

# ── ISO 639-3 → pipeline internal language code ───────────────────────────────
# Only languages with dedicated pipeline prompts (and rule maps) are supported.

ISO3_TO_PIPELINE: dict[str, str | None] = {
    "eng": "EN",
    "deu": "DE",
    "spa": "ES",
    "fra": "FR",
    "pol": "PL",
    # HBS family — Serbo-Croatian (Serbian / Croatian / Bosnian)
    "hbs": "HBS",   # MultiClaim uses this combined code
    "hrv": "HBS",
    "srp": "HBS",
    "bos": "HBS",
    "hin": "HI",
    "ara": "AR",
    "mal": "ML",
    "zho": "ZH",
    "cmn": "ZH",    # Mandarin ISO 639-3 alternate
}

SUPPORTED_ISO3: frozenset[str] = frozenset(
    k for k, v in ISO3_TO_PIPELINE.items() if v is not None
)


def iso1_to_iso3(code: str) -> str | None:
    """Normalise an ISO 639-1 (or BCP-47 variant) tag to ISO 639-3.
    Returns None if unknown or undetermined."""
    return ISO1_TO_ISO3.get(code.strip())


def iso3_to_pipeline(code: str) -> str | None:
    """Map an ISO 639-3 code to the pipeline's internal lang tag (EN, DE, …).
    Returns None for unsupported languages."""
    return ISO3_TO_PIPELINE.get(code.strip())


def get_family(perturbation_name: str) -> str:
    """Return the single-letter family of a perturbation name.

    The first segment before '_' may contain a digit (e.g. 'A1', 'D3').
    We strip it to return only the letter, matching the existing data convention.

    Examples:
        'A1_emoji_relevant'    → 'A'
        'D3_back_translation'  → 'D'
        'P_negation_low'       → 'P'
    """
    prefix = perturbation_name.split("_")[0]   # e.g. 'A1', 'D3', 'P'
    return prefix[0]                            # first char only: 'A', 'D', 'P'


# ── Perturbation helpers ──────────────────────────────────────────────────────

def get_perturbation_names(lang: str) -> list[str]:
    """Return the ordered list of perturbation names applicable for *lang*.

    Non-EN languages get the 15 our-perturbation types.
    EN additionally gets the 12 paper perturbation types.
    """
    # Import lazily so this module can be imported without LM Studio running.
    from perturbations import (
        _LLM_PERTURBATIONS,
        _RULE_PERTURBATIONS,
        _PAPER_LLM_PERTURBATIONS,
        _PAPER_RULE_PERTURBATIONS,
    )
    names: list[str] = list(_LLM_PERTURBATIONS) + list(_RULE_PERTURBATIONS.keys())
    if lang == "EN":
        names += list(_PAPER_LLM_PERTURBATIONS) + list(_PAPER_RULE_PERTURBATIONS.keys())
    return names


def is_llm_perturbation(name: str) -> bool:
    """Return True if this perturbation makes a local LLM call (needs rate-limit delay)."""
    from perturbations import _LLM_PERTURBATIONS, _PAPER_LLM_PERTURBATIONS
    return name in _LLM_PERTURBATIONS or name in _PAPER_LLM_PERTURBATIONS


def perturb_one(name: str, text: str, lang: str) -> tuple[str, str]:
    """Apply a single named perturbation.

    Returns:
        (perturbed_text, error_string)
        error_string is empty on success.
    """
    from perturbations import (
        _LLM_PERTURBATIONS,
        _RULE_PERTURBATIONS,
        _PAPER_LLM_PERTURBATIONS,
        _PAPER_RULE_PERTURBATIONS,
        _apply_llm,
    )
    try:
        if name in _LLM_PERTURBATIONS:
            result = _apply_llm(name, lang, text)
        elif name in _RULE_PERTURBATIONS:
            result = _RULE_PERTURBATIONS[name](text, lang)
        elif name in _PAPER_LLM_PERTURBATIONS:
            result = _apply_llm(name, lang, text, section="paper_perturbations")
        elif name in _PAPER_RULE_PERTURBATIONS:
            result = _PAPER_RULE_PERTURBATIONS[name](text, lang)
        else:
            return "", f"Unknown perturbation type: {name!r}"
        return (result or ""), ""
    except Exception as exc:
        return "", str(exc)


# ── Shared perturber runner ───────────────────────────────────────────────────

def run_perturber(
    input_file: Path,
    output_file: Path,
    id_col: str,
    text_col: str,
    limit: int | None = None,
    workers: int = 1,
    delay: float = 0.5,
    dry_run: bool = False,
) -> None:
    """Main processing loop shared by perturb_multiclaim.py and perturb_posts.py.

    Reads *input_file* (preprocessed CSV), perturbs each row, and writes one
    output row per (claim × perturbation) to *output_file*.

    Resume: on startup, any claim ID already present in *output_file* is skipped.
    All perturbations for one claim are written atomically before moving on.

    Args:
        input_file:  Path to preprocessed CSV (must have 'pipeline_lang' column).
        output_file: Path to write perturbed output CSV.
        id_col:      Name of the unique-ID column ('NID' or 'post_id').
        text_col:    Name of the original-text column ('Claim' or 'post_body').
        limit:       If set, process at most this many input rows.
        workers:     Thread-pool size. With a single-GPU local LLM, >1 mostly
                     helps parallelise back-translation and HTTP overhead.
        delay:       Seconds to sleep between LLM calls (per perturbation).
        dry_run:     Process the first 2 rows, print output, write nothing.
    """
    # ── load input ────────────────────────────────────────────────────────────
    with open(input_file, newline="", encoding="utf-8-sig") as f:
        all_rows = list(csv.DictReader(f))

    if limit:
        all_rows = all_rows[:limit]
    if dry_run:
        all_rows = all_rows[:2]

    if not all_rows:
        print("Input file is empty — nothing to do.")
        return

    # ── determine output schema ───────────────────────────────────────────────
    extra_cols = ["perturbation_name", "family", "perturbed_text", "changed", "error"]
    fieldnames = list(all_rows[0].keys()) + extra_cols

    # ── resume: load already-processed IDs ───────────────────────────────────
    done_ids: set[str] = set()
    output_exists = (not dry_run) and output_file.exists() and output_file.stat().st_size > 0
    if output_exists:
        print(f"Resuming — scanning {output_file.name} for completed IDs …", end=" ", flush=True)
        with open(output_file, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                done_ids.add(row[id_col])
        print(f"{len(done_ids):,} done.")

    todo = [r for r in all_rows if r[id_col] not in done_ids]

    print(f"Total rows  : {len(all_rows):,}")
    print(f"Already done: {len(done_ids):,}")
    print(f"To process  : {len(todo):,}")
    if dry_run:
        print("[DRY RUN] printing first 2 rows — nothing will be written.\n")

    if not todo:
        print("Nothing to do.")
        return

    # ── open output file ──────────────────────────────────────────────────────
    out_file = None
    writer   = None
    if not dry_run:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        mode = "a" if output_exists else "w"
        out_file = open(output_file, mode, newline="", encoding="utf-8")
        writer   = csv.DictWriter(out_file, fieldnames=fieldnames, extrasaction="ignore")
        if mode == "w":
            writer.writeheader()

    write_lock = threading.Lock()
    processed  = 0
    error_ids: list[str] = []
    total = len(todo)

    # ── language-level progress tracking ─────────────────────────────────────
    from collections import Counter
    lang_totals: Counter = Counter(r["pipeline_lang"] for r in todo)
    lang_done:   Counter = Counter()   # updated in main thread — no lock needed
    start_time = time.time()

    # How often to print the full language table (every ~5 %, min 25 claims)
    stats_every = max(25, total // 20)

    # ── display helpers ───────────────────────────────────────────────────────
    def _fmt_eta(seconds: float) -> str:
        if seconds <= 0:
            return "…"
        s = int(seconds)
        if s < 60:
            return f"{s}s"
        if s < 3600:
            return f"{s // 60}m {s % 60:02d}s"
        h, rem = divmod(s, 3600)
        return f"{h}h {rem // 60:02d}m"

    def _fmt_elapsed(seconds: float) -> str:
        s = int(seconds)
        if s < 60:
            return f"{s}s"
        if s < 3600:
            return f"{s // 60}m {s % 60:02d}s"
        h, rem = divmod(s, 3600)
        return f"{h}h {rem // 60:02d}m"

    def _bar(done: int, total_: int, width: int = 28) -> str:
        filled = int(width * done / total_) if total_ > 0 else 0
        return "█" * filled + "░" * (width - filled)

    def _print_lang_table() -> None:
        elapsed = time.time() - start_time
        rate    = processed / elapsed if elapsed > 0 else 0
        eta_sec = (total - processed) / rate if rate > 0 else 0

        # sort langs by total descending so the biggest are on top
        langs = sorted(lang_totals.keys(), key=lambda l: -lang_totals[l])
        width = 68
        sep   = "  " + "─" * width

        print()
        print(sep)
        print(f"  {'Lang':<6} {'Done':>6} {'Total':>6}  {'%':>5}  {'Progress bar':<30}  Left")
        print(sep)
        for lang in langs:
            t = lang_totals[lang]
            d = lang_done[lang]
            r = t - d
            pct = 100 * d / t if t > 0 else 0.0
            bar = _bar(d, t)
            print(f"  {lang:<6} {d:>6,} {t:>6,}  {pct:>5.1f}%  {bar}  {r:,}")
        print(sep)
        overall_pct = 100 * processed / total if total > 0 else 0.0
        print(
            f"  {'Total':<6} {processed:>6,} {total:>6,}  {overall_pct:>5.1f}%"
            f"  elapsed {_fmt_elapsed(elapsed)}  ETA {_fmt_eta(eta_sec)}"
        )
        print(sep)
        print()

    # ── per-claim worker ──────────────────────────────────────────────────────
    def process_claim(row: dict) -> list[dict]:
        text = row[text_col]
        lang = row["pipeline_lang"]
        names = get_perturbation_names(lang)
        out_rows: list[dict] = []
        for name in names:
            perturbed, error = perturb_one(name, text, lang)
            changed = bool(perturbed and perturbed != text and not error)
            out_rows.append({
                **row,
                "perturbation_name": name,
                "family":            get_family(name),
                "perturbed_text":    perturbed,
                "changed":           str(changed),
                "error":             error,
            })
            if is_llm_perturbation(name) and delay > 0:
                time.sleep(delay)
        return out_rows

    # ── processing loop ───────────────────────────────────────────────────────
    try:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_row = {executor.submit(process_claim, row): row for row in todo}
            for future in as_completed(future_to_row):
                row    = future_to_row[future]
                row_id = row[id_col]
                lang   = row["pipeline_lang"]
                exc    = future.exception()
                if exc:
                    print(f"  ERROR {row_id}: {exc}")
                    error_ids.append(row_id)
                    continue

                results = future.result()

                if dry_run:
                    for r in results:
                        print(f"  [{r['perturbation_name']}] changed={r['changed']} "
                              f"| {r['perturbed_text'][:80]}")
                        if r["error"]:
                            print(f"    ERROR: {r['error']}")
                    print()
                else:
                    with write_lock:
                        writer.writerows(results)
                        out_file.flush()

                # ── update counters (main thread only — no lock needed) ────
                processed += 1
                lang_done[lang] += 1

                # ── per-claim line ─────────────────────────────────────────
                elapsed = time.time() - start_time
                rate    = processed / elapsed if elapsed > 0 else 0
                eta_sec = (total - processed) / rate if rate > 0 else 0
                pct     = 100 * processed / total
                n_left  = lang_totals[lang] - lang_done[lang]
                print(
                    f"  [{processed:6,}/{total:,}]  {pct:5.1f}%  "
                    f"ETA {_fmt_eta(eta_sec)}  "
                    f"id={row_id}  lang={lang}  "
                    f"({lang_done[lang]:,}/{lang_totals[lang]:,} {lang}, "
                    f"{n_left:,} left)"
                )

                # ── periodic language table ────────────────────────────────
                if processed % stats_every == 0 or processed == total:
                    _print_lang_table()

    finally:
        if out_file:
            out_file.close()

    # ── final summary ─────────────────────────────────────────────────────────
    if processed > 0:
        _print_lang_table()
    print("─" * 60)
    print(f"Processed : {processed:,} claims")
    print(f"Errors    : {len(error_ids):,}")
    if error_ids:
        print(f"  Failed IDs: {error_ids[:10]}{'…' if len(error_ids) > 10 else ''}")
    if not dry_run:
        print(f"Output    : {output_file}")
    print("─" * 60)
