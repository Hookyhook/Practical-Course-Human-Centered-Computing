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

# ── LM Studio server settings (shared with the perturbation pipeline) ────────
SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))

from perturbations import (  # noqa: E402
    LM_STUDIO_API_KEY,
    LM_STUDIO_BASE_URL,
    LLM_EXTRA_BODY,
    MODEL_NAME,
)

# ── Configuration ─────────────────────────────────────────────────────────────

DATA_DIR      = SCRIPT_DIR.parent / "data"
REQUEST_DELAY = 0.3   # seconds between calls (lower than perturber — verifier calls are cheaper)
DRY_RUN_ROWS  = 5

VERIFIER_TEMPERATURE = 0.2

# llm.prediction.structured from Verifier preset
RESPONSE_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object",
    "properties": {
        "perturbation_applied": {
            "type": "boolean",
            "description": (
                "True if the specified transformation is clearly present in the perturbed text. "
                "False if the texts are identical or the transformation is absent."
            ),
        },
        "meaning_preserved": {
            "type": "boolean",
            "description": (
                "True if the core factual claim is unchanged between original and perturbed. "
                "False if a factual element (entity, date, qualifier, implication) has shifted."
            ),
        },
        "verified": {
            "type": "boolean",
            "description": (
                "True if perturbation_applied=true AND meaning_preserved matches what is "
                "expected for this perturbation type (stated in the criteria)."
            ),
        },
    },
    "required": ["perturbation_applied", "meaning_preserved", "verified"],
    "additionalProperties": False,
}

VERIFIER_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "verifier_response",
        "strict": True,
        "schema": RESPONSE_SCHEMA,
    },
}

#
# description      → criteria sent to the verifier as its task instruction
# expect_preserved → True  = meaning MUST be preserved  (families A, C, D, P_rewrite/dialect/typos)
#                    False = meaning IS intentionally altered (families B, E2, P_negation/entity)

PERTURBATION_META = {

    # ── Family A: Social Media Noise ──────────────────────────────────────────

    "A1_emoji_relevant": {
        "description": (
            "Exactly 2 topic-relevant emojis were appended to the end of the claim. "
            "Check: (1) exactly 2 emojis are present at the end of the text (not inside it), "
            "(2) both emojis relate thematically to the topic of the claim, "
            "(3) the factual content is completely unchanged."
        ),
        "expect_preserved": True,
    },
    "A1_emoji_disruptive": {
        "description": (
            "A random, unrelated emoji was inserted mid-claim to disrupt reading. "
            "Check: (1) at least one emoji appears inside the text body, "
            "(2) the emoji feels unrelated or jarring in context, "
            "(3) the factual content is completely unchanged."
        ),
        "expect_preserved": True,
    },
    "A2_hashtagification": {
        "description": (
            "The original claim text was kept intact, then exactly 3 topic hashtags and "
            "exactly 3 generic spam hashtags were appended (6 hashtags total). "
            "Check: (1) the original claim text is present and unmodified at the start, "
            "(2) exactly 3 topic-derived hashtags appear (e.g. #Government #Policy), "
            "(3) exactly 3 generic spam hashtags follow (e.g. #BreakingNews #Viral #MustRead), "
            "(4) the total hashtag count is 6."
        ),
        "expect_preserved": True,
    },
    "A3_ocr_artifacts": {
        "description": (
            "OCR scanning errors were simulated by substituting characters "
            "(e.g. '@' for 'a', '5' for 's', '!' for 'i', '0' for 'o', '1' for 'l'). "
            "Check: (1) at least one such character substitution is visible, "
            "(2) the text is still parseable by a human, "
            "(3) the factual core meaning is unchanged."
        ),
        "expect_preserved": True,
    },
    "A4_stt_artifacts": {
        "description": (
            "Speech-to-text errors were simulated: phonetic substitutions, "
            "merged words, or missing apostrophes "
            "(e.g. 'u' for 'you', homophones, contracted word boundaries). "
            "Check: (1) at least one such error is visible, "
            "(2) the text is still parseable by a human."
        ),
        "expect_preserved": None,  # neutral — meaning may or may not shift; only application is checked
    },

    # ── Family B: Semantic & Argumentative (meaning intentionally altered) ────

    "B1_qualifier_removal": {
        "description": (
            "A hedging qualifier was removed or weakened to make the claim more assertive. "
            "Qualifiers include: 'allegedly', 'reportedly', 'some claim', 'according to', "
            "'experts say', 'it is believed that', 'possibly', 'may have'. "
            "Check: (1) a qualifier present in the original is absent or softened in the perturbed version, "
            "OR the claim has been made more absolute than the original, "
            "(2) the statement is now more assertive or certain than the source. "
            "IMPORTANT: meaning IS intentionally altered here — that is expected and correct."
        ),
        "expect_preserved": False,
    },
    "B2_temporal_drift": {
        "description": (
            "Dates/years were shifted forward by 3 years AND a short modern-context anchor phrase "
            "was appended (e.g. 'This resurfaced recently and is being widely shared.'). "
            "If the original had no dates, only the anchor phrase was added. "
            "Check: (1a) IF dates were present — each year/date in the original has been increased "
            "by approximately 3 years in the perturbed text; "
            "(1b) IF no dates were present — the text is otherwise unchanged; "
            "(2) a short anchor phrase has been appended to the end in either case, "
            "(3) no other factual content was added or removed beyond the date shift and anchor phrase. "
            "IMPORTANT: meaning IS intentionally altered here — that is expected and correct."
        ),
        "expect_preserved": False,
    },

    # ── Family C: Adversarial / Character-Level Evasion ───────────────────────

    "C1_homoglyphs": {
        "description": (
            "One or more characters were replaced with visually similar Unicode lookalikes "
            "(e.g. Cyrillic 'е' substituted for Latin 'e', 'а' for 'a'). "
            "Check: (1) at least one character in the perturbed text is a Unicode lookalike "
            "(this may be subtle — compare carefully), "
            "(2) the text looks nearly identical to the original at first glance, "
            "(3) the factual meaning is unchanged."
        ),
        "expect_preserved": True,
    },
    "C2_leetspeak": {
        "description": (
            "Letters were substituted with numbers or symbols in leetspeak style "
            "(e.g. '3' for 'e', '4' for 'a', '1' for 'l', '0' for 'o', '@' for 'a'). "
            "Check: (1) at least one such substitution is clearly present, "
            "(2) the claim is still decipherable by a human, "
            "(3) the factual meaning is unchanged."
        ),
        "expect_preserved": True,
    },
    "C3_word_splitting": {
        "description": (
            "Exactly 2 important content words were split mid-token by inserting a space "
            "at a natural-looking position (e.g. 'vaccination' → 'vaccin ation', "
            "'government' → 'govern ment'). "
            "Check: (1) exactly 2 words appear split at unnatural positions, "
            "(2) the splits resemble plausible typos or formatting errors, "
            "(3) the factual meaning is unchanged."
        ),
        "expect_preserved": True,
    },

    # ── Family D: Style / Register ─────────────────────────────────────────────

    "D2_clickbait_llm": {
        "description": (
            "The claim was rewritten in clickbait style by an LLM: one sensationalist hook phrase "
            "was prepended (e.g. 'EXPOSED:', 'BREAKING:', 'You won't believe this:') and exactly "
            "3 key nouns or verbs in the body were written in ALL CAPS. "
            "Check: (1) a sensationalist hook phrase is present at the start, "
            "(2) exactly 3 words in the body are in ALL CAPS while all others keep their original casing, "
            "(3) the underlying factual claim is preserved — no new facts were invented."
        ),
        "expect_preserved": True,
    },
    "D3_back_translation_it": {
        "description": (
            "The claim was translated to Italian and back to its original language, "
            "producing subtle paraphrase artifacts from the round-trip. "
            "Check: (1) the wording differs slightly from the original "
            "(different word choices, minor restructuring typical of translation round-trips), "
            "(2) the difference is consistent with translation artifacts, not intentional edits, "
            "(3) the factual core — all entities, dates, quantities, and claims — is fully preserved."
        ),
        "expect_preserved": True,
    },
    "D3_back_translation_ru": {
        "description": (
            "The claim was translated to Russian and back to its original language, "
            "producing subtle paraphrase artifacts from the round-trip. "
            "Check: (1) the wording differs slightly from the original "
            "(different word choices, minor restructuring typical of translation round-trips), "
            "(2) the difference is consistent with translation artifacts, not intentional edits, "
            "(3) the factual core — all entities, dates, quantities, and claims — is fully preserved."
        ),
        "expect_preserved": True,
    },

    # ── Family E: Rhetorical ───────────────────────────────────────────────────

    "E2_presupposition": {
        "description": (
            "A hidden presupposition was embedded into the claim — a framing that implies "
            "something beyond what the original states "
            "(e.g. 'Why did X cause Y?' presupposes X caused Y; "
            "'Despite the evidence, X claims...' presupposes evidence exists against X). "
            "Check: (1) the perturbed text implies or presupposes something not stated in the original, "
            "(2) the surface phrasing has shifted the claim's implied meaning or burden of proof. "
            "IMPORTANT: meaning IS intentionally altered here — that is expected and correct."
        ),
        "expect_preserved": False,
    },

    # ── Paper perturbations (EN only) ─────────────────────────────────────────
    # Meaning-altering: P_negation_*, P_entity_*
    # Meaning-preserving: P_llm_rewrite_*, P_dialect_*, P_typos_*

    "P_negation_low": {
        "description": (
            "A single negation word ('not', 'never', or equivalent) was inserted to make the claim "
            "assert the opposite of the original. "
            "Check: (1) a negation is clearly present in the perturbed text that was absent in the original, "
            "(2) the core assertion is now reversed. "
            "IMPORTANT: meaning IS intentionally altered here — that is expected and correct."
        ),
        "expect_preserved": False,
    },
    "P_negation_high": {
        "description": (
            "A double negation was applied: the core assertion was negated, then wrapped in a denial frame "
            "('It is not true that [negated claim].'). "
            "Check: (1) the text begins with or contains 'It is not true that', "
            "(2) the embedded claim is itself negated, "
            "(3) the overall assertion is the opposite of the original. "
            "IMPORTANT: meaning IS intentionally altered here — that is expected and correct."
        ),
        "expect_preserved": False,
    },
    "P_entity_low": {
        "description": (
            "The single most prominent named entity (person, organisation, or location) was replaced "
            "with a different but plausible entity of the same type. "
            "Check: (1) exactly one named entity has changed to a different but type-consistent entity, "
            "(2) all other words are unchanged. "
            "IMPORTANT: meaning IS intentionally altered here — that is expected and correct."
        ),
        "expect_preserved": False,
    },
    "P_entity_high": {
        "description": (
            "Every named entity in the claim (persons, organisations, locations, dates) was replaced "
            "with a different but plausible entity of the same type. "
            "Check: (1) multiple named entities have been substituted, "
            "(2) all non-entity words are unchanged. "
            "IMPORTANT: meaning IS intentionally altered here — that is expected and correct."
        ),
        "expect_preserved": False,
    },
    "P_llm_rewrite_low": {
        "description": (
            "The claim was minimally paraphrased: one or two words were swapped for synonyms "
            "or a short phrase was lightly restructured. "
            "Check: (1) the wording differs slightly from the original (at least one word changed), "
            "(2) the texts are not identical, "
            "(3) all factual content — entities, dates, quantities, relationships — is fully preserved."
        ),
        "expect_preserved": True,
    },
    "P_llm_rewrite_high": {
        "description": (
            "The claim was heavily paraphrased: vocabulary, sentence structure, and word order were "
            "changed as much as possible while preserving the meaning. "
            "Check: (1) the wording differs substantially from the original, "
            "(2) the factual content — entities, dates, quantities, relationships — is fully preserved."
        ),
        "expect_preserved": True,
    },
    "P_dialect_aae": {
        "description": (
            "The claim was rewritten in African American English (AAVE). "
            "Check: (1) AAVE grammatical or lexical features are present "
            "(e.g. habitual 'be', copula deletion, double negatives, idiomatic vocabulary), "
            "(2) all factual content is preserved."
        ),
        "expect_preserved": True,
    },
    "P_dialect_jamaican": {
        "description": (
            "The claim was rewritten in Jamaican Patois. "
            "Check: (1) Patois features are present "
            "(e.g. 'mi' for I/me, 'dem' for they, 'nuh' for negation, 'fi' for to/for), "
            "(2) all factual content is preserved."
        ),
        "expect_preserved": True,
    },
    "P_dialect_pidgin": {
        "description": (
            "The claim was rewritten in Nigerian Pidgin English. "
            "Check: (1) Naija Pidgin features are present "
            "(e.g. 'e' for he/she/it, 'dem' for they, 'dey' for is/are, 'wey' as relative pronoun), "
            "(2) all factual content is preserved."
        ),
        "expect_preserved": True,
    },
    "P_dialect_singlish": {
        "description": (
            "The claim was rewritten in Singlish (Singaporean English). "
            "Check: (1) Singlish features are present "
            "(e.g. discourse particles 'lah', 'leh', 'meh', 'lor', topic-fronting, copula omission), "
            "(2) all factual content is preserved."
        ),
        "expect_preserved": True,
    },
    "P_typos_low": {
        "description": (
            "The claim was rewritten by an LLM introducing a small number of social-media-style typos "
            "and abbreviations (1–2 changes, low edit ratio). "
            "Common patterns: letter swaps, shortened words (e.g. 'sighned' for 'signed', "
            "'EO' for 'executive order', 'da' for 'the'). "
            "Check: (1) 1–2 words show typos or informal abbreviations, "
            "(2) the claim is still easily readable, "
            "(3) the factual meaning is unchanged."
        ),
        "expect_preserved": True,
    },
    "P_typos_high": {
        "description": (
            "The claim was rewritten by an LLM introducing multiple social-media-style typos "
            "and abbreviations throughout (high edit ratio). "
            "Common patterns: phonetic spelling, text-speak (e.g. '2' for 'to', 'u' for 'you', "
            "'frm' for 'from', 'signz' for 'signs'), casual misspellings across several words. "
            "Check: (1) multiple words are altered with typos or abbreviations, "
            "(2) the claim still conveys the same core meaning even if messy, "
            "(3) the style reads like a hasty social media post."
        ),
        "expect_preserved": True,
    },
}

# ── System prompt ─────────────────────────────────────────────────────────────
#
# Kept focused on the task only — output format is handled by RESPONSE_SCHEMA.

SYSTEM_PROMPT = """You are a quality auditor for a fact-checking research pipeline.
Your task is to verify whether a text perturbation was correctly applied to a claim.

You will receive:
- The original claim text
- The perturbed claim text
- The perturbation type and its specific verification criteria

Respond with a JSON object containing exactly these three keys:
  perturbation_applied  (boolean) — was the perturbation actually applied?
  meaning_preserved     (boolean) — is the core factual meaning preserved?
  verified              (boolean) — true only if both criteria are met as expected

Rules:
- If the original and perturbed texts are identical, perturbation_applied must be false.
- The claim may be in any language (Arabic, German, Spanish, English) — evaluate in its original language.
- Do not translate the claim before judging."""

# ── LM Studio call ────────────────────────────────────────────────────────────

def call_lm_studio(client: OpenAI, user_prompt: str, debug: bool = False) -> dict:
    """Call LM Studio with structured output — response is guaranteed to match RESPONSE_SCHEMA."""
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_prompt},
        ],
        temperature=VERIFIER_TEMPERATURE,
        response_format=VERIFIER_RESPONSE_FORMAT,
        extra_body=LLM_EXTRA_BODY,
    )

    msg = response.choices[0].message

    # Qwen3 thinking models (LM Studio) put the actual response in reasoning_content
    # when structured output is active, leaving content empty.
    raw = msg.content or getattr(msg, "reasoning_content", "") or ""

    if debug:
        print(f"\n[DEBUG] finish_reason    : {response.choices[0].finish_reason}")
        print(f"[DEBUG] message.content  : {repr(msg.content)}")
        print(f"[DEBUG] reasoning_content: {repr(getattr(msg, 'reasoning_content', None))}")
        print(f"[DEBUG] raw used         : {repr(raw)}")

    return json.loads(raw.strip())


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
        meta          = PERTURBATION_META[name]
        expected_pres = meta["expect_preserved"]
        applied       = bool(verdict["perturbation_applied"])
        if expected_pres is None:
            # Neutral perturbation (A4_stt_artifacts): only check it was applied,
            # don't require a specific meaning_preserved outcome.
            recomputed = applied
        else:
            recomputed = applied and (bool(verdict["meaning_preserved"]) == expected_pres)
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
