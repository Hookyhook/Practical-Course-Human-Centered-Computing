#!/usr/bin/env python3
"""
verify_perturbations.py

Quality-gate verifier for perturbed claims, following the verification
methodology from "When Claims Evolve" (Magomere et al.)

Calls a local LM Studio instance (OpenAI-compatible API, Qwen3-30B-A3B) to judge:
  1. Was the perturbation correctly applied?
  2. Was the factual meaning preserved / altered as expected per perturbation type?

Output CSV adds four columns to the original:
  perturbation_applied  (bool)
  meaning_preserved     (bool)
  verified              (bool)  — true iff both criteria are met as expected
  verify_error          (str)   — non-empty if something went wrong

Usage:
    python verify_perturbations.py              # full run (resumes from checkpoint)
    python verify_perturbations.py --dry-run    # first 5 rows, prints to console only
    python verify_perturbations.py --limit 20   # process only N rows (for testing)
"""

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

try:
    from openai import OpenAI
except ImportError:
    sys.exit("openai package not found. Run: pip install openai --break-system-packages")

# ── Configuration ─────────────────────────────────────────────────────────────

LM_STUDIO_BASE_URL = "http://localhost:1234/v1"
LM_STUDIO_API_KEY  = "lm-studio"      # LM Studio ignores the key but the client requires one
MODEL_NAME         = "qwen/qwen3.6-35b-a3b"  # Match exactly what LM Studio displays as the model ID

SCRIPT_DIR   = Path(__file__).parent
INPUT_FILE   = SCRIPT_DIR / "perturbed_claims_100.csv"
OUTPUT_FILE  = SCRIPT_DIR / "verified_perturbations.csv"
CHECKPOINT   = SCRIPT_DIR / "verification_checkpoint.json"

DRY_RUN_LIMIT = 5    # rows shown in --dry-run mode
REQUEST_DELAY = 0.5  # seconds between API calls (gives the GPU breathing room)

# ── Perturbation metadata ──────────────────────────────────────────────────────
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


def build_user_prompt(row: dict) -> str:
    name = row["perturbation_name"]
    meta = PERTURBATION_META.get(name)
    if meta is None:
        raise ValueError(f"Unknown perturbation type: '{name}'")

    expect_label = (
        "PRESERVED — the core factual meaning should be unchanged"
        if meta["expect_preserved"]
        else "INTENTIONALLY ALTERED — the meaning is expected to have shifted"
    )

    return f"""## Perturbation type: {name}

### Verification criteria:
{meta["description"]}

### Expected outcome for 'meaning_preserved': {expect_label}

---

### Original text:
{row["original_text"]}

---

### Perturbed text:
{row["perturbed_text"]}"""


# ── Checkpoint helpers ────────────────────────────────────────────────────────

def load_checkpoint(path: Path) -> set:
    """Return set of already-processed (claim_id, perturbation_name) tuples."""
    if path.exists():
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return {tuple(item) for item in data}
    return set()


def save_checkpoint(path: Path, done: set):
    with open(path, "w", encoding="utf-8") as f:
        json.dump([list(item) for item in done], f)


# ── LM Studio call ────────────────────────────────────────────────────────────

def call_lm_studio(client: OpenAI, user_prompt: str, debug: bool = False) -> dict:
    """Call LM Studio with structured output — response is guaranteed to match RESPONSE_SCHEMA."""
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_prompt},
        ],
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


# ── Main processing loop ──────────────────────────────────────────────────────

def process(dry_run: bool, limit: int | None):
    client = OpenAI(base_url=LM_STUDIO_BASE_URL, api_key=LM_STUDIO_API_KEY)

    # Load rows
    with open(INPUT_FILE, newline="", encoding="utf-8-sig") as f:
        all_rows = list(csv.DictReader(f))

    if limit:
        all_rows = all_rows[:limit]

    total = len(all_rows)

    # Load checkpoint (skip in dry-run so we always re-process)
    done: set = set() if dry_run else load_checkpoint(CHECKPOINT)
    skipped = sum(
        1 for r in all_rows
        if (r["claim_id"], r["perturbation_name"]) in done
    )
    remaining = total - skipped
    print(f"Total rows : {total}")
    print(f"Already done (checkpoint): {skipped}")
    print(f"To process : {remaining}")
    if dry_run:
        print(f"[DRY RUN] Will process first {DRY_RUN_LIMIT} rows and print results — no files written.\n")
        all_rows = all_rows[:DRY_RUN_LIMIT]

    # Prepare output file (append if checkpoint exists, write fresh otherwise)
    fieldnames = list(all_rows[0].keys()) + [
        "perturbation_applied", "meaning_preserved", "verified", "verify_error"
    ]
    output_mode = "a" if (OUTPUT_FILE.exists() and not dry_run and skipped > 0) else "w"
    out_file = None
    writer   = None

    if not dry_run:
        out_file = open(OUTPUT_FILE, output_mode, newline="", encoding="utf-8")
        writer   = csv.DictWriter(out_file, fieldnames=fieldnames)
        if output_mode == "w":
            writer.writeheader()

    # ── Row loop ──────────────────────────────────────────────────────────────
    processed = 0
    errors    = 0

    for i, row in enumerate(all_rows):
        row_key = (row["claim_id"], row["perturbation_name"])

        if row_key in done:
            print(f"[{i+1:4d}/{total}] SKIP  {row['claim_id']} | {row['perturbation_name']}")
            continue

        label = f"[{i+1:4d}/{total}] {row['claim_id']} | {row['perturbation_name']}"
        print(f"{label} ...", end=" ", flush=True)

        result = {
            "perturbation_applied": None,
            "meaning_preserved":    None,
            "verified":             None,
            "verify_error":         "",
        }

        try:
            user_prompt = build_user_prompt(row)
            verdict     = call_lm_studio(client, user_prompt, debug=dry_run)

            # Validate required keys
            for required_key in ("perturbation_applied", "meaning_preserved", "verified"):
                if required_key not in verdict:
                    raise KeyError(f"Model response missing key: '{required_key}'")

            # Re-derive 'verified' from our own logic (don't fully trust model's self-assessment)
            meta              = PERTURBATION_META[row["perturbation_name"]]
            expected_preserved = meta["expect_preserved"]
            if expected_preserved is None:
                # neutral: only check that the perturbation was applied
                recomputed = bool(verdict["perturbation_applied"])
            else:
                recomputed = (
                    bool(verdict["perturbation_applied"]) and
                    (bool(verdict["meaning_preserved"]) == expected_preserved)
                )
            if recomputed != verdict["verified"]:
                verdict["verified"] = recomputed  # override with ground-truth logic

            result.update(verdict)

            status = "✓ PASS" if verdict["verified"] else "✗ FAIL"
            print(status)

        except json.JSONDecodeError as e:
            result["verify_error"] = f"JSON parse error: {e}"
            errors += 1
            print(f"ERROR (JSON) | {e}")
        except KeyError as e:
            result["verify_error"] = f"Missing key: {e}"
            errors += 1
            print(f"ERROR (key)  | {e}")
        except Exception as e:
            result["verify_error"] = str(e)
            errors += 1
            print(f"ERROR        | {e}")

        # ── Write / print result ──────────────────────────────────────────────
        out_row = {**row, **result}

        if dry_run:
            print(f"         → perturbation_applied={result['perturbation_applied']}, "
                  f"meaning_preserved={result['meaning_preserved']}, "
                  f"verified={result['verified']}")
            if result["verify_error"]:
                print(f"         → error: {result['verify_error']}")
            print()
        else:
            writer.writerow(out_row)
            out_file.flush()
            done.add(row_key)
            save_checkpoint(CHECKPOINT, done)

        processed += 1
        time.sleep(REQUEST_DELAY)

    # ── Summary ───────────────────────────────────────────────────────────────
    if out_file:
        out_file.close()

    print("\n" + "─" * 60)
    print(f"Processed : {processed} rows")
    print(f"Errors    : {errors}")
    if not dry_run:
        print(f"Output    : {OUTPUT_FILE}")
        print(f"Checkpoint: {CHECKPOINT}")
    print("─" * 60)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Verify perturbed claims via LM Studio (Qwen3-30B-A3B)."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=f"Process first {DRY_RUN_LIMIT} rows only; print results, write nothing.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="Process only the first N rows (useful for staging runs).",
    )
    args = parser.parse_args()

    process(dry_run=args.dry_run, limit=args.limit)
