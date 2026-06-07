import json
import random
from pathlib import Path

# ── config ────────────────────────────────────────────────────────────────────

PROMPTS = json.loads((Path(__file__).parent / "perturbation_prompts.json").read_text())
SYSTEM  = PROMPTS["system_prompt"]

LM_STUDIO_BASE_URL = "http://localhost:1234/v1"
LM_STUDIO_API_KEY  = "lm-studio"
MODEL_NAME         = "qwen/qwen3.6-35b-a3b"

_client = None  # lazy-initialised on first LLM call

def _get_client():
    global _client
    if _client is None:
        from openai import OpenAI
        _client = OpenAI(
            base_url=LM_STUDIO_BASE_URL,
            api_key=LM_STUDIO_API_KEY,
            timeout=60.0,        # fail fast instead of hanging indefinitely
        )
    return _client


def check_connection() -> tuple[bool, str]:
    """Ping LM Studio and return (ok, message).  Call this at script startup."""
    try:
        client = _get_client()
        models = client.models.list()
        names = [m.id for m in models.data]
        if not names:
            return False, "LM Studio is reachable but no model is loaded."
        if MODEL_NAME not in names:
            return False, (
                f"Connected to LM Studio, but model '{MODEL_NAME}' is not loaded.\n"
                f"  Available: {names}"
            )
        return True, f"LM Studio OK — model '{MODEL_NAME}' is loaded."
    except Exception as exc:
        return False, (
            f"Cannot reach LM Studio at {LM_STUDIO_BASE_URL}.\n"
            f"  Error: {exc}\n"
            f"  Make sure LM Studio is running and '{MODEL_NAME}' is loaded."
        )

# ── llm helper ────────────────────────────────────────────────────────────────

def _llm(instruction: str, text: str) -> str:
    client = _get_client()
    resp = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user",   "content": f"{instruction}\n\n---\nINPUT TEXT (this is the complete text to transform — it may contain quotes, dialogue, or multiple sentences; apply the perturbation to the entire text, not just one part of it):\n{text}"},
        ],
    )
    return json.loads(resp.choices[0].message.content)["perturbed_text"]

def _apply_llm(name: str, lang: str, text: str, section: str = "perturbations") -> str:
    bucket = PROMPTS[section][name]
    instruction = bucket.get(lang, bucket["EN"])
    return _llm(instruction, text)

# ── rule-based implementations ────────────────────────────────────────────────

_DISRUPTIVE_EMOJIS = ["😱", "🚨", "⚠️", "🔥", "💥", "❗", "🛑", "😤"]

_HOMOGLYPH_MAP = {
    # Latin → Cyrillic lookalikes (works for all Latin-script languages)
    "a": "а", "e": "е", "o": "о", "p": "р",
    "c": "с", "x": "х", "y": "у", "i": "і",
    # Arabic confusable pairs (within-script visual similarity)
    "ي": "ى",   # ya  ↔ alef maqsura (identical without dots)
    "ه": "ة",   # ha  ↔ ta marbuta   (similar shape)
    "و": "ﻭ",   # waw ↔ presentation form
    "ا": "ﺍ",   # alef ↔ alef isolated form
    "ب": "ت",   # ba  ↔ ta (same base shape, different dots)
    "ن": "ي",   # nun ↔ ya (similar curve)
    # Chinese visually similar pairs (simplified ↔ lookalike)
    "人": "入", "土": "士", "己": "已", "末": "未",
    "干": "于", "大": "太", "日": "目", "力": "刀",
    "田": "由", "口": "囗",
    # Devanagari / Hindi visually similar pairs
    "म": "भ", "ग": "प", "ह": "ब", "क": "ख",
    "ण": "न", "ध": "घ", "थ": "ध",
    # Malayalam visually similar pairs
    "ര": "ദ", "ക": "ഥ", "ജ": "ഞ", "ന": "ഩ",
    "ല": "ള", "പ": "ഫ",
}

_LEET_MAP = {
    # Latin
    "a": "4", "e": "3", "i": "1", "o": "0",
    "s": "5", "t": "7", "l": "1", "g": "9",
    # Arabic (Arabizi — established internet transliteration)
    "ع": "3", "ح": "7", "ق": "9", "خ": "5", "ء": "2",
    "ز": "7", "ص": "9", "ط": "6", "غ": "3",
    # Chinese number-character substitutions (internet slang)
    "一": "1", "二": "2", "三": "3", "四": "4", "五": "5",
    "六": "6", "七": "7", "八": "8", "九": "9", "零": "0",
    # Devanagari / Hindi (numeral-shape substitutions)
    "ए": "3",   # e-vowel ↔ 3
    "ओ": "0",   # o-vowel ↔ 0
    "इ": "1",   # i-vowel ↔ 1
    "अ": "4",   # a-vowel ↔ 4
    # Malayalam (numeral-shape substitutions)
    "ഒ": "0",   # o ↔ 0
    "ഇ": "1",   # i ↔ 1
    "ഏ": "3",   # e ↔ 3
}

# OCR: visually similar character confusions, keyed by original char
_OCR_MAP = {
    # Latin
    "o": "0", "O": "0",
    "l": "1", "i": "!",
    "e": "3", "a": "@",
    "s": "5", "t": "7",
    "g": "q", "b": "6",
    # Arabic (common OCR confusion pairs)
    "ر": "ز", "د": "ذ", "ح": "ج", "ب": "ت", "ن": "ي",
    # Chinese (visually similar character pairs)
    "己": "已", "土": "士", "末": "未", "人": "入",
    "大": "太", "天": "夭", "干": "于", "日": "目",
    # Devanagari / Hindi (visually similar pairs)
    "म": "भ", "ग": "प", "ह": "ब", "श": "ष",
    # Malayalam (visually similar pairs)
    "ര": "ദ", "ക": "ഥ", "ജ": "ഞ",
}


def _ocr_artifacts(text: str) -> str:
    chars = list(text)
    eligible = [i for i, c in enumerate(chars) if c in _OCR_MAP]
    # apply up to 3 substitutions, one per unique source character
    seen_sources: set[str] = set()
    applied = 0
    for i in eligible:
        if applied >= 3:
            break
        src = chars[i]
        if src not in seen_sources:
            chars[i] = _OCR_MAP[src]
            seen_sources.add(src)
            applied += 1
    return "".join(chars)


def _emoji_disruptive(text: str) -> str:
    words = text.split()
    emoji = random.choice(_DISRUPTIVE_EMOJIS)
    if len(words) >= 2:
        # Spaced scripts (Latin, Arabic, etc.): insert between words
        pos = random.randint(1, len(words) - 1)
        words.insert(pos, emoji)
        return " ".join(words)
    else:
        # Non-spaced scripts (CJK): insert at a random character position
        if len(text) < 2:
            return text
        pos = random.randint(1, len(text) - 1)
        return text[:pos] + emoji + text[pos:]


def _homoglyphs(text: str) -> str:
    chars = list(text)
    eligible = [i for i, c in enumerate(chars) if c.lower() in _HOMOGLYPH_MAP]
    for i in random.sample(eligible, min(3, len(eligible))):
        chars[i] = _HOMOGLYPH_MAP[chars[i].lower()]
    return "".join(chars)


def _leetspeak(text: str) -> str:
    chars = list(text)
    eligible = [i for i, c in enumerate(chars) if c in _LEET_MAP or c.lower() in _LEET_MAP]
    for i in random.sample(eligible, min(3, len(eligible))):
        key = chars[i] if chars[i] in _LEET_MAP else chars[i].lower()
        chars[i] = _LEET_MAP[key]
    return "".join(chars)


# Google Translate language codes for each pipeline lang tag
_LANG_CODE_MAP: dict[str, str] = {
    "EN": "en", "DE": "de", "ES": "es", "FR": "fr",
    "PL": "pl", "HBS": "sr", "HI": "hi", "AR": "ar",
    "ML": "ml", "ZH": "zh-CN", "PT": "pt",
}

def _back_translate(text: str, pivot: str, lang: str = "EN") -> str:
    from deep_translator import GoogleTranslator
    src = _LANG_CODE_MAP.get(lang.upper(), "auto")
    mid = GoogleTranslator(source=src, target=pivot).translate(text)
    return GoogleTranslator(source=pivot, target=src).translate(mid)

# ── dispatcher ────────────────────────────────────────────────────────────────

_LLM_PERTURBATIONS = [
    "A1_emoji_relevant",
    "A2_hashtagification",
    "A4_stt_artifacts",
    "B1_qualifier_removal",
    "B2_temporal_drift",
    "C3_word_splitting",
    "D2_clickbait_llm",
    "E2_presupposition",
]

_RULE_PERTURBATIONS = {
    "A1_emoji_disruptive":    lambda text, _:    _emoji_disruptive(text),
    "A3_ocr_artifacts":       lambda text, _:    _ocr_artifacts(text),
    "C1_homoglyphs":          lambda text, _:    _homoglyphs(text),
    "C2_leetspeak":           lambda text, _:    _leetspeak(text),
    "D3_back_translation_it": lambda text, lang: _back_translate(text, "it", lang),
    "D3_back_translation_ru": lambda text, lang: _back_translate(text, "ru", lang),
}

# ── paper perturbations (EN only) ─────────────────────────────────────────────
# Prompts sourced from: https://github.com/JabezNzomo99/claim-matching-robustness
# Typos are LLM-based (matching the paper) — the paper uses social-media-style
# abbreviations and phonetic spelling, not simple character-level mutations.

_PAPER_LLM_PERTURBATIONS = [
    "P_negation_low",
    "P_negation_high",
    "P_entity_low",
    "P_entity_high",
    "P_llm_rewrite_low",
    "P_llm_rewrite_high",
    "P_dialect_aae",
    "P_dialect_jamaican",
    "P_dialect_pidgin",
    "P_dialect_singlish",
    "P_typos_low",
    "P_typos_high",
]

_PAPER_RULE_PERTURBATIONS: dict = {}   # all paper perturbations are now LLM-based


def perturb_all(text: str, lang: str = "EN") -> dict[str, str]:
    results = {}

    # our perturbations — all languages
    for name in _LLM_PERTURBATIONS:
        results[name] = _apply_llm(name, lang, text)
    for name, fn in _RULE_PERTURBATIONS.items():
        results[name] = fn(text, lang)

    # paper perturbations — EN only
    if lang == "EN":
        for name in _PAPER_LLM_PERTURBATIONS:
            results[name] = _apply_llm(name, lang, text, section="paper_perturbations")
        for name, fn in _PAPER_RULE_PERTURBATIONS.items():
            results[name] = fn(text, lang)

    return results

# ── cli ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    text = sys.argv[1] if len(sys.argv) > 1 else "The government announced a new policy on climate change."
    lang = sys.argv[2] if len(sys.argv) > 2 else "EN"

    print(f"\nOriginal [{lang}]: {text}\n{'─' * 60}")
    for name, result in perturb_all(text, lang).items():
        print(f"\n[{name}]\n{result}")
