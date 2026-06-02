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
        _client = OpenAI(base_url=LM_STUDIO_BASE_URL, api_key=LM_STUDIO_API_KEY)
    return _client

# ── llm helper ────────────────────────────────────────────────────────────────

def _llm(instruction: str, text: str) -> str:
    client = _get_client()
    resp = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user",   "content": f"{instruction}\n\n{text}"},
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
    "a": "а", "e": "е", "o": "о", "p": "р",
    "c": "с", "x": "х", "y": "у", "i": "і",
}

_LEET_MAP = {
    # Latin
    "a": "4", "e": "3", "i": "1", "o": "0",
    "s": "5", "t": "7", "l": "1", "g": "9",
    # Arabic (Arabizi — established internet transliteration)
    "ع": "3", "ح": "7", "ق": "9", "خ": "5", "ء": "2",
    # Chinese number-character substitutions (internet slang)
    "一": "1", "二": "2", "三": "3", "四": "4", "五": "5",
    "六": "6", "七": "7", "八": "8", "九": "9", "零": "0",
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

# ── paper perturbation rule-based implementations (EN only) ──────────────────

def _introduce_typos(text: str, n: int) -> str:
    """Introduce n character-level typos via insert / delete / substitute / transpose."""
    words = text.split()
    eligible = [i for i, w in enumerate(words) if len(w) > 3 and w.isalpha()]
    targets = random.sample(eligible, min(n, len(eligible)))
    for wi in targets:
        word = list(words[wi])
        op = random.choice(["insert", "delete", "substitute", "transpose"])
        if op == "insert":
            pos = random.randint(0, len(word))
            word.insert(pos, random.choice("abcdefghijklmnopqrstuvwxyz"))
        elif op == "delete" and len(word) > 2:
            word.pop(random.randint(0, len(word) - 1))
        elif op == "substitute":
            pos = random.randint(0, len(word) - 1)
            word[pos] = random.choice("abcdefghijklmnopqrstuvwxyz")
        elif op == "transpose" and len(word) > 1:
            pos = random.randint(0, len(word) - 2)
            word[pos], word[pos + 1] = word[pos + 1], word[pos]
        words[wi] = "".join(word)
    return " ".join(words)


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
    "D3_back_translation_de": lambda text, lang: _back_translate(text, "de", lang),
    "D3_back_translation_es": lambda text, lang: _back_translate(text, "es", lang),
}

# ── paper perturbations (EN only) ─────────────────────────────────────────────

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
]

_PAPER_RULE_PERTURBATIONS = {
    "P_typos_low":  lambda text, _: _introduce_typos(text, 1),
    "P_typos_high": lambda text, _: _introduce_typos(text, 3),
}


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
