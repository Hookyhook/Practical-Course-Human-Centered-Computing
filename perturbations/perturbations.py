import json
import random
from pathlib import Path
from openai import OpenAI

# ── config ────────────────────────────────────────────────────────────────────

PROMPTS = json.loads((Path(__file__).parent / "perturbation_prompts.json").read_text())
SYSTEM  = PROMPTS["system_prompt"]

client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")

# ── llm helper ────────────────────────────────────────────────────────────────

def _llm(instruction: str, text: str) -> str:
    resp = client.chat.completions.create(
        model="qwen/qwen3.6-35b-a3b",
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
    "a": "4", "e": "3", "i": "1", "o": "0",
    "s": "5", "t": "7", "l": "1", "g": "9",
}

_CLICKBAIT_PREFIXES = [
    "EXPOSED:", "BREAKING:", "SHOCKING:", "BOMBSHELL:",
    "YOU WON'T BELIEVE THIS:", "THEY DON'T WANT YOU TO KNOW:",
]

# OCR: visually similar character confusions, keyed by original char
_OCR_MAP = {
    "o": "0", "O": "0",
    "l": "1", "i": "!",
    "e": "3", "a": "@",
    "s": "5", "t": "7",
    "g": "q", "b": "6",
}

def _ocr_artifacts(text: str) -> str:
    words = text.split()
    used_subs = set()
    applied = 0
    result = list(words)

    for wi, word in enumerate(words):
        if applied >= 3:
            break
        for ci, char in enumerate(word):
            sub = _OCR_MAP.get(char)
            if sub and sub not in used_subs:
                chars = list(result[wi])
                chars[ci] = sub
                result[wi] = "".join(chars)
                used_subs.add(sub)
                applied += 1
                break  # one substitution per word

    return " ".join(result)


def _emoji_disruptive(text: str) -> str:
    words = text.split()
    if len(words) < 2:
        return text
    pos = random.randint(1, len(words) - 1)
    words.insert(pos, random.choice(_DISRUPTIVE_EMOJIS))
    return " ".join(words)


def _homoglyphs(text: str) -> str:
    chars = list(text)
    eligible = [i for i, c in enumerate(chars) if c.lower() in _HOMOGLYPH_MAP]
    for i in random.sample(eligible, min(3, len(eligible))):
        chars[i] = _HOMOGLYPH_MAP[chars[i].lower()]
    return "".join(chars)


def _leetspeak(text: str) -> str:
    chars = list(text)
    eligible = [i for i, c in enumerate(chars) if c.lower() in _LEET_MAP]
    for i in random.sample(eligible, min(3, len(eligible))):
        chars[i] = _LEET_MAP[chars[i].lower()]
    return "".join(chars)


def _clickbait_rule(text: str) -> str:
    words = text.split()
    capitalised = " ".join(
        w.upper() if w.isalpha() and len(w) > 4 else w for w in words
    )
    return f"{random.choice(_CLICKBAIT_PREFIXES)} {capitalised}"


def _back_translate(text: str, pivot: str) -> str:
    from deep_translator import GoogleTranslator
    mid = GoogleTranslator(source="en", target=pivot).translate(text)
    return GoogleTranslator(source=pivot, target="en").translate(mid)

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
    "A1_emoji_disruptive":    lambda text, _: _emoji_disruptive(text),
    "A3_ocr_artifacts":       lambda text, _: _ocr_artifacts(text),
    "C1_homoglyphs":          lambda text, _: _homoglyphs(text),
    "C2_leetspeak":           lambda text, _: _leetspeak(text),
    "D2_clickbait_rule":      lambda text, _: _clickbait_rule(text),
    "D3_back_translation_de": lambda text, _: _back_translate(text, "de"),
    "D3_back_translation_es": lambda text, _: _back_translate(text, "es"),
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
