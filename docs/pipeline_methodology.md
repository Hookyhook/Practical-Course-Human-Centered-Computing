# Perturbation Pipeline — Methodology

## 1. Overview

This document describes the multilingual claim perturbation pipeline built for the *When Claims Evolve* practical course (SoSe 2026). The pipeline takes preprocessed social media posts and fact-check claims, applies a taxonomy of systematic text perturbations, and verifies output quality using an automated judge. The goal is to generate a large-scale, multilingual dataset of perturbed claims for training and evaluating claim-matching and misinformation-detection systems.

---

## 2. Data & Preprocessing

### 2.1 Datasets

Two datasets are used:

- **MultiClaim** — a multilingual fact-checking dataset covering claims across 27 languages.
- **Posts** — a social media post dataset linked to fact-check verdicts.

### 2.2 Language Normalization

Both datasets use different language tagging conventions (ISO 639-1, ISO 639-3, free-text labels). A preprocessing step normalizes all language codes to a unified pipeline tag using the following mapping:

| Pipeline Tag | Language | ISO 639-3 |
|---|---|---|
| EN | English | eng |
| DE | German | deu |
| ES | Spanish | spa |
| FR | French | fra |
| PL | Polish | pol |
| HBS | Serbian/Croatian/Bosnian | srp/hrv/bos |
| HI | Hindi | hin |
| AR | Arabic | ara |
| ML | Malayalam | mal |
| ZH | Mandarin Chinese | zho |

Languages outside this set are dropped at preprocessing time. The normalization is handled by `preprocess.py`, which also deduplicates records and writes separate output CSVs per dataset.

### 2.3 Text Field

The field used for perturbation is `post_body` (posts dataset) and `original_text` (MultiClaim). All other original columns are preserved in the output.

---

## 3. Pipeline Architecture

```
preprocessed_posts.csv
        │
        ▼
   perturb_posts.py
        │   (calls utils.run_perturber)
        ▼
   perturbations.py  ◄──── perturbation_prompts.json
   ┌──────────────────────────────────────┐
   │  LLM-based perturbations (8 types)  │
   │  Rule-based perturbations (6 types) │
   │  Paper perturbations (12 types, EN) │
   └──────────────────────────────────────┘
        │
        ▼
  perturbed_posts.csv   (one row per post × perturbation_name)
        │
        ▼
  verify.py
        │
        ▼
  verified_perturbations.csv
```

Each input post produces N output rows, one per perturbation type. All original columns are preserved and three new columns are appended: `perturbation_name`, `perturbed_text`, `changed` (boolean), `error` (non-empty if generation failed).

---

## 4. Perturbation Taxonomy

Perturbations are organized into families. For non-English languages, 14 perturbation types are applied (8 LLM-based + 6 rule-based). English additionally receives 12 paper-sourced perturbation types.

### 4.1 Family A — Social Media Noise

These perturbations simulate surface-level noise typical of social media platforms: emoji usage, hashtagification, OCR scanning errors, and speech-to-text transcription artifacts.

---

#### A1_emoji_relevant *(LLM-based)*

Appends exactly 2 topic-relevant emojis to the end of the claim without changing any words. Tests whether downstream systems are affected by thematically consistent emoji embellishment.

**English prompt:**
> Append exactly 2 emojis to the end of this claim that are thematically relevant to its topic. Do not insert emojis inside the text body. Do not change any words.

*Prompts exist for all 10 pipeline languages.*

---

#### A1_emoji_disruptive *(rule-based)*

Inserts a random disruptive emoji (from `["😱", "🚨", "⚠️", "🔥", "💥", "❗", "🛑", "😤"]`) at a random mid-text position. For space-delimited scripts (Latin, Arabic) it inserts between words; for non-spaced scripts (CJK) it inserts at a random character position.

No LLM call is made. The perturbation is deterministic up to the random position choice.

---

#### A2_hashtagification *(LLM-based)*

Keeps the original claim text intact and appends exactly 3 topic-derived hashtags followed by exactly 3 generic spam hashtags (e.g. `#BreakingNews #MustRead #Viral`).

**English prompt:**
> Keep the original claim text exactly as written. Then append exactly 3 hashtag keywords derived from the most important nouns in the claim (e.g. #Government #Policy #ClimateChange) followed by exactly 3 generic spam hashtags (e.g. #BreakingNews #MustRead #Viral). Do not modify any word in the original sentence.

*Prompts exist for all 10 pipeline languages.*

---

#### A3_ocr_artifacts *(rule-based)*

Simulates OCR scanning errors by substituting up to 3 characters with visually similar lookalikes, applying at most one substitution per unique source character. The substitution map covers Latin, Arabic, Chinese, Devanagari, and Malayalam characters:

```
Latin:     o→0, l→1, i→!, e→3, a→@, s→5, t→7, g→q, b→6
Arabic:    ر→ز, د→ذ, ح→ج, ب→ت, ن→ي
Chinese:   己→已, 土→士, 末→未, 人→入, 大→太, 天→夭, 干→于, 日→目
Devanagari: म→भ, ग→प, ह→ब, श→ष
Malayalam: ര→ദ, ക→ഥ, ജ→ഞ
```

---

#### A4_stt_artifacts *(LLM-based)*

Replaces exactly 3 words with phonetically similar real words that a speech-to-text system would confuse. Each replacement must be an existing valid word in the target language — never a misspelling.

**English prompt:**
> Replace exactly 3 words in this claim with phonetically similar real words that a speech-to-text system would confuse (e.g. 'their' → 'there', 'policy' → 'police', 'won' → 'one', 'weather' → 'whether'). Every replacement must be a valid existing word — never a misspelling. You must always make at least 1 substitution — never return the original text unchanged.

*Prompts exist for all 10 pipeline languages, with language-specific homophone examples.*

> **Design note:** Meaning preservation is treated as **neutral** for this type. STT artifacts can realistically alter meaning (e.g. "vaccinated" → "back sin aided"), and this mirrors real-world transcription failures. Verification only checks that the perturbation was applied, not whether meaning was preserved.

---

### 4.2 Family B — Semantic / Argumentative

These perturbations intentionally alter the factual meaning of the claim. Both types have `expect_preserved = False` in the verifier.

---

#### B1_qualifier_removal *(LLM-based)*

Removes hedging and qualifying language (e.g. *allegedly, reportedly, according to, may, might, appears to*) to make the claim sound like a definitive statement of fact.

**English prompt:**
> Remove all hedging and qualifying language from this claim to make it sound like a definitive statement of fact. Target words and phrases such as: allegedly, reportedly, according to, sources say, may, might, could, would, appears to, seems to, is said to, claims that, suggests. If no qualifying language is present, return the text unchanged.

*Prompts exist for all 10 pipeline languages with language-specific qualifier lists.*

> **Known limitation:** Many social media posts contain no explicit qualifiers, causing the model to return the original text unchanged. This results in a high unchanged rate for B1 (~55% in our test set). This is expected behavior — posts that are already stated as fact simply have nothing to remove.

---

#### B2_temporal_drift *(LLM-based)*

Shifts every year or date in the claim forward by exactly 3 years and appends a short modern-context anchor phrase (e.g. *"This resurfaced recently and is being widely shared."*). If no dates are present, only the anchor phrase is added.

**English prompt:**
> Shift every year or date mentioned in this claim forward by exactly 3 years to create a plausible but incorrect timeline. Then append exactly one short phrase anchoring the claim to a modern context (e.g. 'This resurfaced recently and is being widely shared.' or 'Experts confirmed this again last month.'). If no dates are present, only append the modern anchor phrase.

*Prompts exist for all 10 pipeline languages.*

---

### 4.3 Family C — Character / Token Level

These perturbations operate at the sub-word level to simulate adversarial character manipulations.

---

#### C1_homoglyphs *(rule-based)*

Replaces up to 3 characters with visually identical or near-identical Unicode lookalikes, covering Latin → Cyrillic substitutions and within-script confusable pairs for Arabic, Chinese, Devanagari, and Malayalam:

```
Latin→Cyrillic:  a→а, e→е, o→о, p→р, c→с, x→х, y→у, i→і
Arabic pairs:    ي↔ى, ه↔ة, ب↔ت, ن↔ي, و↔ﻭ, ا↔ﺍ
Chinese pairs:   人↔入, 土↔士, 己↔已, 末↔未, 干↔于, 大↔太, 日↔目, 力↔刀, 田↔由
Devanagari:      म↔भ, ग↔प, ह↔ब, क↔ख, ण↔न, ध↔घ
Malayalam:       ര↔ദ, ക↔ഥ, ജ↔ഞ, ന↔ഩ, ല↔ള
```

The map was extended from a Latin-only baseline after analysis showed ~66% unchanged rate for Arabic and ~54% for Chinese.

---

#### C2_leetspeak *(rule-based)*

Substitutes up to 3 characters with leet-speak equivalents. The map covers Latin, Arabic (Arabizi internet transliteration), Chinese (number-character slang), Devanagari, and Malayalam:

```
Latin:       a→4, e→3, i→1, o→0, s→5, t→7, l→1, g→9
Arabic:      ع→3, ح→7, ق→9, خ→5, ء→2, ز→7, ص→9, ط→6, غ→3
Chinese:     一→1, 二→2, 三→3, 四→4 ... 零→0
Devanagari:  ए→3, ओ→0, इ→1, अ→4
Malayalam:   ഒ→0, ഇ→1, ഏ→3
```

---

#### C3_word_splitting *(LLM-based)*

Selects exactly 2 important content words or named entities and inserts a single space inside each word to create adversarial splits that look like plausible typos (e.g. *vaccination → vaccin ation*, *government → govern ment*).

**English prompt:**
> Select exactly 2 important content words or named entities in this claim and insert a single space inside each word at a natural-looking position to split it adversarially (e.g. 'vaccination' → 'vaccin ation', 'government' → 'govern ment'). The splits should look like plausible typos or formatting errors. Do not change any other words.

*Prompts exist for all 10 pipeline languages.*

---

### 4.4 Family D — Style & Obfuscation

---

#### D2_clickbait_llm *(LLM-based)*

Rewrites the claim in clickbait style: prepends exactly one sensationalist hook phrase and capitalizes exactly 3 key nouns or verbs in the body. No factual content is added or removed.

**English prompt:**
> Rewrite this claim in a clickbait style. Prepend exactly one sensationalist hook phrase (e.g. 'EXPOSED:', 'BREAKING:', 'You won't believe this:', 'SHOCKING TRUTH:'). Then identify exactly 3 key nouns or verbs in the body and write them in ALL CAPS — leave every other word in its original case. Do not add, remove, or change any factual content.

*Prompts exist for all 10 pipeline languages.*

---

#### D3_back_translation_it / D3_back_translation_ru *(rule-based)*

Performs a round-trip translation through a pivot language (Italian or Russian) and back to the source language using Google Translate. The pivot languages were deliberately chosen to be **outside the pipeline's 10 supported languages**, preventing the same-language no-op problem (e.g. translating a German post through German produces an unchanged output).

Implementation:
```python
def _back_translate(text, pivot, lang):
    src = LANG_CODE_MAP[lang]   # e.g. "de" for DE
    mid = GoogleTranslator(source=src, target=pivot).translate(text)
    return GoogleTranslator(source=pivot, target=src).translate(mid)
```

> **Historical note:** An earlier version used German and Spanish as pivot languages. This caused 100% unchanged output for German posts (DE→DE→DE) and 97% unchanged for Spanish posts. The pivot languages were changed to Italian (`it`) and Russian (`ru`).

---

### 4.5 Family E — Rhetorical

---

#### E2_presupposition *(LLM-based)*

Prepends exactly one short loaded presupposition frame that smuggles in an assumption as though it were established fact (e.g. *"As has long been known,"*, *"Despite repeated cover-ups,"*). The original claim text is not changed.

**English prompt:**
> Prepend exactly one short loaded presupposition frame to this claim that smuggles in an assumption as if it were already established fact. Examples: 'In what should come as no surprise,', 'As has long been known,', 'Confirming what many already suspected,', 'Despite repeated cover-ups,'. Choose the frame that fits the topic most naturally. Do not change the original claim text itself.

*Prompts exist for all 10 pipeline languages.*

---

## 5. Paper Perturbations (English Only)

In addition to the core 14 perturbation types, we apply 12 perturbation types from the paper *"Claim Matching Beyond English"* (Nzomo et al., 2023), sourced from the [claim-matching-robustness repository](https://github.com/JabezNzomo99/claim-matching-robustness). These are applied to English posts only and use LLM generation with the paper's original prompt formulations.

All 12 paper perturbations are LLM-based. An earlier design used rule-based typo generation, but the paper's original approach uses LLM-generated social-media-style abbreviations and phonetic spelling rather than simple character mutations, so the implementation was updated to match.

### P_negation_low

Negates the claim with minimal edits by introducing a single "not" or equivalent negation.

> *"You are now a social media user tasked with rewriting a given claim by negating it with minimal edits. Identify the main claim. Negate it by introducing as minimal edits as possible — add 'not', 'never', or an equivalent negation at the most natural grammatical position."*

### P_negation_high

Double-negates: first negates the core assertion, then wraps it in a denial frame.

> *"...Double negate it by: first negating the core assertion, then wrapping the whole thing in a denial frame. Example: 'It is not false that it is safe for individuals infected with COVID-19 to go to work.'"*

### P_entity_low

Substitutes exactly one named entity (person, location, organisation, date) with a similar or related entity of the same type.

> *"...changing only one named entity... to a similar or related entity of the same type (e.g., a synonym, nickname, or alternative). Example: 'Biden signed...' → 'The U.S. leader signed...'"*

### P_entity_high

Substitutes all named entities in the claim.

> *"...changing all named entities (persons, organisations, locations, dates) to similar or related entities of the same type. Example: 'Biden signed...' → 'Sleepy Joe signed an executive order recently banning the term Kung Flu.'"*

### P_llm_rewrite_low

Rewrites the claim with minimal edits: swaps 1–2 words for close synonyms.

### P_llm_rewrite_high

Rewrites the claim using maximum word changes while fully preserving meaning. Different vocabulary, restructured sentences, changed word order.

### P_dialect_aae

Rewrites in African American Vernacular English (AAVE). Uses characteristic features: habitual *be*, copula deletion, double negatives.

### P_dialect_jamaican

Rewrites in Jamaican Patois. Uses: *mi* for I/me, *dem* for they, *nuh* for negation, *fi* for to/for.

### P_dialect_pidgin

Rewrites in Nigerian Pidgin English (Naija). Uses: *e* for he/she/it, *dem* for they, *dey* for is/are, *wey* as relative pronoun.

### P_dialect_singlish

Rewrites in Singlish (Singaporean English). Uses: discourse particles (*lah, leh, meh, lor, sia*), topic-fronting, copula omission.

### P_typos_low

Introduces 1–2 subtle social-media-style typos or abbreviations. Low edit ratio.

### P_typos_high

Introduces multiple typos, phonetic spelling, and text-speak abbreviations throughout. High edit ratio.

---

## 6. LLM Setup

All LLM-based perturbations call a locally hosted model via LM Studio's OpenAI-compatible API.

- **Model:** Qwen3-35B-A3B Q4_K_M (quantized)
- **Endpoint:** `http://localhost:1234/v1`
- **Timeout:** 60 seconds per call
- **Response format:** JSON with a single key `perturbed_text`

### System Prompt

```
You are a text perturbation engine for a multilingual NLP research pipeline. Apply exactly the perturbation described in the user message. If the perturbation cannot be meaningfully applied, place the original text unchanged in the output field. Output only the perturbed text — no explanations, no annotations, no markdown formatting. Never wrap words in ** or any other markdown syntax. Always return the COMPLETE INPUT TEXT with the perturbation applied — never truncate, shorten, or omit any part of the original text.
```

### User Message Format

```
{instruction}

---
INPUT TEXT (this is the complete text to transform — it may contain quotes, dialogue, or
multiple sentences; apply the perturbation to the entire text, not just one part of it):
{text}
```

The `INPUT TEXT` label and its explanation were added after observing that models would sometimes apply perturbations only to one sentence of a multi-sentence post, or would treat quoted dialogue as the only target. The explicit framing ensures the model treats the entire payload as its input.

---

## 7. The Echo Problem

### 7.1 What Is an Echo?

An **echo** occurs when the model returns the perturbation instruction rather than the perturbed text. Instead of applying the transformation, the model outputs the prompt itself — either verbatim or with minor modification.

Example (B2_temporal_drift, German):
```
Expected: perturbed text with dates shifted forward 3 years
Actual:   "Verschiebe jedes in dieser Behauptung genannte Jahr oder Datum um genau 3 Jahre 
           nach vorne, um eine plausible aber falsche Zeitlinie zu erzeugen. Häng..."
```

This is the German B2 instruction text, not a perturbed claim. The model confused its own instruction for the output.

### 7.2 Root Causes

- **Non-English instructions:** The model handles English instructions reliably. When instructions are in Arabic, Polish, or other languages, instruction-following degrades and the model occasionally parrots the instruction text.
- **Thinking mode leakage:** Qwen3 models have an internal chain-of-thought (thinking) mode. When this activates, the actual JSON response sometimes ends up in `reasoning_content` rather than `content`, leaving `content` empty or containing partial thinking output.
- **Ambiguous input framing:** Without an explicit delimiter between instruction and input text, the model sometimes treats the input text as part of the instruction context.

### 7.3 Detection

Echo outputs are detected by a multi-layered filter (`_is_bad_output()`):

1. **Exact match:** `perturbed_text == original_text` (stripped) → echo or no-op
2. **Space-stripped match:** removing all whitespace before comparison catches Traditional/Simplified Chinese variants and whitespace-only differences
3. **Length ratio too high:** `len(perturbed) / len(original) > 4.0` → model appended the instruction to the text
4. **Length ratio too low:** `len(perturbed) / len(original) < 0.4` (LLM types only) → model truncated or returned a fragment

### 7.4 Mitigation

**In the system prompt:** Added the instruction to always return the complete input text, never truncate, and not add explanations. This reduced cases where the model returned partial outputs.

**In the user message:** The `INPUT TEXT` label with explanatory text was added to create an unambiguous boundary between the instruction and the material to be transformed:

```
INPUT TEXT (this is the complete text to transform — it may contain quotes, dialogue, or
multiple sentences; apply the perturbation to the entire text, not just one part of it):
{text}
```

**Post-generation filtering:** All rows where `_is_bad_output()` returns True are excluded from annotation PDFs and downstream analysis. They are retained in the CSV with an `error` flag for transparency.

### 7.5 Remaining Rates

After mitigation, the following echo rates were observed in a 1,000-post per language test run:

| Perturbation | Issue rate | Primary cause |
|---|---|---|
| A4_stt_artifacts | 60.7% | No phonetically eligible words in non-Latin scripts |
| B1_qualifier_removal | 55.7% | Posts contain no qualifiers |
| C1_homoglyphs | 23.1% | Script coverage (pre-fix: Latin only) |
| C3_word_splitting | 16.8% | Mixed echo + model refusal |
| C2_leetspeak | 12.9% | Script coverage (pre-fix) |
| B2_temporal_drift | 7.2% | Over-long outputs (instruction appended) |
| E2_presupposition | 4.6% | Polish instruction echo |
| A1_emoji_relevant | 5.2% | Posts already contain relevant emojis |
| D2_clickbait_llm | 3.6% | Rare refusals |
| A2_hashtagification | 0.6% | Near-perfect |
| A1_emoji_disruptive | 0.1% | Rule-based, near-perfect |

The high A4 and B1 rates are inherent to the perturbation type, not to echo: posts without phonetic homophones or without qualifiers simply cannot have those perturbations applied. The model correctly returns the original text in these cases.

---

## 8. Verification Pipeline

### 8.1 Design

After generation, each perturbed output is verified by the same local LLM acting as a quality judge. The verifier receives:

- The original text
- The perturbed text
- A description of the perturbation type and its verification criteria

It returns three booleans:

| Field | Meaning |
|---|---|
| `perturbation_applied` | Was the transformation actually applied? |
| `meaning_preserved` | Is the core factual meaning preserved? |
| `verified` | Computed: applied AND (meaning_preserved == expected) |

The `verified` field is **recomputed** server-side from the model's individual boolean judgments rather than trusting the model's self-reported `verified` value. This prevents the model from inconsistently marking a row as verified when its own sub-judgments contradict that conclusion.

### 8.2 Meaning Preservation Expectations

Each perturbation type has an `expect_preserved` setting:

| Value | Meaning | Types |
|---|---|---|
| `True` | Meaning must be preserved | A1, A2, A3, C1, C2, C3, D2, D3, E2, P_rewrite, P_dialect, P_typos |
| `False` | Meaning is intentionally altered | B1, B2, E2 (some), P_negation, P_entity |
| `None` | Neutral — only check application | A4_stt_artifacts |

For `None` types, `verified = perturbation_applied` — meaning is ignored entirely. This was chosen for A4 because STT artifacts can realistically change meaning (a phonetic mishearing of "vaccinated" into "back sin aided" is a valid transformation even if the meaning shifts), and forcing a meaning-preservation check adds noise to the verification signal.

### 8.3 Verifier System Prompt

```
You are a quality auditor for a fact-checking research pipeline.
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
- The claim may be in any language — evaluate in its original language.
- Do not translate the claim before judging.
```

### 8.4 Verification Results (Test Set)

On the `perturbed_claims_100.csv` test set (100 EN claims, 17 perturbation types):

| Type | Pass rate | Notes |
|---|---|---|
| A1_emoji_disruptive | 100% | Rule-based, always applies |
| A2_hashtagification | 100% | Clean |
| A3_ocr_artifacts | 100% | Rule-based, always applies |
| B2_temporal_drift | 100% | Temporal anchor correctly detected |
| E2_presupposition | 100% | Presupposition frame correctly detected |
| C1_homoglyphs | 90% | 10% fail: no eligible chars (URL-only posts) |
| C2_leetspeak | 95% | Similar |
| D3_back_translation | 40% | Low due to old DE/ES same-language bug (now fixed) |
| C3_word_splitting | 12% | High echo rate |
| A4_stt_artifacts | 14% | High no-op rate for non-Latin + neutral meaning |
| B1_qualifier_removal | 1% | Near-total echo (no qualifiers in dataset) |

The verifier produced **zero false positives**: no case where the original and perturbed text were identical but `verified=True`.

---

## 9. Annotation PDF Generation

A subset of verified perturbations is compiled into annotation PDFs for human evaluation. Five native-speaker annotators cover Spanish, Polish, Mandarin Chinese, French, and Arabic.

Each PDF contains:
- **Prompt review section:** All LLM prompts for the target language, shown alongside the English reference prompt
- **Post verification section:** 5 posts, each with all applicable perturbation types shown alongside the original

**Selection criteria for posts:**
- Must have at least 5 fully valid (verified = True) perturbations
- Echo outputs and bad-length outputs are excluded
- Specific perturbation types can be skipped per annotator (e.g. C3_word_splitting was removed from the Arabic annotator's form due to systematic echo failures on all Arabic posts)

**Diff highlighting:** Changed words are highlighted in orange (`#B35900`) using `difflib.SequenceMatcher` to make perturbations visually clear.

**Font handling:** DejaVu is used as the base font for all Latin/Arabic/Devanagari/Malayalam text. Chinese content uses DroidSans via inline font tags. Emojis in the original posts are replaced with `[EMOJI_NAME]` labels since reportlab does not render emoji characters.

---

## 10. Key Design Decisions & Fixes

### Same-language back-translation bug
The original D3 implementation used German and Spanish as pivot languages. Since both are in the 10-language pipeline, German posts were routed DE→DE→DE (100% unchanged) and Spanish posts ES→ES→ES (97% unchanged). Fixed by switching pivots to Italian and Russian, which are outside the pipeline.

### Non-Latin homoglyph / leet coverage
The initial `_HOMOGLYPH_MAP` and `_LEET_MAP` contained only Latin characters. Arabic, Chinese, Devanagari, and Malayalam posts produced 50–72% unchanged outputs. Both maps were extended with within-script confusable pairs and numeral substitutions for these scripts.

### Dialog and multi-sentence truncation
Several long French posts (containing extended dialogue) were being perturbed in only the first sentence. The system prompt was updated to explicitly instruct the model to apply the perturbation to the entire input text. The `INPUT TEXT:` delimiter in the user message reinforces this. The PDF generator's truncation threshold was also raised from 700 to 100,000 characters to prevent long posts from being silently cut.

### Thinking mode field routing
Qwen3 thinking models in LM Studio sometimes route the JSON response to `reasoning_content` instead of `content` when structured output is enabled. The LM Studio call was updated to fall back: `raw = msg.content or getattr(msg, "reasoning_content", "") or ""`.

### Verifier field naming
The verifier's explanation field was originally named `reasoning`. Because the word "reasoning" in a prompt can activate the model's extended chain-of-thought mode, this field was removed entirely from the verifier output. The verifier now returns only three booleans with no free-text explanation.
