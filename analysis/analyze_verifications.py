#!/usr/bin/env python3
"""
analyze_verifications.py

Computes statistics on verified_perturbations.csv produced by verify_perturbations.py.

Outputs:
  - Console summary
  - stats_summary.csv            — per-perturbation-type metrics
  - stats_family.csv             — per-family aggregates
  - stats_meaning_shift.csv      — semantic-altering types (B1, B2, E2): shift success rate
  - stats_failure_breakdown.csv  — why rows failed (not applied / wrong meaning / API error)
  - plot_verified_by_type.png    — bar chart: verified rate per perturbation type, grouped by family
  - plot_failure_breakdown.png   — stacked bar: failure reasons per type
  - plot_family_heatmap.png      — heatmap: verified rate across all types
  - plot_meaning_shift.png       — bar chart: meaning shift success for B1, B2, E2

Usage:
    python analyze_verifications.py
    python analyze_verifications.py --input path/to/verified_perturbations.csv
    python analyze_verifications.py --no-plots   # skip diagram generation
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

try:
    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend, safe on any machine
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import numpy as np
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

# ── Configuration ─────────────────────────────────────────────────────────────

SCRIPT_DIR   = Path(__file__).parent
INPUT_FILE   = SCRIPT_DIR / "verified_perturbations.csv"
OUT_SUMMARY  = SCRIPT_DIR / "stats_summary.csv"
OUT_FAMILY   = SCRIPT_DIR / "stats_family.csv"
OUT_SHIFT    = SCRIPT_DIR / "stats_meaning_shift.csv"
OUT_PLOT_TYPES    = SCRIPT_DIR / "plot_verified_by_type.png"
OUT_PLOT_FAILURE  = SCRIPT_DIR / "plot_failure_breakdown.png"
OUT_PLOT_HEATMAP  = SCRIPT_DIR / "plot_family_heatmap.png"
OUT_PLOT_SHIFT    = SCRIPT_DIR / "plot_meaning_shift.png"
OUT_FAILURE  = SCRIPT_DIR / "stats_failure_breakdown.csv"

# Perturbation types where meaning is INTENTIONALLY altered
MEANING_ALTERING = {"B1_qualifier_removal", "B2_temporal_drift", "E2_presupposition"}

# Family membership
FAMILY_MAP = {
    "A1_emoji_relevant":    "A",
    "A1_emoji_disruptive":  "A",
    "A2_hashtagification":  "A",
    "A3_ocr_artifacts":     "A",
    "A4_stt_artifacts":     "A",
    "B1_qualifier_removal": "B",
    "B2_temporal_drift":    "B",
    "C1_homoglyphs":        "C",
    "C2_leetspeak":         "C",
    "C3_word_splitting":    "C",
    "D1_formal_to_casual":  "D",
    "D1_casual_to_formal":  "D",
    "D2_clickbait":         "D",
    "D3_back_translation_es": "D",
    "D3_back_translation_de": "D",
    "E1_voice_transform":   "E",
    "E2_presupposition":    "E",
}

FAMILY_LABELS = {
    "A": "Social Media Noise",
    "B": "Semantic & Argumentative",
    "C": "Adversarial / Evasion",
    "D": "Style / Register",
    "E": "Rhetorical",
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def parse_bool(val: str) -> bool | None:
    if val is None or val.strip() == "":
        return None
    return val.strip().lower() in ("true", "1", "yes")


def pct(num: int, denom: int) -> str:
    if denom == 0:
        return "—"
    return f"{100 * num / denom:.1f}%"


def write_csv(path: Path, fieldnames: list, rows: list[dict]):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  → saved {path.name}")


# ── Loading ───────────────────────────────────────────────────────────────────

def load(input_file: Path) -> list[dict]:
    with open(input_file, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    print(f"Loaded {len(rows)} rows from {input_file.name}\n")
    return rows


# ── Analysis functions ────────────────────────────────────────────────────────

def overall_stats(rows: list[dict]) -> dict:
    total       = len(rows)
    errors      = sum(1 for r in rows if r.get("verify_error", "").strip())
    verified    = sum(1 for r in rows if parse_bool(r.get("verified")) is True)
    not_applied = sum(1 for r in rows if parse_bool(r.get("perturbation_applied")) is False
                                     and not r.get("verify_error", "").strip())
    wrong_meaning = sum(
        1 for r in rows
        if parse_bool(r.get("perturbation_applied")) is True
        and parse_bool(r.get("verified")) is False
        and not r.get("verify_error", "").strip()
    )
    return {
        "total":          total,
        "errors":         errors,
        "verified":       verified,
        "not_applied":    not_applied,
        "wrong_meaning":  wrong_meaning,
    }


def per_type_stats(rows: list[dict]) -> list[dict]:
    """Pass rate and breakdown by individual perturbation type."""
    buckets: dict[str, list] = defaultdict(list)
    for r in rows:
        buckets[r["perturbation_name"]].append(r)

    results = []
    for pname in sorted(buckets):
        group   = buckets[pname]
        total   = len(group)
        errors  = sum(1 for r in group if r.get("verify_error", "").strip())
        valid   = total - errors
        applied = sum(1 for r in group if parse_bool(r.get("perturbation_applied")) is True)
        verified_count = sum(1 for r in group if parse_bool(r.get("verified")) is True)
        meaning_preserved_count = sum(
            1 for r in group if parse_bool(r.get("meaning_preserved")) is True
        )
        results.append({
            "perturbation_name":      pname,
            "family":                 FAMILY_MAP.get(pname, "?"),
            "meaning_altering":       pname in MEANING_ALTERING,
            "total":                  total,
            "errors":                 errors,
            "valid_responses":        valid,
            "perturbation_applied":   applied,
            "verified":               verified_count,
            "meaning_preserved":      meaning_preserved_count,
            "applied_rate":           pct(applied, valid),
            "verified_rate":          pct(verified_count, valid),
            "meaning_preserved_rate": pct(meaning_preserved_count, valid),
        })
    return results


def per_family_stats(per_type: list[dict]) -> list[dict]:
    """Aggregate per-type stats up to family level."""
    buckets: dict[str, list] = defaultdict(list)
    for row in per_type:
        buckets[row["family"]].append(row)

    results = []
    for fam in sorted(buckets):
        group = buckets[fam]
        total    = sum(r["total"]           for r in group)
        errors   = sum(r["errors"]          for r in group)
        valid    = sum(r["valid_responses"]  for r in group)
        applied  = sum(r["perturbation_applied"] for r in group)
        verified = sum(r["verified"]        for r in group)
        results.append({
            "family":         fam,
            "family_label":   FAMILY_LABELS.get(fam, ""),
            "perturbation_types": len(group),
            "total":          total,
            "errors":         errors,
            "valid_responses": valid,
            "perturbation_applied": applied,
            "verified":       verified,
            "applied_rate":   pct(applied, valid),
            "verified_rate":  pct(verified, valid),
        })
    return results


def meaning_shift_stats(rows: list[dict]) -> list[dict]:
    """
    For semantic-altering perturbation types (B, E2):
    how often did the meaning actually shift as intended?
    (meaning_preserved == False AND perturbation_applied == True)
    """
    results = []
    for pname in sorted(MEANING_ALTERING):
        group  = [r for r in rows if r["perturbation_name"] == pname]
        total  = len(group)
        errors = sum(1 for r in group if r.get("verify_error", "").strip())
        valid  = total - errors

        applied          = sum(1 for r in group if parse_bool(r.get("perturbation_applied")) is True)
        meaning_shifted  = sum(
            1 for r in group
            if parse_bool(r.get("perturbation_applied")) is True
            and parse_bool(r.get("meaning_preserved")) is False
        )
        applied_but_not_shifted = applied - meaning_shifted

        results.append({
            "perturbation_name":         pname,
            "family":                    FAMILY_MAP.get(pname, "?"),
            "total":                     total,
            "errors":                    errors,
            "valid_responses":           valid,
            "perturbation_applied":      applied,
            "meaning_shifted":           meaning_shifted,
            "applied_but_not_shifted":   applied_but_not_shifted,
            "shift_success_rate":        pct(meaning_shifted, applied),
            "note": (
                "shift_success_rate = meaning shifted as intended / perturbation applied"
            ),
        })
    return results


def failure_breakdown(rows: list[dict]) -> list[dict]:
    """
    For each perturbation type, break down WHY rows failed:
      1. API / parse error
      2. Perturbation not applied
      3. Applied but meaning wrong (preserved when it should shift, or shifted when it should preserve)
      4. Passed
    """
    buckets: dict[str, list] = defaultdict(list)
    for r in rows:
        buckets[r["perturbation_name"]].append(r)

    results = []
    for pname in sorted(buckets):
        group   = buckets[pname]
        total   = len(group)

        n_error         = sum(1 for r in group if r.get("verify_error", "").strip())
        n_not_applied   = sum(
            1 for r in group
            if not r.get("verify_error", "").strip()
            and parse_bool(r.get("perturbation_applied")) is False
        )
        n_wrong_meaning = sum(
            1 for r in group
            if not r.get("verify_error", "").strip()
            and parse_bool(r.get("perturbation_applied")) is True
            and parse_bool(r.get("verified")) is False
        )
        n_passed = sum(1 for r in group if parse_bool(r.get("verified")) is True)

        results.append({
            "perturbation_name":   pname,
            "family":              FAMILY_MAP.get(pname, "?"),
            "total":               total,
            "passed":              n_passed,
            "failed_api_error":    n_error,
            "failed_not_applied":  n_not_applied,
            "failed_wrong_meaning": n_wrong_meaning,
            "passed_rate":         pct(n_passed, total),
            "error_rate":          pct(n_error, total),
            "not_applied_rate":    pct(n_not_applied, total),
            "wrong_meaning_rate":  pct(n_wrong_meaning, total),
        })
    return results


# ── Pretty printing ───────────────────────────────────────────────────────────

def print_section(title: str):
    print("\n" + "═" * 60)
    print(f"  {title}")
    print("═" * 60)


def print_table(rows: list[dict], cols: list[str], col_widths: list[int] | None = None):
    if not rows:
        print("  (no data)")
        return
    if col_widths is None:
        col_widths = [max(len(str(r.get(c, ""))) for r in rows + [{"": c}]) + 2 for c in cols]
    header = "  " + "".join(str(c).ljust(w) for c, w in zip(cols, col_widths))
    print(header)
    print("  " + "─" * (sum(col_widths)))
    for row in rows:
        line = "  " + "".join(str(row.get(c, "")).ljust(w) for c, w in zip(cols, col_widths))
        print(line)


# ── Plotting ──────────────────────────────────────────────────────────────────

# Consistent colour per family
FAMILY_COLORS = {
    "A": "#4DA6FF",   # blue
    "B": "#FF6B6B",   # red
    "C": "#FFD93D",   # yellow
    "D": "#6BCB77",   # green
    "E": "#C77DFF",   # purple
}

DPI = 150


def _rate(num: int, denom: int) -> float:
    return (100 * num / denom) if denom else 0.0


def plot_verified_by_type(per_type: list[dict]):
    """Bar chart: verified rate per perturbation type, colour-coded by family."""
    names  = [r["perturbation_name"] for r in per_type]
    rates  = [_rate(r["verified"], r["valid_responses"]) for r in per_type]
    colors = [FAMILY_COLORS.get(r["family"], "#aaa") for r in per_type]

    fig, ax = plt.subplots(figsize=(14, 5))
    bars = ax.bar(names, rates, color=colors, edgecolor="white", linewidth=0.5)

    ax.set_ylim(0, 110)
    ax.set_ylabel("Verified rate (%)", fontsize=11)
    ax.set_title("Verified Rate by Perturbation Type", fontsize=13, fontweight="bold")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=40, ha="right", fontsize=8)
    ax.axhline(100, color="grey", linewidth=0.5, linestyle="--")
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)

    # Value labels on bars
    for bar, rate in zip(bars, rates):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1.5,
            f"{rate:.0f}%",
            ha="center", va="bottom", fontsize=7,
        )

    # Family legend
    legend_patches = [
        mpatches.Patch(color=FAMILY_COLORS[f], label=f"{f}: {FAMILY_LABELS[f]}")
        for f in sorted(FAMILY_COLORS)
    ]
    ax.legend(handles=legend_patches, fontsize=8, loc="lower right")

    fig.tight_layout()
    fig.savefig(OUT_PLOT_TYPES, dpi=DPI)
    plt.close(fig)
    print(f"  → saved {OUT_PLOT_TYPES.name}")


def plot_failure_breakdown(failures: list[dict]):
    """Stacked bar: passed / not applied / wrong meaning / error per type."""
    names   = [r["perturbation_name"] for r in failures]
    totals  = [r["total"] for r in failures]
    passed  = [r["passed"] for r in failures]
    not_app = [r["failed_not_applied"] for r in failures]
    wrong   = [r["failed_wrong_meaning"] for r in failures]
    errors  = [r["failed_api_error"] for r in failures]

    x = range(len(names))
    fig, ax = plt.subplots(figsize=(14, 5))

    b1 = ax.bar(x, [p / t * 100 if t else 0 for p, t in zip(passed, totals)],
                color="#6BCB77", label="Passed")
    b2 = ax.bar(x, [n / t * 100 if t else 0 for n, t in zip(not_app, totals)],
                bottom=[p / t * 100 if t else 0 for p, t in zip(passed, totals)],
                color="#FFD93D", label="Not applied")
    b3 = ax.bar(x, [w / t * 100 if t else 0 for w, t in zip(wrong, totals)],
                bottom=[(p + n) / t * 100 if t else 0 for p, n, t in zip(passed, not_app, totals)],
                color="#FF6B6B", label="Wrong meaning")
    b4 = ax.bar(x, [e / t * 100 if t else 0 for e, t in zip(errors, totals)],
                bottom=[(p + n + w) / t * 100 if t else 0
                        for p, n, w, t in zip(passed, not_app, wrong, totals)],
                color="#aaaaaa", label="API error")

    ax.set_ylim(0, 110)
    ax.set_ylabel("Share of rows (%)", fontsize=11)
    ax.set_title("Failure Breakdown by Perturbation Type", fontsize=13, fontweight="bold")
    ax.set_xticks(list(x))
    ax.set_xticklabels(names, rotation=40, ha="right", fontsize=8)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    ax.legend(fontsize=9, loc="lower right")

    fig.tight_layout()
    fig.savefig(OUT_PLOT_FAILURE, dpi=DPI)
    plt.close(fig)
    print(f"  → saved {OUT_PLOT_FAILURE.name}")


def plot_heatmap(per_type: list[dict]):
    """Heatmap: verified rate for each type, arranged by family rows."""
    # Build a 2-D grid: rows = families, cols = type slots within family
    from collections import OrderedDict
    by_family: dict[str, list] = OrderedDict()
    for fam in sorted(FAMILY_LABELS):
        by_family[fam] = [r for r in per_type if r["family"] == fam]

    max_cols = max(len(v) for v in by_family.values())
    families = list(by_family.keys())

    grid   = np.full((len(families), max_cols), np.nan)
    labels = [[""] * max_cols for _ in families]

    for fi, fam in enumerate(families):
        for ci, r in enumerate(by_family[fam]):
            val = _rate(r["verified"], r["valid_responses"])
            grid[fi, ci]   = val
            labels[fi][ci] = r["perturbation_name"].split("_", 1)[1] if "_" in r["perturbation_name"] else r["perturbation_name"]

    fig, ax = plt.subplots(figsize=(max_cols * 1.4 + 1.5, len(families) * 1.2 + 1.5))
    im = ax.imshow(grid, aspect="auto", vmin=0, vmax=100,
                   cmap="RdYlGn", interpolation="nearest")

    ax.set_yticks(range(len(families)))
    ax.set_yticklabels([f"{f} — {FAMILY_LABELS[f]}" for f in families], fontsize=9)
    ax.set_xticks(range(max_cols))
    ax.set_xticklabels([f"slot {i+1}" for i in range(max_cols)], fontsize=8)
    ax.set_title("Verified Rate Heatmap (green = 100%, red = 0%)",
                 fontsize=12, fontweight="bold", pad=12)

    # Cell annotations
    for fi in range(len(families)):
        for ci in range(max_cols):
            val = grid[fi, ci]
            lbl = labels[fi][ci]
            if not np.isnan(val):
                ax.text(ci, fi, f"{lbl}\n{val:.0f}%",
                        ha="center", va="center", fontsize=7,
                        color="black" if 30 < val < 85 else "white")

    fig.colorbar(im, ax=ax, label="Verified rate (%)", fraction=0.03, pad=0.04)
    fig.tight_layout()
    fig.savefig(OUT_PLOT_HEATMAP, dpi=DPI)
    plt.close(fig)
    print(f"  → saved {OUT_PLOT_HEATMAP.name}")


def plot_meaning_shift(shift: list[dict]):
    """Bar chart: meaning shift success rate for B1, B2, E2."""
    names       = [r["perturbation_name"] for r in shift]
    success     = [r["meaning_shifted"] for r in shift]
    not_shifted = [r["applied_but_not_shifted"] for r in shift]
    totals      = [r["perturbation_applied"] for r in shift]

    x = range(len(names))
    fig, ax = plt.subplots(figsize=(7, 4))

    ax.bar(x, [s / t * 100 if t else 0 for s, t in zip(success, totals)],
           color="#FF6B6B", label="Meaning shifted (intended)")
    ax.bar(x, [n / t * 100 if t else 0 for n, t in zip(not_shifted, totals)],
           bottom=[s / t * 100 if t else 0 for s, t in zip(success, totals)],
           color="#4DA6FF", label="Applied but meaning unchanged (failure)")

    ax.set_ylim(0, 110)
    ax.set_ylabel("Share of applied rows (%)", fontsize=11)
    ax.set_title("Meaning Shift Success\n(semantic-altering types: B1, B2, E2)",
                 fontsize=12, fontweight="bold")
    ax.set_xticks(list(x))
    ax.set_xticklabels(names, fontsize=10)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    ax.legend(fontsize=9)

    # Rate labels
    for i, (s, t) in enumerate(zip(success, totals)):
        rate = s / t * 100 if t else 0
        ax.text(i, rate + 2, f"{rate:.0f}%", ha="center", fontsize=10, fontweight="bold")

    fig.tight_layout()
    fig.savefig(OUT_PLOT_SHIFT, dpi=DPI)
    plt.close(fig)
    print(f"  → saved {OUT_PLOT_SHIFT.name}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main(input_file: Path, no_plots: bool = False):
    rows = load(input_file)

    # ── 1. Overall ────────────────────────────────────────────────────────────
    print_section("OVERALL STATISTICS")
    ov = overall_stats(rows)
    total = ov["total"]
    print(f"  Total rows          : {total}")
    print(f"  API / parse errors  : {ov['errors']}  ({pct(ov['errors'], total)})")
    print(f"  Verified (passed)   : {ov['verified']}  ({pct(ov['verified'], total)})")
    print(f"  Failed — not applied: {ov['not_applied']}  ({pct(ov['not_applied'], total)})")
    print(f"  Failed — wrong mean.: {ov['wrong_meaning']}  ({pct(ov['wrong_meaning'], total)})")

    # ── 2. Per perturbation type ──────────────────────────────────────────────
    print_section("PASS RATE BY PERTURBATION TYPE")
    per_type = per_type_stats(rows)
    print_table(
        per_type,
        cols=["perturbation_name", "family", "total", "errors", "applied_rate", "verified_rate"],
        col_widths=[26, 8, 8, 8, 14, 14],
    )

    # ── 3. Per family ─────────────────────────────────────────────────────────
    print_section("PASS RATE BY FAMILY")
    per_fam = per_family_stats(per_type)
    print_table(
        per_fam,
        cols=["family", "family_label", "total", "errors", "applied_rate", "verified_rate"],
        col_widths=[8, 26, 8, 8, 14, 14],
    )

    # ── 4. Meaning shift (B, E2) ──────────────────────────────────────────────
    print_section("MEANING SHIFT SUCCESS (semantic-altering types: B1, B2, E2)")
    shift = meaning_shift_stats(rows)
    print_table(
        shift,
        cols=["perturbation_name", "applied_but_not_shifted", "meaning_shifted", "shift_success_rate"],
        col_widths=[26, 26, 16, 20],
    )

    # ── 5. Failure breakdown ──────────────────────────────────────────────────
    print_section("FAILURE BREAKDOWN BY TYPE")
    failures = failure_breakdown(rows)
    print_table(
        failures,
        cols=["perturbation_name", "passed", "failed_not_applied", "failed_wrong_meaning", "failed_api_error"],
        col_widths=[26, 8, 20, 22, 18],
    )

    # ── Save CSVs ─────────────────────────────────────────────────────────────
    print_section("SAVING OUTPUT FILES")
    write_csv(OUT_SUMMARY, list(per_type[0].keys()), per_type)
    write_csv(OUT_FAMILY,  list(per_fam[0].keys()),  per_fam)
    write_csv(OUT_SHIFT,   list(shift[0].keys()),    shift)
    write_csv(OUT_FAILURE, list(failures[0].keys()), failures)

    # ── Plots ──────────────────────────────────────────────────────────────────
    if not no_plots:
        if not HAS_MATPLOTLIB:
            print("\n  matplotlib not installed — skipping plots.")
            print("  Run: pip install matplotlib numpy")
        else:
            print_section("GENERATING PLOTS")
            plot_verified_by_type(per_type)
            plot_failure_breakdown(failures)
            plot_heatmap(per_type)
            plot_meaning_shift(shift)

    print("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze verification results.")
    parser.add_argument(
        "--input",
        type=Path,
        default=INPUT_FILE,
        help="Path to verified_perturbations.csv (default: same folder as script)",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip diagram generation (CSV outputs still produced).",
    )
    args = parser.parse_args()

    if not args.input.exists():
        print(f"Input file not found: {args.input}")
        print("Run verify_perturbations.py first to generate it.")
        raise SystemExit(1)

    main(args.input, no_plots=args.no_plots)
