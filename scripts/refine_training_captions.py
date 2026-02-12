#!/usr/bin/env python
"""Refine training captions in data/training/ for Flux LoRA training.

This script applies targeted text substitutions to improve caption
consistency and fix known issues discovered during smoke testing:

  1. AGE ANCHORING: "young wood elf" → "young adult wood elf" to prevent
     the model from interpreting "young" as teenage/child-like.

  2. CLOTHING CONSISTENCY: Normalize "dress" to "gown" in contexts where
     it conflicts with the corset+skirt outfit (only when NOT describing
     the actual garment).

  3. DANGLING FRAGMENTS: Clean up incomplete sentences and trailing
     artifacts from the Florence-2 caption generator.

NOTE: This only modifies data/training/*.txt files.
      Original captions in data/labels/ are NEVER touched.

Usage:
    python scripts/refine_training_captions.py [--dry-run]
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


def _project_root() -> Path:
    p = Path(__file__).resolve().parent
    while p != p.parent:
        if (p / "src").is_dir():
            return p
        p = p.parent
    return Path(__file__).resolve().parent.parent


def refine_caption(text: str) -> str:
    """Apply all refinement rules to a single caption string."""
    original = text

    # ── 1. AGE ANCHORING ──────────────────────────────────────────
    # "a young wood elf" → "a young adult wood elf"
    # "a young, nude female elf" → "a young adult, nude female elf"
    # Only if "young adult" isn't already present
    if "young adult" not in text.lower():
        text = re.sub(
            r'\byoung\b(?=\s+(?:wood\s+)?(?:female\s+)?(?:nude\s+)?(?:elf|elven))',
            'young adult',
            text,
            flags=re.IGNORECASE,
        )

    # Also anchor "young" when followed by comma patterns like "young, nude female elf"
    if "young adult" not in text.lower():
        text = re.sub(
            r'\byoung\b(?=,?\s+(?:nude|naked|clothed|semi-nude))',
            'young adult',
            text,
            flags=re.IGNORECASE,
        )

    # ── 2. EXPLICIT AGE PHRASING ──────────────────────────────────
    # Where "the central figure is a female elf" appears without age,
    # add "adult" to anchor it.  Use "an adult" after "a" for grammar.
    text = re.sub(
        r'\b(a|the)\s+female\s+elf\b',
        r'\1 adult female elf',
        text,
        flags=re.IGNORECASE,
    )
    # Fix "a adult" → "an adult"
    text = re.sub(r'\ba adult\b', 'an adult', text, flags=re.IGNORECASE)
    # Prevent double-adult
    text = re.sub(r'\badult\s+adult\b', 'adult', text, flags=re.IGNORECASE)

    # ── 3. CLOTHING: "flowing dress" → "flowing gown" ─────────────
    # "dress" in many captions describes what Florence-2 saw, but the
    # actual character design is corset+skirt. We remap "flowing dress"
    # and "long dress" to "gown" to reduce confusion, since "gown" is
    # more distinct and won't compete with "green skirt"
    text = re.sub(
        r'\b(flowing|long|elegant|sheer)\s+dress\b',
        r'\1 gown',
        text,
        flags=re.IGNORECASE,
    )

    # ── 4. DANGLING FRAGMENTS ─────────────────────────────────────
    # Florence-2 sometimes cuts off mid-sentence at the token limit.
    # Remove trailing fragments that end with prepositions or articles.
    text = re.sub(
        r',\s*(?:the|a|an|with|and|or|of|in|on|at|to|for|by|from|into|is|are|was)\s*$',
        '',
        text,
        flags=re.IGNORECASE,
    )

    # Remove trailing fragments ending with "making her" / "creating a" etc.
    text = re.sub(
        r',\s*(?:making|creating|giving|casting|suggesting|indicating|adding|enhancing|providing)\s+(?:her|a|the|it|his)\b.*$',
        '',
        text,
        flags=re.IGNORECASE,
    )

    # ── 5. DOUBLE DESCRIPTIONS ────────────────────────────────────
    # "a highly detailed digital illustration ... a highly detailed digital illustration"
    # Remove second occurrence of repeated phrases
    text = re.sub(
        r'(a\s+highly\s+detailed\s+(?:digital\s+)?(?:illustration|painting|artwork|rendering))'
        r'(.*?)\1',
        r'\1\2',
        text,
        flags=re.IGNORECASE,
    )

    # ── 6. CLEANUP ────────────────────────────────────────────────
    # Collapse multiple commas/spaces
    text = re.sub(r'(,\s*){2,}', ', ', text)
    text = re.sub(r'\s{2,}', ' ', text)
    text = text.strip().rstrip(',').strip()

    return text


def main():
    parser = argparse.ArgumentParser(description="Refine training captions")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show changes without writing")
    parser.add_argument("--training-dir", type=str, default=None,
                        help="Path to training dir (default: data/training)")
    args = parser.parse_args()

    root = _project_root()
    training_dir = Path(args.training_dir) if args.training_dir else root / "data" / "training"

    if not training_dir.exists():
        print(f"Error: Training dir not found: {training_dir}", file=sys.stderr)
        sys.exit(1)

    txt_files = sorted(training_dir.glob("*.txt"))
    print(f"Found {len(txt_files)} caption files in {training_dir}")

    changed_count = 0
    age_fixes = 0
    clothing_fixes = 0
    fragment_fixes = 0

    for txt_path in txt_files:
        original = txt_path.read_text(encoding="utf-8")
        refined = refine_caption(original)

        if refined != original:
            changed_count += 1

            # Count specific fix types
            if "young adult" in refined and "young adult" not in original:
                age_fixes += 1
            if "gown" in refined and "gown" not in original:
                clothing_fixes += 1
            if len(refined) < len(original) - 5:
                fragment_fixes += 1

            if args.dry_run:
                print(f"\n--- {txt_path.name} ---")
                # Show first diff
                orig_words = original.split()
                new_words = refined.split()
                for i, (o, n) in enumerate(zip(orig_words, new_words)):
                    if o != n:
                        ctx_start = max(0, i - 3)
                        ctx_end = min(len(orig_words), i + 4)
                        print(f"  OLD: ...{' '.join(orig_words[ctx_start:ctx_end])}...")
                        ctx_end = min(len(new_words), i + 4)
                        print(f"  NEW: ...{' '.join(new_words[ctx_start:ctx_end])}...")
                        break
            else:
                txt_path.write_text(refined, encoding="utf-8")

    print(f"\n{'[DRY RUN] ' if args.dry_run else ''}Results:")
    print(f"  Total files: {len(txt_files)}")
    print(f"  Changed:     {changed_count}")
    print(f"  Age fixes:   {age_fixes}")
    print(f"  Clothing:    {clothing_fixes}")
    print(f"  Fragments:   {fragment_fixes}")
    print(f"  Unchanged:   {len(txt_files) - changed_count}")


if __name__ == "__main__":
    main()
