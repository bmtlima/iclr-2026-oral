#!/usr/bin/env python3
"""Rebuild data/enriched.json from .cache/enriched/*.json.

Safe to run repeatedly; it's just a gather-and-emit step. Papers whose cache
files are missing or invalid are skipped (the site renders "Enrichment pending"
for them).
"""
from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from lib import claude as claude_lib  # noqa: E402

CACHE_DIR = REPO_ROOT / ".cache" / "enriched"
DATA_DIR = REPO_ROOT / "data"
OUT_PATH = DATA_DIR / "enriched.json"
SCHEMA_VERSION = 1

REQUIRED_KEYS = {
    "id",
    "one_sentence_summary",
    "contributions",
    "methods_used",
    "datasets_used",
    "limitations",
    "future_work",
    "source_sections_found",
    "pdf_extraction_status",
    "enriched_at",
}


def _valid(rec: dict) -> bool:
    return (
        rec.get("schema_version") == SCHEMA_VERSION
        and isinstance(rec.get("id"), str)
        and REQUIRED_KEYS.issubset(rec.keys())
    )


def main() -> int:
    enriched: dict[str, dict] = {}
    bad: list[str] = []
    total = 0
    for p in sorted(CACHE_DIR.glob("*.json")):
        total += 1
        try:
            rec = json.loads(p.read_text())
        except Exception:
            bad.append(p.name)
            continue
        if not _valid(rec):
            bad.append(p.name)
            continue
        enriched[rec["id"]] = rec

    doc = {
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "model": claude_lib.HAIKU_MODEL,
        "schema_version": SCHEMA_VERSION,
        "enriched": enriched,
    }
    OUT_PATH.write_text(
        json.dumps(doc, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    )
    print(f"[aggregate] {len(enriched)}/{total} records valid → {OUT_PATH.relative_to(REPO_ROOT)}")
    if bad:
        print(f"[aggregate] {len(bad)} invalid files (skipped): {bad[:5]}{'…' if len(bad) > 5 else ''}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
