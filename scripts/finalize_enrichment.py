#!/usr/bin/env python3
"""Finalize subagent-produced extractions into canonical enriched records.

Reads .cache/enriched_raw/{id}.json (raw extraction fields from subagents) and
.cache/prepared/manifest.json (metadata from prep step), merges them into
canonical .cache/enriched/{id}.json records, then rebuilds data/enriched.json.
"""
from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from lib import claude as claude_lib  # noqa: E402

DATA_DIR = REPO_ROOT / "data"
RAW_DIR = REPO_ROOT / ".cache" / "enriched_raw"
OUT_DIR = REPO_ROOT / ".cache" / "enriched"
PREPARED_DIR = REPO_ROOT / ".cache" / "prepared"

OUT_DIR.mkdir(parents=True, exist_ok=True)
SCHEMA_VERSION = 1
REQUIRED_EXTRACTION = {
    "one_sentence_summary",
    "contributions",
    "methods_used",
    "datasets_used",
    "limitations",
    "future_work",
}


def _clean_list(v) -> list[str]:
    if not isinstance(v, list):
        return []
    return [str(x).strip() for x in v if isinstance(x, str) and x.strip()]


def _stub(pid: str, reason: str, sections_found=None, extraction_status="failed", input_chars=0) -> dict:
    return {
        "id": pid,
        "schema_version": SCHEMA_VERSION,
        "one_sentence_summary": "",
        "contributions": [],
        "methods_used": [],
        "datasets_used": [],
        "limitations": [],
        "future_work": [],
        "source_sections_found": sections_found or {
            "conclusion": False,
            "limitations": False,
            "future_work": False,
            "discussion": False,
        },
        "pdf_extraction_status": extraction_status,
        "enriched_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "input_char_count": input_chars,
        "model_stop_reason": f"stub: {reason}",
        "cost_usd_estimate": 0.0,
    }


def main() -> int:
    manifest = json.loads((PREPARED_DIR / "manifest.json").read_text())["papers"]
    papers = json.loads((DATA_DIR / "papers.json").read_text())["papers"]
    paper_ids = [p["id"] for p in papers]

    finalized = 0
    stubbed = 0
    for pid in paper_ids:
        mf = manifest.get(pid, {})
        sections_found = mf.get("found", {
            "conclusion": False,
            "limitations": False,
            "future_work": False,
            "discussion": False,
        })
        extraction_status = mf.get("status", "failed")
        input_chars = mf.get("char_count", 0)

        raw_path = RAW_DIR / f"{pid}.json"
        if not raw_path.exists():
            record = _stub(pid, "no raw output", sections_found, extraction_status, input_chars)
            stubbed += 1
        else:
            try:
                raw = json.loads(raw_path.read_text())
            except Exception as e:
                record = _stub(pid, f"unreadable raw json: {e}", sections_found, extraction_status, input_chars)
                stubbed += 1
            else:
                if not REQUIRED_EXTRACTION.issubset(raw.keys()):
                    missing = REQUIRED_EXTRACTION - set(raw.keys())
                    record = _stub(pid, f"missing keys: {sorted(missing)}", sections_found, extraction_status, input_chars)
                    stubbed += 1
                else:
                    record = {
                        "id": pid,
                        "schema_version": SCHEMA_VERSION,
                        "one_sentence_summary": str(raw.get("one_sentence_summary", "")).strip(),
                        "contributions": _clean_list(raw.get("contributions")),
                        "methods_used": _clean_list(raw.get("methods_used")),
                        "datasets_used": _clean_list(raw.get("datasets_used")),
                        "limitations": _clean_list(raw.get("limitations")),
                        "future_work": _clean_list(raw.get("future_work")),
                        "source_sections_found": sections_found,
                        "pdf_extraction_status": extraction_status,
                        "enriched_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                        "input_char_count": input_chars,
                        "model_stop_reason": "ok",
                        "cost_usd_estimate": 0.0,   # via subagents, accounted for elsewhere
                    }
                    finalized += 1

        out_path = OUT_DIR / f"{pid}.json"
        out_path.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n")

    # Aggregate
    enriched: dict[str, dict] = {}
    for p in sorted(OUT_DIR.glob("*.json")):
        try:
            rec = json.loads(p.read_text())
            if rec.get("schema_version") == SCHEMA_VERSION and rec.get("id"):
                enriched[rec["id"]] = rec
        except Exception:
            continue

    doc = {
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "model": claude_lib.HAIKU_MODEL,
        "schema_version": SCHEMA_VERSION,
        "enriched": enriched,
    }
    out = DATA_DIR / "enriched.json"
    out.write_text(json.dumps(doc, indent=2, sort_keys=True, ensure_ascii=False) + "\n")
    print(f"[finalize] {finalized} ok, {stubbed} stubbed → {out.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
