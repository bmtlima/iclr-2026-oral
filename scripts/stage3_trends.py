#!/usr/bin/env python3
"""Stage 3: cross-paper trend meta-clustering via Claude Sonnet.

Reads .cache/enriched/*.json, builds an XML blob of all papers with non-empty
future_work or limitations, calls Sonnet once, validates paper_ids, writes
data/trends.json.

Run:
    ANTHROPIC_API_KEY=sk-... .venv/bin/python scripts/stage3_trends.py
"""
from __future__ import annotations

import datetime as dt
import json
import os
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))
load_dotenv(REPO_ROOT / ".env")

from lib import claude as claude_lib  # noqa: E402

DATA_DIR = REPO_ROOT / "data"
CACHE_DIR = REPO_ROOT / ".cache" / "enriched"
TRENDS_PATH = DATA_DIR / "trends.json"

MODEL = claude_lib.SONNET_MODEL


def _xml_escape(s: str) -> str:
    return (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def load_enrichments() -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for p in sorted(CACHE_DIR.glob("*.json")):
        try:
            data = json.loads(p.read_text())
            pid = data.get("id")
            if pid:
                out[pid] = data
        except Exception:
            continue
    return out


def load_papers() -> dict[str, dict[str, Any]]:
    papers = json.loads((DATA_DIR / "papers.json").read_text())["papers"]
    return {p["id"]: p for p in papers}


def build_blob(enrichments: dict[str, dict[str, Any]], papers: dict[str, dict[str, Any]]) -> tuple[str, int]:
    """Build the XML input blob. Include only papers with non-empty future_work or limitations."""
    parts: list[str] = ["<papers>"]
    n_included = 0
    for pid, e in enrichments.items():
        fw = e.get("future_work") or []
        lim = e.get("limitations") or []
        if not fw and not lim:
            continue
        title = papers.get(pid, {}).get("title", "") or ""
        parts.append(f'  <paper id="{_xml_escape(pid)}" title="{_xml_escape(title)}">')
        if fw:
            parts.append("    <future_work>")
            for item in fw:
                parts.append(f"      <item>{_xml_escape(item)}</item>")
            parts.append("    </future_work>")
        if lim:
            parts.append("    <limitations>")
            for item in lim:
                parts.append(f"      <item>{_xml_escape(item)}</item>")
            parts.append("    </limitations>")
        parts.append("  </paper>")
        n_included += 1
    parts.append("</papers>")
    return "\n".join(parts), n_included


def validate_themes(result: Any, valid_ids: set[str]) -> tuple[list[dict[str, Any]], list[str]]:
    warnings: list[str] = []
    themes_out: list[dict[str, Any]] = []
    for theme in result.themes:
        valid_paper_ids = [pid for pid in theme.paper_ids if pid in valid_ids]
        dropped = set(theme.paper_ids) - set(valid_paper_ids)
        if dropped:
            warnings.append(
                f"theme '{theme.slug}': dropped {len(dropped)} invalid paper_ids: {sorted(dropped)[:5]}"
            )
        if len(valid_paper_ids) < 2:
            warnings.append(f"theme '{theme.slug}': <2 valid paper_ids, dropping theme")
            continue
        valid_quotes: list[dict[str, str]] = []
        for q in theme.representative_quotes:
            if q.get("paper_id") in valid_paper_ids and q.get("quote"):
                valid_quotes.append({"paper_id": q["paper_id"], "quote": q["quote"]})
            else:
                warnings.append(
                    f"theme '{theme.slug}': dropped quote for paper_id={q.get('paper_id')}"
                )
        themes_out.append({
            "slug": theme.slug,
            "headline": theme.headline,
            "explanation": theme.explanation,
            "paper_ids": valid_paper_ids,
            "representative_quotes": valid_quotes[:3],
        })
    return themes_out, warnings


def main() -> int:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("[stage3] ERROR: ANTHROPIC_API_KEY not set (put it in .env)")
        return 2

    enrichments = load_enrichments()
    papers = load_papers()
    if not enrichments:
        print("[stage3] No enrichments found in .cache/enriched/. Run stage2 first.")
        return 1

    blob, n_included = build_blob(enrichments, papers)
    if n_included < 10:
        print(f"[stage3] Only {n_included} papers have future_work or limitations. Trends will be weak.")

    print(f"[stage3] Sending {n_included} papers to {MODEL}…")
    result, meta = claude_lib.call_stage3(blob, n_included, model=MODEL)

    valid_ids = set(enrichments.keys())
    themes, warnings = validate_themes(result, valid_ids)
    for w in warnings:
        print(f"[stage3]   warning: {w}")

    cost = claude_lib.estimate_cost(MODEL, meta.get("input_tokens", 0), meta.get("output_tokens", 0))
    print(f"[stage3] Generated {len(themes)} themes. Tokens: in={meta.get('input_tokens', 0)} out={meta.get('output_tokens', 0)}. Cost est: ${cost:.4f}")

    doc = {
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "model": MODEL,
        "schema_version": 1,
        "themes": themes,
    }
    TRENDS_PATH.write_text(
        json.dumps(doc, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    )
    print(f"[stage3] Wrote {TRENDS_PATH.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
