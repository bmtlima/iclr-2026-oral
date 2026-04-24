#!/usr/bin/env python3
"""Stage 2: per-paper enrichment via Claude Haiku.

Reads data/papers.json, downloads PDFs, extracts sections, calls Haiku,
caches per-paper results to .cache/enriched/{id}.json, and writes the
aggregate data/enriched.json.

Run:
    ANTHROPIC_API_KEY=sk-... .venv/bin/python scripts/stage2_enrich.py
    .venv/bin/python scripts/stage2_enrich.py --only <paper_id>
    .venv/bin/python scripts/stage2_enrich.py --force
    .venv/bin/python scripts/stage2_enrich.py --limit 3    # for smoke-test

Environment:
    ANTHROPIC_API_KEY    required for live calls (skipped for --dry-run)
"""
from __future__ import annotations

import argparse
import asyncio
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

from lib import pdf as pdf_lib  # noqa: E402
from lib import claude as claude_lib  # noqa: E402

DATA_DIR = REPO_ROOT / "data"
CACHE_DIR = REPO_ROOT / ".cache" / "enriched"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

SCHEMA_VERSION = 1
MODEL = claude_lib.HAIKU_MODEL


def _cache_path(pid: str) -> Path:
    return CACHE_DIR / f"{pid}.json"


def _load_cached(pid: str) -> dict[str, Any] | None:
    p = _cache_path(pid)
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text())
        if data.get("schema_version") != SCHEMA_VERSION:
            return None
        return data
    except Exception:
        return None


def _write_cache(pid: str, data: dict[str, Any]) -> None:
    _cache_path(pid).write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")


def _stub(pid: str, reason: str) -> dict[str, Any]:
    return {
        "id": pid,
        "schema_version": SCHEMA_VERSION,
        "one_sentence_summary": "",
        "contributions": [],
        "methods_used": [],
        "datasets_used": [],
        "limitations": [],
        "future_work": [],
        "source_sections_found": {"conclusion": False, "limitations": False, "future_work": False, "discussion": False},
        "pdf_extraction_status": "failed",
        "enriched_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "input_char_count": 0,
        "model_stop_reason": f"stub: {reason}",
        "cost_usd_estimate": 0.0,
    }


def prepare_paper_input(paper: dict[str, Any]) -> tuple[str, dict[str, bool], str, int]:
    """Download PDF, extract text, assemble model input.

    Returns (input_text, found_flags, extraction_status, input_char_count).
    Never raises — on failure, returns title+abstract only with status='failed'.
    """
    pid = paper["id"]
    title = paper.get("title") or ""
    authors = paper.get("authors") or []
    abstract = paper.get("abstract") or ""
    pdf_url = paper.get("pdf_url")

    text = ""
    extract_status = "failed"
    if pdf_url:
        path = pdf_lib.download_pdf(pdf_url, pid)
        if path is not None:
            text, extract_status = pdf_lib.extract_text(path, pid)

    body, found, status = pdf_lib.assemble_input(title, authors, abstract, text)
    # If text extraction completely failed, status will be "failed" already.
    if extract_status == "failed" and not text:
        status = "failed"
    return body, found, status, len(body)


def enrich_one(paper: dict[str, Any], force: bool = False, dry_run: bool = False) -> tuple[str, dict[str, Any]]:
    """Enrich a single paper. Returns (status, record) where status is one of
    'cached', 'enriched', 'stubbed', 'dry'.
    """
    pid = paper["id"]

    if not force:
        cached = _load_cached(pid)
        if cached is not None:
            return "cached", cached

    body, found, extract_status, input_chars = prepare_paper_input(paper)

    if dry_run:
        return "dry", {
            "id": pid,
            "input_char_count": input_chars,
            "pdf_extraction_status": extract_status,
            "source_sections_found": found,
            "preview": body[:400],
        }

    try:
        result, meta = claude_lib.call_stage2(body, model=MODEL)
    except Exception as e:
        record = _stub(pid, str(e)[:200])
        record["source_sections_found"] = found
        record["pdf_extraction_status"] = "failed"
        record["input_char_count"] = input_chars
        _write_cache(pid, record)
        return "stubbed", record

    cost = claude_lib.estimate_cost(MODEL, meta.get("input_tokens", 0), meta.get("output_tokens", 0))
    record = {
        "id": pid,
        "schema_version": SCHEMA_VERSION,
        "one_sentence_summary": result.one_sentence_summary.strip(),
        "contributions": [s.strip() for s in result.contributions if s and s.strip()],
        "methods_used": [s.strip() for s in result.methods_used if s and s.strip()],
        "datasets_used": [s.strip() for s in result.datasets_used if s and s.strip()],
        "limitations": [s.strip() for s in result.limitations if s and s.strip()],
        "future_work": [s.strip() for s in result.future_work if s and s.strip()],
        "source_sections_found": found,
        "pdf_extraction_status": extract_status,
        "enriched_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "input_char_count": input_chars,
        "model_stop_reason": meta.get("model_stop_reason", ""),
        "cost_usd_estimate": cost,
    }
    _write_cache(pid, record)
    return "enriched", record


async def enrich_many(papers: list[dict[str, Any]], force: bool, concurrency: int, dry_run: bool) -> dict[str, dict[str, Any]]:
    sem = asyncio.Semaphore(concurrency)
    out: dict[str, dict[str, Any]] = {}
    stats = {"cached": 0, "enriched": 0, "stubbed": 0, "dry": 0}
    total = len(papers)
    idx = 0

    async def worker(paper: dict[str, Any]) -> None:
        nonlocal idx
        async with sem:
            # enrich_one is sync; wrap in to_thread to stay off the event loop.
            status, rec = await asyncio.to_thread(enrich_one, paper, force, dry_run)
            idx += 1
            stats[status] += 1
            out[paper["id"]] = rec
            summary = rec.get("one_sentence_summary", "")[:70]
            print(f"  [{idx:3d}/{total}] {status:8s} {paper['id']:16s}  {summary}")

    await asyncio.gather(*(worker(p) for p in papers))
    print(f"[stage2] Done. {stats}")
    return out


def write_aggregate() -> None:
    """Rebuild data/enriched.json from all cache files."""
    enriched: dict[str, Any] = {}
    for p in sorted(CACHE_DIR.glob("*.json")):
        try:
            data = json.loads(p.read_text())
            if data.get("schema_version") == SCHEMA_VERSION and data.get("id"):
                enriched[data["id"]] = data
        except Exception:
            continue
    doc = {
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "model": MODEL,
        "schema_version": SCHEMA_VERSION,
        "enriched": enriched,
    }
    out_path = DATA_DIR / "enriched.json"
    out_path.write_text(
        json.dumps(doc, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    )
    print(f"[stage2] Wrote {out_path.relative_to(REPO_ROOT)} ({len(enriched)} papers)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="Re-enrich even if cache exists")
    ap.add_argument("--only", type=str, default=None, help="Enrich a single paper ID")
    ap.add_argument("--limit", type=int, default=None, help="Smoke-test: enrich first N papers")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--dry-run", action="store_true", help="Download and extract but skip Haiku call")
    args = ap.parse_args()

    papers_path = DATA_DIR / "papers.json"
    papers = json.loads(papers_path.read_text())["papers"]

    if args.only:
        papers = [p for p in papers if p["id"] == args.only]
        if not papers:
            print(f"[stage2] No paper with id={args.only}")
            return 1

    if args.limit:
        papers = papers[: args.limit]

    print(f"[stage2] Enriching {len(papers)} paper(s) with {MODEL} (concurrency={args.concurrency}, dry_run={args.dry_run})")
    if not args.dry_run and not os.environ.get("ANTHROPIC_API_KEY"):
        print("[stage2] ERROR: ANTHROPIC_API_KEY not set (put it in .env)")
        return 2

    asyncio.run(enrich_many(papers, force=args.force, concurrency=args.concurrency, dry_run=args.dry_run))

    if not args.dry_run:
        write_aggregate()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
