"""LLM-based topic classification for ICLR 2026 oral papers via OpenRouter.

Run:
    python scripts/assign_topics_llm.py                  # full run
    python scripts/assign_topics_llm.py --limit 3        # smoke-test (writes papers.json)
    python scripts/assign_topics_llm.py --limit 3 --dry-run  # no API calls, no write
    python scripts/assign_topics_llm.py --only VJZ477R89F --force

Environment:
    OPENROUTER_API_KEY    required for live calls (put it in .env)
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))
load_dotenv(REPO_ROOT / ".env")

from lib import openrouter as or_lib  # noqa: E402
from assign_topics import assign_topic  # noqa: E402

DATA_DIR = REPO_ROOT / "data"
PAPERS_PATH = DATA_DIR / "papers.json"
TOPICS_PATH = DATA_DIR / "topics.json"


def classify_one(
    paper: dict[str, Any],
    topics: list[dict[str, Any]],
    overrides: dict[str, str],
    force: bool,
    dry_run: bool,
) -> tuple[str, dict[str, Any], tuple[int, int]]:
    """Classify one paper. Returns (status, field_updates, (input_tokens, output_tokens))."""
    pid = paper.get("id", "")

    if pid in overrides or paper.get("topic_confidence") == "manual":
        return "skipped_manual", {}, (0, 0)

    if not force and paper.get("topic_confidence") == "llm":
        return "skipped_llm", {}, (0, 0)

    if dry_run:
        return "dry", {}, (0, 0)

    try:
        result, meta = or_lib.classify_paper(paper, topics)
        updates = {
            "topic_slug": result.primary,
            "topic_slug_secondary": result.secondary,
            "topic_confidence": "llm",
        }
        tokens = (meta.get("input_tokens", 0), meta.get("output_tokens", 0))
        return "classified", updates, tokens
    except RuntimeError:
        fallback_slug, fallback_conf = assign_topic(paper, topics, overrides)
        updates = {
            "topic_slug": fallback_slug,
            "topic_slug_secondary": None,
            "topic_confidence": fallback_conf,
        }
        print(f"  [warn] LLM failed for {pid}, falling back to heuristic → {fallback_slug}")
        return "fallback", updates, (0, 0)


async def classify_many(
    papers: list[dict[str, Any]],
    topics: list[dict[str, Any]],
    overrides: dict[str, str],
    force: bool,
    concurrency: int,
    dry_run: bool,
) -> tuple[list[dict[str, Any]], dict[str, int], tuple[int, int]]:
    sem = asyncio.Semaphore(concurrency)
    results: list[dict[str, Any]] = [dict(p) for p in papers]
    id_to_idx = {p["id"]: i for i, p in enumerate(results)}
    stats: dict[str, int] = {
        "skipped_manual": 0, "skipped_llm": 0, "dry": 0, "classified": 0, "fallback": 0,
    }
    total = len(papers)
    processed = 0
    total_input = 0
    total_output = 0
    lock = asyncio.Lock()

    async def worker(paper: dict[str, Any]) -> None:
        nonlocal processed, total_input, total_output
        async with sem:
            status, updates, tokens = await asyncio.to_thread(
                classify_one, paper, topics, overrides, force, dry_run
            )
            async with lock:
                processed += 1
                stats[status] += 1
                total_input += tokens[0]
                total_output += tokens[1]
                if updates:
                    results[id_to_idx[paper["id"]]].update(updates)
                slug = updates.get("topic_slug", paper.get("topic_slug", ""))
                secondary = updates.get("topic_slug_secondary") or ""
                sec_display = f" + {secondary}" if secondary else ""
                print(f"  [{processed:3d}/{total}] {status:16s} {paper['id']:16s}  {slug}{sec_display}")

    await asyncio.gather(*(worker(p) for p in papers))
    print(f"[llm-topics] Done. {stats}")
    return results, stats, (total_input, total_output)


def write_papers_atomic(doc: dict[str, Any], papers: list[dict[str, Any]]) -> None:
    doc_out = dict(doc)
    doc_out["papers"] = papers
    tmp_path = PAPERS_PATH.with_suffix(".json.tmp")
    tmp_path.write_text(
        json.dumps(doc_out, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    )
    tmp_path.replace(PAPERS_PATH)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="LLM-based topic classification for ICLR 2026 oral papers via OpenRouter."
    )
    ap.add_argument("--force", action="store_true",
                    help="Re-classify even papers already tagged topic_confidence='llm'")
    ap.add_argument("--only", type=str, default=None, help="Classify a single paper by ID")
    ap.add_argument("--limit", type=int, default=None, help="Classify first N papers (smoke-test)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print what would be classified without calling OpenRouter")
    ap.add_argument("--concurrency", type=int, default=20,
                    help="Max parallel OpenRouter calls (default: 20)")
    args = ap.parse_args()

    if not args.dry_run and not os.environ.get("OPENROUTER_API_KEY"):
        print("[llm-topics] ERROR: OPENROUTER_API_KEY not set (put it in .env)")
        return 2

    doc = json.loads(PAPERS_PATH.read_text())
    all_papers = doc["papers"]
    topics_doc = json.loads(TOPICS_PATH.read_text())
    topics = topics_doc["topics"]
    overrides = topics_doc.get("overrides") or {}

    target = all_papers
    if args.only:
        target = [p for p in all_papers if p["id"] == args.only]
        if not target:
            print(f"[llm-topics] No paper with id={args.only!r}")
            return 1
    elif args.limit:
        target = all_papers[: args.limit]

    will_classify = sum(
        1 for p in target
        if p["id"] not in overrides
        and p.get("topic_confidence") != "manual"
        and (args.force or p.get("topic_confidence") != "llm")
    )
    print(
        f"[llm-topics] {will_classify} paper(s) to classify with {or_lib.MODEL} "
        f"(concurrency={args.concurrency}, dry_run={args.dry_run})"
    )

    updated, stats, (in_tok, out_tok) = asyncio.run(
        classify_many(target, topics, overrides, args.force, args.concurrency, args.dry_run)
    )

    if args.only or args.limit:
        updated_by_id = {p["id"]: p for p in updated}
        final_papers = [updated_by_id.get(p["id"], p) for p in all_papers]
    else:
        final_papers = updated

    if not args.dry_run:
        write_papers_atomic(doc, final_papers)
        print(f"[llm-topics] Wrote {PAPERS_PATH.relative_to(REPO_ROOT)}")
    else:
        print("[llm-topics] Dry run — no files written.")

    if in_tok or out_tok:
        cost = or_lib.estimate_cost(in_tok, out_tok)
        print(
            f"[llm-topics] Cost estimate: ${cost:.4f} "
            f"({in_tok/1000:.1f}k input + {out_tok/1000:.1f}k output tokens, "
            f"{or_lib.MODEL} @ ${or_lib.PRICING_PER_M['input']}/${or_lib.PRICING_PER_M['output']} per 1M)"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
