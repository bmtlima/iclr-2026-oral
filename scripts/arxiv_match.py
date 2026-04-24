#!/usr/bin/env python3
"""Title-search arXiv for each paper in data/papers.json and write data/arxiv_matches.json.

Designed to be interruptible and cache-safe: each successful query is committed to the
output file every few papers, so Ctrl-C mid-run doesn't lose progress, and re-running
resumes from the existing cache.

Respects arXiv's rate-limit guidance (≥3s between requests, sequential).

Usage:
    python scripts/arxiv_match.py              # fill in missing entries only
    python scripts/arxiv_match.py --force      # re-check every paper
    python scripts/arxiv_match.py --limit 5    # smoke test on first 5
    python scripts/arxiv_match.py --sleep 3.0  # tune the per-request sleep
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from lib import arxiv as arxiv_lib  # noqa: E402

DATA_DIR = REPO_ROOT / "data"
PAPERS_PATH = DATA_DIR / "papers.json"
MATCHES_PATH = DATA_DIR / "arxiv_matches.json"

THRESHOLD = 90.0


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def load_matches() -> dict:
    if not MATCHES_PATH.exists():
        return {"generated_at": _now(), "matches": {}}
    try:
        data = json.loads(MATCHES_PATH.read_text())
        if "matches" not in data:
            data["matches"] = {}
        return data
    except Exception:
        return {"generated_at": _now(), "matches": {}}


def write_matches(data: dict) -> None:
    data["generated_at"] = _now()
    MATCHES_PATH.write_text(
        json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    )


def query_one(paper: dict) -> dict:
    """Query arXiv for a single paper. Returns the match record (may have null fields)."""
    title = (paper.get("title") or "").strip()
    try:
        entries = arxiv_lib.search_by_title(title, limit=5)
    except Exception as e:
        # Transient error: do NOT cache (so next run retries).
        raise RuntimeError(f"arXiv query failed for {paper.get('id')}: {e}") from e

    match = arxiv_lib.best_match(title, entries, threshold=THRESHOLD)
    if match is None:
        return {
            "arxiv_id": None,
            "arxiv_url": None,
            "match_confidence": (
                max((arxiv_lib.fuzz.WRatio(arxiv_lib.normalize_title(title),
                                           arxiv_lib.normalize_title(e.title))
                     for e in entries), default=None)
                if entries else None
            ),
            "matched_title": None,
            "checked_at": _now(),
        }

    entry, score = match
    return {
        "arxiv_id": entry.arxiv_id,
        "arxiv_url": f"https://arxiv.org/abs/{entry.arxiv_id}",
        "match_confidence": round(float(score), 1),
        "matched_title": entry.title,
        "checked_at": _now(),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="Re-query every paper, even if cached")
    ap.add_argument("--limit", type=int, default=None, help="Only process the first N papers")
    ap.add_argument("--sleep", type=float, default=3.0, help="Seconds between requests (arXiv asks ≥3)")
    args = ap.parse_args()

    papers = json.loads(PAPERS_PATH.read_text())["papers"]
    if args.limit:
        papers = papers[: args.limit]

    data = load_matches()
    matches = data["matches"]

    total = len(papers)
    to_query = [p for p in papers if args.force or p["id"] not in matches]
    print(f"[arxiv] {len(to_query)}/{total} papers to query (others cached)")
    if not to_query:
        print("[arxiv] Nothing to do. Use --force to re-query.")
        return 0

    matched = 0
    unmatched = 0
    errors = 0
    for i, p in enumerate(to_query, start=1):
        pid = p["id"]
        try:
            rec = query_one(p)
        except Exception as e:
            errors += 1
            print(f"  [{i:3d}/{len(to_query)}] ERROR  {pid}  {e}")
            # Keep going; don't cache the failure.
            time.sleep(args.sleep)
            continue

        matches[pid] = rec
        if rec.get("arxiv_id"):
            matched += 1
            title = (rec.get("matched_title") or "")[:60]
            score = rec.get("match_confidence")
            print(f"  [{i:3d}/{len(to_query)}] ok    {pid}  {score}  {title}")
        else:
            unmatched += 1
            best = rec.get("match_confidence")
            tag = f"no-match (best={best})" if best is not None else "no-match (no results)"
            print(f"  [{i:3d}/{len(to_query)}] --    {pid}  {tag}")

        # Persist every 10 papers so Ctrl-C doesn't lose progress.
        if i % 10 == 0:
            write_matches(data)

        if i < len(to_query):  # don't sleep after the last one
            time.sleep(args.sleep)

    write_matches(data)
    print(f"[arxiv] Done. matched={matched} unmatched={unmatched} errors={errors}")
    return 0 if errors == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
