#!/usr/bin/env python3
"""Stage 1: scrape OpenReview + iclr.cc, join, assign topics, write data/papers.json."""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

from lib import openreview, iclrcc, matching
from assign_topics import assign_all

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
PAPERS_PATH = DATA_DIR / "papers.json"
UNMATCHED_PATH = DATA_DIR / "unmatched.json"


def main() -> int:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("[stage1] Fetching OpenReview notes…")
    or_papers = openreview.fetch_all_oral_papers()
    print(f"[stage1]   got {len(or_papers)} papers from OpenReview")

    print("[stage1] Fetching iclr.cc oral schedule…")
    try:
        iclrcc_entries = iclrcc.fetch_and_parse()
        print(f"[stage1]   got {len(iclrcc_entries)} entries from iclr.cc")
    except Exception as e:
        print(f"[stage1]   WARNING iclr.cc fetch failed ({e}); continuing without session data")
        iclrcc_entries = []

    print("[stage1] Joining by fuzzy title…")
    papers, orphans = matching.join(or_papers, iclrcc_entries)

    match_counts: dict[str, int] = {}
    for p in papers:
        match_counts[p["match_method"]] = match_counts.get(p["match_method"], 0) + 1
    print(f"[stage1]   join results: {match_counts}")
    if orphans:
        print(f"[stage1]   {len(orphans)} iclr.cc entries with no OpenReview match")

    print("[stage1] Assigning topics…")
    papers = assign_all(papers)

    topic_counts: dict[str, int] = {}
    for p in papers:
        topic_counts[p["topic_slug"]] = topic_counts.get(p["topic_slug"], 0) + 1
    print(f"[stage1]   topic distribution: {topic_counts}")

    # Merge arXiv matches (optional; written by scripts/arxiv_match.py).
    arxiv_path = DATA_DIR / "arxiv_matches.json"
    arxiv_map: dict[str, dict] = {}
    if arxiv_path.exists():
        try:
            arxiv_map = json.loads(arxiv_path.read_text()).get("matches", {}) or {}
            have = sum(1 for v in arxiv_map.values() if v.get("arxiv_id"))
            print(f"[stage1]   merging arXiv matches: {have}/{len(arxiv_map)} with IDs")
        except Exception as e:
            print(f"[stage1]   WARNING could not read {arxiv_path.name}: {e}")
            arxiv_map = {}
    for p in papers:
        entry = arxiv_map.get(p["id"]) or {}
        p["arxiv_id"] = entry.get("arxiv_id")
        p["arxiv_url"] = entry.get("arxiv_url")
        p["arxiv_match_confidence"] = entry.get("match_confidence")

    # Merge reviewer ratings (optional; written by scripts/fetch_ratings.py).
    ratings_path = DATA_DIR / "ratings.json"
    ratings_map: dict[str, dict] = {}
    if ratings_path.exists():
        try:
            ratings_map = json.loads(ratings_path.read_text()).get("ratings", {}) or {}
            have = sum(1 for v in ratings_map.values() if v.get("ratings_n"))
            print(f"[stage1]   merging ratings: {have}/{len(ratings_map)} with reviews")
        except Exception as e:
            print(f"[stage1]   WARNING could not read {ratings_path.name}: {e}")
            ratings_map = {}
    for p in papers:
        entry = ratings_map.get(p["id"]) or {}
        p["ratings_avg"] = entry.get("ratings_avg")
        p["ratings_min"] = entry.get("ratings_min")
        p["ratings_max"] = entry.get("ratings_max")
        p["ratings_n"] = entry.get("ratings_n", 0)

    # Stable ordering: by session_date then title (unknowns last).
    papers.sort(key=lambda p: (
        p.get("session_date") or "9999-99-99",
        p.get("session_start") or "99:99",
        (p.get("title") or "").lower(),
    ))

    out_papers = {
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "papers": papers,
    }
    PAPERS_PATH.write_text(
        json.dumps(out_papers, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    )
    print(f"[stage1] Wrote {PAPERS_PATH.relative_to(REPO_ROOT)} ({len(papers)} papers)")

    out_unmatched = {
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "iclrcc_orphans": orphans,
        "enrichment_failures": [],
    }
    # Preserve any existing enrichment_failures if file already exists.
    if UNMATCHED_PATH.exists():
        try:
            existing = json.loads(UNMATCHED_PATH.read_text())
            out_unmatched["enrichment_failures"] = existing.get("enrichment_failures", [])
        except Exception:
            pass
    UNMATCHED_PATH.write_text(
        json.dumps(out_unmatched, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    )
    print(f"[stage1] Wrote {UNMATCHED_PATH.relative_to(REPO_ROOT)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
