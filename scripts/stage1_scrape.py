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
