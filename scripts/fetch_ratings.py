#!/usr/bin/env python3
"""Fetch average reviewer ratings from OpenReview for each paper in data/papers.json.

Writes both:
  - data/ratings.json (side-channel, merged by stage1_scrape.py on re-runs)
  - data/papers.json  (patched in place with ratings_avg/min/max/n fields)
"""
from __future__ import annotations

import datetime as dt
import json
import time
from pathlib import Path

from lib import openreview

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
PAPERS_PATH = DATA_DIR / "papers.json"
RATINGS_PATH = DATA_DIR / "ratings.json"


def _stats(ratings: list[float]) -> dict[str, float | int | None]:
    if not ratings:
        return {"ratings_avg": None, "ratings_min": None, "ratings_max": None, "ratings_n": 0}
    return {
        "ratings_avg": round(sum(ratings) / len(ratings), 2),
        "ratings_min": min(ratings),
        "ratings_max": max(ratings),
        "ratings_n": len(ratings),
    }


def main() -> int:
    doc = json.loads(PAPERS_PATH.read_text())
    papers = doc.get("papers", [])
    print(f"[ratings] fetching reviews for {len(papers)} papers…")

    # Resume from any prior run so rate-limited batches can be retried without losing progress.
    ratings_map: dict[str, dict] = {}
    if RATINGS_PATH.exists():
        try:
            ratings_map = json.loads(RATINGS_PATH.read_text()).get("ratings", {}) or {}
            resumed = sum(1 for v in ratings_map.values() if v.get("ratings_n"))
            print(f"[ratings] resuming: {resumed}/{len(ratings_map)} already have reviews")
        except Exception:
            ratings_map = {}

    for i, p in enumerate(papers, 1):
        forum = p.get("forum") or p.get("id")
        if not forum:
            continue
        existing = ratings_map.get(p["id"])
        if existing and existing.get("ratings_n"):
            continue  # already fetched successfully; skip
        try:
            values = openreview.fetch_forum_ratings(forum)
        except Exception as e:
            print(f"[ratings]   {i}/{len(papers)} {forum} FAILED: {e}")
            values = []
        s = _stats(values)
        ratings_map[p["id"]] = {"forum": forum, **s, "ratings_raw": values}
        if i % 25 == 0 or i == len(papers):
            print(f"[ratings]   {i}/{len(papers)}")
        time.sleep(0.5)  # polite pacing

    # Patch papers.json in place.
    for p in papers:
        entry = ratings_map.get(p["id"]) or {}
        p["ratings_avg"] = entry.get("ratings_avg")
        p["ratings_min"] = entry.get("ratings_min")
        p["ratings_max"] = entry.get("ratings_max")
        p["ratings_n"] = entry.get("ratings_n", 0)

    PAPERS_PATH.write_text(
        json.dumps(doc, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    )
    print(f"[ratings] patched {PAPERS_PATH.relative_to(REPO_ROOT)}")

    RATINGS_PATH.write_text(
        json.dumps(
            {
                "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                "ratings": ratings_map,
            },
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
        )
        + "\n"
    )
    print(f"[ratings] wrote {RATINGS_PATH.relative_to(REPO_ROOT)}")

    with_ratings = sum(1 for v in ratings_map.values() if v.get("ratings_n"))
    print(f"[ratings] done: {with_ratings}/{len(ratings_map)} papers have at least one review")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
