"""Heuristic topic assignment across title + abstract + keywords.

Manual overrides in data/topics.json.overrides always win.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
TOPICS_PATH = REPO_ROOT / "data" / "topics.json"


def _load_topics() -> tuple[list[dict[str, Any]], dict[str, str]]:
    data = json.loads(TOPICS_PATH.read_text())
    return data["topics"], data.get("overrides", {}) or {}


def _score(text: str, keywords: list[str]) -> int:
    if not text:
        return 0
    score = 0
    for kw in keywords:
        # Word-boundary match for single tokens; substring match for phrases with spaces.
        kw_l = kw.lower()
        if " " in kw_l or "-" in kw_l:
            if kw_l in text:
                score += 3
        else:
            if re.search(rf"\b{re.escape(kw_l)}\b", text):
                score += 2
    return score


def assign_topic(paper: dict[str, Any], topics: list[dict[str, Any]] | None = None, overrides: dict[str, str] | None = None) -> tuple[str, str]:
    """Return (topic_slug, topic_confidence)."""
    if topics is None or overrides is None:
        topics, overrides = _load_topics()

    pid = paper.get("id")
    if pid and pid in overrides:
        return overrides[pid], "manual"

    text_parts = [
        paper.get("title") or "",
        paper.get("abstract") or "",
        " ".join(paper.get("keywords") or []),
        paper.get("tldr") or "",
        paper.get("primary_area") or "",
    ]
    text = " ".join(text_parts).lower()

    best_slug = "uncategorized"
    best_score = 0
    for t in topics:
        s = _score(text, t.get("keywords", []))
        if s > best_score:
            best_score = s
            best_slug = t["slug"]

    return best_slug, "auto"


def assign_all(papers: list[dict[str, Any]]) -> list[dict[str, Any]]:
    topics, overrides = _load_topics()
    out = []
    for p in papers:
        slug, conf = assign_topic(p, topics, overrides)
        p2 = dict(p)
        p2["topic_slug"] = slug
        p2["topic_confidence"] = conf
        out.append(p2)
    return out


if __name__ == "__main__":
    papers_path = REPO_ROOT / "data" / "papers.json"
    doc = json.loads(papers_path.read_text())
    doc["papers"] = assign_all(doc["papers"])
    papers_path.write_text(
        json.dumps(doc, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    )
    print(f"Re-assigned topics for {len(doc['papers'])} papers")
