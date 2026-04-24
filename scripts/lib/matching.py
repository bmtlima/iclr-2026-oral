"""Fuzzy title join between OpenReview papers and iclr.cc entries."""
from __future__ import annotations

import re
from typing import Any

from rapidfuzz import fuzz, process


def normalize_title(title: str) -> str:
    if not title:
        return ""
    t = title.lower()
    t = re.sub(r"[^\w\s]", " ", t)
    t = re.sub(r"\s+", " ", t)
    return t.strip()


def join(
    or_papers: list[dict[str, Any]],
    iclrcc_entries: list[Any],  # list[IclrccEntry]
    exact_threshold: int = 95,
    fuzzy_threshold: int = 88,
    ambiguous_gap: int = 2,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return (papers_with_session, iclrcc_orphans).

    Each paper dict is augmented with session_date/start/end, room, iclrcc_url,
    match_confidence, match_method.
    """
    iclrcc_by_norm: dict[str, Any] = {}
    for e in iclrcc_entries:
        key = normalize_title(e.title)
        if key and key not in iclrcc_by_norm:
            iclrcc_by_norm[key] = e

    iclrcc_norm_titles = list(iclrcc_by_norm.keys())
    matched_iclrcc_keys: set[str] = set()

    augmented: list[dict[str, Any]] = []
    for p in or_papers:
        out = dict(p)
        out["session_date"] = None
        out["session_start"] = None
        out["session_end"] = None
        out["room"] = None
        out["iclrcc_url"] = None
        out["match_confidence"] = None
        out["match_method"] = "unmatched"

        norm = normalize_title(p.get("title") or "")
        if not norm or not iclrcc_norm_titles:
            augmented.append(out)
            continue

        # Top-2 to detect ambiguity.
        hits = process.extract(
            norm, iclrcc_norm_titles, scorer=fuzz.WRatio, limit=2
        )
        if not hits:
            augmented.append(out)
            continue

        best = hits[0]
        best_title, best_score, _ = best
        second_score = hits[1][1] if len(hits) > 1 else 0.0

        if best_score >= exact_threshold:
            method = "exact"
        elif best_score >= fuzzy_threshold:
            method = "fuzzy"
        else:
            method = "unmatched"

        # Ambiguity check only when we'd otherwise accept.
        if method in ("exact", "fuzzy") and (best_score - second_score) < ambiguous_gap and second_score >= fuzzy_threshold:
            method = "ambiguous"

        if method in ("exact", "fuzzy"):
            entry = iclrcc_by_norm[best_title]
            out["session_date"] = entry.session_date
            out["session_start"] = entry.session_start
            out["session_end"] = entry.session_end
            out["room"] = entry.room
            out["iclrcc_url"] = entry.iclrcc_url
            matched_iclrcc_keys.add(best_title)

        out["match_confidence"] = round(float(best_score), 1)
        out["match_method"] = method
        augmented.append(out)

    # iclr.cc entries without a corresponding OpenReview paper
    orphans: list[dict[str, Any]] = []
    for key, e in iclrcc_by_norm.items():
        if key in matched_iclrcc_keys:
            continue
        orphans.append({
            "title": e.title,
            "authors": e.authors,
            "session_date": e.session_date,
            "session_start": e.session_start,
            "session_end": e.session_end,
            "room": e.room,
            "iclrcc_url": e.iclrcc_url,
        })

    return augmented, orphans
