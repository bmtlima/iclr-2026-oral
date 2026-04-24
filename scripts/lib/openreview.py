"""OpenReview API v2 client for ICLR 2026 Oral papers."""
from __future__ import annotations

import time
from typing import Any, Iterator

import httpx

API_BASE = "https://api2.openreview.net"
VENUE_STRING = "ICLR 2026 Oral"


def _val(field: Any) -> Any:
    """OpenReview v2 wraps every content field as {'value': ...}. Unwrap safely."""
    if isinstance(field, dict) and "value" in field:
        return field["value"]
    return field


def fetch_notes(limit: int = 1000, timeout: float = 30.0) -> list[dict[str, Any]]:
    """Paginate notes for venue=ICLR 2026 Oral. Returns list of raw note dicts."""
    all_notes: list[dict[str, Any]] = []
    offset = 0
    with httpx.Client(timeout=timeout) as client:
        while True:
            resp = client.get(
                f"{API_BASE}/notes",
                params={
                    "content.venue": VENUE_STRING,
                    "limit": limit,
                    "offset": offset,
                },
            )
            resp.raise_for_status()
            data = resp.json()
            notes = data.get("notes", [])
            all_notes.extend(notes)
            if len(notes) < limit:
                break
            offset += len(notes)
            time.sleep(0.3)  # polite pacing
    return all_notes


def flatten_note(note: dict[str, Any]) -> dict[str, Any]:
    """Flatten a raw OpenReview note into our canonical paper shape (pre-join)."""
    content = note.get("content", {}) or {}

    def g(key: str, default: Any = None) -> Any:
        return _val(content.get(key, default))

    nid = note.get("id")
    pdf_rel = g("pdf")
    pdf_url: str | None = None
    if isinstance(pdf_rel, str) and pdf_rel:
        if pdf_rel.startswith("http"):
            pdf_url = pdf_rel
        else:
            pdf_url = f"https://openreview.net{pdf_rel}"

    authors = g("authors", []) or []
    if isinstance(authors, str):
        authors = [authors]
    authorids = g("authorids", []) or []
    if isinstance(authorids, str):
        authorids = [authorids]
    keywords = g("keywords", []) or []
    if isinstance(keywords, str):
        keywords = [keywords]

    return {
        "id": nid,
        "forum": note.get("forum") or nid,
        "openreview_url": f"https://openreview.net/forum?id={nid}" if nid else None,
        "pdf_url": pdf_url,
        "title": (g("title") or "").strip(),
        "authors": [str(a).strip() for a in authors if a],
        "authorids": [str(a).strip() for a in authorids if a],
        "abstract": (g("abstract") or "").strip(),
        "tldr": (g("TLDR") or None) or None,
        "keywords": [str(k).strip() for k in keywords if k],
        "primary_area": g("primary_area"),
    }


def fetch_all_oral_papers() -> list[dict[str, Any]]:
    notes = fetch_notes()
    return [flatten_note(n) for n in notes if n.get("id")]


def fetch_forum_ratings(forum_id: str, timeout: float = 30.0, max_retries: int = 5) -> list[float]:
    """Return the list of numeric ratings from Official_Review notes on a forum.

    Retries with exponential backoff on 429/5xx.
    """
    notes: list[dict[str, Any]] = []
    with httpx.Client(timeout=timeout) as client:
        for attempt in range(max_retries):
            resp = client.get(f"{API_BASE}/notes", params={"forum": forum_id})
            if resp.status_code == 429 or resp.status_code >= 500:
                backoff = min(60.0, 2.0 ** attempt)
                time.sleep(backoff)
                continue
            resp.raise_for_status()
            notes = resp.json().get("notes", []) or []
            break
        else:
            resp.raise_for_status()  # re-raise last error

    ratings: list[float] = []
    for n in notes:
        invs = n.get("invitations") or ([n.get("invitation")] if n.get("invitation") else [])
        if not any("Official_Review" in str(i) for i in invs):
            continue
        raw = _val((n.get("content") or {}).get("rating"))
        if raw is None:
            continue
        try:
            ratings.append(float(raw))
        except (TypeError, ValueError):
            continue
    return ratings
