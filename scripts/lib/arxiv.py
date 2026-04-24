"""arXiv API client: title-search + match validation.

Usage:
    entries = search_by_title("Some paper title", limit=5)
    best = best_match("ICLR paper title", entries, threshold=90)
    if best:
        arxiv_id, confidence = best["id"], best["score"]

arXiv asks for ≥3s between requests (no concurrent calls from the same IP).
This module does not sleep on its own — the caller is responsible.
"""
from __future__ import annotations

import re
import urllib.parse
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Any

import httpx
from rapidfuzz import fuzz

from .matching import normalize_title

API_URL = "https://export.arxiv.org/api/query"
ATOM_NS = "{http://www.w3.org/2005/Atom}"

# Characters that tend to break arXiv's ti: search — strip before wrapping in quotes.
# Curly quotes, colons, slashes, pipes, and some LaTeX residue get stripped; spaces are kept.
_QUERY_STRIP_RE = re.compile(r'[:\\|"\'“”‘’]')


@dataclass
class ArxivEntry:
    arxiv_id: str          # version-stripped, e.g. "2506.01732"
    full_id: str           # with version, e.g. "2506.01732v2"
    title: str
    authors: list[str]
    updated: str           # ISO8601 string from Atom <updated>


def _query_string(title: str) -> str:
    """Produce a safe arXiv ti: query string for a given paper title."""
    cleaned = _QUERY_STRIP_RE.sub(" ", title)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    # ti:"..." forces a title-field match, much more accurate than an all-field search.
    return f'ti:"{cleaned}"'


def _parse_atom(xml_text: str) -> list[ArxivEntry]:
    """Parse arXiv's Atom response into a list of entries."""
    root = ET.fromstring(xml_text)
    entries: list[ArxivEntry] = []
    for entry in root.findall(f"{ATOM_NS}entry"):
        id_el = entry.find(f"{ATOM_NS}id")
        title_el = entry.find(f"{ATOM_NS}title")
        updated_el = entry.find(f"{ATOM_NS}updated")
        if id_el is None or title_el is None or id_el.text is None:
            continue
        # id is "http://arxiv.org/abs/2506.01732v2"
        url = id_el.text.strip()
        full_id = url.rsplit("/", 1)[-1]
        arxiv_id = re.sub(r"v\d+$", "", full_id)
        title = re.sub(r"\s+", " ", (title_el.text or "").strip())
        authors: list[str] = []
        for a in entry.findall(f"{ATOM_NS}author"):
            name_el = a.find(f"{ATOM_NS}name")
            if name_el is not None and name_el.text:
                authors.append(name_el.text.strip())
        entries.append(ArxivEntry(
            arxiv_id=arxiv_id,
            full_id=full_id,
            title=title,
            authors=authors,
            updated=(updated_el.text or "").strip() if updated_el is not None else "",
        ))
    return entries


def search_by_title(title: str, limit: int = 5, timeout: float = 30.0) -> list[ArxivEntry]:
    """Query arXiv API by title. Returns [] on empty results or transient errors."""
    if not title or not title.strip():
        return []
    params = {
        "search_query": _query_string(title),
        "start": 0,
        "max_results": limit,
    }
    url = f"{API_URL}?{urllib.parse.urlencode(params)}"
    with httpx.Client(timeout=timeout, follow_redirects=True, headers={
        "User-Agent": "iclr-2026-oral-explorer/0.1 (+https://github.com/bmtlima/iclr-2026-oral)"
    }) as client:
        resp = client.get(url)
        resp.raise_for_status()
        return _parse_atom(resp.text)


def best_match(
    or_title: str,
    entries: list[ArxivEntry],
    threshold: float = 90.0,
) -> tuple[ArxivEntry, float] | None:
    """Return (entry, score) if the best entry's title similarity >= threshold."""
    if not entries:
        return None
    target = normalize_title(or_title)
    best: tuple[ArxivEntry, float] | None = None
    for e in entries:
        score = fuzz.WRatio(target, normalize_title(e.title))
        if best is None or score > best[1]:
            best = (e, score)
    if best is not None and best[1] >= threshold:
        return best
    return None
