"""Scraper for https://iclr.cc/virtual/2026/events/oral.

The page wraps each oral session in a card with the following structure:

  <h3 class="event-title">
    <a href="/virtual/2026/oral/<id>">{title}</a>
  </h3>
  <div class="event-speakers">Author1 · Author2 · …</div>
  <div class="event-meta-row">
    <span class="meta-pill time">
      <span class="touchup-time">Apr 23, 6:30 AM - 6:40 AM</span>
    </span>
    <span class="meta-pill"><span>201 C</span></span>
  </div>

We parse each h3.event-title as an anchor and walk to the next meta-row / speakers
inside the same card. The card container isn't guaranteed to have a stable class,
so we anchor off event-title instead.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import httpx
from selectolax.parser import HTMLParser, Node

URL = "https://iclr.cc/virtual/2026/events/oral"
BASE = "https://iclr.cc"

# Printed format: "Apr 23, 6:30 AM - 6:40 AM"
DATE_RE = re.compile(
    r"(?P<mon>Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+(?P<day>\d{1,2}),\s*"
    r"(?P<start_h>\d{1,2}):(?P<start_m>\d{2})\s*(?P<start_mer>AM|PM)\s*[-–]\s*"
    r"(?P<end_h>\d{1,2}):(?P<end_m>\d{2})\s*(?P<end_mer>AM|PM)",
    re.IGNORECASE,
)

MONTHS = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}


@dataclass
class IclrccEntry:
    title: str
    authors: list[str]
    session_date: str | None
    session_start: str | None
    session_end: str | None
    room: str | None
    iclrcc_url: str | None


def _to_24h(h: int, m: int, meridian: str) -> str:
    meridian = meridian.upper()
    if meridian == "AM":
        hh = 0 if h == 12 else h
    else:
        hh = 12 if h == 12 else h + 12
    return f"{hh:02d}:{m:02d}"


def _parse_date(text: str, year: int = 2026) -> tuple[str | None, str | None, str | None]:
    m = DATE_RE.search(text or "")
    if not m:
        return None, None, None
    mon = MONTHS[m.group("mon").lower()]
    day = int(m.group("day"))
    start = _to_24h(int(m.group("start_h")), int(m.group("start_m")), m.group("start_mer"))
    end = _to_24h(int(m.group("end_h")), int(m.group("end_m")), m.group("end_mer"))
    return f"{year:04d}-{mon:02d}-{day:02d}", start, end


def fetch_html(timeout: float = 30.0) -> str:
    with httpx.Client(timeout=timeout, follow_redirects=True, headers={
        "User-Agent": "iclr-2026-oral-explorer/0.1 (+https://github.com/bmtlima/iclr-2026-oral)"
    }) as client:
        resp = client.get(URL)
        resp.raise_for_status()
        return resp.text


def _card_container(title_node: Node) -> Node | None:
    """Walk up from an event-title h3 until we find an ancestor that also contains
    an event-meta-row (i.e., the full card)."""
    node: Node | None = title_node.parent
    for _ in range(6):
        if node is None:
            return None
        meta = node.css_first("div.event-meta-row")
        if meta is not None:
            return node
        node = node.parent
    return None


def parse_entries(html: str) -> list[IclrccEntry]:
    tree = HTMLParser(html)
    entries: list[IclrccEntry] = []
    seen_titles: set[str] = set()

    for h3 in tree.css("h3.event-title"):
        a = h3.css_first("a")
        if a is None:
            continue
        href = a.attributes.get("href") or ""
        if not href or "/virtual/2026/" not in href:
            continue
        title = re.sub(r"\s+", " ", (a.text(deep=True) or "").strip())
        if not title or title in seen_titles:
            continue

        card = _card_container(h3) or h3.parent
        if card is None:
            continue

        # Authors
        authors: list[str] = []
        sp = card.css_first("div.event-speakers")
        if sp is not None:
            raw = re.sub(r"\s+", " ", (sp.text(deep=True) or "").strip())
            parts = re.split(r"\s*[·•⋅]\s*", raw)
            authors = [p.strip() for p in parts if p.strip() and len(p.strip()) < 160]

        # Time
        session_date = session_start = session_end = None
        t = card.css_first("span.touchup-time, span.meta-pill.time")
        if t is not None:
            session_date, session_start, session_end = _parse_date(t.text(deep=True) or "")
        if session_date is None:
            # Last resort: scan the whole card text.
            session_date, session_start, session_end = _parse_date(card.text(deep=True) or "")

        # Location: the non-time meta-pill. Try scoped first, then fall back to any
        # meta-pill whose text doesn't parse as a date.
        room: str | None = None
        meta_row = card.css_first("div.event-meta-row")
        if meta_row is not None:
            for pill in meta_row.css("span.meta-pill"):
                cls = pill.attributes.get("class") or ""
                if "time" in cls:
                    continue
                pill_text = re.sub(r"\s+", " ", (pill.text(deep=True) or "").strip())
                if pill_text and not DATE_RE.search(pill_text):
                    room = pill_text
                    break

        entries.append(IclrccEntry(
            title=title,
            authors=authors,
            session_date=session_date,
            session_start=session_start,
            session_end=session_end,
            room=room,
            iclrcc_url=(BASE + href) if href.startswith("/") else href,
        ))
        seen_titles.add(title)

    return entries


def fetch_and_parse() -> list[IclrccEntry]:
    return parse_entries(fetch_html())
