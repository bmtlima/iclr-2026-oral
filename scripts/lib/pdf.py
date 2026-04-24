"""PDF download + text extraction + section targeting for ICLR 2026 Orals."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import httpx

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CACHE_PDFS = REPO_ROOT / ".cache" / "pdfs"
CACHE_TEXT = REPO_ROOT / ".cache" / "pdf_text"
CACHE_PDFS.mkdir(parents=True, exist_ok=True)
CACHE_TEXT.mkdir(parents=True, exist_ok=True)

# Header prefix: "5", "5.1", "D", or none. Followed by target keyword(s).
# Matches e.g. "5 DISCUSSION OF OUR RESULTS", "D LIMITATIONS AND FUTURE DIRECTIONS",
# "7 CONCLUSION", "Future Work". Line must be short (<= ~80 chars) to count as a header.
_HEADER_PREFIX = r"(?:\d+(?:\.\d+)*|[A-Z])\.?"
SECTION_HEADER_RE = re.compile(
    r"^[ \t]*(?:" + _HEADER_PREFIX + r"[ \t]+)?"
    r"(?P<key>conclusions?|discussions?|limitations?|future\s+work|"
    r"future\s+directions?|broader\s+impacts?|remarks?)"
    r"(?P<tail>[^\n]{0,60})?[ \t]*$",
    re.IGNORECASE | re.MULTILINE,
)
# Any header-like short line for terminating sections.
ANY_HEADER_RE = re.compile(
    r"^[ \t]*(?:\d+(?:\.\d+)*|[A-Z])\.?[ \t]+[A-Z][A-Z a-z\-&/:,]{2,60}[ \t]*$",
    re.MULTILINE,
)
REFERENCES_RE = re.compile(r"^\s*references\s*$", re.IGNORECASE | re.MULTILINE)


def _canonicalize(key: str, tail: str) -> list[str]:
    """Map a matched header to one or more canonical section names."""
    combined = f"{key} {tail or ''}".lower()
    out: list[str] = []
    if re.search(r"\blimit", combined):
        out.append("limitations")
    if re.search(r"\bfuture\s+(work|direction)", combined):
        out.append("future_work")
    if re.search(r"\bdiscuss", combined):
        out.append("discussion")
    if re.search(r"\bconclu", combined):
        out.append("conclusion")
    if re.search(r"\bbroader\s+impact", combined) and not out:
        # Broader impact often reads like limitations/future-work adjacent material.
        out.append("discussion")
    if re.search(r"\bremark", combined) and not out:
        out.append("discussion")
    return out


def download_pdf(pdf_url: str, paper_id: str, timeout: float = 60.0) -> Path | None:
    """Download PDF to .cache/pdfs/{id}.pdf. Skip if already present."""
    path = CACHE_PDFS / f"{paper_id}.pdf"
    if path.exists() and path.stat().st_size > 1024:
        return path
    try:
        with httpx.Client(timeout=timeout, follow_redirects=True, headers={
            "User-Agent": "iclr-2026-oral-explorer/0.1"
        }) as client:
            resp = client.get(pdf_url)
            resp.raise_for_status()
            ctype = resp.headers.get("content-type", "")
            if "pdf" not in ctype.lower() and not resp.content.startswith(b"%PDF"):
                return None
            path.write_bytes(resp.content)
            return path
    except Exception:
        return None


def extract_text(pdf_path: Path, paper_id: str) -> tuple[str, str]:
    """Return (text, status) where status is 'ok' or 'failed'. Caches to .txt."""
    txt_path = CACHE_TEXT / f"{paper_id}.txt"
    if txt_path.exists() and txt_path.stat().st_size > 500:
        return txt_path.read_text(encoding="utf-8", errors="replace"), "ok"

    # Primary: pdfplumber
    text = ""
    try:
        import pdfplumber  # type: ignore
        with pdfplumber.open(str(pdf_path)) as pdf:
            parts = []
            for page in pdf.pages:
                try:
                    t = page.extract_text() or ""
                except Exception:
                    t = ""
                parts.append(t)
            text = "\n".join(parts)
    except Exception:
        text = ""

    # Fallback: pypdf
    if not text.strip():
        try:
            from pypdf import PdfReader  # type: ignore
            reader = PdfReader(str(pdf_path))
            parts = [(page.extract_text() or "") for page in reader.pages]
            text = "\n".join(parts)
        except Exception:
            text = ""

    if not text.strip():
        return "", "failed"

    txt_path.write_text(text, encoding="utf-8")
    return text, "ok"


def _find_sections(text: str, max_section_chars: int = 4000) -> tuple[dict[str, str], dict[str, bool]]:
    """Locate Conclusion/Discussion/Limitations/Future Work sections.

    Returns (sections_dict, found_flags).
    """
    found = {"conclusion": False, "limitations": False, "future_work": False, "discussion": False}
    sections: dict[str, str] = {}

    # Do NOT cut at "References" — appendices after references often contain the
    # most useful Limitations / Future Work content (ICLR conventions). The header
    # regex is strict enough that reference-list entries rarely false-match.
    body = text

    headers: list[tuple[int, list[str]]] = []  # (body_end_pos, canonical_names)
    for m in SECTION_HEADER_RE.finditer(body):
        canons = _canonicalize(m.group("key"), m.group("tail") or "")
        if canons:
            headers.append((m.end(), canons))

    if not headers:
        return sections, found

    for i, (start, canons) in enumerate(headers):
        # Next section header of any kind in our target set.
        next_target = len(body)
        for j in range(i + 1, len(headers)):
            if headers[j][0] > start:
                next_target = headers[j][0]
                break
        # Also stop at any other header-like line encountered in between.
        m_any = ANY_HEADER_RE.search(body, pos=start + 1)
        next_pos = next_target
        if m_any and start < m_any.start() < next_pos:
            next_pos = m_any.start()

        chunk = body[start:next_pos].strip()
        if len(chunk) > max_section_chars:
            chunk = chunk[:max_section_chars]
        if not chunk:
            continue
        for canon in canons:
            if canon in sections:
                if len(chunk) > len(sections[canon]):
                    sections[canon] = chunk
            else:
                sections[canon] = chunk
            found[canon] = True

    return sections, found


def assemble_input(
    title: str,
    authors: list[str],
    abstract: str,
    text: str,
    fallback_tail_chars: int = 3000,
) -> tuple[str, dict[str, bool], str]:
    """Build the Haiku input.

    Returns (input_text, found_flags, extraction_status).
    extraction_status is 'ok' when at least one target section was found,
    'partial' when we used the tail-of-paper fallback, 'failed' when even that
    was unavailable (empty text → only title+abstract).
    """
    sections, found = _find_sections(text)

    parts: list[str] = []
    parts.append(f"Paper title: {title}")
    parts.append(f"Authors: {', '.join(authors[:20])}")
    if abstract:
        parts.append("Abstract:\n" + abstract.strip())

    order = ["discussion", "conclusion", "limitations", "future_work"]
    for key in order:
        if key in sections:
            parts.append(f"=== {key.upper()} ===\n{sections[key]}")

    status = "ok" if any(found.values()) else "partial"

    if not any(found.values()):
        # Fallback: last N chars before references.
        if text:
            refs_match = REFERENCES_RE.search(text)
            body = text[: refs_match.start()] if refs_match else text
            tail = body[-fallback_tail_chars:].strip()
            if tail:
                parts.append(f"=== TAIL OF PAPER (pre-references, fallback) ===\n{tail}")
                status = "partial"
        if not text:
            status = "failed"  # only title+abstract available

    return "\n\n".join(parts), found, status
