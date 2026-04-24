#!/usr/bin/env python3
"""Prepare per-paper input text files for subagent-based enrichment.

For every paper in data/papers.json, downloads the PDF (if not cached),
extracts text, finds Discussion/Conclusion/Limitations/Future-Work sections,
and writes the model-ready input to .cache/prepared/{id}.txt.

Also writes .cache/prepared/manifest.json with per-paper status + found flags
so the downstream subagents (and the aggregate step) know what to expect.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from lib import pdf as pdf_lib  # noqa: E402

DATA_DIR = REPO_ROOT / "data"
PREPARED_DIR = REPO_ROOT / ".cache" / "prepared"
PREPARED_DIR.mkdir(parents=True, exist_ok=True)


def main() -> int:
    papers = json.loads((DATA_DIR / "papers.json").read_text())["papers"]
    manifest: dict[str, dict] = {}
    stats = {"ok": 0, "partial": 0, "failed": 0, "skipped": 0}

    for i, p in enumerate(papers):
        pid = p["id"]
        title = p.get("title") or ""
        authors = p.get("authors") or []
        abstract = p.get("abstract") or ""
        pdf_url = p.get("pdf_url")

        text = ""
        pdf_ok = False
        if pdf_url:
            path = pdf_lib.download_pdf(pdf_url, pid)
            if path is not None:
                text, _ = pdf_lib.extract_text(path, pid)
                pdf_ok = bool(text.strip())

        body, found, status = pdf_lib.assemble_input(title, authors, abstract, text)
        stats[status] = stats.get(status, 0) + 1

        (PREPARED_DIR / f"{pid}.txt").write_text(body, encoding="utf-8")
        manifest[pid] = {
            "title": title,
            "status": status,
            "found": found,
            "char_count": len(body),
            "pdf_downloaded": pdf_ok,
        }
        if (i + 1) % 25 == 0:
            print(f"[prep]   {i + 1}/{len(papers)} papers prepared")

    (PREPARED_DIR / "manifest.json").write_text(
        json.dumps({"papers": manifest, "stats": stats}, indent=2, ensure_ascii=False) + "\n"
    )
    print(f"[prep] Done. {stats}")
    print(f"[prep] Wrote {len(manifest)} files to {PREPARED_DIR.relative_to(REPO_ROOT)}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
