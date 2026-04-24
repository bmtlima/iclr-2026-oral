# ICLR 2026 Orals

An unofficial browser for the 223 Oral papers at ICLR 2026.

Live: (deploy to Vercel to get a URL)

## Stack

- **Astro** static site (no server, no DB)
- **Tailwind** for styling, dark-mode default
- **MiniSearch** client-side fuzzy search (keyboard `/` to focus)
- **Python** pipeline that scrapes OpenReview + iclr.cc, extracts PDF sections, calls **Claude Haiku 4.5** for per-paper summaries, and **Claude Sonnet 4.6** for cross-paper trend themes

## Local dev

```bash
# one-time: Node 20 + Python 3.12
nvm use            # .nvmrc says 20
python3 -m venv .venv
.venv/bin/pip install -r scripts/requirements.txt
npm install

# data pipeline (stage 1 is free, stages 2+3 need an API key)
.venv/bin/python scripts/stage1_scrape.py       # -> data/papers.json

cp .env.example .env                            # then add ANTHROPIC_API_KEY
.venv/bin/python scripts/stage2_enrich.py       # -> .cache/enriched/*.json + data/enriched.json (~$2)
.venv/bin/python scripts/stage3_trends.py       # -> data/trends.json (~$0.12)

# site
npm run dev                                     # http://localhost:4321
npm run build                                   # static output in dist/
```

## Deploy

Push to GitHub, import the repo in Vercel. Astro is auto-detected; no env vars needed (the LLM pipeline runs offline and commits its JSON output).

GitHub Actions re-scrapes Stage 1 every 6 hours and auto-commits `data/papers.json` if it changed; Vercel redeploys on each commit. Stages 2 and 3 stay manual (they call paid APIs).

## Data labeling

- Plain prose = original from OpenReview.
- Dashed-border "Auto-generated" blocks = AI output (model named inline).
- Blockquotes "from the paper" inside AI blocks = verbatim excerpts selected by the AI.

See `/about` on the live site for the full methodology.

## Data sources

- OpenReview API v2: `content.venue = "ICLR 2026 Oral"`
- iclr.cc virtual schedule: `https://iclr.cc/virtual/2026/events/oral`

Join is fuzzy by title (rapidfuzz WRatio ≥ 88).

## Layout

```
scripts/                  # Python data pipeline
  stage1_scrape.py
  stage2_enrich.py
  stage3_trends.py
  lib/                    # openreview, iclrcc, matching, pdf, claude helpers
data/                     # committed JSON the site reads
src/                      # Astro site
.cache/enriched/          # COMMITTED per-paper LLM cache (keyed by OpenReview id)
.cache/pdfs/              # gitignored
.cache/pdf_text/          # gitignored
.github/workflows/        # re-scrape cron
```

Not affiliated with ICLR or Anthropic.
