"""OpenRouter client for LLM-based paper topic classification."""
from __future__ import annotations

import json
import os
import re
from typing import Any

from pydantic import BaseModel, ConfigDict
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

try:
    import openai  # type: ignore
except ImportError:  # pragma: no cover
    openai = None  # type: ignore

MODEL = "google/gemini-2.5-flash"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
PRICING_PER_M = {"input": 0.15, "output": 0.60}

_client: Any = None


class TopicResult(BaseModel):
    model_config = ConfigDict(extra="ignore")  # Gemini may return extra "reasoning" key
    primary: str
    secondary: str | None = None


SYSTEM_PROMPT = """You are a research librarian organizing ICLR 2026 oral papers by topic.
You assign each paper to exactly one primary topic category, and optionally one secondary category.
You only return valid JSON. No prose, no markdown fences, no commentary.
Your goal is to help researchers *discover* papers when browsing by topic — so file under the category a browser would find most useful, not necessarily the paper's primary methodological contribution."""

USER_TEMPLATE = """Classify the following ICLR 2026 oral paper into topic categories.

Available topic categories (slug → description):
{topic_list}

Return a single JSON object:
{{"primary": "<slug>", "secondary": "<slug or null>"}}

Rules:
- "primary" must be exactly one slug from the list above. No other value is valid.
- "secondary" is a second slug (different from primary) if the paper meaningfully spans two categories; otherwise null.
- File under the category a *researcher browsing by topic* would look in.

Paper to classify:
Title: {title}
Authors keywords: {keywords}
Primary area (author-declared): {primary_area}
TL;DR: {tldr}
Abstract: {abstract}"""


def _require_client() -> Any:
    global _client
    if _client is not None:
        return _client
    if openai is None:
        raise RuntimeError("openai SDK not installed. Add 'openai>=1.0' to scripts/requirements.txt and pip install.")
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY not set. Put it in .env (at repo root).")
    _client = openai.OpenAI(api_key=api_key, base_url=OPENROUTER_BASE_URL)
    return _client


def _extract_json_object(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```(?:json|javascript|js)?\s*\n?", "", t, count=1)
        t = re.sub(r"\n?```\s*$", "", t, count=1)
    return t.strip()


def _retryable_exceptions() -> tuple[type[Exception], ...]:
    if openai is None:
        return ()
    return (
        openai.RateLimitError,
        openai.APIConnectionError,
        openai.InternalServerError,
        openai.APITimeoutError,
    )


def estimate_cost(input_tokens: int, output_tokens: int) -> float:
    return round(
        (input_tokens * PRICING_PER_M["input"] + output_tokens * PRICING_PER_M["output"]) / 1_000_000,
        6,
    )


def classify_paper(
    paper: dict[str, Any],
    topics: list[dict[str, Any]],
    model: str = MODEL,
) -> tuple[TopicResult, dict[str, Any]]:
    """Classify a paper into topic categories. Returns (TopicResult, metadata)."""
    client = _require_client()
    valid_slugs = {t["slug"] for t in topics}

    topic_list = "\n".join(f"{t['slug']} → {t['description']}" for t in topics)
    user_msg = USER_TEMPLATE.format(
        topic_list=topic_list,
        title=paper.get("title") or "",
        keywords=", ".join(paper.get("keywords") or []),
        primary_area=paper.get("primary_area") or "",
        tldr=paper.get("tldr") or "",
        abstract=paper.get("abstract") or "",
    )

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]

    meta = {"input_tokens": 0, "output_tokens": 0}

    @retry(
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=1, min=2, max=20),
        retry=retry_if_exception_type(_retryable_exceptions()),
        reraise=True,
    )
    def _one_shot(msgs: list[dict[str, Any]]) -> Any:
        return client.chat.completions.create(
            model=model,
            messages=msgs,
            max_tokens=60,
            temperature=0.0,
        )

    resp = _one_shot(messages)
    text = resp.choices[0].message.content or ""
    if hasattr(resp, "usage") and resp.usage:
        meta["input_tokens"] += getattr(resp.usage, "prompt_tokens", 0)
        meta["output_tokens"] += getattr(resp.usage, "completion_tokens", 0)

    def _parse_and_validate(raw: str) -> TopicResult:
        data = json.loads(_extract_json_object(raw))
        result = TopicResult.model_validate(data)
        if result.primary not in valid_slugs:
            raise ValueError(f"'{result.primary}' is not a valid slug")
        if result.secondary is not None and result.secondary not in valid_slugs:
            result = TopicResult(primary=result.primary, secondary=None)
        return result

    try:
        return _parse_and_validate(text), meta
    except (json.JSONDecodeError, ValueError, Exception) as first_err:
        valid_list = ", ".join(sorted(valid_slugs))
        messages.append({"role": "assistant", "content": text})
        messages.append({
            "role": "user",
            "content": (
                f"Your previous response failed validation: {first_err}. "
                f"Valid slugs are: {valid_list}. "
                "Return only the corrected JSON object, nothing else."
            ),
        })
        resp2 = _one_shot(messages)
        text2 = resp2.choices[0].message.content or ""
        if hasattr(resp2, "usage") and resp2.usage:
            meta["input_tokens"] += getattr(resp2.usage, "prompt_tokens", 0)
            meta["output_tokens"] += getattr(resp2.usage, "completion_tokens", 0)
        try:
            return _parse_and_validate(text2), meta
        except Exception as e:
            raise RuntimeError(f"Topic classification failed twice: {e}") from e
