"""Anthropic SDK wrapper with retry, JSON validation, and Stage 2/3 prompts."""
from __future__ import annotations

import json
import os
import re
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

try:
    import anthropic  # type: ignore
except ImportError:  # pragma: no cover
    anthropic = None  # type: ignore


# Pricing (per million tokens), current ICLR-week rates.
PRICING = {
    "claude-haiku-4-5-20251001": {"input": 1.0, "output": 5.0},
    "claude-haiku-4-5": {"input": 1.0, "output": 5.0},
    "claude-sonnet-4-6": {"input": 3.0, "output": 15.0},
}

HAIKU_MODEL = "claude-haiku-4-5-20251001"
SONNET_MODEL = "claude-sonnet-4-6"


STAGE2_SYSTEM = """You are an information extraction assistant for ICLR 2026 papers.
You only return valid JSON matching the user-specified schema.
You do not add commentary, preamble, or markdown fences.
You extract information only if it is explicitly stated in the provided text.
If information is not explicitly stated, return an empty array or null as specified.
Never paraphrase or repeat the abstract in any field other than `one_sentence_summary`.
Never invent dataset names, method names, or citations.
Never use marketing language ("novel", "state-of-the-art", "groundbreaking").
Preserve the authors' own phrasing for limitations and future_work; shorten only by trimming, not by rewording."""


STAGE2_USER_TEMPLATE = """Extract structured information from the following ICLR 2026 paper.

Return a single JSON object matching EXACTLY this schema (no extra keys, no missing keys):

{{
  "one_sentence_summary": string,   // <=25 words, factual. Do not copy the abstract.
  "contributions": string[],        // 2-5 items, each <=160 chars. [] if not stated.
  "methods_used": string[],         // short noun phrases USED by this work. [] if unclear.
  "datasets_used": string[],        // explicit named datasets only. [] if none named.
  "limitations": string[],          // author-stated, lightly trimmed. [] if not stated.
  "future_work": string[]           // author-stated, lightly trimmed. [] if none stated.
}}

Rules:
- Return [] when the source text does not explicitly contain the information. Do NOT guess.
- Do NOT paraphrase the abstract into limitations or future_work.
- Do NOT merge multiple future_work points into one; preserve each as a separate bullet.
- Output the JSON object only. No code fences. No prose.

Example of desired output for a hypothetical paper:
{{
  "one_sentence_summary": "Introduces a token-level entropy regularizer for DPO that reduces reward hacking on open-ended tasks.",
  "contributions": ["A token-level entropy penalty for DPO", "Analysis showing reward hacking correlates with entropy collapse", "Evaluation on MT-Bench and AlpacaEval 2"],
  "methods_used": ["DPO", "entropy regularization", "pairwise preference data"],
  "datasets_used": ["UltraFeedback", "MT-Bench", "AlpacaEval 2"],
  "limitations": ["Experiments restricted to 7B-scale models", "Regularizer coefficient requires per-dataset tuning"],
  "future_work": ["Extend to multi-turn preference data", "Investigate interaction with KL-controlled RLHF"]
}}

Paper content:
---
{body}
---"""


STAGE3_SYSTEM = """You analyze structured extracts from a set of ICLR 2026 oral papers and identify cross-paper themes in their stated future work and limitations.
Return ONLY valid JSON matching the schema provided.
Each theme must cite the paper IDs that actually support it.
Do not invent paper IDs; only use IDs that appear in the input.
Themes should describe specific research directions, not vague categories like "improving models".
Aim for 5 to 10 themes. Prefer fewer high-signal themes over many weak ones."""


STAGE3_USER_TEMPLATE = """Here are future_work and limitations bullets from {n} ICLR 2026 oral papers. Identify 5–10 cross-paper themes.

Return JSON matching this schema exactly:
{{
  "themes": [
    {{
      "slug": "kebab-case-identifier",
      "headline": string,          // <=80 chars, specific
      "explanation": string,       // 2–4 sentences, plain prose, no hype
      "paper_ids": string[],       // 2+ IDs; all must appear in the input
      "representative_quotes": [
        {{"paper_id": string, "quote": string}}  // 0–3 items, verbatim from the input
      ]
    }}
  ]
}}

Input:
{blob}"""


class Stage2Output(BaseModel):
    model_config = ConfigDict(extra="forbid")
    one_sentence_summary: str = Field(default="")
    contributions: list[str] = Field(default_factory=list)
    methods_used: list[str] = Field(default_factory=list)
    datasets_used: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    future_work: list[str] = Field(default_factory=list)


class Stage3Theme(BaseModel):
    model_config = ConfigDict(extra="forbid")
    slug: str
    headline: str
    explanation: str
    paper_ids: list[str]
    representative_quotes: list[dict[str, str]] = Field(default_factory=list)


class Stage3Output(BaseModel):
    model_config = ConfigDict(extra="forbid")
    themes: list[Stage3Theme]


def _require_client() -> Any:
    if anthropic is None:
        raise RuntimeError(
            "anthropic SDK not installed. Add 'anthropic' to scripts/requirements.txt and pip install."
        )
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError(
            "ANTHROPIC_API_KEY not set. Put it in .env (at repo root) and re-run."
        )
    return anthropic.Anthropic()


def _extract_json_object(text: str) -> str:
    """Strip code fences if present; return the JSON blob."""
    t = text.strip()
    if t.startswith("```"):
        # Remove opening fence (optionally with language) and closing fence.
        t = re.sub(r"^```(?:json|javascript|js)?\s*\n?", "", t, count=1)
        t = re.sub(r"\n?```\s*$", "", t, count=1)
    return t.strip()


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    price = PRICING.get(model, {"input": 1.0, "output": 5.0})
    return round(
        (input_tokens * price["input"] + output_tokens * price["output"]) / 1_000_000,
        6,
    )


def _retryable_exceptions() -> tuple[type[Exception], ...]:
    if anthropic is None:
        return ()
    return (
        anthropic.RateLimitError,
        anthropic.APIConnectionError,
        anthropic.InternalServerError,
    )


def call_stage2(body: str, model: str = HAIKU_MODEL, max_tokens: int = 1500) -> tuple[Stage2Output, dict[str, Any]]:
    """Call Haiku once; on validation failure, retry once with feedback."""
    client = _require_client()
    user = STAGE2_USER_TEMPLATE.format(body=body)

    @retry(
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=2, min=2, max=30),
        retry=retry_if_exception_type(_retryable_exceptions()),
        reraise=True,
    )
    def _one_shot(messages: list[dict[str, Any]]) -> Any:
        return client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=STAGE2_SYSTEM,
            messages=messages,
        )

    messages: list[dict[str, Any]] = [{"role": "user", "content": user}]
    resp = _one_shot(messages)
    text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
    meta = {
        "model_stop_reason": getattr(resp, "stop_reason", "") or "",
        "input_tokens": getattr(resp.usage, "input_tokens", 0) if getattr(resp, "usage", None) else 0,
        "output_tokens": getattr(resp.usage, "output_tokens", 0) if getattr(resp, "usage", None) else 0,
    }

    json_text = _extract_json_object(text)
    try:
        data = json.loads(json_text)
        return Stage2Output.model_validate(data), meta
    except (json.JSONDecodeError, ValidationError) as first_err:
        # One retry with error feedback.
        messages.append({"role": "assistant", "content": text})
        messages.append({
            "role": "user",
            "content": f"Your previous response failed validation: {first_err}. Return only the JSON object matching the schema, nothing else.",
        })
        resp2 = _one_shot(messages)
        text2 = "".join(b.text for b in resp2.content if getattr(b, "type", "") == "text")
        meta["model_stop_reason"] = getattr(resp2, "stop_reason", "") or meta["model_stop_reason"]
        if getattr(resp2, "usage", None):
            meta["input_tokens"] += getattr(resp2.usage, "input_tokens", 0)
            meta["output_tokens"] += getattr(resp2.usage, "output_tokens", 0)
        try:
            data2 = json.loads(_extract_json_object(text2))
            return Stage2Output.model_validate(data2), meta
        except Exception as e:
            raise RuntimeError(f"Stage2 validation failed twice: {e}") from e


def call_stage3(blob: str, n_papers: int, model: str = SONNET_MODEL, max_tokens: int = 4000) -> tuple[Stage3Output, dict[str, Any]]:
    client = _require_client()
    user = STAGE3_USER_TEMPLATE.format(n=n_papers, blob=blob)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=2, max=30),
        retry=retry_if_exception_type(_retryable_exceptions()),
        reraise=True,
    )
    def _one_shot(messages: list[dict[str, Any]]) -> Any:
        return client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=STAGE3_SYSTEM,
            messages=messages,
        )

    messages: list[dict[str, Any]] = [{"role": "user", "content": user}]
    resp = _one_shot(messages)
    text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
    meta = {
        "model_stop_reason": getattr(resp, "stop_reason", "") or "",
        "input_tokens": getattr(resp.usage, "input_tokens", 0) if getattr(resp, "usage", None) else 0,
        "output_tokens": getattr(resp.usage, "output_tokens", 0) if getattr(resp, "usage", None) else 0,
    }
    try:
        data = json.loads(_extract_json_object(text))
        return Stage3Output.model_validate(data), meta
    except (json.JSONDecodeError, ValidationError) as first_err:
        messages.append({"role": "assistant", "content": text})
        messages.append({
            "role": "user",
            "content": f"Your previous response failed validation: {first_err}. Return only the JSON object matching the schema.",
        })
        resp2 = _one_shot(messages)
        text2 = "".join(b.text for b in resp2.content if getattr(b, "type", "") == "text")
        if getattr(resp2, "usage", None):
            meta["input_tokens"] += getattr(resp2.usage, "input_tokens", 0)
            meta["output_tokens"] += getattr(resp2.usage, "output_tokens", 0)
        data2 = json.loads(_extract_json_object(text2))
        return Stage3Output.model_validate(data2), meta
