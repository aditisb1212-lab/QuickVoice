"""
Langfuse evaluation-layer integration for the QuickVoice AI service.

This module wires finished calls (and, optionally, RAG retrievals) into a
self-hosted or cloud Langfuse project so calls can be traced, scored, and
reviewed alongside the rest of the LLM engineering workflow.

Design goals:
- Opt-in: disabled unless LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY are set,
  or LANGFUSE_ENABLED=false is set explicitly.
- Never break a live call or call-log delivery. Every public function here
  swallows its own exceptions and logs a warning instead of raising.
- No hard dependency: if the `langfuse` package is not installed, every
  function becomes a no-op after logging once.

Wire-up points (see the rest of this PR):
- handlers/finalization_handler.py calls `log_call_trace(payload)` once the
  call-log payload has been built, so every finished call becomes a Langfuse
  trace with per-turn observations and heuristic evaluation scores.
- handlers/rag_handler.py calls `log_retrieval_span(...)` around Pinecone
  lookups, so knowledge-base retrieval quality is inspectable per agent.

Extending evaluation:
- `DEFAULT_EVALUATORS` holds cheap, local heuristics that run on every call
  with no extra LLM cost. Add new callables there to extend scoring.
- For LLM-as-judge scoring (e.g. "was the agent helpful?", "did the agent
  follow the script?"), run Langfuse's evaluation pipelines against the
  ingested traces/datasets from the Langfuse UI or API, rather than adding
  another live-call LLM call to this hot path.
"""

from __future__ import annotations

import os
from typing import Any

from utils.logger import logger, redact_sensitive

_client: Any = None
_client_checked = False


def _enabled() -> bool:
    return os.getenv("LANGFUSE_ENABLED", "true").strip().lower() not in {"0", "false", "no", "off"}


def get_client() -> Any:
    """Lazily build a singleton Langfuse client, or None if disabled/unavailable."""
    global _client, _client_checked
    if _client_checked:
        return _client
    _client_checked = True

    if not _enabled():
        logger.info("[langfuse] disabled via LANGFUSE_ENABLED")
        return None

    public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
    secret_key = os.getenv("LANGFUSE_SECRET_KEY")
    if not public_key or not secret_key:
        logger.info("[langfuse] disabled: LANGFUSE_PUBLIC_KEY/LANGFUSE_SECRET_KEY not set")
        return None

    try:
        from langfuse import Langfuse  # noqa: WPS433 (intentionally lazy import)
    except ImportError:
        logger.warning("[langfuse] `langfuse` package not installed; evaluation layer disabled")
        return None

    try:
        _client = Langfuse(
            public_key=public_key,
            secret_key=secret_key,
            host=os.getenv("LANGFUSE_BASE_URL", "https://cloud.langfuse.com"),
        )
    except Exception as error:  # noqa: BLE001 - never let init break the caller
        logger.warning("[langfuse] client init failed: {}", redact_sensitive(str(error)))
        _client = None

    return _client


# --------------------------------------------------------------------------
# Heuristic call-quality evaluators. Each takes the normalized transcript
# list (see calllog_handler._normalize_transcript_item) and returns a score
# in the 0..1 range. These are intentionally cheap (no network/LLM calls) so
# they can run on every single call without adding latency or cost.
# --------------------------------------------------------------------------

def _score_has_transcript(transcript: list[dict[str, Any]]) -> float:
    return 1.0 if transcript else 0.0


def _score_agent_responded(transcript: list[dict[str, Any]]) -> float:
    return 1.0 if any(item.get("role") == "agent" for item in transcript) else 0.0


def _score_conversation_balance(transcript: list[dict[str, Any]]) -> float:
    """Flags monologue-only calls (agent talks, user never responds, or vice versa)."""
    if not transcript:
        return 0.0
    roles = {item.get("role") for item in transcript}
    return 1.0 if {"user", "agent"}.issubset(roles) else 0.0


DEFAULT_EVALUATORS = {
    "has_transcript": _score_has_transcript,
    "agent_responded": _score_agent_responded,
    "conversation_balance": _score_conversation_balance,
}


def _first_message(transcript: list[dict[str, Any]], role: str) -> str | None:
    for item in transcript:
        if item.get("role") == role:
            return item.get("message")
    return None


def _last_message(transcript: list[dict[str, Any]], role: str) -> str | None:
    for item in reversed(transcript):
        if item.get("role") == role:
            return item.get("message")
    return None


def log_call_trace(payload: dict[str, Any]) -> None:
    """
    Send a finished call to Langfuse as a trace: one observation per
    transcript turn, plus heuristic evaluation scores from
    `DEFAULT_EVALUATORS`. Call this once the call-log payload has been
    built (see handlers/calllog_handler.build_call_log_payload).

    Best-effort only: any failure here is logged and swallowed so it can
    never affect call-log delivery to the QuickVoice API.
    """
    client = get_client()
    if client is None:
        return

    try:
        transcript = payload.get("transcripts") or []
        trace = client.trace(
            id=payload.get("callId"),
            name="quickvoice-call",
            session_id=payload.get("callId"),
            user_id=payload.get("userId"),
            metadata={
                "agentId": payload.get("agentId"),
                "organizationId": payload.get("organizationId"),
                "provider": payload.get("provider"),
                "direction": payload.get("direction"),
                "toNumber": payload.get("toNumber"),
                "fromNumber": payload.get("fromNumber"),
                "durationSeconds": payload.get("durationSeconds"),
            },
            tags=[str(payload.get("direction") or "unknown"), str(payload.get("provider") or "unknown")],
            input=_first_message(transcript, "user"),
            output=_last_message(transcript, "agent"),
        )

        for index, item in enumerate(transcript):
            role = item.get("role", "agent")
            trace.event(
                name=f"turn-{index}:{role}",
                input=item.get("message") if role == "user" else None,
                output=item.get("message") if role == "agent" else None,
                metadata={"messageId": item.get("messageId"), "timestamp": item.get("timestamp")},
            )

        for name, evaluator in DEFAULT_EVALUATORS.items():
            try:
                trace.score(name=name, value=evaluator(transcript))
            except Exception as eval_error:  # noqa: BLE001
                logger.warning(
                    "[langfuse] evaluator '{}' failed: {}",
                    name,
                    redact_sensitive(str(eval_error)),
                )

        duration_seconds = payload.get("durationSeconds")
        if isinstance(duration_seconds, (int, float)):
            trace.score(name="call_duration_seconds", value=float(duration_seconds))

        client.flush()
    except Exception as error:  # noqa: BLE001 - eval layer must never break call finalization
        logger.warning("[langfuse] failed to log call trace: {}", redact_sensitive(str(error)))


def log_retrieval_span(
    *,
    agent_id: str,
    query: str,
    status: str,
    matches: int,
    latency_ms: int,
) -> None:
    """
    Best-effort: record a single RAG (Pinecone) retrieval as a Langfuse span
    so knowledge-base quality can be reviewed per agent alongside call
    traces. Call this from handlers/rag_handler.get_rag_context.
    """
    client = get_client()
    if client is None:
        return
    try:
        span_trace = client.trace(name="quickvoice-rag-retrieval", metadata={"agentId": agent_id})
        span_trace.span(
            name="pinecone-retrieval",
            input={"query": query},
            output={"status": status, "matches": matches},
            metadata={"latencyMs": latency_ms, "agentId": agent_id},
        )
    except Exception as error:  # noqa: BLE001
        logger.warning("[langfuse] failed to log retrieval span: {}", redact_sensitive(str(error)))
