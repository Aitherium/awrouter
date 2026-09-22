"""MicroScheduler adapter -- the platform's one mandated LLM front door.

`from_snapshot` turns a MicroScheduler `GET /llm/backends/snapshot` response
into a list of `Backend` objects, WITHOUT ever inventing a per-backend URL:
the snapshot carries none (measured live, 2026-09-19: 16 backends, shape
`{"backends": {name: {healthy, models, latency_ms, last_check_age_s,
check_interval_s}}}` -- no `url` field of any kind), and MicroScheduler is
the platform's single mandated LLM front door ("All LLM calls route through
MicroScheduler. Never bypass it." -- root CLAUDE.md).

So every `Backend` this returns gets the SAME `base_url` -- the configured
MicroScheduler endpoint, `https://` always -- and the snapshot's `name` key
(e.g. `vllm_gemma4_12b`, `deepseek_api`) becomes `Backend.id`/an alias for
MODEL ROUTING ONLY. It is never substituted into `base_url` and never used
to construct a direct backend host:port. `stream_completion`'s target is
therefore always MicroScheduler's own endpoint; the resolved backend/model
name travels in the REQUEST BODY instead, the field MicroScheduler's own
`/v1/chat/completions` uses to pick the real backend -- so streaming through
this adapter can never bypass MicroScheduler, by construction, not by
convention.

`unwrap_microscheduler_stream` is the second half of the same boundary.
Measured live 2026-09-19 against a real, healthy backend (`vllm_gemma4_12b`,
`deepseek_api`, `llamacpp_pool`, all three): `/v1/chat/completions` with
`stream: true` does NOT emit OpenAI delta chunks. The route's own handler
(`_dispatch_openai_chat` in AitherMicroScheduler.py) wraps a plain dict
result in an OpenAI `chat.completion.chunk` shape ONLY when the underlying
call returns one; every live backend actually returns an
already-streaming response, which is passed straight through untouched --
and that response speaks Aither's own native event protocol
(`event: session_start` / `token` / `complete`, with payloads shaped
`{"t": ..., "type": "token"}` and a `complete` event carrying the full
text). The generic OpenAI-chunk wrapping in that same handler is a fallback
branch this route does not reach in practice.

`wire.iter_sse` already parses this stream correctly -- it reads any
`data:` line and ignores the `event:` line ahead of it, so nothing breaks
there. Only the PAYLOAD SHAPE differs from an OpenAI delta, so only that
translation lives here, in the adapter for this one platform surface: the
generic core (`wire.unwrap_tool_calls`) stays OpenAI-only and untouched.
"""

from __future__ import annotations

from typing import Any, Iterator, Optional

from ..registry import Backend

#: MicroScheduler is reached in-network, TLS always -- plain http:// on this
#: port hangs rather than refusing (measured), which reads as "wedged"
#: rather than "wrong scheme". Never default to http://, never pass
#: verify=False: trust the internal CA, per root CLAUDE.md and
#: .claude/rules/secrets-access.md.
DEFAULT_MICROSCHEDULER_BASE_URL = "https://127.0.0.1:8150"

#: The snapshot has no context-window field for any backend. Declaring 0
#: would refuse every request through this resolver's own local
#: `fit_context` check; declaring a plausible-looking number (8192, say)
#: would fabricate precision this adapter does not have. This sentinel says
#: "unknown at this routing layer" honestly -- MicroScheduler enforces the
#: real per-model window server-side, on the request it actually serves.
UNKNOWN_CONTEXT_WINDOW = 2_000_000_000

#: Same reasoning for cost: the snapshot reports no price. An explicit 0.0
#: is "unknown", never a fabricated estimate -- a cost-weighted policy over
#: an all-zero-cost registry degrades to latency-only ranking, which is
#: honest given what is actually known.
UNKNOWN_COST_PER_1K = 0.0


def from_snapshot(
    snapshot: dict[str, Any],
    *,
    microscheduler_base_url: str = DEFAULT_MICROSCHEDULER_BASE_URL,
) -> list[Backend]:
    """Build `Backend` objects from a parsed `/llm/backends/snapshot` body.

    A snapshot with no `"backends"` key, or a non-dict value there, yields an
    empty list rather than guessing. A per-backend entry that is not a dict
    is skipped, not fabricated into one.

    `health_check` on each `Backend` is a closure over the snapshot's OWN
    `healthy` flag at adapt time -- not a live probe. That is deliberate:
    this is a point-in-time snapshot, and re-checking here would silently
    diverge from what the caller actually read. `Resolver.resolve`'s
    failover then only ever selects a backend the snapshot itself reported
    healthy.
    """
    backends: list[Backend] = []
    raw = snapshot.get("backends") if isinstance(snapshot, dict) else None
    if not isinstance(raw, dict):
        return backends

    for name, info in raw.items():
        if not isinstance(name, str) or not isinstance(info, dict):
            continue

        models = info.get("models")
        if not isinstance(models, list):
            models = []
        aliases = sorted({name, *(m for m in models if isinstance(m, str))})

        healthy = bool(info.get("healthy", False))

        latency_ms = info.get("latency_ms")
        latency_p50_ms = int(latency_ms) if isinstance(latency_ms, (int, float)) else 0

        backends.append(
            Backend(
                id=name,
                base_url=microscheduler_base_url,
                aliases=aliases,
                #: Every snapshot backend answers /v1/chat/completions --
                #: that is the one thing this endpoint is known to serve.
                #: Anything more specific (tools, thinking) is not reported
                #: by the snapshot and is not claimed here.
                capabilities={"chat"},
                context_window=UNKNOWN_CONTEXT_WINDOW,
                cost_per_1k_input=UNKNOWN_COST_PER_1K,
                cost_per_1k_output=UNKNOWN_COST_PER_1K,
                latency_p50_ms=latency_p50_ms,
                health_check=(lambda _h=healthy: _h),
            )
        )
    return backends


def unwrap_microscheduler_stream(events: Iterator[dict[str, Any]]) -> dict[str, Any]:
    """Fold MicroScheduler's OWN native SSE event stream into the same
    `{"content": str, "tool_calls": [...]}` shape `wire.unwrap_tool_calls`
    returns, so a caller can treat either source identically.

    Reads `type == "token"` events' `"t"` field as incremental deltas, and a
    `type == "complete"` event's `"full_content"` (falling back to
    `"content"`) as the authoritative final text -- MicroScheduler sends the
    whole answer there, so it is preferred over the token concatenation
    when both are present. `tool_calls` comes from the same `complete`
    event. A stream that ends with no `complete` event (cut off mid-flight)
    still returns whatever text the `token` events carried, never raises.
    """
    content_parts: list[str] = []
    final_content: Optional[str] = None
    tool_calls: list[Any] = []

    for event in events:
        if not isinstance(event, dict):
            continue
        etype = event.get("type")
        if etype == "token":
            text = event.get("t")
            if text:
                content_parts.append(str(text))
        elif etype == "complete":
            final_content = event.get("full_content")
            if final_content is None:
                final_content = event.get("content")
            calls = event.get("tool_calls")
            if isinstance(calls, list):
                tool_calls = calls

    content = final_content if final_content is not None else "".join(content_parts)
    return {"content": content, "tool_calls": tool_calls}
