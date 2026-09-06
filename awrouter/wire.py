"""The OpenAI-compatible wire shape, stdlib-only.

Three pieces: SSE serialization (what the caller receives), SSE parsing (what
the upstream backend sends), and a streaming proxy that carries one to the
other without buffering the body. Tool-call deltas are accumulated and
re-emitted intact — unwrapping means the caller gets complete tool_calls,
not half-assembled fragments.
"""

from __future__ import annotations

import json
import urllib.request
from typing import Any, Generator, Iterator, Optional

from .registry import Backend

DONE_MARKER = "[DONE]"


def sse_chunk(
    model: str,
    *,
    delta: dict[str, Any],
    index: int,
    finish_reason: Optional[str] = None,
) -> str:
    """One OpenAI chat.completion.chunk event, serialized."""
    payload = {
        "id": f"chatcmpl-{index}",
        "object": "chat.completion.chunk",
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }
    return f"data: {json.dumps(payload)}\n\n"


def iter_sse(stream: Iterator[bytes]) -> Generator[dict[str, Any], None, None]:
    """Parse an SSE byte stream into JSON events. [DONE] terminates.

    Tolerates the two real-world line shapers: \n\n and \r\n\r\n splits.
    """
    buffer = b""
    for chunk in stream:
        buffer += chunk
        while True:
            split = buffer.find(b"\n\n")
            if split == -1:
                split = buffer.find(b"\r\n\r\n")
            if split == -1:
                break
            event, buffer = buffer[:split], buffer[split + 2 :]
            for line in event.splitlines():
                if not line.startswith(b"data:"):
                    continue
                data = line[5:].strip()
                if data == b"":
                    continue
                text = data.decode("utf-8", errors="replace")
                if text == DONE_MARKER:
                    return
                yield json.loads(text)


def unwrap_tool_calls(events: Iterator[dict[str, Any]]) -> dict[str, Any]:
    """Fold a chat completion event stream into one result object.

    Returns {"content": str, "tool_calls": [{"id","name","arguments"}...]}.
    Text deltas and tool-call deltas are kept apart; a stream with neither
    yields empty strings rather than raising.
    """
    content_parts: list[str] = []
    calls: dict[int, dict[str, Any]] = {}
    for event in events:
        for choice in event.get("choices", []):
            delta = choice.get("delta", {})
            text = delta.get("content")
            if text:
                content_parts.append(text)
            for call in delta.get("tool_calls", []):
                slot = calls.setdefault(
                    call.get("index", 0), {"id": "", "name": "", "arguments": ""}
                )
                if call.get("id"):
                    slot["id"] = call["id"]
                fn = call.get("function", {})
                if fn.get("name"):
                    slot["name"] = fn["name"]
                if fn.get("arguments"):
                    slot["arguments"] += fn["arguments"]
    ordered = [calls[i] for i in sorted(calls)]
    return {"content": "".join(content_parts), "tool_calls": ordered}


def stream_completion(
    backend: Backend,
    payload: dict[str, Any],
    *,
    headers: Optional[dict[str, str]] = None,
    timeout: float = 60.0,
) -> Generator[dict[str, Any], None, None]:
    """POST a chat completion to the backend and yield parsed SSE events.

    The body is streamed line by line — never buffered — so a long
    generation starts reaching the caller as soon as the first token lands.
    """
    body = json.dumps(payload).encode("utf-8")
    merged = {"Content-Type": "application/json", "Accept": "text/event-stream"}
    merged.update(headers or {})
    url = backend.base_url.rstrip("/") + "/v1/chat/completions"
    request = urllib.request.Request(url, data=body, headers=merged, method="POST")
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
        yield from iter_sse(response)
