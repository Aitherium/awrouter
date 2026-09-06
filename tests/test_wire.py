"""The wire shape: SSE out, SSE in, tool-call folding, streaming proxy."""

import json

from awrouter.registry import Backend
from awrouter.wire import DONE_MARKER, iter_sse, sse_chunk, stream_completion, unwrap_tool_calls


def _event_stream(events: list[dict]) -> list[bytes]:
    out = []
    for event in events:
        out.append(f"data: {json.dumps(event)}\n\n".encode())
    out.append(f"data: {DONE_MARKER}\n\n".encode())
    return out


def test_sse_chunk_shape() -> None:
    chunk = sse_chunk("m", delta={"content": "hi"}, index=1)
    assert chunk.startswith("data: ")
    assert '"object": "chat.completion.chunk"' in chunk
    assert '"model": "m"' in chunk
    assert chunk.endswith("\n\n")


def test_iter_sse_parses_events_and_stops_at_done() -> None:
    events = [
        {"choices": [{"delta": {"content": "a"}}]},
        {"choices": [{"delta": {"content": "b"}}]},
    ]
    parsed = list(iter_sse(iter(_event_stream(events))))
    assert [e["choices"][0]["delta"]["content"] for e in parsed] == ["a", "b"]


def test_iter_sse_handles_crlf_split() -> None:
    raw = b'data: {"choices": []}\r\n\r\ndata: [DONE]\r\n\r\n'
    parsed = list(iter_sse(iter([raw])))
    assert parsed == [{"choices": []}]


def test_iter_sse_handles_partial_chunks_across_reads() -> None:
    raw = _event_stream([{"choices": [{"delta": {"content": "x"}}]}])
    joined = b"".join(raw)
    # Deliver one byte at a time: the parser must reassemble the event.
    parsed = list(iter_sse(iter(joined[i : i + 1] for i in range(len(joined)))))
    assert parsed == [{"choices": [{"delta": {"content": "x"}}]}]


def test_unwrap_tool_calls_folds_deltas() -> None:
    events = [
        {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_1",
                                "function": {"name": "search", "arguments": '{"q":'},
                            }
                        ]
                    }
                }
            ]
        },
        {
            "choices": [
                {"delta": {"tool_calls": [{"index": 0, "function": {"arguments": ' "x"}'}}]}}
            ]
        },
        {"choices": [{"delta": {"content": "let me look"}}]},
    ]
    result = unwrap_tool_calls(iter(events))
    assert result["content"] == "let me look"
    assert result["tool_calls"] == [{"id": "call_1", "name": "search", "arguments": '{"q": "x"}'}]


def test_unwrap_tool_calls_empty_stream() -> None:
    assert unwrap_tool_calls(iter([])) == {"content": "", "tool_calls": []}


def test_stream_completion_posts_and_yields(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    events = [{"choices": [{"delta": {"content": "hi"}}]}]
    raw = b"".join(_event_stream(events))

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *args):  # type: ignore[no-untyped-def]
            return False

        def __iter__(self):
            return iter([raw])

    seen: dict = {}

    def fake_urlopen(request, timeout):  # type: ignore[no-untyped-def]
        seen["url"] = request.full_url
        seen["method"] = request.get_method()
        seen["body"] = request.data
        return FakeResponse()

    monkeypatch.setattr("awrouter.wire.urllib.request.urlopen", fake_urlopen)

    backend = Backend(id="b", base_url="http://127.0.0.1:9000", aliases=["m"])
    parsed = list(stream_completion(backend, {"model": "m", "messages": []}))
    assert seen["url"] == "http://127.0.0.1:9000/v1/chat/completions"
    assert seen["method"] == "POST"
    assert json.loads(seen["body"]) == {"model": "m", "messages": []}
    assert parsed[0]["choices"][0]["delta"]["content"] == "hi"
