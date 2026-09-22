"""The MicroScheduler adapter: no-bypass pin + native-event translation."""

from awrouter.adapters.microscheduler import (
    DEFAULT_MICROSCHEDULER_BASE_URL,
    from_snapshot,
    unwrap_microscheduler_stream,
)
from awrouter.registry import Registry
from awrouter.resolver import ResolutionPolicy, Resolver

_LIVE_SHAPED_SNAPSHOT = {
    "backends": {
        "vllm_gemma4_12b": {
            "healthy": True,
            "models": ["gemma4-12b"],
            "latency_ms": 9.6,
            "last_check_age_s": 0.6,
            "check_interval_s": 5.0,
        },
        "deepseek_api": {
            "healthy": True,
            "models": [],
            "latency_ms": None,
            "last_check_age_s": 12.1,
            "check_interval_s": 60.0,
        },
        "vllm_reasoning": {
            "healthy": False,
            "models": [],
            "latency_ms": 600.5,
            "last_check_age_s": 0.6,
            "check_interval_s": 15.0,
        },
    },
    "healthy": ["vllm_gemma4_12b", "deepseek_api"],
    "healthy_count": 2,
    "total": 3,
}


def test_from_snapshot_pins_every_backend_to_the_configured_url() -> None:
    """The snapshot has no url field. Every Backend must get the SAME,
    caller-configured base_url -- never one derived from the snapshot,
    which has nothing to derive from."""
    backends = from_snapshot(
        _LIVE_SHAPED_SNAPSHOT, microscheduler_base_url="https://10.0.0.9:8150"
    )
    assert len(backends) == 3
    assert {b.base_url for b in backends} == {"https://10.0.0.9:8150"}


def test_from_snapshot_defaults_to_the_documented_default_url() -> None:
    backends = from_snapshot(_LIVE_SHAPED_SNAPSHOT)
    assert {b.base_url for b in backends} == {DEFAULT_MICROSCHEDULER_BASE_URL}


def test_from_snapshot_id_and_aliases_are_the_snapshot_name_for_routing_only() -> None:
    backends = {b.id: b for b in from_snapshot(_LIVE_SHAPED_SNAPSHOT)}
    gemma = backends["vllm_gemma4_12b"]
    assert gemma.id == "vllm_gemma4_12b"
    # The snapshot name itself must be an alias too, or Registry.serving()
    # (which checks aliases, not id) cannot resolve a request naming it.
    assert "vllm_gemma4_12b" in gemma.aliases
    assert "gemma4-12b" in gemma.aliases


def test_from_snapshot_resolving_by_the_snapshot_name_works() -> None:
    """The whole point of aliasing the name: Registry.serving() looks up by
    alias, not by id, so a caller resolving 'deepseek_api' must find it."""
    registry = Registry(from_snapshot(_LIVE_SHAPED_SNAPSHOT))
    resolver = Resolver(registry, ResolutionPolicy())
    resolution = resolver.resolve("deepseek_api", prompt_chars=10)
    assert resolution.backend.id == "deepseek_api"
    assert resolution.backend.base_url == DEFAULT_MICROSCHEDULER_BASE_URL


def test_from_snapshot_health_check_reflects_the_snapshot_not_a_live_probe() -> None:
    backends = {b.id: b for b in from_snapshot(_LIVE_SHAPED_SNAPSHOT)}
    assert backends["vllm_gemma4_12b"].health_check() is True
    assert backends["vllm_reasoning"].health_check() is False


def test_from_snapshot_unhealthy_backend_is_never_resolved() -> None:
    """Resolver.resolve() must refuse rather than route to a backend the
    snapshot itself reported unhealthy."""
    registry = Registry(from_snapshot(_LIVE_SHAPED_SNAPSHOT))
    resolver = Resolver(registry, ResolutionPolicy())
    try:
        resolver.resolve("vllm_reasoning", prompt_chars=10)
        raised = False
    except Exception:
        raised = True
    assert raised, "an unhealthy-per-snapshot backend must be refused, not routed to"


def test_from_snapshot_fields_not_in_the_snapshot_are_explicit_unknown_not_fabricated() -> None:
    backend = from_snapshot(_LIVE_SHAPED_SNAPSHOT)[0]
    assert backend.cost_per_1k_input == 0.0
    assert backend.cost_per_1k_output == 0.0
    # A precise-looking number here would be fabricated; this must be the
    # documented "unknown, treat as unbounded at this layer" sentinel, not
    # some small number that would spuriously refuse requests.
    from awrouter.adapters.microscheduler import UNKNOWN_CONTEXT_WINDOW

    assert backend.context_window == UNKNOWN_CONTEXT_WINDOW


def test_from_snapshot_missing_backends_key_yields_empty_list() -> None:
    assert from_snapshot({}) == []
    assert from_snapshot({"backends": "not-a-dict"}) == []
    assert from_snapshot({"backends": {"x": "not-a-dict-either"}}) == []


def test_from_snapshot_latency_falls_back_to_zero_when_null() -> None:
    backends = {b.id: b for b in from_snapshot(_LIVE_SHAPED_SNAPSHOT)}
    # deepseek_api's latency_ms is JSON null in the live shape.
    assert backends["deepseek_api"].latency_p50_ms == 0


# ---------------------------------------------------------------------------
# unwrap_microscheduler_stream -- the native event-shape translation.
# ---------------------------------------------------------------------------

_NATIVE_EVENTS = [
    {"model": "gemma4-12b", "agent": "openai_compat", "type": "session_start"},
    {"t": "O", "n": 1, "type": "token"},
    {"t": "K", "n": 2, "type": "token"},
    {
        "finish_reason": "stop",
        "tokens": 2,
        "content": "OK",
        "full_content": "OK",
        "tool_calls": [],
        "type": "complete",
    },
]


def test_unwrap_microscheduler_stream_prefers_the_complete_events_full_content() -> None:
    result = unwrap_microscheduler_stream(iter(_NATIVE_EVENTS))
    assert result == {"content": "OK", "tool_calls": []}


def test_unwrap_microscheduler_stream_falls_back_to_token_concatenation() -> None:
    """A stream cut off before its `complete` event must still surface
    whatever text the token deltas carried -- never raise, never return
    None."""
    cut_off = [e for e in _NATIVE_EVENTS if e.get("type") != "complete"]
    result = unwrap_microscheduler_stream(iter(cut_off))
    assert result == {"content": "OK", "tool_calls": []}


def test_unwrap_microscheduler_stream_empty_stream_is_empty_not_an_error() -> None:
    assert unwrap_microscheduler_stream(iter([])) == {"content": "", "tool_calls": []}


def test_unwrap_microscheduler_stream_carries_tool_calls_from_complete() -> None:
    events = [
        {"t": "", "type": "token"},
        {
            "content": "",
            "full_content": "",
            "tool_calls": [{"id": "call_1", "name": "search", "arguments": "{}"}],
            "type": "complete",
        },
    ]
    result = unwrap_microscheduler_stream(iter(events))
    assert result["tool_calls"] == [{"id": "call_1", "name": "search", "arguments": "{}"}]
