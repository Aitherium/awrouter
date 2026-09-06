"""The resolver contract: resolve or refuse, never guess."""

import pytest
from awrouter.registry import Backend, Registry
from awrouter.resolver import (
    ModelSpec,
    RefusalError,
    Resolver,
    TierMap,
    fit_context,
)


def _registry() -> Registry:
    registry = Registry()
    registry.register(
        Backend(
            id="fast",
            base_url="http://127.0.0.1:9000",
            aliases=["quick-model"],
            capabilities={"chat", "tools"},
            context_window=8192,
            cost_per_1k_input=0.5,
            cost_per_1k_output=1.5,
            latency_p50_ms=200,
        )
    )
    registry.register(
        Backend(
            id="big",
            base_url="http://127.0.0.1:9001",
            aliases=["big-model", "quick-model"],
            capabilities={"chat", "tools", "thinking"},
            context_window=65536,
            max_output_tokens=8192,
            cost_per_1k_input=3.0,
            cost_per_1k_output=6.0,
            latency_p50_ms=2500,
        )
    )
    return registry


def test_resolve_picks_cheapest_live_backend() -> None:
    resolver = Resolver(_registry())
    resolution = resolver.resolve("quick-model")
    assert resolution.backend.id == "fast"
    assert resolution.ranked == ["fast", "big"]


def test_failover_skips_dead_backend() -> None:
    registry = _registry()
    registry.get("fast").health_check = lambda: False  # type: ignore[union-attr]
    resolver = Resolver(registry)
    resolution = resolver.resolve("quick-model")
    assert resolution.backend.id == "big"


def test_failover_refuses_when_all_probed_dead() -> None:
    registry = _registry()
    for backend_id in ("fast", "big"):
        registry.get(backend_id).health_check = lambda: False  # type: ignore[union-attr]
    with pytest.raises(RefusalError, match="no live backend"):
        Resolver(registry).resolve("quick-model")


def test_unknown_model_refused() -> None:
    with pytest.raises(RefusalError, match="no backend serves"):
        Resolver(_registry()).resolve("nope")


def test_capability_gap_refused() -> None:
    resolver = Resolver(_registry())
    spec = ModelSpec(id="big-model", requirements={"vision"})
    with pytest.raises(RefusalError, match="requirements"):
        resolver.resolve("big-model", spec=spec)


def test_thinking_model_needs_thinking_backend() -> None:
    registry = Registry()
    registry.register(
        Backend(id="chat", base_url="http://127.0.0.1:1", aliases=["brain"], capabilities={"chat"})
    )
    with pytest.raises(RefusalError, match="thinking"):
        Resolver(registry).resolve("brain", spec=ModelSpec(id="brain", thinking=True))


def test_tier_map_blocks_model() -> None:
    tier_map = TierMap({"free": {"quick-model"}})
    resolver = Resolver(_registry())
    with pytest.raises(RefusalError, match="not allowed on tier"):
        resolver.resolve("big-model", tier="free", tier_map=tier_map)


def test_tier_map_unknown_tier_blocks() -> None:
    tier_map = TierMap({"free": {"quick-model"}})
    with pytest.raises(RefusalError, match="not allowed on tier"):
        Resolver(_registry()).resolve("quick-model", tier="platinum", tier_map=tier_map)


def test_context_overflow_refused_not_truncated() -> None:
    resolver = Resolver(_registry())
    with pytest.raises(RefusalError, match="context overflow"):
        resolver.resolve("quick-model", prompt_chars=100_000)


def test_context_fit_reports_headroom() -> None:
    tokens, headroom = fit_context(100, 8192, 1024, 0.25)
    assert tokens == 25
    assert headroom == 8192 - 25 - 1024


def test_duplicate_backend_id_raises() -> None:
    registry = _registry()
    with pytest.raises(ValueError, match="duplicate backend"):
        registry.register(Backend(id="fast", base_url="http://127.0.0.1:9", aliases=["other"]))


def test_shared_alias_yields_both_backends_for_failover() -> None:
    registry = _registry()
    assert [b.id for b in registry.serving("quick-model")] == ["fast", "big"]


def test_snapshot_is_serializable() -> None:
    snapshot = _registry().snapshot()
    assert len(snapshot["backends"]) == 2
    assert {b["id"] for b in snapshot["backends"]} == {"fast", "big"}
