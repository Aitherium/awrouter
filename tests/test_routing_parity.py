"""Fleet routing parity: stand-in chains, request priority, the aither_route marker.

The fleet scheduler's semantics (AitherLLMQueue stand-ins / interactive
fallback, AitherMicroScheduler's route marker), reimplemented here. Every arm
is driven through Resolver.resolve, the path a caller actually uses.
"""

import json

import pytest
from awrouter.failover import LoadTracker
from awrouter.registry import Backend, Registry
from awrouter.resolver import (
    ModelSpec,
    RefusalError,
    ResolutionPolicy,
    Resolver,
    TierMap,
)
from awrouter.routing import (
    lane_state,
    normalize_priority,
    parse_fallbacks,
    parse_standins,
    route_marker,
)
from awrouter.wire import compose_route, stream_completion


@pytest.fixture(autouse=True)
def _no_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """The host's posture env must never leak into these arms."""
    monkeypatch.delenv("AITHER_MODEL_STANDINS", raising=False)
    monkeypatch.delenv("AITHER_INTERACTIVE_BUSY_FALLBACKS", raising=False)


def _registry(**kwargs: object) -> Registry:
    """pool (1 slot) and bonsai (1 slot) and cloud (unbounded); nobody serves gemma."""
    registry = Registry(**kwargs)  # type: ignore[arg-type]
    registry.register(
        Backend(id="pool", base_url="http://127.0.0.1:9000", aliases=["pool-model"],
                capabilities={"chat"}, max_concurrent=1)
    )
    registry.register(
        Backend(id="bonsai", base_url="http://127.0.0.1:9001", aliases=["bonsai-model"],
                capabilities={"chat"}, max_concurrent=1)
    )
    registry.register(
        Backend(id="cloud", base_url="http://127.0.0.1:9002", aliases=["cloud-model"],
                capabilities={"chat"})
    )
    return registry


CHAIN = {"gemma": ["pool-model", "bonsai-model"]}


# -- parsers -------------------------------------------------------------------


def test_parse_standins_fleet_syntax() -> None:
    parsed = parse_standins("gemma=pool|bonsai, x = y ,self=self|z,gemma=ignored,,bad")
    assert parsed == {"gemma": ["pool", "bonsai"], "x": ["y"], "self": ["z"]}
    assert parse_standins("") == {}


def test_parse_fallbacks_fleet_syntax() -> None:
    assert parse_fallbacks("pool=bonsai,me=me,pool=other, a = b") == {"pool": "bonsai", "a": "b"}
    assert parse_fallbacks("") == {}


def test_priority_normalized_and_unknown_raises() -> None:
    assert normalize_priority(None) == "agent"
    assert normalize_priority("USER") == "user"
    with pytest.raises(ValueError):
        normalize_priority("urgent")


# -- stand-in chain walk --------------------------------------------------------


def test_chain_walk_takes_first_when_all_free() -> None:
    res = Resolver(_registry(standins=CHAIN)).resolve("gemma", load=LoadTracker())
    assert res.model_id == "pool-model"
    assert res.backend.id == "pool"
    assert res.reason == "standin"


def test_chain_walk_skips_busy_for_free() -> None:
    load = LoadTracker()
    load.set_pending("pool", 1)
    res = Resolver(_registry(standins=CHAIN)).resolve("gemma", load=load)
    assert res.model_id == "bonsai-model"
    assert res.backend.id == "bonsai"


def test_chain_walk_all_busy_takes_first() -> None:
    load = LoadTracker()
    load.set_pending("pool", 1)
    load.set_pending("bonsai", 3)
    res = Resolver(_registry(standins=CHAIN)).resolve("gemma", load=load)
    assert res.model_id == "pool-model"


def test_chain_walk_prefers_busy_over_down() -> None:
    registry = _registry(standins=CHAIN)
    registry.get("pool").health_check = lambda: False  # type: ignore[union-attr]
    load = LoadTracker()
    load.set_pending("bonsai", 1)
    res = Resolver(registry).resolve("gemma", load=load)
    assert res.model_id == "bonsai-model"


def test_reservations_count_against_capacity() -> None:
    load = LoadTracker()
    resolver = Resolver(_registry(standins=CHAIN))
    first = resolver.resolve("gemma", load=load)
    second = resolver.resolve("gemma", load=load)
    assert (first.model_id, second.model_id) == ("pool-model", "bonsai-model")


def test_chain_all_down_refuses_naming_the_standin() -> None:
    registry = _registry(standins=CHAIN)
    for bid in ("pool", "bonsai"):
        registry.get(bid).health_check = lambda: False  # type: ignore[union-attr]
    with pytest.raises(RefusalError, match="standin 'pool-model'"):
        Resolver(registry).resolve("gemma")


def test_env_standins_override_registry(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AITHER_MODEL_STANDINS", "gemma=cloud-model|pool-model")
    res = Resolver(_registry(standins=CHAIN), use_env=True).resolve("gemma")
    assert res.model_id == "cloud-model"


def test_env_standins_alone_work(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AITHER_MODEL_STANDINS", "gemma=bonsai-model")
    assert Resolver(_registry(), use_env=True).resolve("gemma").model_id == "bonsai-model"


def test_env_ignored_without_opt_in(monkeypatch: pytest.MonkeyPatch) -> None:
    """An embedded Resolver never picks up a host's posture env on its own."""
    monkeypatch.setenv("AITHER_MODEL_STANDINS", "gemma=bonsai-model")
    monkeypatch.setenv("AITHER_INTERACTIVE_BUSY_FALLBACKS", "pool-model=cloud-model")
    resolver = Resolver(_registry())
    with pytest.raises(RefusalError, match="no backend serves model 'gemma'"):
        resolver.resolve("gemma")
    res = resolver.resolve("pool-model", load=_busy_pool(), priority="user")
    assert res.model_id == "pool-model"


def test_pinned_unserved_model_refuses_not_standin() -> None:
    """A pin is an attribution promise: no stand-in for a pinned request."""
    with pytest.raises(RefusalError) as info:
        Resolver(_registry(standins=CHAIN)).resolve("gemma", pinned="pool")
    assert info.value.route == {
        "requested": "gemma", "served_by": "", "cross_model": False, "refused": True,
    }


def test_served_model_ignores_its_chain() -> None:
    registry = _registry(standins={"pool-model": ["bonsai-model"]})
    load = LoadTracker()
    load.set_pending("pool", 5)
    res = Resolver(registry).resolve("pool-model", load=load)
    assert res.model_id == "pool-model"
    assert res.reason is None


def test_standin_must_be_allowed_on_tier() -> None:
    tiers = TierMap({"free": {"gemma", "bonsai-model"}})
    res = Resolver(_registry(standins=CHAIN)).resolve("gemma", tier="free", tier_map=tiers)
    assert res.model_id == "bonsai-model"


def test_standin_must_meet_requirements() -> None:
    registry = _registry(standins={"gemma": ["pool-model"]})
    with pytest.raises(RefusalError, match="requirements"):
        Resolver(registry).resolve("gemma", spec=ModelSpec(id="gemma", requirements={"vision"}))


# -- priority + interactive fallback --------------------------------------------


def _busy_pool() -> LoadTracker:
    load = LoadTracker()
    load.set_pending("pool", 1)
    return load


POLICY = ResolutionPolicy(interactive_fallbacks={"pool-model": "bonsai-model"})


def test_user_turn_takes_fallback_off_busy_lane() -> None:
    res = Resolver(_registry(), POLICY).resolve("pool-model", load=_busy_pool(), priority="user")
    assert res.model_id == "bonsai-model"
    assert res.reason == "interactive_fallback"


@pytest.mark.parametrize("priority", ["background", "agent", None])
def test_non_user_never_takes_fallback(priority: object) -> None:
    res = Resolver(_registry(), POLICY).resolve(
        "pool-model", load=_busy_pool(), priority=priority  # type: ignore[arg-type]
    )
    assert res.model_id == "pool-model"
    assert res.reason is None


def test_user_turn_stays_when_lane_free() -> None:
    res = Resolver(_registry(), POLICY).resolve("pool-model", load=LoadTracker(), priority="user")
    assert res.model_id == "pool-model"


def test_user_turn_stays_when_fallback_busy() -> None:
    load = _busy_pool()
    load.set_pending("bonsai", 1)
    res = Resolver(_registry(), POLICY).resolve("pool-model", load=load, priority="user")
    assert res.model_id == "pool-model"


def test_user_turn_leaves_down_lane() -> None:
    registry = _registry()
    registry.get("pool").health_check = lambda: False  # type: ignore[union-attr]
    res = Resolver(registry, POLICY).resolve("pool-model", priority="user")
    assert res.model_id == "bonsai-model"


def test_pinned_user_turn_never_moves() -> None:
    res = Resolver(_registry(), POLICY).resolve(
        "pool-model", load=_busy_pool(), priority="user", pinned="pool"
    )
    assert res.model_id == "pool-model"


def test_env_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AITHER_INTERACTIVE_BUSY_FALLBACKS", "pool-model=cloud-model")
    res = Resolver(_registry(), POLICY, use_env=True).resolve(
        "pool-model", load=_busy_pool(), priority="user"
    )
    assert res.model_id == "cloud-model"


def test_lane_state_reads() -> None:
    registry = _registry()
    load = _busy_pool()
    assert lane_state(registry, "pool-model", load) == "busy"
    assert lane_state(registry, "pool-model") == "free"
    assert lane_state(registry, "cloud-model", load) == "free"
    assert lane_state(registry, "nobody") == "down"


def test_lane_reads_reuse_a_recent_probe() -> None:
    """Choosing between lanes never re-probes a backend inside the TTL window."""
    now = [0.0]
    registry = _registry(standins=CHAIN, clock=lambda: now[0])
    calls: dict[str, int] = {}
    for bid in ("pool", "bonsai"):
        def probe(_b: str = bid) -> bool:
            calls[_b] = calls.get(_b, 0) + 1
            return True
        registry.get(bid).health_check = probe  # type: ignore[union-attr]
    resolver = Resolver(registry)
    for _ in range(3):
        resolver.resolve("gemma")
    assert calls["bonsai"] == 1          # read for lane state once, cached after
    assert calls["pool"] == 1 + 3        # one lane read + the winner's live probe each time
    now[0] = 10.0                        # past the TTL: one fresh probe
    lane_state(registry, "bonsai-model")
    assert calls["bonsai"] == 2


def test_max_concurrent_appended_after_health_check() -> None:
    """Positional construction through health_check still means health_check."""
    b = Backend("b", "http://127.0.0.1:1", ["m"], {"chat"}, 8192, 2048, 1.0, 2.0, 1000,
                lambda: False)
    assert b.max_concurrent == 0
    assert Registry([b]).alive(b) is False


# -- the route marker ------------------------------------------------------------


def test_marker_shape_matches_fleet() -> None:
    assert route_marker("a", "a") == {"requested": "a", "served_by": "a", "cross_model": False}
    assert route_marker("a", "b") == {"requested": "a", "served_by": "b", "cross_model": True}
    assert route_marker(None, "b") == {"requested": "", "served_by": "b", "cross_model": False}


def test_resolution_route_cross_model_true_for_standin() -> None:
    res = Resolver(_registry(standins=CHAIN)).resolve("gemma")
    assert res.route == {"requested": "gemma", "served_by": "pool-model", "cross_model": True}


def test_resolution_route_cross_model_false_when_served() -> None:
    res = Resolver(_registry()).resolve("cloud-model")
    assert res.route == {"requested": "cloud-model", "served_by": "cloud-model",
                         "cross_model": False}


def test_stream_completion_stamps_route(monkeypatch: pytest.MonkeyPatch) -> None:
    body = [b'data: {"choices": []}\n\n', b'data: {"aither_route": {"x": 1}}\n\n',
            b"data: [DONE]\n\n"]
    sent: list = []

    class _Resp:
        def __enter__(self) -> "_Resp":
            return self

        def __exit__(self, *exc: object) -> None:
            return None

        def __iter__(self):  # type: ignore[no-untyped-def]
            return iter(body)

    def _urlopen(request: object, *a: object, **k: object) -> _Resp:
        sent.append(json.loads(request.data))  # type: ignore[attr-defined]
        return _Resp()

    monkeypatch.setattr("urllib.request.urlopen", _urlopen)
    backend = Backend(id="b", base_url="http://127.0.0.1:1")
    route = route_marker("gemma", "pool-model")
    payload = {"model": "gemma", "messages": []}
    events = list(stream_completion(backend, payload, route=route))
    assert sent[-1]["model"] == "pool-model"     # the substitute is what is ASKED for
    assert payload["model"] == "gemma"           # caller's payload untouched
    assert events[0]["aither_route"] == route
    # An upstream marker is composed, never allowed to hide our substitution.
    assert events[1]["aither_route"] == {
        "x": 1, "requested": "gemma", "served_by": "pool-model", "cross_model": True,
    }
    plain = list(stream_completion(backend, payload))
    assert sent[-1]["model"] == "gemma"
    assert "aither_route" not in plain[0]


def test_compose_route_keeps_caller_request_and_upstream_server() -> None:
    ours = route_marker("gemma", "pool-model")
    # The fleet saw only pool-model and served it as asked: still cross-model.
    fleet = route_marker("pool-model", "pool-model")
    assert compose_route(ours, fleet) == {
        "requested": "gemma", "served_by": "pool-model", "cross_model": True,
    }
    # The fleet substituted again: its served_by is the truth.
    again = compose_route(ours, route_marker("pool-model", "bonsai-model"))
    assert again["served_by"] == "bonsai-model"
    # Neither substituted: not cross-model.
    same = route_marker("cloud-model", "cloud-model")
    assert compose_route(same, same)["cross_model"] is False
    assert compose_route(ours, "junk") == ours


# -- no config means no change ----------------------------------------------------


def test_no_config_behaviour_unchanged() -> None:
    registry = _registry()
    resolver = Resolver(registry)
    with pytest.raises(RefusalError, match="no backend serves model 'gemma'"):
        resolver.resolve("gemma")
    load = _busy_pool()
    res = resolver.resolve("pool-model", load=load, priority="user")
    assert (res.model_id, res.backend.id, res.reason) == ("pool-model", "pool", None)
    assert res.route["cross_model"] is False
    assert res.requested == "pool-model"


def test_serve_response_carries_route(monkeypatch: pytest.MonkeyPatch) -> None:
    """The `serve` chat route puts aither_route on its response (needs the serve extra)."""
    pytest.importorskip("fastapi")
    uvicorn = pytest.importorskip("uvicorn")
    from awrouter import cli
    from fastapi.testclient import TestClient

    captured: dict = {}
    monkeypatch.setattr(uvicorn, "run", lambda app, **_k: captured.setdefault("app", app))
    assert cli.main(["serve"]) == 0
    resp = TestClient(captured["app"]).post(
        "/v1/chat/completions", json={"model": "quick-model", "priority": "user"}
    )
    assert resp.json()["aither_route"] == {
        "requested": "quick-model", "served_by": "quick-model", "cross_model": False,
    }


def _serve_app(monkeypatch: pytest.MonkeyPatch, *argv: str) -> object:
    pytest.importorskip("fastapi")
    uvicorn = pytest.importorskip("uvicorn")
    from awrouter import cli
    from fastapi.testclient import TestClient

    captured: dict = {}
    monkeypatch.setattr(uvicorn, "run", lambda app, **_k: captured.setdefault("app", app))
    assert cli.main(["serve", *argv]) == 0
    return TestClient(captured["app"])


def test_serve_refusal_carries_route_and_attempted(
    monkeypatch: pytest.MonkeyPatch, tmp_path: object
) -> None:
    """A refusal is a non-2xx with aither_route naming the refused substitute."""
    cfg = tmp_path / "router.json"  # type: ignore[operator]
    cfg.write_text(json.dumps({
        "backends": [{"id": "pool", "base_url": "http://127.0.0.1:1",
                      "aliases": ["pool-model"], "capabilities": ["chat"],
                      "context_window": 64}],
        "standins": {"gemma": ["pool-model"]},
    }))
    client = _serve_app(monkeypatch, "--config", str(cfg))
    resp = client.post("/v1/chat/completions",  # type: ignore[attr-defined]
                       json={"model": "gemma", "messages": [{"content": "x" * 400}]})
    assert resp.status_code == 503
    assert resp.json()["aither_route"] == {
        "requested": "gemma", "served_by": "", "cross_model": False, "refused": True,
        "attempted": "pool-model", "reason": "standin",
    }
    plain = client.post("/v1/chat/completions", json={"model": "nope"})  # type: ignore[attr-defined]
    assert plain.status_code == 503
    assert "attempted" not in plain.json()["aither_route"]


def test_serve_priority_is_capped_server_side() -> None:
    from awrouter.cli import effective_priority

    assert effective_priority("user", "agent") == "agent"             # cannot claim up
    assert effective_priority("background", "agent") == "background"  # may lower
    assert effective_priority(None, "user") == "user"
    assert effective_priority("user", "user") == "user"
    with pytest.raises(ValueError):
        effective_priority("urgent", "user")


def test_serve_forward_busy_lane_takes_fallback(
    monkeypatch: pytest.MonkeyPatch, tmp_path: object
) -> None:
    """End to end from the CLI: a config with max_concurrent, a held slot,
    and a user turn moves to its fallback and forwards on THAT backend."""
    import threading

    from awrouter import cli

    gate = threading.Event()
    sent: list = []

    def fake_stream(backend, payload, *, route, timeout):  # type: ignore[no-untyped-def]
        sent.append((backend.id, route["served_by"]))
        if payload.get("hold"):
            gate.wait(5)
        yield {"choices": [{"delta": {"content": "hi"}}], "aither_route": dict(route)}

    monkeypatch.setattr(cli, "stream_completion", fake_stream)
    cfg = tmp_path / "router.json"  # type: ignore[operator]
    cfg.write_text(json.dumps({
        "backends": [
            {"id": "pool", "base_url": "http://127.0.0.1:1", "aliases": ["pool-model"],
             "capabilities": ["chat"], "max_concurrent": 1},
            {"id": "bonsai", "base_url": "http://127.0.0.1:2", "aliases": ["bonsai-model"],
             "capabilities": ["chat"], "max_concurrent": 1},
        ],
        "interactive_fallbacks": {"pool-model": "bonsai-model"},
    }))
    client = _serve_app(monkeypatch, "--config", str(cfg), "--forward", "--max-priority", "user")
    holder = threading.Thread(target=lambda: client.post(  # type: ignore[attr-defined]
        "/v1/chat/completions", json={"model": "pool-model", "hold": True}))
    holder.start()
    for _ in range(100):
        if sent:
            break
        gate.wait(0.05)
    try:
        resp = client.post("/v1/chat/completions",  # type: ignore[attr-defined]
                           json={"model": "pool-model", "priority": "user"})
    finally:
        gate.set()
        holder.join(5)
    assert resp.status_code == 200
    assert sent[0] == ("pool", "pool-model")
    assert sent[-1] == ("bonsai", "bonsai-model")
    body = resp.json()
    assert body["choices"][0]["message"]["content"] == "hi"
    assert body["aither_route"] == {
        "requested": "pool-model", "served_by": "bonsai-model", "cross_model": True,
    }
