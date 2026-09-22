"""The failover contract: one ordered list per request, load-aware, deterministic.

Each test names the property it pins. The burst test is the one that matters:
without reservations every request in a burst sees the same idle backend and
piles onto it (the upstream design's stated regression).
"""

import pytest
from awrouter.failover import (
    UNKNOWN_PRESSURE,
    LoadTracker,
    Reservation,
    is_retryable,
    order_candidates,
    pressure_band,
    pressure_with_hysteresis,
)
from awrouter.registry import Backend, Registry
from awrouter.resolver import RefusalError, Resolver


class FakeClock:
    def __init__(self, t: float = 1000.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t

    def advance(self, s: float) -> None:
        self.t += s


def _backend(bid: str, **kw) -> Backend:
    base = dict(
        id=bid,
        base_url=f"http://127.0.0.1:{9000 + hash(bid) % 100}",
        aliases=["m"],
        capabilities={"chat"},
        context_window=8192,
    )
    base.update(kw)
    return Backend(**base)


# -- pressure bands -------------------------------------------------------------


@pytest.mark.parametrize(
    "util,band",
    [(0, 0), (39.9, 0), (40, 1), (69.9, 1), (70, 2), (84.9, 2), (85, 3), (100, 3)],
)
def test_pressure_band_thresholds(util, band):
    assert pressure_band(util) == band


def test_hysteresis_rises_on_upper_and_falls_only_past_lower():
    assert pressure_with_hysteresis(40, 0) == 1
    # 37% is below the 40 rise threshold but above the 35 fall threshold:
    # a backend already at 1 stays at 1.
    assert pressure_with_hysteresis(37, 1) == 1
    assert pressure_with_hysteresis(34.9, 1) == 0
    # Can climb several bands in one step.
    assert pressure_with_hysteresis(90, 0) == 3
    # And fall several.
    assert pressure_with_hysteresis(10, 3) == 0


def test_hysteresis_with_garbage_previous_falls_back_to_band():
    assert pressure_with_hysteresis(72, -1) == 2
    assert pressure_with_hysteresis(72, 9) == 2


# -- telemetry freshness -----------------------------------------------------------


def test_missing_telemetry_is_neutral_not_idle():
    t = LoadTracker(clock=FakeClock())
    assert t.pressure("nobody") == UNKNOWN_PRESSURE
    assert t.load("nobody") == UNKNOWN_PRESSURE


def test_stale_telemetry_decays_to_neutral():
    clock = FakeClock()
    t = LoadTracker(clock=clock)
    t.observe_utilization("a", 5.0)
    assert t.pressure("a") == 0
    clock.advance(10.1)
    assert t.pressure("a") == UNKNOWN_PRESSURE


def test_sample_already_too_old_on_arrival_is_invalid():
    t = LoadTracker(clock=FakeClock())
    t.observe_utilization("a", 5.0, age_s=11.0)
    assert t.pressure("a") == UNKNOWN_PRESSURE


def test_none_sample_invalidates():
    t = LoadTracker(clock=FakeClock())
    t.observe_utilization("a", 95.0)
    assert t.pressure("a") == 3
    t.observe_utilization("a", None)
    assert t.pressure("a") == UNKNOWN_PRESSURE


def test_ewma_smooths_and_hysteresis_holds_the_band():
    clock = FakeClock()
    t = LoadTracker(clock=clock)
    t.observe_utilization("a", 60.0)  # first fresh sample seeds the EWMA
    assert t.pressure("a") == 1
    clock.advance(1)
    # One 100% spike: ewma = 0.35*100 + 0.65*60 = 74 -> crosses 70 -> 2
    assert t.observe_utilization("a", 100.0) is True
    assert t.pressure("a") == 2
    clock.advance(1)
    # Back to 60: ewma = 0.35*60 + 0.65*74 = 69.1, above the 65 fall line -> stays 2
    assert t.observe_utilization("a", 60.0) is False
    assert t.pressure("a") == 2


def test_utilization_is_clamped():
    t = LoadTracker(clock=FakeClock())
    t.observe_utilization("a", 250.0)
    assert t.pressure("a") == 3
    t2 = LoadTracker(clock=FakeClock())
    t2.observe_utilization("a", -5.0)
    assert t2.pressure("a") == 0


# -- reservations + snapshots -------------------------------------------------------


def test_burst_spreads_across_equal_backends_via_reservations():
    t = LoadTracker(clock=FakeClock())
    a, b = _backend("a"), _backend("b")
    for bid in ("a", "b"):
        t.observe_utilization(bid, 10.0)
    picks = []
    for _ in range(4):
        first = order_candidates([a, b], t)[0]
        t.reserve(first.id)
        picks.append(first.id)
    # Without reservations this would be ["a", "a", "a", "a"].
    assert picks == ["a", "b", "a", "b"]


def test_release_restores_the_order():
    t = LoadTracker(clock=FakeClock())
    a, b = _backend("a"), _backend("b")
    r = t.reserve("a")
    assert order_candidates([a, b], t)[0].id == "b"
    assert t.release(r) is True
    assert order_candidates([a, b], t)[0].id == "a"
    # Double release is a no-op, never a negative count.
    assert t.release(r) is False
    assert t.snapshot().get("a", {}).get("reservations", 0) == 0


def test_snapshot_supersedes_reservations_and_late_release_is_ignored():
    t = LoadTracker(clock=FakeClock())
    r = t.reserve("a")
    gen = t.apply_snapshot({"a": 0, "b": 0})
    assert gen == 1
    assert t.load("a") == UNKNOWN_PRESSURE  # pending 0 + neutral 1 + reservations 0
    # The release belongs to generation 0; the snapshot already counted it.
    assert t.release(r) is False
    # A fresh reservation on the new generation still counts.
    r2 = t.reserve("a")
    assert t.load("a") == UNKNOWN_PRESSURE + 1
    assert t.release(r2) is True


def test_move_transfers_to_the_landed_backend():
    t = LoadTracker(clock=FakeClock())
    r = t.reserve("a")
    r2 = t.move(r, "b")
    assert t.snapshot()["b"]["reservations"] == 1
    assert "a" not in t.snapshot() or t.snapshot()["a"]["reservations"] == 0
    assert r2.backend_id == "b"


def test_unheld_reservation_release_is_noop():
    t = LoadTracker(clock=FakeClock())
    assert t.release(Reservation(backend_id="x", generation=0, held=False)) is False


def test_pending_counts_toward_load():
    t = LoadTracker(clock=FakeClock())
    t.set_pending("a", 3)
    t.observe_utilization("a", 0.0)
    assert t.load("a") == 3


# -- ordering -----------------------------------------------------------------------


def test_stable_id_tiebreak_without_tracker_or_score():
    c, a, b = _backend("c"), _backend("a"), _backend("b")
    assert [x.id for x in order_candidates([c, a, b])] == ["a", "b", "c"]


def test_pressure_breaks_a_load_tie():
    t = LoadTracker(clock=FakeClock())
    # a: pending 2, pressure 0 -> load 2 ; b: pending 0, pressure 2 -> load 2
    t.set_pending("a", 2)
    t.observe_utilization("a", 0.0)
    t.observe_utilization("b", 75.0)
    assert order_candidates([_backend("b"), _backend("a")], t)[0].id == "a"


def test_pinned_leads_only_if_in_the_set():
    a, b = _backend("a"), _backend("b")
    assert order_candidates([a, b], pinned="b")[0].id == "b"
    # A pin outside the (capability-gated) set cannot pull in a stranger.
    assert [x.id for x in order_candidates([a, b], pinned="zzz")] == ["a", "b"]


def test_score_orders_within_equal_load():
    cheap = _backend("z", cost_per_1k_input=0.1, cost_per_1k_output=0.1)
    dear = _backend("a", cost_per_1k_input=5.0, cost_per_1k_output=5.0)
    ordered = order_candidates([dear, cheap], score=lambda b: b.score(1.0, 0.0))
    assert ordered[0].id == "z"


# -- retry policy -------------------------------------------------------------------


@pytest.mark.parametrize("status", [408, 429, 500, 502, 503, 504, 599])
def test_retryable_statuses(status):
    assert is_retryable(status, inference=True)
    assert is_retryable(status, inference=False)


@pytest.mark.parametrize("status", [400, 401, 403, 413, 422, 200])
def test_client_errors_are_not_retried(status):
    assert not is_retryable(status, inference=True)


def test_404_retries_only_on_inference():
    assert is_retryable(404, inference=True)
    assert not is_retryable(404, inference=False)


# -- resolver integration ---------------------------------------------------------


def _two_backend_registry() -> Registry:
    reg = Registry()
    reg.register(_backend("a", cost_per_1k_input=1.0, cost_per_1k_output=1.0))
    reg.register(_backend("b", cost_per_1k_input=1.0, cost_per_1k_output=1.0))
    return reg


def test_resolver_without_tracker_is_unchanged():
    res = Resolver(_two_backend_registry()).resolve("m")
    assert res.backend.id == "a"
    assert res.reservation is None
    assert res.ranked == ["a", "b"]


def test_resolver_with_tracker_reserves_and_spreads():
    t = LoadTracker(clock=FakeClock())
    r = Resolver(_two_backend_registry())
    first = r.resolve("m", load=t)
    second = r.resolve("m", load=t)
    assert first.backend.id == "a"
    assert first.reservation is not None and first.reservation.backend_id == "a"
    assert second.backend.id == "b"
    assert second.ranked == ["b", "a"]
    t.release(first.reservation)
    assert r.resolve("m", load=t).backend.id == "a"


def test_resolver_pin_cannot_bypass_capability_gate():
    reg = Registry()
    reg.register(_backend("a"))
    reg.register(Backend(id="other", base_url="http://127.0.0.1:1", aliases=["different"]))
    res = Resolver(reg).resolve("m", pinned="other")
    assert res.backend.id == "a"
    with pytest.raises(RefusalError):
        Resolver(reg).resolve("nope", pinned="other")
