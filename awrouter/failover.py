"""Load-aware failover ordering — the per-request decision, made once, walked in order.

Adapted (reimplemented on awrouter's own primitives, no code copied) from the
routing plane of NVIDIA Personal AI Router (Apache-2.0, https://github.com/NVIDIA/Personal-AI-Router):
`services/nvpair-job-scheduler/{schedule,telemetry}.go` and the reservation
logic in `services/nvpair-proxy/proxy.go`.

What it adds to the resolver, which until now ranked on a STATIC cost/latency
score and pre-probed the top few:

* A coarse, smoothed GPU **pressure** (0-3) per backend with hysteresis, so
  rank does not thrash on a noisy utilization feed. Missing, invalid or
  older-than-ten-second telemetry is a NEUTRAL 1 — never "idle".
* **Pending** work attributed to the backend it was placed on.
* Process-local **reservations**: a dispatch this process just made counts
  against its target immediately, so a burst spreads before any telemetry
  or workload report can catch up. Reservations are stamped with the snapshot
  generation they were counted against; a newer snapshot supersedes them, and
  a late release from an older generation is ignored rather than
  double-counted.
* A **stable id** tiebreak so cold start and unranked backends are
  predictable rather than random.
* A **status-class retry policy**: which upstream statuses mean "try the next
  candidate" and which would fail identically everywhere.

The tracker is deliberately not thread-safe beyond a single lock: it is one
process's view. Two routers dispatching at the same instant can still both
pick the same idle backend — that is corrected by the next snapshot, not
prevented, exactly as the upstream design accepts.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

from .registry import Backend

#: Telemetry older than this contributes the neutral pressure, not its value.
TELEMETRY_FRESHNESS_S = 10.0
#: EWMA weight on the newest utilization sample.
EWMA_ALPHA = 0.35
#: What a backend with no usable telemetry costs. Not 0: an unknown box is
#: not an idle box, and treating it as idle is how a dead node collects work.
UNKNOWN_PRESSURE = 1

_UP = (40.0, 70.0, 85.0)
_DOWN = (0.0, 35.0, 65.0, 80.0)


def pressure_band(utilization_pct: float) -> int:
    """Map a utilization percentage to 0-3 with no history."""
    if utilization_pct < _UP[0]:
        return 0
    if utilization_pct < _UP[1]:
        return 1
    if utilization_pct < _UP[2]:
        return 2
    return 3


def pressure_with_hysteresis(utilization_pct: float, previous: int) -> int:
    """Move at most as far as the thresholds allow; drop only past the LOWER band.

    Rising crosses 40/70/85; falling crosses 35/65/80. A backend hovering at
    68% therefore stays at pressure 1 once it is there and stays at 2 once it
    got there — the rank does not flap on a boundary.
    """
    if previous < 0 or previous > 3:
        return pressure_band(utilization_pct)
    pressure = previous
    while pressure < 3 and utilization_pct >= _UP[pressure]:
        pressure += 1
    while pressure > 0 and utilization_pct < _DOWN[pressure]:
        pressure -= 1
    return pressure


def is_retryable(status: int, *, inference: bool) -> bool:
    """Should an upstream status send the request to the NEXT candidate?

    Busy/unavailable/gateway statuses and any 5xx are retryable. A 404 is
    retryable only on an inference call: it means an advertised owner's
    inventory went stale, and another owner may still hold the model.
    Genuine client errors (400, 401, 403, 413, 422, ...) are not retried —
    they would fail identically on every backend.
    """
    if status in (408, 429, 502, 503, 504):
        return True
    if status == 404:
        return inference
    return status >= 500


@dataclass(frozen=True)
class Reservation:
    """One in-flight dispatch this process made, counted against a backend."""

    backend_id: str
    generation: int
    #: False means "nothing was reserved" — release is then a no-op.
    held: bool = True


@dataclass
class _Telemetry:
    ewma: float = 0.0
    pressure: int = UNKNOWN_PRESSURE
    has_ewma: bool = False
    valid: bool = False
    age_at_receipt: float = 0.0
    received_at: float = 0.0


@dataclass
class LoadTracker:
    """This process's view of every backend's load. Feed it; ask it to order."""

    clock: Callable[[], float] = time.monotonic
    freshness_s: float = TELEMETRY_FRESHNESS_S
    alpha: float = EWMA_ALPHA
    _pending: dict[str, int] = field(default_factory=dict)
    _telemetry: dict[str, _Telemetry] = field(default_factory=dict)
    _reservations: dict[str, int] = field(default_factory=dict)
    _generation: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock)

    # -- telemetry -----------------------------------------------------------

    def observe_utilization(
        self,
        backend_id: str,
        utilization_pct: Optional[float],
        *,
        age_s: float = 0.0,
    ) -> bool:
        """Fold one utilization sample in. Returns True if the pressure changed.

        `utilization_pct` None means the source reported nothing usable this
        tick; `age_s` is how old the sample already was when it arrived. Either
        past the freshness window makes the sample invalid: the backend then
        contributes UNKNOWN_PRESSURE until a fresh sample arrives.
        """
        now = self.clock()
        age = max(0.0, float(age_s))
        fresh = utilization_pct is not None and age <= self.freshness_s
        util = min(100.0, max(0.0, float(utilization_pct or 0.0)))
        with self._lock:
            prev = self._telemetry.get(backend_id, _Telemetry())
            before = self._effective_pressure(prev, now)
            nxt = _Telemetry(
                ewma=prev.ewma,
                pressure=prev.pressure,
                has_ewma=prev.has_ewma,
                valid=fresh,
                age_at_receipt=age,
                received_at=now,
            )
            if fresh:
                prev_fresh = self._is_fresh(prev, now)
                if not prev_fresh or not prev.has_ewma:
                    nxt.ewma = util
                    nxt.pressure = pressure_band(util)
                else:
                    nxt.ewma = self.alpha * util + (1.0 - self.alpha) * prev.ewma
                    nxt.pressure = pressure_with_hysteresis(nxt.ewma, prev.pressure)
                nxt.has_ewma = True
            self._telemetry[backend_id] = nxt
            return before != self._effective_pressure(nxt, now)

    def _is_fresh(self, t: _Telemetry, now: float) -> bool:
        if not t.valid:
            return False
        elapsed = max(0.0, now - t.received_at)
        return t.age_at_receipt + elapsed <= self.freshness_s

    def _effective_pressure(self, t: _Telemetry, now: float) -> int:
        return t.pressure if self._is_fresh(t, now) else UNKNOWN_PRESSURE

    def pressure(self, backend_id: str) -> int:
        now = self.clock()
        with self._lock:
            t = self._telemetry.get(backend_id)
            return UNKNOWN_PRESSURE if t is None else self._effective_pressure(t, now)

    # -- pending + snapshots ---------------------------------------------------

    def set_pending(self, backend_id: str, count: int) -> None:
        with self._lock:
            self._pending[backend_id] = max(0, int(count))

    def apply_snapshot(self, pending: dict[str, int]) -> int:
        """Replace pending counts with a newer view; clear every reservation.

        A snapshot already accounts for the dispatches the reservations were
        standing in for. Advancing the generation is what makes a release
        that arrives AFTER this call a no-op instead of an undercount.
        Returns the new generation.
        """
        with self._lock:
            self._pending = {k: max(0, int(v)) for k, v in pending.items()}
            self._reservations.clear()
            self._generation += 1
            return self._generation

    @property
    def generation(self) -> int:
        with self._lock:
            return self._generation

    # -- reservations -----------------------------------------------------------

    def reserve(self, backend_id: str) -> Reservation:
        with self._lock:
            self._reservations[backend_id] = self._reservations.get(backend_id, 0) + 1
            return Reservation(backend_id=backend_id, generation=self._generation)

    def release(self, reservation: Reservation) -> bool:
        """Drop a reservation. Returns True only if it was actually counted."""
        if not reservation.held:
            return False
        with self._lock:
            if reservation.generation != self._generation:
                return False
            n = self._reservations.get(reservation.backend_id, 0)
            if n <= 0:
                return False
            if n == 1:
                del self._reservations[reservation.backend_id]
            else:
                self._reservations[reservation.backend_id] = n - 1
            return True

    def move(self, reservation: Reservation, backend_id: str) -> Reservation:
        """Transfer a reservation to the backend a failover actually landed on.

        Across a snapshot boundary the old one is already gone, so this simply
        takes a fresh reservation on the new backend.
        """
        self.release(reservation)
        return self.reserve(backend_id)

    # -- the number the order is built on ------------------------------------------

    def load(self, backend_id: str) -> int:
        now = self.clock()
        with self._lock:
            t = self._telemetry.get(backend_id)
            pressure = UNKNOWN_PRESSURE if t is None else self._effective_pressure(t, now)
            return (
                self._pending.get(backend_id, 0)
                + pressure
                + self._reservations.get(backend_id, 0)
            )

    def snapshot(self) -> dict[str, dict[str, int]]:
        """Serializable view: per backend pending / pressure / reservations / load."""
        now = self.clock()
        with self._lock:
            ids = set(self._pending) | set(self._telemetry) | set(self._reservations)
            out: dict[str, dict[str, int]] = {}
            for bid in sorted(ids):
                t = self._telemetry.get(bid)
                pressure = UNKNOWN_PRESSURE if t is None else self._effective_pressure(t, now)
                pending = self._pending.get(bid, 0)
                reserved = self._reservations.get(bid, 0)
                out[bid] = {
                    "pending": pending,
                    "pressure": pressure,
                    "reservations": reserved,
                    "load": pending + pressure + reserved,
                }
            return out


def order_candidates(
    candidates: Sequence[Backend],
    tracker: Optional[LoadTracker] = None,
    *,
    pinned: Optional[str] = None,
    score: Optional[Callable[[Backend], float]] = None,
) -> list[Backend]:
    """Produce the ordered failover list for one request.

    Order: an explicitly pinned candidate (only if it is IN the candidate set —
    a pin cannot override the capability gate that built the set), then by
    load (pending + pressure + reservations), then by pressure alone, then by
    the caller's score, then by stable id. With no tracker every backend is
    load-equal and the score decides, which is the resolver's prior behaviour.
    """
    ordered = sorted(
        candidates,
        key=lambda b: (
            tracker.load(b.id) if tracker else 0,
            tracker.pressure(b.id) if tracker else 0,
            score(b) if score else 0.0,
            b.id,
        ),
    )
    if pinned:
        for i, b in enumerate(ordered):
            if b.id == pinned:
                ordered.insert(0, ordered.pop(i))
                break
    return ordered
