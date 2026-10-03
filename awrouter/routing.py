"""Routing semantics the fleet scheduler applies — stand-ins, priority, the route marker.

Reimplemented (not imported: this package stays free of any hosting platform)
from the AitherOS MicroScheduler so a router running on a laptop routes the
way the fleet does:

* **Stand-in chains.** A model can name the models that answer for it, in
  preference order. The resolver walks the chain when nobody serves the
  requested model: the first stand-in whose lane is FREE wins, else the first
  BUSY one (it queues, as before), and a DOWN lane is used only when every
  stand-in is down. Env ``AITHER_MODEL_STANDINS`` uses the fleet syntax,
  ``name=a|b[,name=c...]``, read only by a Resolver built with ``use_env=True``.
* **Priority.** ``user`` / ``agent`` / ``background`` (default ``agent``). A
  USER request whose model's lane is busy or down moves to its interactive
  fallback, and only when that fallback's lane is free. Agent and background
  work keep waiting for the lane they asked for. Env
  ``AITHER_INTERACTIVE_BUSY_FALLBACKS`` is ``name=fallback[,...]`` (same opt-in).
* **The route marker.** ``aither_route = {requested, served_by, cross_model}``
  on every response, so a caller always knows whether it got the model it
  asked for.

A lane is the set of backends serving one model. It is "free" when at least
one live, capable backend has spare concurrency, "busy" when every live one is
at its ``max_concurrent``, and "down" when none is live. Without a LoadTracker
or with no declared ``max_concurrent`` a live lane is free: nothing here
invents saturation it cannot measure.
"""

from __future__ import annotations

import os
from typing import Iterable, Optional

from .failover import LoadTracker
from .registry import Backend, Registry

PRIORITIES = ("user", "agent", "background")
DEFAULT_PRIORITY = "agent"

LANE_FREE = "free"
LANE_BUSY = "busy"
LANE_DOWN = "down"

STANDINS_ENV = "AITHER_MODEL_STANDINS"
FALLBACKS_ENV = "AITHER_INTERACTIVE_BUSY_FALLBACKS"


def normalize_priority(priority: Optional[str]) -> str:
    """Lower-cased priority name; None means the default. Unknown names raise."""
    name = (priority or DEFAULT_PRIORITY).strip().lower()
    if name not in PRIORITIES:
        raise ValueError(f"unknown priority {priority!r}; expected one of {PRIORITIES}")
    return name


def parse_standins(raw: str) -> dict[str, list[str]]:
    """``name=a|b,other=c`` -> ``{"name": ["a", "b"], "other": ["c"]}``.

    A name never stands in for itself; blank entries are dropped; the first
    declaration of a name wins, as the fleet's first-match scan does.
    """
    out: dict[str, list[str]] = {}
    for pair in (raw or "").split(","):
        src, _, dst = pair.partition("=")
        name = src.strip()
        if not name or name in out:
            continue
        chain = [d.strip() for d in dst.split("|") if d.strip() and d.strip() != name]
        if chain:
            out[name] = chain
    return out


def parse_fallbacks(raw: str) -> dict[str, str]:
    """``name=fallback,other=x`` -> ``{"name": "fallback", "other": "x"}``. First wins."""
    out: dict[str, str] = {}
    for pair in (raw or "").split(","):
        src, _, dst = pair.partition("=")
        name, target = src.strip(), dst.strip()
        if name and target and target != name and name not in out:
            out[name] = target
    return out


def env_standins() -> dict[str, list[str]]:
    """The operator's stand-in chains from ``AITHER_MODEL_STANDINS`` (read per call)."""
    return parse_standins(os.environ.get(STANDINS_ENV, ""))


def env_fallbacks() -> dict[str, str]:
    """The operator's interactive fallbacks from ``AITHER_INTERACTIVE_BUSY_FALLBACKS``."""
    return parse_fallbacks(os.environ.get(FALLBACKS_ENV, ""))


def backend_state(registry: Registry, backend: Backend, load: Optional[LoadTracker]) -> str:
    """One backend's lane state: down (probe failed), busy (at max_concurrent), else free.

    Liveness is ``Registry.last_alive``: a recent probe is reused, never re-run.
    """
    if not registry.last_alive(backend):
        return LANE_DOWN
    if load is not None and backend.max_concurrent > 0:
        if load.in_flight(backend.id) >= backend.max_concurrent:
            return LANE_BUSY
    return LANE_FREE


def lane_state(
    registry: Registry,
    model_id: str,
    load: Optional[LoadTracker] = None,
    *,
    requirements: Iterable[str] = (),
) -> str:
    """The lane serving ``model_id``: free if any capable backend is free, busy if
    any is merely busy, down when none is live (or nobody serves it at all)."""
    needed = set(requirements)
    states = {
        backend_state(registry, b, load)
        for b in registry.serving(model_id)
        if b.covers(needed)
    }
    for want in (LANE_FREE, LANE_BUSY):
        if want in states:
            return want
    return LANE_DOWN


def pick_standin(chain: list[str], states: list[str]) -> Optional[str]:
    """First free stand-in, else first busy, else the head of the chain."""
    if not chain:
        return None
    for want in (LANE_FREE, LANE_BUSY):
        if want in states:
            return chain[states.index(want)]
    return chain[0]


def route_marker(requested: Optional[str], served: Optional[str]) -> dict:
    """The one field that tells a caller whether it got the model it asked for.

    Same shape as the fleet scheduler's ``aither_route``.
    """
    requested = str(requested or "")
    served = str(served or "")
    return {
        "requested": requested,
        "served_by": served,
        "cross_model": bool(requested and served and served != requested),
    }
