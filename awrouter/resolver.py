"""The resolver — decide which backend serves a model, or refuse.

Everything here is pure logic over the registry: capability filter, then a
policy-weighted cost/latency score, then failover through health probes, then
a fail-closed context-window fit. A request that cannot be served raises
RefusalError with the reason; nothing is truncated or silently downgraded.

Before that, the fleet's routing semantics pick WHICH model answers (see
routing.py): a declared stand-in when nobody serves the requested model, an
interactive fallback for a user turn on a busy lane. Either is visible on the
Resolution's route marker — a stand-in is a disclosed substitution, never a
silent one.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Optional

from .failover import LoadTracker, Reservation, order_candidates
from .registry import Backend, Registry
from .routing import (
    LANE_FREE,
    env_fallbacks,
    env_standins,
    lane_state,
    normalize_priority,
    pick_standin,
    route_marker,
)


@dataclass
class ModelSpec:
    """What a model requires and how it behaves. Served by matching backends."""

    id: str
    #: Hard requirements a backend's capabilities must cover.
    requirements: set[str] = field(default_factory=set)
    context_window: int = 8192
    max_output_tokens: int = 2048
    #: A thinking model needs a backend that can hold a chain of thought;
    #: routing it to a plain chat backend is refused, not downgraded.
    thinking: bool = False


@dataclass
class TierMap:
    """Plan -> allowed model ids. The entitlement plane is pluggable: this is
    the pure mapping the platform's own auth layer feeds into."""

    tiers: dict[str, set[str]] = field(default_factory=dict)

    def allows(self, tier: Optional[str], model_id: str) -> bool:
        if tier is None:
            return True
        allowed = self.tiers.get(tier)
        if allowed is None:
            return False
        return model_id in allowed


@dataclass
class ResolutionPolicy:
    """How the resolver weighs candidates and how hard it probes."""

    cost_weight: float = 1.0
    latency_weight: float = 0.0
    #: How many scored candidates to health-probe before giving up.
    failover_probe_limit: int = 3
    #: Rough tokens-per-character estimate used only when no tokenizer is
    #: plugged in. Callers with a real tokenizer should pass exact counts.
    tokens_per_char: float = 0.25
    #: model id -> the model a USER-priority turn may use while that model's
    #: lane is busy or down. With Resolver(use_env=True),
    #: AITHER_INTERACTIVE_BUSY_FALLBACKS overrides per name.
    interactive_fallbacks: dict[str, str] = field(default_factory=dict)


@dataclass
class Resolution:
    #: The model that SERVES the request (a stand-in or fallback when rerouted).
    model_id: str
    backend: Backend
    #: The candidate order the winner came from (audit trail). With a load
    #: tracker this is the FAILOVER LIST: walk it on a retryable status.
    ranked: list[str] = field(default_factory=list)
    #: Taken against the winner when a LoadTracker was supplied, so the
    #: dispatch counts immediately. Release it when the request ends; move
    #: it if failover lands elsewhere.
    reservation: Optional[Reservation] = None
    #: The model the caller asked for. Empty means "the same as model_id".
    requested: str = ""
    #: Why model_id differs from requested: "standin", "interactive_fallback",
    #: or None when the requested model serves.
    reason: Optional[str] = None

    @property
    def route(self) -> dict:
        """The ``aither_route`` marker: {requested, served_by, cross_model}."""
        return route_marker(self.requested or self.model_id, self.model_id)


class RefusalError(Exception):
    """A request that cannot be served. The reason is the message.

    ``requested`` / ``attempted`` say which model was refused: when a stand-in
    or fallback was tried, ``attempted`` names it, so a caller can tell a
    refused substitute from a refused original. Empty when not known.
    """

    def __init__(self, message: str, *, requested: str = "", attempted: str = "",
                 reason: Optional[str] = None) -> None:
        super().__init__(message)
        self.requested = requested
        self.attempted = attempted
        self.reason = reason

    @property
    def route(self) -> dict:
        """An ``aither_route`` for the refusal: nothing served it.

        Same three keys as a served route (``served_by`` empty), plus
        ``refused`` and, for a refused substitute, ``attempted`` + ``reason``.
        """
        marker = route_marker(self.requested, "")
        marker["refused"] = True
        if self.attempted and self.attempted != self.requested:
            marker["attempted"] = self.attempted
            marker["reason"] = self.reason
        return marker


def estimate_tokens(text: str, tokens_per_char: float) -> int:
    """Deterministic stand-in for a tokenizer when none is injected."""
    return max(1, int(len(text) * tokens_per_char))


def fit_context(
    prompt_chars: int,
    model_window: int,
    requested_output: int,
    tokens_per_char: float,
) -> tuple[int, int]:
    """Return (input_tokens, headroom) or raise RefusalError.

    headroom is what remains of the window after input + requested output.
    A non-positive headroom refuses: the caller asked for more than the
    backend can hold, and silently trimming is exactly the failure mode this
    package exists to prevent.
    """
    input_tokens = estimate_tokens("x" * prompt_chars, tokens_per_char)
    headroom = model_window - input_tokens - requested_output
    if headroom < 0:
        raise RefusalError(
            f"context overflow: ~{input_tokens} input + {requested_output} "
            f"output > window {model_window}"
        )
    return input_tokens, headroom


class Resolver:
    """Stateless decision maker over a Registry. One instance per policy.

    ``use_env=True`` lets the operator's posture env (AITHER_MODEL_STANDINS,
    AITHER_INTERACTIVE_BUSY_FALLBACKS) override the registry and policy per
    name. Off by default: an embedded library never picks up a host
    process's posture without asking for it. The CLI turns it on.
    """

    def __init__(
        self,
        registry: Registry,
        policy: Optional[ResolutionPolicy] = None,
        *,
        use_env: bool = False,
    ) -> None:
        self.registry = registry
        self.policy = policy or ResolutionPolicy()
        self.use_env = use_env

    def resolve(
        self,
        model_id: str,
        *,
        tier: Optional[str] = None,
        tier_map: Optional[TierMap] = None,
        spec: Optional[ModelSpec] = None,
        prompt_chars: int = 0,
        requested_output_tokens: Optional[int] = None,
        load: Optional[LoadTracker] = None,
        pinned: Optional[str] = None,
        priority: Optional[str] = None,
    ) -> Resolution:
        """Resolve model_id to a live backend, or raise RefusalError.

        With `load`, the capable set is ordered by live load (pending +
        smoothed GPU pressure + this process's own in-flight reservations),
        then pressure, then the policy score, then stable id — and the winner
        is reserved before returning. Without it, the policy score alone
        orders, which is the original behaviour. `pinned` puts one backend
        first, but only if it survived the capability gate.

        `priority` is user | agent | background (default agent). Only a user
        request takes an interactive fallback. When nobody serves model_id, a
        declared stand-in chain is walked (registry declaration, overridden by
        AITHER_MODEL_STANDINS when `use_env`). A pinned request never moves to
        another model. The Resolution's `route` says which happened; a
        RefusalError's `route` says which model was refused.
        """
        prio = normalize_priority(priority)
        spec = spec or ModelSpec(id=model_id)
        requested = spec.id

        if tier_map is not None and not tier_map.allows(tier, spec.id):
            raise RefusalError(
                f"model {spec.id!r} not allowed on tier {tier!r}", requested=requested
            )

        served, reason = self._route_model(spec, prio, tier, tier_map, load, pinned)
        if served == requested:
            try:
                resolution = self._resolve_spec(
                    spec, prompt_chars, requested_output_tokens, load, pinned
                )
            except RefusalError as exc:
                raise RefusalError(str(exc), requested=requested, attempted=requested) from exc
        else:
            try:
                resolution = self._resolve_spec(
                    replace(spec, id=served), prompt_chars, requested_output_tokens, load, pinned
                )
            except RefusalError as exc:
                raise RefusalError(
                    f"{requested!r} -> {reason} {served!r}: {exc}",
                    requested=requested,
                    attempted=served,
                    reason=reason,
                ) from exc
        resolution.requested = requested
        resolution.reason = reason
        return resolution

    def standins_for(self, model_id: str) -> list[str]:
        """The stand-in chain for model_id: with use_env, the operator's env wins."""
        from_env = env_standins().get(model_id) if self.use_env else None
        return from_env or self.registry.standins_for(model_id)

    def fallback_for(self, model_id: str) -> Optional[str]:
        """The interactive fallback for model_id: with use_env, env wins over the policy."""
        from_env = env_fallbacks().get(model_id) if self.use_env else None
        return from_env or self.policy.interactive_fallbacks.get(model_id)

    def _route_model(
        self,
        spec: ModelSpec,
        priority: str,
        tier: Optional[str],
        tier_map: Optional[TierMap],
        load: Optional[LoadTracker],
        pinned: Optional[str],
    ) -> tuple[str, Optional[str]]:
        """Pick the model that answers: (model_id, reason) — reason None = as asked.

        A substitute must itself be allowed on the caller's tier; one that is
        not is skipped, never served.
        """
        requested = spec.id
        needs = set(spec.requirements) | ({"thinking"} if spec.thinking else set())

        def allowed(model: str) -> bool:
            return tier_map is None or tier_map.allows(tier, model)

        # A pin is an attribution promise: it never moves to another model --
        # not to a stand-in, not to a fallback. An unserved pinned model refuses.
        if pinned:
            return requested, None

        if not self.registry.serving(requested):
            chain = [m for m in self.standins_for(requested) if allowed(m)]
            states = [lane_state(self.registry, m, load, requirements=needs) for m in chain]
            pick = pick_standin(chain, states)
            return (pick, "standin") if pick else (requested, None)

        if priority != "user":
            return requested, None
        fallback = self.fallback_for(requested)
        if not fallback or not allowed(fallback):
            return requested, None
        if lane_state(self.registry, requested, load, requirements=needs) == LANE_FREE:
            return requested, None
        if lane_state(self.registry, fallback, load, requirements=needs) != LANE_FREE:
            return requested, None
        return fallback, "interactive_fallback"

    def _resolve_spec(
        self,
        spec: ModelSpec,
        prompt_chars: int,
        requested_output_tokens: Optional[int],
        load: Optional[LoadTracker],
        pinned: Optional[str],
    ) -> Resolution:
        """The capability / thinking / failover / fit pipeline for one model."""
        candidates = self.registry.serving(spec.id)
        if not candidates:
            raise RefusalError(f"no backend serves model {spec.id!r}")

        capable = [b for b in candidates if b.covers(spec.requirements)]
        if not capable:
            names = ", ".join(sorted(b.id for b in candidates))
            raise RefusalError(
                f"no backend serving {spec.id!r} meets requirements "
                f"{sorted(spec.requirements)} (candidates: {names})"
            )

        if spec.thinking:
            thinkers = [b for b in capable if "thinking" in b.capabilities]
            if not thinkers:
                raise RefusalError(
                    f"model {spec.id!r} is a thinking model; no thinking backend serves it"
                )
            capable = thinkers

        # Order all capable candidates (load first when a tracker is supplied,
        # else the policy score), then health-probe the top few in order.
        ranked = order_candidates(
            capable,
            load,
            pinned=pinned,
            score=lambda b: b.score(self.policy.cost_weight, self.policy.latency_weight),
        )
        probe_limit = max(1, self.policy.failover_probe_limit)
        for backend in ranked[:probe_limit]:
            if self.registry.alive(backend):
                requested = requested_output_tokens or spec.max_output_tokens
                fit_context(
                    prompt_chars, backend.context_window, requested, self.policy.tokens_per_char
                )
                reservation = load.reserve(backend.id) if load is not None else None
                return Resolution(
                    model_id=spec.id,
                    backend=backend,
                    ranked=[b.id for b in ranked],
                    reservation=reservation,
                )

        names = ", ".join(b.id for b in ranked[:probe_limit])
        raise RefusalError(f"no live backend among probed candidates: {names}")
