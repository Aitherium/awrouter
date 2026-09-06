"""The resolver — decide which backend serves a model, or refuse.

Everything here is pure logic over the registry: capability filter, then a
policy-weighted cost/latency score, then failover through health probes, then
a fail-closed context-window fit. A request that cannot be served raises
RefusalError with the reason; nothing is truncated or silently downgraded.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .registry import Backend, Registry


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


@dataclass
class Resolution:
    model_id: str
    backend: Backend
    #: The candidate order the winner came from (audit trail).
    ranked: list[str] = field(default_factory=list)


class RefusalError(Exception):
    """A request that cannot be served. The reason is the message."""


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
    """Stateless decision maker over a Registry. One instance per policy."""

    def __init__(self, registry: Registry, policy: Optional[ResolutionPolicy] = None) -> None:
        self.registry = registry
        self.policy = policy or ResolutionPolicy()

    def resolve(
        self,
        model_id: str,
        *,
        tier: Optional[str] = None,
        tier_map: Optional[TierMap] = None,
        spec: Optional[ModelSpec] = None,
        prompt_chars: int = 0,
        requested_output_tokens: Optional[int] = None,
    ) -> Resolution:
        """Resolve model_id to a live backend, or raise RefusalError."""
        spec = spec or ModelSpec(id=model_id)

        if tier_map is not None and not tier_map.allows(tier, spec.id):
            raise RefusalError(f"model {spec.id!r} not allowed on tier {tier!r}")

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

        # Score all capable candidates, then health-probe the top few in order.
        ranked = sorted(
            capable,
            key=lambda b: b.score(self.policy.cost_weight, self.policy.latency_weight),
        )
        probe_limit = max(1, self.policy.failover_probe_limit)
        for backend in ranked[:probe_limit]:
            if self.registry.alive(backend):
                requested = requested_output_tokens or spec.max_output_tokens
                fit_context(
                    prompt_chars, backend.context_window, requested, self.policy.tokens_per_char
                )
                return Resolution(model_id=spec.id, backend=backend, ranked=[b.id for b in ranked])

        names = ", ".join(b.id for b in ranked[:probe_limit])
        raise RefusalError(f"no live backend among probed candidates: {names}")
