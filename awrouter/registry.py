"""Backend registry — the pluggable inventory the resolver chooses from.

A Backend declares what it can serve (aliases, capabilities, dimensions) and
how expensive/slow it is. Nothing here reaches the network; health checking
is an injected callable so the registry stays a pure data plane.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional


@dataclass
class Backend:
    """One model-serving endpoint (vLLM, llama.cpp, any OpenAI-compatible)."""

    id: str
    base_url: str
    #: Model ids this backend can serve, in the backend's own spelling.
    aliases: list[str] = field(default_factory=list)
    #: Hard capability requirements. A model is only routable to a backend
    #: whose capabilities are a superset of the model's requirements.
    capabilities: set[str] = field(default_factory=set)
    #: Context window the backend actually allocates (tokens). The resolver
    #: REFUSES a request that does not fit — never silently truncates.
    context_window: int = 8192
    max_output_tokens: int = 2048
    #: Cost per 1k tokens, input and output, in arbitrary consistent units.
    cost_per_1k_input: float = 1.0
    cost_per_1k_output: float = 2.0
    #: Median latency for a first token, ms. Used only for scoring, never
    #: for correctness.
    latency_p50_ms: int = 1000
    #: Optional liveness probe: called with no arguments, returns True if the
    #: backend answers. None means "assume up" (declared, not verified).
    health_check: Optional[Callable[[], bool]] = None

    def covers(self, requirement: set[str]) -> bool:
        """True when this backend satisfies every hard requirement."""
        return requirement <= self.capabilities

    def score(self, cost_weight: float, latency_weight: float) -> float:
        """Lower is better. The blend is the policy's, not the backend's."""
        cost = self.cost_per_1k_input + self.cost_per_1k_output
        return cost_weight * cost + latency_weight * self.latency_p50_ms


class Registry:
    """Ordered collection of backends, addressable by id and by alias."""

    def __init__(self, backends: Optional[list[Backend]] = None) -> None:
        self._by_id: dict[str, Backend] = {}
        #: One alias may be served by SEVERAL backends — that is the failover
        #: case (two engines serving one model id), not an error.
        self._by_alias: dict[str, list[Backend]] = {}
        for backend in backends or []:
            self.register(backend)

    def register(self, backend: Backend) -> None:
        """Add a backend. A duplicate id raises (fail loud)."""
        if backend.id in self._by_id:
            raise ValueError(f"duplicate backend id: {backend.id}")
        self._by_id[backend.id] = backend
        for alias in backend.aliases:
            self._by_alias.setdefault(alias, []).append(backend)

    def get(self, backend_id: str) -> Optional[Backend]:
        return self._by_id.get(backend_id)

    def serving(self, model_id: str) -> list[Backend]:
        """Every backend that can serve model_id, by alias or declared id."""
        if model_id in self._by_alias:
            return list(self._by_alias[model_id])
        return [b for b in self._by_id.values() if model_id in b.aliases]

    def alive(self, backend: Backend) -> bool:
        """The injected probe decides; an absent probe declares up."""
        if backend.health_check is None:
            return True
        try:
            return bool(backend.health_check())
        except Exception:
            return False

    def snapshot(self) -> dict:
        """Serializable inventory for /backends/snapshot."""
        return {
            "backends": [
                {
                    "id": b.id,
                    "aliases": b.aliases,
                    "capabilities": sorted(b.capabilities),
                    "context_window": b.context_window,
                    "max_output_tokens": b.max_output_tokens,
                    "cost_per_1k_input": b.cost_per_1k_input,
                    "cost_per_1k_output": b.cost_per_1k_output,
                    "latency_p50_ms": b.latency_p50_ms,
                }
                for b in self._by_id.values()
            ]
        }
