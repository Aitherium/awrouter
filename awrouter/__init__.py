"""awrouter — Aither World Router.

The STATELESS LLM routing plane: given a model id and a request, decide which
backend should serve it right now (capability filter, then cost/latency
score, then failover), fit the context window, and stream the completion
over the OpenAI wire shape.

Deliberately standalone: zero imports from any hosting platform. A backend
registry is a plain dict; auth is a pluggable callable; the stateful
scheduler plane (queues, budgets, heartbeat) belongs to whoever consumes this
package, not to it.
"""

from .registry import Backend, Registry
from .resolver import (
    ModelSpec,
    Resolution,
    ResolutionPolicy,
    Resolver,
    TierMap,
    fit_context,
)
from .wire import stream_completion

__all__ = [
    "Backend",
    "ModelSpec",
    "Registry",
    "Resolution",
    "ResolutionPolicy",
    "Resolver",
    "TierMap",
    "fit_context",
    "stream_completion",
]

__version__ = "0.1.0"
