# awrouter — Aither World Router

The **stateless LLM routing plane**: given a model id and a request, decide
which backend should serve it right now — capability filter, then a
policy-weighted cost/latency score, then failover through health probes —
fit the context window, and stream the completion over the OpenAI wire shape.

OpenRouter for your own fleet. Standalone, OpenAI-compatible, no platform
dependencies: the registry is a plain dict, auth is a pluggable callable,
and the stateful scheduler plane (queues, budgets, heartbeats) belongs to
whoever consumes this package, not to it.

## Install

```bash
pip install awrouter              # core: stdlib-only
pip install "awrouter[serve]"     # + the OpenAI wire server (fastapi/uvicorn)
```

## Use

```python
from awrouter import Backend, ModelSpec, ResolutionPolicy, Resolver, Registry

registry = Registry()
registry.register(Backend(
    id="fast",
    base_url="http://127.0.0.1:9000",
    aliases=["quick-model"],
    capabilities={"chat", "tools"},
    context_window=8192,
))
resolver = Resolver(registry, ResolutionPolicy(cost_weight=1.0, latency_weight=0.5))

# Resolve, or raise Refusal with the reason — never a silent downgrade.
resolution = resolver.resolve("quick-model", prompt_chars=2_000)
print(resolution.backend.id)
```

## CLI

```bash
awrouter resolve quick-model                 # -> fast
awrouter resolve big-model --tier free       # refused: not allowed on tier 'free'
awrouter fit --prompt-chars 50000 --window 8192   # refused: context overflow
awrouter backends --json                     # the registry inventory
awrouter serve --port 8240                   # /v1/models + /v1/chat/completions
awrouter self-test                           # prove the refusals can fire (exit 1 on any silent pass)
```

## The contract

- **Refuse, never guess.** Unknown model, capability gap, tier block,
  thinking model with no thinking backend, context overflow, all-dead
  backends — each is a `Refusal` with the reason. Nothing is truncated,
  downgraded, or silently re-routed.
- **Stateless.** The resolver holds no queue, no budget, no in-flight state.
  Scale it horizontally; the state lives in the caller.
- **Pluggable seams.** Backend discovery, auth, token counting, and health
  probes are injectable; the shipped defaults are honest stand-ins (the
  token estimate is a chars-per-token ratio, declared as such).

## License

Apache-2.0.
