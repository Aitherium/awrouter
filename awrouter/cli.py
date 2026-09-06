"""awrouter CLI — resolve, backends, serve, and a self-test that can fail.

The `serve` subcommand needs the optional `serve` extra (fastapi + uvicorn);
everything else is stdlib-only.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Optional

from .registry import Backend, Registry
from .resolver import (
    ModelSpec,
    RefusalError,
    ResolutionPolicy,
    Resolver,
    TierMap,
    fit_context,
)


def _demo_registry() -> Registry:
    """A tiny example inventory, used by resolve/backends/serve/self-test."""
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
    registry.register(
        Backend(
            id="tiny",
            base_url="http://127.0.0.1:9002",
            aliases=["tiny-model"],
            capabilities={"chat"},
            context_window=4096,
            cost_per_1k_input=0.1,
            cost_per_1k_output=0.2,
            latency_p50_ms=100,
        )
    )
    return registry


def _cmd_resolve(args: argparse.Namespace) -> int:
    registry = _demo_registry()
    resolver = Resolver(registry, ResolutionPolicy(args.cost_weight, args.latency_weight))
    tier_map = TierMap({"free": {"quick-model"}})
    spec = ModelSpec(id=args.model, thinking=args.thinking, requirements=set(args.requires))
    try:
        resolution = resolver.resolve(
            args.model,
            tier=args.tier,
            tier_map=tier_map if args.tier else None,
            spec=spec,
            prompt_chars=args.prompt_chars,
        )
    except Exception as exc:
        print(f"refused: {exc}")
        return 1
    result = {
        "model": resolution.model_id,
        "backend": resolution.backend.id,
        "base_url": resolution.backend.base_url,
        "ranked": resolution.ranked,
    }
    print(json.dumps(result, indent=2) if args.json else result["backend"])
    return 0


def _cmd_backends(args: argparse.Namespace) -> int:
    registry = _demo_registry()
    snapshot = registry.snapshot()
    print(json.dumps(snapshot, indent=2) if args.json else json.dumps(snapshot))
    return 0


def _cmd_fit(args: argparse.Namespace) -> int:
    policy = ResolutionPolicy()
    try:
        tokens, headroom = fit_context(
            args.prompt_chars, args.window, args.output, policy.tokens_per_char
        )
    except Exception as exc:
        print(f"refused: {exc}")
        return 1
    print(f"input ~{tokens} tokens, headroom {headroom} tokens")
    return 0


def _cmd_serve(args: argparse.Namespace) -> int:
    try:
        import uvicorn  # type: ignore[import-not-found]
        from fastapi import FastAPI, Request  # type: ignore[import-not-found]
    except ImportError as exc:
        print(f"serve needs the 'serve' extra: pip install 'awrouter[serve]' ({exc})")
        return 2

    registry = _demo_registry()
    resolver = Resolver(registry, ResolutionPolicy())
    app = FastAPI(title="awrouter", version="0.1.0")

    @app.get("/v1/models")
    def models() -> dict:
        return registry.snapshot()

    @app.post("/v1/chat/completions")
    async def chat(request: Request) -> dict:
        body = await request.json()
        model_id = body.get("model", "")
        try:
            resolution = resolver.resolve(model_id, prompt_chars=len(json.dumps(body)))
        except Exception as exc:
            return {"error": {"message": str(exc), "type": "refusal"}}
        return {
            "id": "chatcmpl-resolved",
            "object": "chat.completion",
            "model": resolution.model_id,
            "backend": resolution.backend.id,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": ""}}],
        }

    uvicorn.run(app, host=args.host, port=args.port)
    return 0


def _cmd_self_test() -> int:
    """Prove the refusals can fire. Any silent pass here is a broken gate."""
    registry = _demo_registry()
    policy = ResolutionPolicy()
    tier_map = TierMap({"free": {"quick-model"}})
    failures: list[str] = []

    def expect_refusal(label: str, fn: object) -> None:
        try:
            fn()  # type: ignore[operator]
            failures.append(f"{label}: did NOT refuse")
        except Exception as exc:
            if not isinstance(exc, RefusalError):
                failures.append(f"{label}: refused with {type(exc).__name__}, not RefusalError")

    resolver = Resolver(registry, policy)

    def unknown() -> object:
        return resolver.resolve("nope")

    def tier_blocked() -> object:
        return resolver.resolve("big-model", tier="free", tier_map=tier_map)

    def thinking_mismatch() -> object:
        return resolver.resolve("tiny-model", spec=ModelSpec(id="tiny-model", thinking=True))

    def window_overflow() -> object:
        return resolver.resolve(
            "quick-model", spec=ModelSpec(id="quick-model"), prompt_chars=100_000
        )

    def no_live_backend() -> object:
        dead = Registry()
        dead.register(
            Backend(
                id="dead", base_url="http://127.0.0.1:1", aliases=["m"], health_check=lambda: False
            )
        )
        return Resolver(dead, policy).resolve("m")

    expect_refusal("unknown model", unknown)
    expect_refusal("tier block", tier_blocked)
    expect_refusal("thinking without a thinking backend", thinking_mismatch)
    expect_refusal("context overflow", window_overflow)
    expect_refusal("no live backend", no_live_backend)

    # And the positive arm: a routable request resolves.
    resolution = resolver.resolve("quick-model")
    if resolution.backend.id not in ("fast", "big"):
        failures.append(f"resolve returned {resolution.backend.id}")

    if failures:
        print("\n".join(f"FAIL: {f}" for f in failures))
        return 1
    print("self-test ok: 5 refusal arms + 1 positive arm")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="awrouter", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    resolve = sub.add_parser("resolve", help="resolve a model to a backend (or refuse)")
    resolve.add_argument("model")
    resolve.add_argument("--tier")
    resolve.add_argument("--thinking", action="store_true")
    resolve.add_argument("--requires", nargs="*", default=[])
    resolve.add_argument("--prompt-chars", type=int, default=0)
    resolve.add_argument("--cost-weight", type=float, default=1.0)
    resolve.add_argument("--latency-weight", type=float, default=0.0)
    resolve.add_argument("--json", action="store_true")
    resolve.set_defaults(func=_cmd_resolve)

    backends = sub.add_parser("backends", help="show the registry inventory")
    backends.add_argument("--json", action="store_true")
    backends.set_defaults(func=_cmd_backends)

    fit = sub.add_parser("fit", help="context-window fit check")
    fit.add_argument("--prompt-chars", type=int, required=True)
    fit.add_argument("--window", type=int, required=True)
    fit.add_argument("--output", type=int, default=1024)
    fit.set_defaults(func=_cmd_fit)

    serve = sub.add_parser("serve", help="serve the OpenAI wire shape (needs serve extra)")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8240)
    serve.set_defaults(func=_cmd_serve)

    selftest = sub.add_parser("self-test", help="prove the refusals can fire")
    selftest.set_defaults(func=lambda _args: _cmd_self_test())
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
