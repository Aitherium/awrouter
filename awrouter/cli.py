"""awrouter CLI — resolve, backends, serve, and a self-test that can fail.

The `serve` subcommand needs the optional `serve` extra (fastapi + uvicorn);
everything else is stdlib-only.

`--config FILE` (JSON) replaces the demo inventory with your own:
``{"backends": [{"id", "base_url", "aliases", "max_concurrent", "health_url",
...}], "standins": {"model": ["a", "b"]}, "interactive_fallbacks": {"model":
"fallback"}, "health_ttl_s": 5}``. The CLI also honours the posture env
(AITHER_MODEL_STANDINS, AITHER_INTERACTIVE_BUSY_FALLBACKS) over the file.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from typing import Any, Callable, Optional

from .failover import LoadTracker
from .registry import Backend, Registry
from .resolver import (
    ModelSpec,
    RefusalError,
    ResolutionPolicy,
    Resolver,
    TierMap,
    fit_context,
)
from .routing import PRIORITIES, normalize_priority, route_marker
from .wire import stream_completion, unwrap_tool_calls

#: Backend fields a config entry may set (health_url is the CLI's own probe).
_BACKEND_KEYS = {
    "id", "base_url", "aliases", "capabilities", "context_window", "max_output_tokens",
    "cost_per_1k_input", "cost_per_1k_output", "latency_p50_ms", "max_concurrent",
}


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


def _http_probe(url: str, timeout: float = 2.0) -> Callable[[], bool]:
    """A liveness probe for a config backend: GET url answers 2xx in time."""

    def probe() -> bool:
        with urllib.request.urlopen(url, timeout=timeout) as resp:  # noqa: S310
            return 200 <= int(resp.status) < 300

    return probe


def load_config(path: str) -> tuple[Registry, dict[str, str]]:
    """Read a JSON router config: (registry, interactive_fallbacks).

    Unknown backend keys raise (a typo never silently drops a cap); a
    ``health_url`` becomes the backend's probe, else it is declared up.
    """
    with open(path, encoding="utf-8") as fh:
        raw = json.load(fh)
    registry = Registry(
        standins=raw.get("standins") or {},
        health_ttl_s=float(raw.get("health_ttl_s", 5.0)),
    )
    for entry in raw.get("backends") or []:
        entry = dict(entry)
        health_url = entry.pop("health_url", None)
        unknown = set(entry) - _BACKEND_KEYS
        if unknown:
            raise ValueError(f"backend {entry.get('id')!r}: unknown keys {sorted(unknown)}")
        entry["capabilities"] = set(entry.get("capabilities") or ())
        if health_url:
            entry["health_check"] = _http_probe(str(health_url))
        registry.register(Backend(**entry))
    return registry, dict(raw.get("interactive_fallbacks") or {})


def _build_resolver(args: argparse.Namespace) -> Resolver:
    """The demo inventory, or --config's; the posture env is honoured either way."""
    fallbacks: dict[str, str] = {}
    registry = _demo_registry()
    if getattr(args, "config", None):
        registry, fallbacks = load_config(args.config)
    policy = ResolutionPolicy(
        cost_weight=getattr(args, "cost_weight", 1.0),
        latency_weight=getattr(args, "latency_weight", 0.0),
        interactive_fallbacks=fallbacks,
    )
    return Resolver(registry, policy, use_env=True)


def effective_priority(asked: Optional[str], ceiling: str) -> str:
    """The priority a serve request gets: a client may LOWER its own
    priority, never raise it above the server's ``--max-priority``.

    Priority decides who may take an interactive fallback, so it is set
    server-side, as the fleet does; a body cannot claim ``user`` on its own.
    """
    asked_name = normalize_priority(asked) if asked else ceiling
    rank = PRIORITIES.index
    return asked_name if rank(asked_name) >= rank(ceiling) else ceiling


def _completion_from_events(events: list[dict[str, Any]], model: str, route: dict) -> dict:
    """Fold forwarded SSE events into one chat.completion body."""
    for event in reversed(events):
        if isinstance(event.get("aither_route"), dict):
            route = event["aither_route"]
            break
    folded = unwrap_tool_calls(iter(events))
    message: dict[str, Any] = {"role": "assistant", "content": folded["content"]}
    if folded["tool_calls"]:
        message["tool_calls"] = [
            {"id": c["id"], "type": "function",
             "function": {"name": c["name"], "arguments": c["arguments"]}}
            for c in folded["tool_calls"]
        ]
    return {
        "id": "chatcmpl-awrouter",
        "object": "chat.completion",
        "model": model,
        "aither_route": route,
        "choices": [{
            "index": 0,
            "message": message,
            "finish_reason": "tool_calls" if folded["tool_calls"] else "stop",
        }],
    }


def _cmd_resolve(args: argparse.Namespace) -> int:
    resolver = _build_resolver(args)
    tier_map = TierMap({"free": {"quick-model"}})
    spec = ModelSpec(id=args.model, thinking=args.thinking, requirements=set(args.requires))
    try:
        resolution = resolver.resolve(
            args.model,
            tier=args.tier,
            tier_map=tier_map if args.tier else None,
            spec=spec,
            prompt_chars=args.prompt_chars,
            priority=args.priority,
        )
    except Exception as exc:
        print(f"refused: {exc}")
        return 1
    result = {
        "model": resolution.model_id,
        "backend": resolution.backend.id,
        "base_url": resolution.backend.base_url,
        "ranked": resolution.ranked,
        "aither_route": resolution.route,
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

    from fastapi.responses import (  # type: ignore[import-not-found]
        JSONResponse,
        StreamingResponse,
    )
    from starlette.concurrency import (
        run_in_threadpool,  # type: ignore[import-not-found]
    )

    resolver = _build_resolver(args)
    registry = resolver.registry
    #: In-flight per backend for this process. With Backend.max_concurrent it
    #: is what makes a lane BUSY: a forwarded request holds its slot until done.
    load = LoadTracker()
    app = FastAPI(title="awrouter", version="0.1.0")

    def refused(status: int, message: str, kind: str, route: dict) -> Any:
        # Every response carries aither_route, a refusal included.
        return JSONResponse(
            status_code=status,
            content={"error": {"message": message, "type": kind}, "aither_route": route},
        )

    def release(resolution: Any) -> None:
        if resolution.reservation is not None:
            load.release(resolution.reservation)

    @app.get("/v1/models")
    def models() -> dict:
        return registry.snapshot()

    async def chat(request: Request) -> Any:
        body = await request.json()
        model_id = str(body.get("model", ""))
        try:
            priority = effective_priority(body.get("priority"), args.max_priority)
        except ValueError as exc:
            return refused(400, str(exc), "invalid_request", route_marker(model_id, ""))
        try:
            resolution = resolver.resolve(
                model_id, prompt_chars=len(json.dumps(body)), priority=priority, load=load
            )
        except RefusalError as exc:
            return refused(503, str(exc), "refusal", exc.route)
        route = resolution.route

        if not args.forward:
            release(resolution)
            return {
                "id": "chatcmpl-resolved",
                "object": "chat.completion",
                "model": resolution.model_id,
                "backend": resolution.backend.id,
                "aither_route": route,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": ""}}],
            }

        payload = {k: v for k, v in body.items() if k != "priority"}
        payload["stream"] = True
        events = stream_completion(resolution.backend, payload, route=route, timeout=args.timeout)
        streaming = bool(body.get("stream"))
        try:
            if streaming:
                first = await run_in_threadpool(next, events, None)
            else:
                collected = await run_in_threadpool(list, events)
        except Exception as exc:
            release(resolution)
            return refused(502, f"upstream {resolution.backend.id}: {exc}", "upstream", route)
        if not streaming:
            release(resolution)
            return _completion_from_events(collected, resolution.model_id, route)

        def relay() -> Any:
            try:
                if first is not None:
                    yield f"data: {json.dumps(first)}\n\n"
                    for event in events:
                        yield f"data: {json.dumps(event)}\n\n"
                yield "data: [DONE]\n\n"
            finally:
                release(resolution)

        return StreamingResponse(relay(), media_type="text/event-stream")

    # PEP 563 makes `Request` the STRING "Request", which FastAPI resolves against
    # this module's globals, where the lazily imported class is absent: the body
    # param then reads as a required query field and every POST was a 422.
    chat.__annotations__["request"] = Request
    app.post("/v1/chat/completions")(chat)

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
    resolve.add_argument("--priority", choices=["user", "agent", "background"], default=None)
    resolve.add_argument("--json", action="store_true")
    resolve.add_argument("--config", help="router config JSON (default: the demo inventory)")
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
    serve.add_argument("--config", help="router config JSON (default: the demo inventory)")
    serve.add_argument(
        "--max-priority", choices=list(PRIORITIES), default="agent",
        help="highest priority a request body may claim (user: a single-user local router)",
    )
    serve.add_argument(
        "--forward", action="store_true",
        help="proxy to the resolved backend (default: answer with the resolution only)",
    )
    serve.add_argument("--timeout", type=float, default=60.0, help="upstream timeout, seconds")
    serve.set_defaults(func=_cmd_serve)

    selftest = sub.add_parser("self-test", help="prove the refusals can fire")
    selftest.set_defaults(func=lambda _args: _cmd_self_test())
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    # GENERATED doctor intercept (gen_aw_doctor.py) -- do not edit
    _dv = locals().get("argv")
    if (_dv if _dv is not None else __import__("sys").argv[1:])[:1] == ["doctor"]:
        from ._doctor import report
        return report()
    # GENERATED repo-state intercept (gen_aw_doctor.py) -- do not edit
    try:
        from awgit import state as _aw_state
    except Exception:
        _aw_state = None
    if _aw_state is not None:
        _sv = locals().get("argv")
        if _aw_state.cli_banner(_sv if _sv is not None else __import__("sys").argv[1:]):
            return 0
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
