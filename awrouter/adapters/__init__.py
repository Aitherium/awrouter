"""Platform-specific glue, kept OUT of the generic core on purpose.

`awrouter.registry` / `.resolver` / `.wire` are pure OpenAI-shape, stdlib,
standalone -- they know nothing about any one platform. An adapter module
bridges one real platform's actual surface (its inventory shape, its actual
wire behavior) into that generic core, so the core never grows a special
case for a platform it should not know exists.
"""

from __future__ import annotations

__all__: list[str] = []
