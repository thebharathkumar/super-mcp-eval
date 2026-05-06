"""OpenTelemetry tracing for the eval harness.

The default configuration installs a TracerProvider with no exporters, so
production runs incur no tracing overhead and emit no output. Set
``MCP_EVAL_TRACE_CONSOLE=1`` while debugging to dump spans to stderr.
"""

from __future__ import annotations

import os

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor

_initialized = False


def setup_tracing() -> trace.Tracer:
    """Idempotently install the mcp-eval TracerProvider and return the tracer."""
    global _initialized
    if not _initialized:
        provider = TracerProvider(resource=Resource.create({"service.name": "mcp-eval"}))
        if os.getenv("MCP_EVAL_TRACE_CONSOLE"):
            provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
        trace.set_tracer_provider(provider)
        _initialized = True
    return trace.get_tracer("mcp_eval")


def get_tracer() -> trace.Tracer:
    return setup_tracing()
