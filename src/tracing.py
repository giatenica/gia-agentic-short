"""
OpenTelemetry Tracing Setup
===========================
Configures distributed tracing for the research agent workflow.

Sends traces to AI Toolkit's OTLP endpoint for visualization.

Author: Gia Tenica*
*Gia Tenica is an anagram for Agentic AI. Gia is a fully autonomous AI researcher,
for more information see: https://giatenica.com
"""

import atexit
import json
import os
from typing import Any, Mapping, Optional

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource, SERVICE_NAME
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor

from src.config import TRACING

# Use centralized config
SERVICE_NAME_VALUE = TRACING.SERVICE_NAME
OTLP_ENDPOINT = TRACING.OTLP_ENDPOINT
ENABLE_TRACING = TRACING.ENABLED

# Track provider for cleanup
_provider: Optional[TracerProvider] = None


def _cleanup_tracing() -> None:
    """Shutdown the tracer provider to flush pending spans."""
    global _provider
    if _provider is not None:
        try:
            _provider.shutdown()
        except Exception:
            pass  # Best effort cleanup


def setup_tracing(service_name: str = SERVICE_NAME_VALUE) -> trace.Tracer:
    """
    Initialize OpenTelemetry tracing with OTLP export.
    
    Args:
        service_name: Name of the service for trace identification
        
    Returns:
        Configured tracer instance
    """
    global _provider
    
    # Create resource with service name
    resource = Resource.create({
        SERVICE_NAME: service_name,
    })
    
    # Set up tracer provider
    _provider = TracerProvider(resource=resource)
    
    # Configure OTLP exporter (HTTP)
    otlp_exporter = OTLPSpanExporter(
        endpoint=OTLP_ENDPOINT,
    )
    
    # Add batch processor for efficient export
    _provider.add_span_processor(BatchSpanProcessor(otlp_exporter))
    
    # Set as global tracer provider
    trace.set_tracer_provider(_provider)
    
    # Instrument HTTP clients (for Anthropic API calls)
    HTTPXClientInstrumentor().instrument()
    
    # Register cleanup on exit to flush pending spans
    atexit.register(_cleanup_tracing)
    
    return trace.get_tracer(service_name)


def get_tracer(name: str = SERVICE_NAME_VALUE) -> trace.Tracer:
    """
    Get a tracer instance for creating spans.
    
    Args:
        name: Tracer name (usually module or component name)
        
    Returns:
        Tracer instance
    """
    return trace.get_tracer(name)


def safe_set_span_attributes(span: Any, attributes: Mapping[str, Any]) -> None:
    """Best-effort attribute setter.

    This helper is safe to call when tracing is disabled, when the active span is
    a no-op, or when values are not directly serializable.
    """

    if span is None:
        return

    setter = getattr(span, "set_attribute", None)
    if not callable(setter):
        return

    for key, value in attributes.items():
        if not isinstance(key, str) or not key:
            continue

        try:
            v = value
            if isinstance(v, str):
                setter(key, v[:2048])
                continue

            if isinstance(v, (bool, int, float)) or v is None:
                setter(key, v)
                continue

            if isinstance(v, (list, tuple)):
                # Keep sequences small and scalar.
                trimmed = list(v)[:25]
                scalar_ok = all(isinstance(x, (str, bool, int, float)) or x is None for x in trimmed)
                if scalar_ok:
                    out = [x[:256] if isinstance(x, str) else x for x in trimmed]
                    setter(key, out)
                else:
                    setter(key, [str(x)[:256] for x in trimmed])
                continue

            if isinstance(v, dict):
                try:
                    setter(key, json.dumps(v, sort_keys=True)[:2048])
                except Exception:
                    setter(key, str(v)[:2048])
                continue

            setter(key, str(v)[:2048])
        except Exception:
            # Never break workflows due to tracing.
            continue


def safe_set_current_span_attributes(attributes: Mapping[str, Any]) -> None:
    """Set attributes on the current span when available."""

    try:
        span = trace.get_current_span()
    except Exception:
        return

    safe_set_span_attributes(span, attributes)


# Initialize tracing on module import
_tracer = None


def init_tracing() -> trace.Tracer:
    """
    Initialize tracing if not already done.
    
    Returns:
        The global tracer instance (or NoOp tracer if disabled)
    """
    global _tracer
    if _tracer is None:
        if ENABLE_TRACING:
            _tracer = setup_tracing()
        else:
            # Return NoOp tracer when tracing is disabled
            _tracer = trace.get_tracer(SERVICE_NAME_VALUE)
    return _tracer
