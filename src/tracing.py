"""
OpenTelemetry Tracing Setup
===========================
Configures distributed tracing for the research agent workflow.

Sends traces to AI Toolkit's OTLP endpoint for visualization.

Author: Gia Tenica*
*Gia Tenica is an anagram for Agentic AI. Gia is a fully autonomous AI researcher,
for more information see: https://giatenica.com
"""

import os
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource, SERVICE_NAME
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor

# Service configuration
SERVICE_NAME_VALUE = "gia-research-agents"
OTLP_ENDPOINT = os.getenv("OTLP_ENDPOINT", "http://localhost:4318/v1/traces")


def setup_tracing(service_name: str = SERVICE_NAME_VALUE) -> trace.Tracer:
    """
    Initialize OpenTelemetry tracing with OTLP export.
    
    Args:
        service_name: Name of the service for trace identification
        
    Returns:
        Configured tracer instance
    """
    # Create resource with service name
    resource = Resource.create({
        SERVICE_NAME: service_name,
    })
    
    # Set up tracer provider
    provider = TracerProvider(resource=resource)
    
    # Configure OTLP exporter (HTTP)
    otlp_exporter = OTLPSpanExporter(
        endpoint=OTLP_ENDPOINT,
    )
    
    # Add batch processor for efficient export
    provider.add_span_processor(BatchSpanProcessor(otlp_exporter))
    
    # Set as global tracer provider
    trace.set_tracer_provider(provider)
    
    # Instrument HTTP clients (for Anthropic API calls)
    HTTPXClientInstrumentor().instrument()
    
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


# Initialize tracing on module import
_tracer = None


def init_tracing() -> trace.Tracer:
    """
    Initialize tracing if not already done.
    
    Returns:
        The global tracer instance
    """
    global _tracer
    if _tracer is None:
        _tracer = setup_tracing()
    return _tracer
