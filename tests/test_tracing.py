"""
Tracing Module Tests
====================
Tests for OpenTelemetry tracing setup.

Author: Gia Tenica*
*Gia Tenica is an anagram for Agentic AI. Gia is a fully autonomous AI researcher,
for more information see: https://giatenica.com
"""

import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestTracingSetup:
    """Tests for tracing module configuration."""
    
    @pytest.mark.unit
    def test_service_name_constant(self):
        """Service name should be defined."""
        from src.tracing import SERVICE_NAME_VALUE
        
        assert SERVICE_NAME_VALUE == "gia-research-agents"
    
    @pytest.mark.unit
    def test_otlp_endpoint_default(self):
        """OTLP endpoint should have sensible default."""
        from src.tracing import OTLP_ENDPOINT
        
        assert "localhost" in OTLP_ENDPOINT or "4318" in OTLP_ENDPOINT
    
    @pytest.mark.unit
    @patch('src.tracing.TracerProvider')
    @patch('src.tracing.OTLPSpanExporter')
    @patch('src.tracing.BatchSpanProcessor')
    @patch('src.tracing.trace')
    @patch('src.tracing.HTTPXClientInstrumentor')
    def test_setup_tracing_creates_provider(
        self, mock_httpx, mock_trace, mock_processor, mock_exporter, mock_provider
    ):
        """setup_tracing should create and configure TracerProvider."""
        from src.tracing import setup_tracing
        
        mock_trace.get_tracer.return_value = MagicMock()
        
        tracer = setup_tracing()
        
        mock_provider.assert_called_once()
        mock_trace.set_tracer_provider.assert_called_once()
    
    @pytest.mark.unit
    @patch('src.tracing.trace')
    def test_get_tracer_returns_tracer(self, mock_trace):
        """get_tracer should return a tracer instance."""
        from src.tracing import get_tracer
        
        mock_tracer = MagicMock()
        mock_trace.get_tracer.return_value = mock_tracer
        
        tracer = get_tracer("test-component")
        
        mock_trace.get_tracer.assert_called_with("test-component")
        assert tracer == mock_tracer
    
    @pytest.mark.unit
    @patch('src.tracing.trace')
    def test_get_tracer_with_default_name(self, mock_trace):
        """get_tracer should use service name as default."""
        from src.tracing import get_tracer, SERVICE_NAME_VALUE
        
        mock_trace.get_tracer.return_value = MagicMock()
        
        get_tracer()
        
        mock_trace.get_tracer.assert_called_with(SERVICE_NAME_VALUE)


class TestTracingIntegration:
    """Integration tests for tracing with workflow."""
    
    @pytest.mark.unit
    def test_workflow_imports_tracing(self):
        """Workflow module should import tracing functions."""
        from src.agents.workflow import init_tracing, get_tracer
        
        assert callable(init_tracing)
        assert callable(get_tracer)


class TestTracingAttributeHelpers:
    """Tests for safe span attribute setters."""

    @pytest.mark.unit
    def test_safe_set_span_attributes_noop_span(self):
        from src.tracing import safe_set_span_attributes

        class NoopSpan:
            pass

        safe_set_span_attributes(NoopSpan(), {"k": "v"})

    @pytest.mark.unit
    def test_safe_set_current_span_attributes_does_not_raise(self):
        from src.tracing import safe_set_current_span_attributes

        safe_set_current_span_attributes({"k": "v", "n": 1, "flag": True})

    @pytest.mark.unit
    def test_safe_set_span_attributes_truncates_long_strings(self):
        from src.tracing import safe_set_span_attributes

        span = MagicMock()
        long_value = "x" * 5000
        safe_set_span_attributes(span, {"long": long_value})

        span.set_attribute.assert_called()
        args, _kwargs = span.set_attribute.call_args
        assert args[0] == "long"
        assert isinstance(args[1], str)
        assert len(args[1]) == 2048

    @pytest.mark.unit
    def test_safe_set_span_attributes_handles_non_serializable(self):
        from src.tracing import safe_set_span_attributes

        span = MagicMock()
        safe_set_span_attributes(
            span,
            {
                "path": Path("foo"),
                "obj": object(),
                "nested": {"x": object()},
                "seq": ["a" * 5000, object()],
                123: "ignored",
            },
        )

        # Best-effort: should not crash; may skip some keys.
        assert span.set_attribute.call_count >= 1
