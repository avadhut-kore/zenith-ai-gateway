"""Telemetry, OpenTelemetry tracing, and Prometheus metrics for Zenith AI Gateway."""

import logging
from typing import Optional
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.sdk.resources import Resource
from prometheus_client import Counter, Histogram, Gauge

logger = logging.getLogger(__name__)

# Prometheus Metrics
REQUEST_COUNT = Counter(
    "zenith_requests_total",
    "Total chat completion requests processed by Zenith Gateway",
    ["model", "cache_status", "status_code"],
)

CACHE_LATENCY = Histogram(
    "zenith_cache_latency_seconds",
    "Latency of semantic vector cache lookup in seconds",
    buckets=[0.002, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
)

TTFT_HISTOGRAM = Histogram(
    "zenith_time_to_first_token_seconds",
    "Time to first token (TTFT) in seconds for streaming responses",
    buckets=[0.05, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0],
)

TOKEN_COUNT = Counter(
    "zenith_tokens_total",
    "Total tokens consumed and generated",
    ["model", "token_type"],  # token_type: prompt, completion
)

ACTIVE_REQUESTS = Gauge(
    "zenith_active_requests",
    "Number of active in-flight requests currently being served",
)


def setup_telemetry(service_name: str = "zenith-ai-gateway", otlp_endpoint: Optional[str] = None) -> None:
    """Initialize OpenTelemetry tracer provider with resources and processors."""
    resource = Resource.create({"service.name": service_name})
    provider = TracerProvider(resource=resource)

    if otlp_endpoint:
        try:
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
            otlp_exporter = OTLPSpanExporter(endpoint=otlp_endpoint, insecure=True)
            provider.add_span_processor(BatchSpanProcessor(otlp_exporter))
            logger.info(f"OpenTelemetry OTLP exporter configured for endpoint: {otlp_endpoint}")
        except Exception as e:
            logger.warning(f"Could not initialize OTLP exporter: {e}. Falling back to default.")
    else:
        # In non-configured environments, use lightweight console or no-op processor in debug
        logger.debug("OpenTelemetry initialized with default tracer provider.")

    trace.set_tracer_provider(provider)


def get_tracer(module_name: str = "zenith-ai-gateway") -> trace.Tracer:
    """Obtain an OpenTelemetry tracer for custom instrumented spans."""
    return trace.get_tracer(module_name)
