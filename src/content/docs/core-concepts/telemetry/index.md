---
title: "Telemetry"
description: "Documentation for Telemetry"
---

The Rizk SDK provides comprehensive telemetry capabilities through OpenTelemetry integration. This guide explains how to use and configure telemetry in your applications.

## Overview

Telemetry in the Rizk SDK includes:

1. **Traces**: Detailed request/response flow tracking
2. **Metrics**: Performance and usage measurements
3. **Logs**: Application events and errors

## Basic Usage

### Enabling Telemetry

Telemetry is enabled by default when initializing the SDK:

```python
from rizk.sdk import Rizk

client = Rizk.init(
    app_name="my_app",
    api_key="your_api_key",
    telemetry_enabled=True  # Enabled by default
)
```

### Setting Resource Attributes

Add context to your telemetry data:

```python
client = Rizk.init(
    app_name="my_app",
    api_key="your_api_key",
    resource_attributes={
        "service.name": "my_service",
        "service.version": "1.0.0",
        "deployment.environment": "production"
    }
)
```

## Tracing

### Automatic Tracing

The SDK automatically traces:

- API calls
- Guardrails processing
- Policy evaluations
- LLM interactions

### Custom Tracing

Add custom traces to your code:

```python
from opentelemetry import trace

tracer = trace.get_tracer("my_app")

async def my_function():
    with tracer.start_as_current_span("my_operation") as span:
        span.set_attribute("custom.attribute", "value")
        # Your code here
```

## Metrics

### Built-in Metrics

The SDK collects metrics for:

- Request latency
- Policy evaluation times
- Guardrails processing duration
- Error rates

### Custom Metrics

Add custom metrics:

```python
from opentelemetry import metrics

meter = metrics.get_meter("my_app")
counter = meter.create_counter(
    "my_counter",
    description="Custom counter metric"
)

# Record metrics
counter.add(1, {"label": "value"})
```

## Logging

### Configuration

Configure logging levels and format:

```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Get logger for your application
logger = logging.getLogger("my_app")
```

### Logging Events

Log important events:

```python
logger.info("Application started")
logger.warning("Policy violation detected")
logger.error("Error processing request", exc_info=True)
```

## OpenTelemetry Integration

### Collector Configuration

Configure the OpenTelemetry collector endpoint:

```python
client = Rizk.init(
    app_name="my_app",
    api_key="your_api_key",
    opentelemetry_endpoint="https://api.rizk.tools"
)
```

### Custom Exporters

Use custom exporters for different backends:

```python
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

exporter = OTLPSpanExporter(endpoint="your-collector-endpoint")
processor = BatchSpanProcessor(exporter)

client = Rizk.init(
    app_name="my_app",
    api_key="your_api_key",
    processor=processor
)
```

## Best Practices

1. **Use Meaningful Context**:
   ```python
   Rizk.set_association_properties({
       "organization_id": "org123",
       "project_id": "project456",
       "agent_id": "agent789"
   })
   ```

2. **Handle Errors Properly**:
   ```python
   try:
       result = await process_request()
   except Exception as e:
       logger.error("Request failed", exc_info=True)
       span.record_exception(e)
       span.set_status(Status(StatusCode.ERROR))
   ```

3. **Monitor Performance**:
   ```python
   with tracer.start_as_current_span("operation") as span:
       start_time = time.time()
       result = await process_request()
       duration = time.time() - start_time
       span.set_attribute("duration", duration)
   ```

## Environment Variables

Configure telemetry through environment variables:

```env
RIZK_TELEMETRY_ENABLED=true
RIZK_OPENTELEMETRY_ENDPOINT=https://api.rizk.tools
RIZK_DISABLE_BATCH=false
```

## Next Steps

- [Using Telemetry Guide](../guides/using-telemetry.md)
- [Debugging Guide](../troubleshooting/debugging.md)
- [API Reference](../api/rizk.md)
- [Examples](../examples/basic-usage.md) 