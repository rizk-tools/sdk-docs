---
title: "Tracing"
description: "Documentation for Tracing"
---

Tracing in the Rizk SDK provides detailed visibility into your application's execution flow, helping you understand and debug your AI applications. This guide explains how to use and configure tracing.

## Overview

Tracing in the Rizk SDK is built on OpenTelemetry and provides:

1. **Distributed Tracing**: Track requests across service boundaries
2. **Span Attributes**: Add context to your traces
3. **Error Tracking**: Capture and analyze errors
4. **Performance Monitoring**: Measure execution times

## Basic Usage

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

## Span Management

### Creating Spans

Create spans for custom operations:

```python
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

tracer = trace.get_tracer("my_app")

async def process_request():
    with tracer.start_as_current_span("process_request") as span:
        try:
            # Your code here
            span.set_status(Status(StatusCode.OK))
        except Exception as e:
            span.set_status(Status(StatusCode.ERROR))
            span.record_exception(e)
            raise
```

### Span Attributes

Add context to your spans:

```python
with tracer.start_as_current_span("operation") as span:
    span.set_attribute("user.id", "user123")
    span.set_attribute("request.type", "query")
    span.set_attribute("timestamp", datetime.now().isoformat())
```

## Context Management

### Setting Context

Set context for the current trace:

```python
from rizk.sdk import Rizk

# Set context for the current conversation
Rizk.set_association_properties({
    "organization_id": "org123",
    "project_id": "project456",
    "agent_id": "agent789"
})
```

### Context Propagation

Context is automatically propagated across service boundaries:

```python
# Context is propagated in headers
headers = {
    "traceparent": span.get_span_context().trace_id,
    "tracestate": span.get_span_context().trace_state
}
```

## Error Tracking

### Recording Errors

Capture errors in your traces:

```python
try:
    result = await process_request()
except Exception as e:
    span.record_exception(e)
    span.set_status(Status(StatusCode.ERROR))
    span.set_attribute("error.message", str(e))
    raise
```

### Error Attributes

Add error context:

```python
span.set_attribute("error.type", type(e).__name__)
span.set_attribute("error.stack_trace", traceback.format_exc())
```

## Performance Monitoring

### Measuring Duration

Track operation duration:

```python
with tracer.start_as_current_span("operation") as span:
    start_time = time.time()
    result = await process_request()
    duration = time.time() - start_time
    span.set_attribute("duration", duration)
```

### Custom Metrics

Add custom performance metrics:

```python
from opentelemetry import metrics

meter = metrics.get_meter("my_app")
duration_histogram = meter.create_histogram(
    "operation_duration",
    description="Duration of operations"
)

duration_histogram.record(duration, {"operation": "process_request"})
```

## OpenTelemetry Integration

### Collector Configuration

Configure the OpenTelemetry collector:

```python
from rizk.sdk import Rizk

client = Rizk.init(
    app_name="my_app",
    api_key="your_api_key",
    opentelemetry_endpoint="https://api.rizk.tools"
)
```

### Custom Exporters

Use custom exporters:

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

1. **Use Meaningful Span Names**:
   ```python
   with tracer.start_as_current_span("process_user_query") as span:
       # Use descriptive names that indicate the operation
   ```

2. **Add Relevant Attributes**:
   ```python
   span.set_attribute("user.id", user_id)
   span.set_attribute("request.type", request_type)
   span.set_attribute("timestamp", datetime.now().isoformat())
   ```

3. **Handle Errors Properly**:
   ```python
   try:
       result = await process_request()
   except Exception as e:
       span.record_exception(e)
       span.set_status(Status(StatusCode.ERROR))
       span.set_attribute("error.message", str(e))
       raise
   ```

## Environment Variables

Configure tracing through environment variables:

```env
RIZK_OPENTELEMETRY_ENDPOINT=https://api.rizk.tools
RIZK_ENABLED=true
RIZK_DISABLE_BATCH=false
```

## Next Steps

- [Using Telemetry Guide](../guides/using-telemetry.md)
- [Debugging Guide](../troubleshooting/debugging.md)
- [API Reference](../api/rizk.md)
- [Examples](../examples/basic-usage.md) 