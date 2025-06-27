---
title: "Analytics and Event Tracking"
description: "Analytics and Event Tracking"
---

# Analytics and Event Tracking

Rizk SDK provides basic analytics and event tracking capabilities through its built-in analytics framework. This gives you essential insights into your LLM application's behavior, usage patterns, and guardrails decisions.

## Overview

Rizk's analytics capabilities enable you to:

- **Track Events**: Monitor workflow executions, guardrails decisions, and errors
- **Collect Usage Data**: Framework detection, adapter usage, and operation counts
- **Monitor Guardrails**: Policy decisions and violation tracking
- **Custom Events**: Track business-specific events and metrics

## Quick Start

### Basic Analytics Setup

Enable analytics collection with minimal configuration:

```python
from rizk.sdk import Rizk

# Initialize with analytics enabled
rizk = Rizk.init(
    app_name="MyLLMApp",
    api_key="your-rizk-api-key",
    
    # Basic telemetry (uses Traceloop)
    telemetry_enabled=True,
    
    # Enable SDK analytics
    enabled=True
)

# Your decorated functions automatically generate analytics events
@workflow(name="customer_support", organization_id="acme", project_id="helpdesk")
def handle_support_request(request: dict) -> dict:
    # Automatically tracked:
    # - Workflow execution events
    # - Success/failure status
    # - Basic timing information
    
    return process_support_request(request)

# Use the function - analytics are collected automatically
result = handle_support_request({"query": "How do I reset my password?"})
```

## Analytics Events

### Automatic Event Collection

Rizk automatically collects the following event types:

```python
# SDK Lifecycle Events
- "sdk.initialized"
- "sdk.shutdown"

# Decorator Events  
- "workflow.started"
- "workflow.completed"
- "workflow.failed"
- "task.started"
- "task.completed"
- "task.failed"
- "agent.started"
- "agent.completed" 
- "agent.failed"
- "tool.called"
- "tool.completed"
- "tool.failed"

# Framework Events
- "framework.detected"
- "adapter.loaded"
- "adapter.failed"

# Guardrails Events
- "guardrails.input.checked"
- "guardrails.output.checked"
- "guardrails.blocked"
- "guardrails.allowed"
- "policy.matched"
- "policy.violated"

# Performance Events
- "performance.metric"
- "latency.measured"

# Error Events
- "error.occurred"
- "exception.raised"
```

### Event Data Structure

Each analytics event contains:

```python
{
    "event_id": "uuid-string",
    "event_type": "workflow.completed",
    "timestamp": "2024-01-15T10:30:45.123Z",
    "data": {
        "workflow_name": "customer_support",
        "duration_ms": 1250,
        "success": true
    },
    "context": {
        "organization_id": "acme",
        "project_id": "helpdesk",
        "conversation_id": "conv_123",
        "user_id": "user_456"
    },
    "sdk_version": "1.0.0",
    "framework": "langchain"
}
```

## Custom Analytics

### Adding Custom Events

Track custom business events:

```python
from rizk.sdk.analytics import track_event, EventType

# Track custom events
def process_order(order_data: dict) -> dict:
    # Track order processing start
    track_event(
        EventType.CUSTOM,
        data={
            "order_id": order_data["id"],
            "order_value": order_data["total"],
            "customer_type": order_data["customer_type"]
        },
        organization_id="ecommerce",
        project_id="orders"
    )
    
    result = handle_order_processing(order_data)
    
    # Track completion
    track_event(
        "order.processed",
        data={
            "order_id": order_data["id"],
            "processing_time_ms": result["duration"],
            "success": result["success"]
        }
    )
    
    return result
```

### Setting Analytics Context

Set persistent context for all events:

```python
from rizk.sdk.analytics import set_analytics_context

# Set context that applies to all subsequent events
set_analytics_context(
    organization_id="my_org",
    project_id="my_project",
    user_id="user_123",
    session_id="session_456"
)

# All events will now include this context automatically
@workflow(name="data_processing")
def process_data(data: dict) -> dict:
    # Events automatically include the context set above
    return analyze_data(data)
```

### Custom Analytics Processors

Create custom processors to send analytics data to your systems:

```python
from rizk.sdk.analytics import get_global_collector, AnalyticsProcessor

class CustomAnalyticsProcessor(AnalyticsProcessor):
    @property
    def name(self) -> str:
        return "custom_processor"
    
    def process_event(self, event) -> None:
        # Send to your analytics system
        send_to_analytics_platform(event.to_dict())
    
    def process_metric(self, metric) -> None:
        # Send to your metrics system
        send_to_metrics_platform(metric.to_dict())
    
    def flush(self) -> None:
        # Flush any pending data
        pass
    
    def close(self) -> None:
        # Clean up resources
        pass

# Add your custom processor
collector = get_global_collector()
collector.add_processor(CustomAnalyticsProcessor())
```

## Performance Tracking

### Basic Performance Instrumentation

Rizk provides basic performance tracking through OpenTelemetry:

```python
from rizk.sdk.performance import timed_operation, performance_instrumented

# Manual timing for specific operations
def complex_business_logic(data: dict) -> dict:
    with timed_operation("business_logic", "custom"):
        # Your code here - timing is automatic
        result = process_data(data)
        return result

# Decorator-based instrumentation
@performance_instrumented("guardrails", "policy_check")
def check_policy(message: str) -> bool:
    # Performance metrics automatically collected
    return evaluate_policy(message)
```

### Available Performance Decorators

```python
from rizk.sdk.performance import (
    guardrails_instrumented,
    adapter_instrumented,
    llm_fallback_instrumented,
    framework_detection_instrumented
)

# Instrument guardrails operations
@guardrails_instrumented("fast_rules")
def evaluate_fast_rules(input_text: str) -> dict:
    return run_fast_rules(input_text)

# Instrument adapter operations
@adapter_instrumented("langchain", "agent_execution")
def run_langchain_agent(query: str) -> str:
    return agent.run(query)

# Instrument LLM fallback
@llm_fallback_instrumented("policy_evaluation")
def llm_policy_check(content: str) -> bool:
    return llm_evaluate(content)
```

## Viewing Analytics Data

### Development and Testing

During development, analytics events are logged to the console:

```python
# Enable verbose logging to see analytics events
rizk = Rizk.init(
    app_name="DevApp",
    verbose=True,  # Shows analytics events in console
    telemetry_enabled=True
)
```

### Production Data Collection

In production, analytics data is sent via:

1. **Traceloop SDK**: Telemetry events are sent through Traceloop's system
2. **OpenTelemetry**: Performance metrics are sent to your OTLP endpoint
3. **Custom Processors**: You can add custom analytics processors

## Configuration

### Environment Variables

Configure analytics through environment variables:

```bash
# Enable/disable telemetry
export RIZK_TELEMETRY=true

# OpenTelemetry configuration
export RIZK_TRACING_ENABLED=true
export RIZK_OPENTELEMETRY_ENDPOINT=https://your-otlp-endpoint.com

# Performance instrumentation
export RIZK_PERFORMANCE_ENABLED=true
```

### Programmatic Configuration

```python
from rizk.sdk import Rizk
from rizk.sdk.performance import set_instrumentation_enabled

# Initialize with analytics configuration
rizk = Rizk.init(
    app_name="MyApp",
    
    # Basic telemetry via Traceloop
    telemetry_enabled=True,
    
    # OpenTelemetry tracing
    enabled=True,
    opentelemetry_endpoint="https://your-collector.com"
)

# Configure performance instrumentation
set_instrumentation_enabled(True)
```

## Analytics Collector API

### Getting Analytics Stats

```python
from rizk.sdk.analytics import get_global_collector

collector = get_global_collector()

# Get collector statistics
stats = collector.get_stats()
print(f"Events collected: {stats['total_events']}")
print(f"Metrics collected: {stats['total_metrics']}")
print(f"Active processors: {stats['active_processors']}")

# Get recent events
recent_events = collector.get_recent_events(limit=10)
for event in recent_events:
    print(f"Event: {event.event_type} at {event.timestamp}")

# Get recent metrics
recent_metrics = collector.get_recent_metrics(limit=10)
for metric in recent_metrics:
    print(f"Metric: {metric.name} = {metric.value}")
```

### Filtering Events and Metrics

```python
# Add event filters
collector.add_event_filter(
    lambda event: event.event_type != "performance.metric"
)

# Add metric filters
collector.add_metric_filter(
    lambda metric: metric.value > 0  # Only positive values
)
```

## Best Practices

### Event Collection

1. **Use Standard Event Types**: Prefer built-in event types over custom ones
2. **Include Context**: Always set organization, project, and user context
3. **Avoid PII**: Don't include sensitive data in analytics events
4. **Batch Processing**: Use analytics processors for efficient data handling

### Performance Tracking

1. **Selective Instrumentation**: Only instrument critical operations
2. **Avoid Over-Instrumentation**: Too much instrumentation can impact performance
3. **Use Sampling**: In high-traffic scenarios, consider sampling traces

### Production Deployment

1. **Configure Endpoints**: Set up proper OTLP endpoints for production
2. **Monitor Collection**: Ensure analytics data is being collected properly
3. **Set Up Processors**: Use custom processors for your analytics platform
4. **Test Thoroughly**: Verify analytics collection in staging environments

## Limitations

The current analytics system has these limitations:

- **No Built-in Dashboards**: You need to use external tools for visualization
- **Basic Metrics Only**: Only timing and event counting, no complex aggregations
- **No Alerting**: No built-in alerting system
- **Limited Querying**: No built-in query interface for historical data

For advanced analytics, metrics, and monitoring, integrate with external platforms like:
- DataDog
- New Relic  
- Prometheus + Grafana
- Elastic Stack
- Custom analytics platforms

## Troubleshooting

### Analytics Not Collecting

```python
# Check if telemetry is enabled
import os
print(f"RIZK_TELEMETRY: {os.getenv('RIZK_TELEMETRY')}")

# Enable verbose logging
rizk = Rizk.init(app_name="Debug", verbose=True, telemetry_enabled=True)

# Check analytics collector status
from rizk.sdk.analytics import get_global_collector
collector = get_global_collector()
print(f"Analytics stats: {collector.get_stats()}")
```

### Performance Issues

```python
# Disable performance instrumentation if causing issues
from rizk.sdk.performance import set_instrumentation_enabled
set_instrumentation_enabled(False)

# Check OpenTelemetry configuration
print(f"OTLP Endpoint: {os.getenv('RIZK_OPENTELEMETRY_ENDPOINT')}")
```

### Custom Processor Issues

```python
# Test your custom processor
processor = CustomAnalyticsProcessor()
test_event = AnalyticsEvent(event_type=EventType.CUSTOM, data={"test": True})
processor.process_event(test_event)
```

---

**Note**: This analytics system provides basic event tracking and performance measurement. For comprehensive metrics, dashboards, and alerting, integrate with dedicated observability platforms. 

