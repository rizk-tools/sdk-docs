---
title: "Observability"
description: "Observability"
---

# Observability

Rizk SDK provides comprehensive observability for LLM applications through OpenTelemetry integration, distributed tracing, and hierarchical context management. This document explains how the observability system works and how to leverage it effectively.

## Overview

The observability system automatically instruments your LLM applications with:

- **Distributed Tracing**: Track requests across components and services
- **Hierarchical Context**: Organize traces by organization, project, and agent
- **Performance Metrics**: Monitor latency, throughput, and error rates
- **Custom Attributes**: Add business-specific metadata to traces
- **Framework Integration**: Native support for all major LLM frameworks

```python
from rizk.sdk import Rizk
from rizk.sdk.decorators import workflow

# Initialize observability
rizk = Rizk.init(
    app_name="MyLLMApp",
    api_key="your-api-key",
    enabled=True
)

@workflow(name="customer_support", organization_id="acme", project_id="support")
def handle_customer_query(query: str) -> str:
    # Automatically traced with hierarchical context
    return process_query(query)
```

## OpenTelemetry Integration

### Automatic Instrumentation

Rizk SDK integrates with OpenTelemetry through the Traceloop SDK, providing automatic instrumentation:

```python
# Automatic instrumentation includes:
# - HTTP requests and responses
# - Database queries  
# - LLM API calls (OpenAI, Anthropic, etc.)
# - Framework-specific operations
# - Custom application logic
```

### Trace Structure

Each trace follows a hierarchical structure:

```
Trace: customer_support_workflow
â”œâ”€â”€ Span: input_validation
â”œâ”€â”€ Span: llm_processing
â”‚   â”œâ”€â”€ Span: openai_chat_completion
â”‚   â””â”€â”€ Span: response_formatting
â”œâ”€â”€ Span: output_guardrails
â””â”€â”€ Span: result_logging
```

### Span Attributes

Spans are enriched with contextual attributes:

```python
# Automatic attributes added to spans:
span_attributes = {
    # Hierarchical context
    "organization.id": "acme_corp",
    "project.id": "customer_support", 
    "agent.id": "support_assistant",
    "conversation.id": "conv_12345",
    "user.id": "user_6789",
    
    # Framework information
    "framework.name": "langchain",
    "framework.version": "0.1.0",
    
    # Function metadata
    "function.name": "handle_customer_query",
    "function.version": 1,
    
    # Performance metrics
    "duration.ms": 1250,
    "tokens.input": 150,
    "tokens.output": 75,
    "cost.usd": 0.0023
}
```

## Hierarchical Context Management

### Context Levels

Rizk SDK supports multiple levels of hierarchical context:

```python
# Level 1: Organization
organization_id = "acme_corp"

# Level 2: Project  
project_id = "customer_support"

# Level 3: Agent/Service
agent_id = "support_assistant"

# Level 4: Conversation
conversation_id = "conv_12345"

# Level 5: User
user_id = "user_6789"
```

### Context Propagation

Context is automatically propagated through your application:

```python
from rizk.sdk.decorators import workflow, task, agent

@workflow(
    name="support_workflow",
    organization_id="acme_corp",
    project_id="customer_support"
)
def handle_support_request(request: dict) -> dict:
    """Top-level workflow with context."""
    
    @agent(
        name="support_agent",
        organization_id="acme_corp",      # Inherited
        project_id="customer_support",   # Inherited  
        agent_id="support_assistant"
    )
    def create_support_agent():
        """Agent inherits workflow context."""
        return {"agent": "support_assistant"}
    
    @task(
        name="process_request",
        organization_id="acme_corp",      # Inherited
        project_id="customer_support",   # Inherited
        task_id="request_processing"
    )
    def process_request(data: dict):
        """Task inherits workflow context."""
        return {"processed": True}
    
    agent = create_support_agent()
    result = process_request(request)
    return result
```

### Dynamic Context

Context can be set dynamically during execution:

```python
from rizk.sdk import Rizk

def handle_user_request(user_id: str, request: str):
    """Handle request with dynamic user context."""
    
    # Set dynamic context
    Rizk.set_association_properties({
        "user.id": user_id,
        "conversation.id": f"conv_{user_id}_{int(time.time())}",
        "request.type": "support_inquiry"
    })
    
    @workflow(name="user_request_workflow")
    def process_user_request():
        # Context automatically included in traces
        return f"Processed request for user {user_id}"
    
    return process_user_request()
```

## Tracing Configuration

### Basic Configuration

```python
from rizk.sdk import Rizk

# Basic tracing setup
rizk = Rizk.init(
    app_name="MyApp",
    api_key="your-api-key",
    enabled=True,                    # Enable tracing
    telemetry_enabled=False,         # Disable anonymous telemetry
    trace_content=True               # Include content in traces
)
```

### Custom OTLP Endpoints

```python
# Send traces to custom endpoint
rizk = Rizk.init(
    app_name="MyApp",
    api_key="your-api-key",
    opentelemetry_endpoint="https://your-otlp-endpoint.com",
    headers={
        "Authorization": "Bearer your-token",
        "X-Custom-Header": "custom-value"
    }
)
```

### Advanced Configuration

```python
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

# Custom exporter and processor
custom_exporter = OTLPSpanExporter(
    endpoint="https://your-endpoint.com",
    headers={"authorization": "Bearer token"}
)

custom_processor = BatchSpanProcessor(
    span_exporter=custom_exporter,
    max_queue_size=2048,
    max_export_batch_size=512,
    export_timeout_millis=30000
)

rizk = Rizk.init(
    app_name="MyApp",
    api_key="your-api-key",
    processor=custom_processor,
    exporter=custom_exporter
)
```

## Framework-Specific Observability

### OpenAI Agents

```python
from rizk.sdk.decorators import workflow, agent, tool

@tool(name="search_tool", organization_id="demo", project_id="agents")
def search_web(query: str) -> str:
    """Search tool with automatic tracing."""
    # Tool usage automatically traced
    return f"Search results for: {query}"

@agent(name="research_agent", organization_id="demo", project_id="agents")
def create_research_agent():
    """Agent creation with tracing."""
    # Agent lifecycle automatically traced
    return {"agent": "researcher", "tools": [search_web]}

@workflow(name="research_workflow", organization_id="demo", project_id="agents")
def run_research(topic: str):
    """Complete workflow tracing."""
    # Full workflow execution traced
    agent = create_research_agent()
    return f"Research completed on: {topic}"
```

### LangChain Integration

```python
from rizk.sdk.decorators import workflow, agent
from langchain.callbacks import get_openai_callback

@workflow(name="langchain_workflow", organization_id="demo", project_id="langchain")
def run_langchain_process(query: str):
    """LangChain process with enhanced tracing."""
    
    # Automatic callback integration
    with get_openai_callback() as cb:
        # LangChain operations automatically traced
        # Token usage and costs captured
        result = process_with_langchain(query)
        
        # Cost information added to trace
        Rizk.set_association_properties({
            "tokens.total": cb.total_tokens,
            "tokens.prompt": cb.prompt_tokens,
            "tokens.completion": cb.completion_tokens,
            "cost.total_usd": cb.total_cost
        })
        
        return result

def process_with_langchain(query: str):
    """Simulate LangChain processing."""
    return f"LangChain processed: {query}"
```

### CrewAI Integration

```python
from rizk.sdk.decorators import crew, agent, task

@agent(name="writer", organization_id="demo", project_id="crewai")
def create_writer():
    """Writer agent with tracing."""
    return {"role": "writer", "goal": "create content"}

@task(name="writing_task", organization_id="demo", project_id="crewai")
def create_writing_task():
    """Writing task with tracing."""
    return {"task": "write article", "agent": "writer"}

@crew(name="content_crew", organization_id="demo", project_id="crewai")
def run_content_crew(topic: str):
    """Crew execution with comprehensive tracing."""
    writer = create_writer()
    task = create_writing_task()
    
    # Crew execution automatically traced
    # Individual agent and task performance captured
    return f"Content crew completed work on: {topic}"
```

## Custom Metrics and Attributes

### Adding Custom Attributes

```python
from rizk.sdk import Rizk
from opentelemetry import trace

@workflow(name="custom_attributes_workflow")
def process_with_custom_attributes(user_data: dict):
    """Add custom attributes to traces."""
    
    # Get current span
    current_span = trace.get_current_span()
    
    # Add custom attributes
    current_span.set_attribute("user.tier", user_data.get("tier", "standard"))
    current_span.set_attribute("request.priority", "high")
    current_span.set_attribute("feature.flags", "new_ui,beta_features")
    
    # Or use Rizk's association properties
    Rizk.set_association_properties({
        "business.unit": "customer_success",
        "geo.region": "us-west-2",
        "experiment.variant": "control"
    })
    
    return {"processed": True}
```

### Performance Metrics

```python
import time
from rizk.sdk.decorators import workflow

@workflow(name="performance_monitoring")
def monitor_performance():
    """Monitor custom performance metrics."""
    start_time = time.time()
    
    # Simulate processing
    time.sleep(0.1)
    
    processing_time = time.time() - start_time
    
    # Add performance metrics
    Rizk.set_association_properties({
        "performance.processing_time_ms": processing_time * 1000,
        "performance.memory_usage_mb": 45.2,
        "performance.cpu_usage_percent": 23.5
    })
    
    return {"status": "completed"}
```

## Error Tracking and Debugging

### Automatic Error Capture

```python
from rizk.sdk.decorators import workflow

@workflow(name="error_handling_workflow")
def process_with_error_handling(data: dict):
    """Automatic error capture and tracing."""
    try:
        # Simulate processing that might fail
        if data.get("invalid"):
            raise ValueError("Invalid data provided")
        
        return {"processed": True}
        
    except Exception as e:
        # Errors automatically captured in traces
        # Stack traces and error details included
        current_span = trace.get_current_span()
        current_span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
        current_span.set_attribute("error.type", type(e).__name__)
        current_span.set_attribute("error.message", str(e))
        
        # Re-raise or handle as needed
        raise
```

### Debug Tracing

```python
import logging
from rizk.sdk.decorators import workflow

# Enable debug logging
logging.getLogger("rizk").setLevel(logging.DEBUG)
logging.getLogger("opentelemetry").setLevel(logging.DEBUG)

@workflow(name="debug_workflow")
def debug_process():
    """Process with debug-level tracing."""
    # Detailed trace information logged
    # Span creation and completion logged
    # Attribute setting logged
    return {"debug": "enabled"}
```

## Observability Best Practices

### 1. Meaningful Span Names

```python
# âœ… Good - Descriptive, consistent naming
@workflow(name="customer_onboarding_v2")
@task(name="validate_customer_email")
@agent(name="onboarding_assistant")

# âŒ Avoid - Generic or unclear names
@workflow(name="process")
@task(name="step1") 
@agent(name="bot")
```

### 2. Appropriate Context Granularity

```python
# âœ… Good - Balanced context depth
@workflow(
    name="order_processing",
    organization_id="ecommerce",
    project_id="orders"
)
def process_order():
    
    @task(
        name="payment_validation",
        task_id="payment_check"
    )
    def validate_payment():
        pass

# âŒ Avoid - Too much or too little context
@workflow(name="order_processing")  # Missing context
def process_order():
    
    @task(
        name="payment_validation",
        organization_id="ecommerce",
        project_id="orders", 
        service_id="payments",
        region_id="us-west",
        datacenter_id="dc1"  # Too granular
    )
    def validate_payment():
        pass
```

### 3. Sensitive Data Handling

```python
# âœ… Good - Exclude sensitive data
@workflow(name="user_authentication")
def authenticate_user(username: str, password: str):
    """Authenticate user without logging sensitive data."""
    
    # Don't include password in traces
    Rizk.set_association_properties({
        "user.username": username,
        "auth.method": "password",
        # "user.password": password  # âŒ Never do this
    })
    
    return {"authenticated": True}

# âœ… Good - Hash or mask sensitive data
def process_payment(card_number: str):
    """Process payment with masked card data."""
    
    masked_card = f"****-****-****-{card_number[-4:]}"
    
    Rizk.set_association_properties({
        "payment.card_last_four": card_number[-4:],
        "payment.card_masked": masked_card,
        # "payment.card_number": card_number  # âŒ Never do this
    })
```

### 4. Performance Optimization

```python
# âœ… Good - Conditional detailed tracing
import os

@workflow(name="performance_optimized")
def optimized_process():
    """Optimize tracing for performance."""
    
    # Detailed tracing only in development
    if os.getenv("ENVIRONMENT") == "development":
        Rizk.set_association_properties({
            "debug.detailed_timing": True,
            "debug.memory_tracking": True
        })
    
    # Always include essential metrics
    Rizk.set_association_properties({
        "request.id": "req_12345",
        "user.tier": "premium"
    })
```

## Monitoring and Alerting

### Key Metrics to Monitor

```python
# Essential observability metrics:
metrics_to_monitor = {
    "request_latency": "95th percentile response time",
    "error_rate": "Percentage of failed requests",
    "throughput": "Requests per second",
    "token_usage": "LLM token consumption",
    "cost_per_request": "Average cost per request",
    "guardrails_violations": "Policy violation rate",
    "framework_errors": "Framework-specific errors"
}
```

### Custom Dashboards

```python
# Dashboard queries for common metrics:
dashboard_queries = {
    "latency_p95": "histogram_quantile(0.95, rate(request_duration_seconds_bucket[5m]))",
    "error_rate": "rate(request_errors_total[5m]) / rate(requests_total[5m])",
    "cost_trend": "rate(llm_cost_usd_total[1h])",
    "top_users": "topk(10, sum by (user_id) (rate(requests_total[24h])))",
    "framework_distribution": "sum by (framework_name) (rate(requests_total[24h]))"
}
```

## Integration with Observability Platforms

### Popular Platforms

```python
# Jaeger
rizk = Rizk.init(
    app_name="MyApp",
    opentelemetry_endpoint="http://jaeger:14268/api/traces"
)

# Datadog
rizk = Rizk.init(
    app_name="MyApp", 
    opentelemetry_endpoint="https://trace.agent.datadoghq.com",
    headers={"DD-API-KEY": "your-dd-api-key"}
)

# New Relic
rizk = Rizk.init(
    app_name="MyApp",
    opentelemetry_endpoint="https://otlp.nr-data.net:4317",
    headers={"api-key": "your-nr-license-key"}
)

# Honeycomb
rizk = Rizk.init(
    app_name="MyApp",
    opentelemetry_endpoint="https://api.honeycomb.io",
    headers={"x-honeycomb-team": "your-honeycomb-key"}
)
```

## Summary

Rizk SDK's observability system provides:

âœ… **Comprehensive Tracing** - Full request lifecycle visibility  
âœ… **Hierarchical Context** - Enterprise-grade organization  
âœ… **Framework Integration** - Native support for all LLM frameworks  
âœ… **Performance Monitoring** - Latency, cost, and usage metrics  
âœ… **Error Tracking** - Automatic error capture and debugging  
âœ… **Custom Attributes** - Business-specific metadata support  
âœ… **Platform Integration** - Works with all major observability platforms  

The observability system gives you complete visibility into your LLM applications, enabling proactive monitoring, debugging, and optimization at enterprise scale. 

