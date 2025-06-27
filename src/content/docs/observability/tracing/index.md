---
title: "Distributed Tracing"
description: "Distributed Tracing"
---

# Distributed Tracing

Rizk SDK provides comprehensive distributed tracing capabilities through OpenTelemetry integration, giving you complete visibility into your LLM application's behavior, performance, and decision-making processes.

## Overview

Distributed tracing in Rizk enables you to:

- **Track Request Flows**: Follow requests through complex multi-agent workflows
- **Monitor Performance**: Identify bottlenecks and optimization opportunities
- **Debug Issues**: Trace errors and unexpected behavior across systems
- **Analyze Patterns**: Understand usage patterns and system behavior
- **Ensure Compliance**: Maintain audit trails for regulatory requirements

## Quick Start

### Basic Tracing Setup

Enable tracing with minimal configuration:

```python
from rizk.sdk import Rizk

# Initialize with tracing enabled
rizk = Rizk.init(
    app_name="MyLLMApp",
    api_key="your-rizk-api-key",
    
    # Enable tracing
    tracing_enabled=True,
    
    # Optional: Custom OTLP endpoint
    opentelemetry_endpoint="https://your-otlp-collector.com",
    
    # Trace configuration
    trace_content=True,  # Include content in traces
    trace_sampling_rate=1.0  # 100% sampling
)

# Your decorated functions automatically create spans
@workflow(name="customer_chat", organization_id="acme", project_id="support")
def handle_customer_query(query: str) -> str:
    # This creates a span automatically
    return process_query(query)

# Use the function - traces are created automatically
result = handle_customer_query("How can I reset my password?")
```

### Viewing Traces

Traces are automatically sent to:
1. **Rizk Dashboard**: View at [dashboard.rizk.tools](https://dashboard.rizk.tools)
2. **Custom OTLP Endpoint**: Your configured observability platform
3. **Local Development**: Console output when in debug mode

## Trace Hierarchy

Rizk creates a hierarchical trace structure that mirrors your application's logical flow:

```
Organization: acme
â”œâ”€â”€ Project: customer_support
    â”œâ”€â”€ Workflow: customer_chat
        â”œâ”€â”€ Task: query_analysis
        â”œâ”€â”€ Agent: support_assistant
            â”œâ”€â”€ Tool: knowledge_search
            â”œâ”€â”€ LLM Call: openai.chat.completions
        â””â”€â”€ Task: response_generation
```

### Automatic Span Creation

Rizk automatically creates spans for:

```python
# Workflow-level spans
@workflow(name="order_processing", organization_id="ecommerce", project_id="backend")
def process_order(order_data: dict) -> dict:
    # Span: workflow.order_processing
    return handle_order(order_data)

# Task-level spans  
@task(name="inventory_check", organization_id="ecommerce", project_id="backend")
def check_inventory(product_id: str) -> bool:
    # Span: task.inventory_check
    return verify_stock(product_id)

# Agent-level spans
@agent(name="sales_agent", organization_id="ecommerce", project_id="sales")
def sales_assistant(customer_query: str) -> str:
    # Span: agent.sales_agent
    return generate_sales_response(customer_query)

# Tool-level spans
@tool(name="price_calculator", organization_id="ecommerce", project_id="sales")
def calculate_price(product_id: str, quantity: int) -> float:
    # Span: tool.price_calculator
    return compute_price(product_id, quantity)
```

### Framework-Specific Spans

Rizk creates framework-specific spans automatically:

```python
# LangChain - automatic span creation
from langchain.agents import AgentExecutor
from rizk.sdk import Rizk

rizk = Rizk.init(app_name="LangChainApp", tracing_enabled=True)

# Spans created automatically for:
# - Agent execution
# - Tool calls
# - LLM interactions
# - Chain operations
agent_executor = AgentExecutor(agent=agent, tools=tools)
result = agent_executor.invoke({"input": user_query})

# CrewAI - automatic span creation
from crewai import Crew
crew = Crew(agents=[agent1, agent2], tasks=[task1, task2])
# Spans created for:
# - Crew execution
# - Agent interactions
# - Task completion
# - Inter-agent communication
result = crew.kickoff()
```

## Custom Span Creation

### Manual Span Creation

Create custom spans for specific operations:

```python
from rizk.sdk.tracing import get_tracer

tracer = get_tracer(__name__)

def complex_business_logic(data: dict) -> dict:
    # Create a custom span
    with tracer.start_as_current_span("business_logic.complex_operation") as span:
        # Add attributes to the span
        span.set_attribute("operation.type", "data_processing")
        span.set_attribute("data.size", len(data))
        span.set_attribute("user.id", data.get("user_id"))
        
        try:
            # Your business logic
            result = process_complex_data(data)
            
            # Add result attributes
            span.set_attribute("operation.success", True)
            span.set_attribute("result.count", len(result))
            
            return result
            
        except Exception as e:
            # Record the error
            span.record_exception(e)
            span.set_status(Status(StatusCode.ERROR, str(e)))
            raise
```

### Async Span Creation

Handle async operations with proper span context:

```python
import asyncio
from rizk.sdk.tracing import get_tracer

tracer = get_tracer(__name__)

async def async_llm_operation(prompt: str) -> str:
    with tracer.start_as_current_span("llm.async_call") as span:
        span.set_attribute("llm.provider", "openai")
        span.set_attribute("llm.model", "gpt-4")
        span.set_attribute("prompt.length", len(prompt))
        
        # Async LLM call
        response = await async_llm_client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}]
        )
        
        span.set_attribute("response.length", len(response.choices[0].message.content))
        span.set_attribute("llm.tokens_used", response.usage.total_tokens)
        
        return response.choices[0].message.content
```

## Span Attributes and Context

### Standard Attributes

Rizk automatically adds standard attributes to spans:

```python
# Organizational context
span.set_attribute("organization.id", "acme_corp")
span.set_attribute("project.id", "customer_support")
span.set_attribute("agent.id", "support_assistant")

# Request context
span.set_attribute("request.id", "req_12345")
span.set_attribute("conversation.id", "conv_67890")
span.set_attribute("user.id", "user_abcdef")

# Framework context
span.set_attribute("framework.name", "langchain")
span.set_attribute("framework.version", "0.1.0")

# LLM context
span.set_attribute("llm.provider", "openai")
span.set_attribute("llm.model", "gpt-4")
span.set_attribute("llm.temperature", 0.7)
```

### Custom Attributes

Add domain-specific attributes:

```python
@workflow(name="financial_analysis", organization_id="fintech", project_id="analytics")
def analyze_portfolio(portfolio_data: dict) -> dict:
    # Get current span
    span = trace.get_current_span()
    
    # Add custom attributes
    span.set_attribute("portfolio.value", portfolio_data["total_value"])
    span.set_attribute("portfolio.assets_count", len(portfolio_data["assets"]))
    span.set_attribute("analysis.type", "risk_assessment")
    span.set_attribute("compliance.required", True)
    
    # Your analysis logic
    analysis = perform_analysis(portfolio_data)
    
    # Add result attributes
    span.set_attribute("analysis.risk_score", analysis["risk_score"])
    span.set_attribute("analysis.recommendations_count", len(analysis["recommendations"]))
    
    return analysis
```

### Hierarchical Context

Maintain context across nested operations:

```python
from rizk.sdk.tracing import set_span_context

@workflow(name="document_processing", organization_id="legal", project_id="contracts")
def process_legal_document(document: dict) -> dict:
    # Set workflow context
    set_span_context({
        "document.type": document["type"],
        "document.id": document["id"],
        "document.classification": document["classification"]
    })
    
    # Extract text - inherits context
    text = extract_text(document)
    
    # Analyze content - inherits context
    analysis = analyze_content(text)
    
    # Generate summary - inherits context
    summary = generate_summary(analysis)
    
    return {
        "text": text,
        "analysis": analysis,
        "summary": summary
    }

@task(name="text_extraction", organization_id="legal", project_id="contracts")
def extract_text(document: dict) -> str:
    # Automatically inherits document context from parent span
    span = trace.get_current_span()
    
    # Add task-specific attributes
    span.set_attribute("extraction.method", "ocr")
    span.set_attribute("document.pages", document.get("page_count", 0))
    
    return perform_ocr(document)
```

## Content Tracing

### Enabling Content Tracing

Control what content is included in traces:

```python
# Full content tracing (development)
rizk = Rizk.init(
    app_name="DevApp",
    trace_content=True,
    trace_llm_inputs=True,
    trace_llm_outputs=True,
    trace_user_inputs=True
)

# Limited content tracing (production)
rizk = Rizk.init(
    app_name="ProdApp",
    trace_content=False,  # Disable content tracing
    trace_metadata_only=True,  # Only metadata
    trace_content_hashes=True  # Content hashes for debugging
)

# Selective content tracing
rizk = Rizk.init(
    app_name="SelectiveApp",
    trace_user_inputs=False,  # Don't trace user inputs (privacy)
    trace_llm_outputs=True,   # Trace LLM outputs
    trace_system_prompts=True,  # Trace system prompts
    content_redaction_enabled=True  # Redact sensitive content
)
```

### Content Redaction

Automatically redact sensitive information:

```python
from rizk.sdk.tracing import configure_content_redaction

# Configure content redaction
configure_content_redaction({
    "pii_redaction": True,
    "financial_data_redaction": True,
    "custom_patterns": [
        r"\b\d{4}-\d{4}-\d{4}-\d{4}\b",  # Credit card numbers
        r"\b\d{3}-\d{2}-\d{4}\b",        # SSN
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"  # Email
    ],
    "replacement_text": "[REDACTED]"
})

@workflow(name="customer_service", organization_id="bank", project_id="support")
def handle_banking_query(query: str) -> str:
    # Content is automatically redacted before tracing
    # "My SSN is 123-45-6789" becomes "My SSN is [REDACTED]"
    return process_banking_query(query)
```

## Performance Monitoring

### Latency Tracking

Automatic latency measurement for all operations:

```python
@workflow(name="image_generation", organization_id="creative", project_id="ai_art")
def generate_artwork(prompt: str) -> str:
    # Latency automatically tracked:
    # - Total workflow duration
    # - Individual task durations
    # - LLM call latencies
    # - Tool execution times
    
    style = analyze_style_preferences(prompt)  # Tracked
    image = generate_image(prompt, style)      # Tracked  
    metadata = extract_metadata(image)        # Tracked
    
    return {
        "image_url": image,
        "metadata": metadata,
        "generation_time": "automatically_tracked"
    }
```

### Resource Usage Tracking

Monitor resource consumption:

```python
from rizk.sdk.tracing import track_resource_usage

@track_resource_usage
@workflow(name="data_analysis", organization_id="analytics", project_id="ml")
def analyze_large_dataset(dataset_path: str) -> dict:
    # Automatically tracks:
    # - Memory usage
    # - CPU utilization  
    # - I/O operations
    # - Network requests
    
    data = load_dataset(dataset_path)
    features = extract_features(data)
    model = train_model(features)
    results = evaluate_model(model)
    
    return results
```

## Integration with Observability Platforms

### Jaeger Integration

Configure Jaeger for trace visualization:

```python
rizk = Rizk.init(
    app_name="MyApp",
    tracing_enabled=True,
    
    # Jaeger configuration
    opentelemetry_endpoint="http://jaeger-collector:14268/api/traces",
    trace_exporter="jaeger",
    
    # Jaeger-specific settings
    jaeger_agent_host="localhost",
    jaeger_agent_port=6831,
    jaeger_service_name="rizk-llm-app"
)
```

### Zipkin Integration

Configure Zipkin for distributed tracing:

```python
rizk = Rizk.init(
    app_name="MyApp",
    tracing_enabled=True,
    
    # Zipkin configuration
    opentelemetry_endpoint="http://zipkin:9411/api/v2/spans",
    trace_exporter="zipkin",
    
    # Zipkin-specific settings
    zipkin_endpoint="http://zipkin:9411/api/v2/spans",
    zipkin_service_name="rizk-llm-app"
)
```

### DataDog Integration

Configure DataDog APM:

```python
rizk = Rizk.init(
    app_name="MyApp",
    tracing_enabled=True,
    
    # DataDog configuration
    opentelemetry_endpoint="https://trace.agent.datadoghq.com",
    trace_exporter="datadog",
    
    # DataDog-specific settings
    datadog_api_key="your-datadog-api-key",
    datadog_service="rizk-llm-app",
    datadog_env="production"
)
```

### New Relic Integration

Configure New Relic for monitoring:

```python
rizk = Rizk.init(
    app_name="MyApp",
    tracing_enabled=True,
    
    # New Relic configuration
    opentelemetry_endpoint="https://otlp.nr-data.net:4317",
    trace_exporter="otlp",
    
    # New Relic-specific headers
    otlp_headers={
        "api-key": "your-new-relic-license-key"
    }
)
```

## Sampling Strategies

### Basic Sampling

Control trace sampling to manage volume and costs:

```python
# Fixed rate sampling
rizk = Rizk.init(
    app_name="MyApp",
    trace_sampling_rate=0.1  # Sample 10% of traces
)

# Environment-based sampling
import os

sampling_rate = {
    "development": 1.0,  # 100% sampling in dev
    "staging": 0.5,      # 50% sampling in staging  
    "production": 0.01   # 1% sampling in production
}.get(os.getenv("ENVIRONMENT"), 0.1)

rizk = Rizk.init(
    app_name="MyApp",
    trace_sampling_rate=sampling_rate
)
```

### Intelligent Sampling

Sample based on content and context:

```python
from rizk.sdk.tracing import configure_intelligent_sampling

# Configure intelligent sampling
configure_intelligent_sampling({
    "error_sampling_rate": 1.0,      # Always sample errors
    "slow_request_sampling_rate": 1.0,  # Always sample slow requests
    "slow_request_threshold_ms": 1000,  # Define slow requests
    
    # Sample more for specific operations
    "high_priority_operations": {
        "financial_transactions": 1.0,
        "user_authentication": 1.0,
        "policy_violations": 1.0
    },
    
    # Sample less for routine operations
    "low_priority_operations": {
        "health_checks": 0.01,
        "static_content": 0.001
    }
})
```

### Custom Sampling Logic

Implement custom sampling decisions:

```python
from rizk.sdk.tracing import CustomSampler

class BusinessLogicSampler(CustomSampler):
    def should_sample(self, span_context, operation_name, attributes):
        # Always sample errors
        if attributes.get("error", False):
            return True
            
        # Sample based on user tier
        user_tier = attributes.get("user.tier")
        if user_tier == "premium":
            return True
        elif user_tier == "enterprise":
            return True
        elif user_tier == "free":
            return random.random() < 0.1  # 10% sampling for free users
            
        # Sample based on operation importance
        if operation_name.startswith("financial"):
            return True
        elif operation_name.startswith("auth"):
            return True
            
        # Default sampling
        return random.random() < 0.05  # 5% default

# Apply custom sampler
rizk = Rizk.init(
    app_name="MyApp",
    custom_sampler=BusinessLogicSampler()
)
```

## Troubleshooting Tracing

### Debug Mode

Enable debug mode for tracing issues:

```python
import logging

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)
trace_logger = logging.getLogger("rizk.tracing")
trace_logger.setLevel(logging.DEBUG)

rizk = Rizk.init(
    app_name="DebugApp",
    tracing_enabled=True,
    trace_debug=True,  # Enable trace debugging
    console_exporter=True  # Export traces to console
)
```

### Common Issues

**1. Traces Not Appearing**

```python
# Check trace configuration
from rizk.sdk.tracing import get_trace_config

config = get_trace_config()
print(f"Tracing enabled: {config.enabled}")
print(f"Endpoint: {config.endpoint}")
print(f"Sampling rate: {config.sampling_rate}")

# Verify span creation
from opentelemetry import trace
tracer = trace.get_tracer(__name__)

with tracer.start_as_current_span("test_span") as span:
    span.set_attribute("test", "value")
    print(f"Span created: {span.get_span_context().span_id}")
```

**2. Missing Context**

```python
# Ensure proper context propagation
from rizk.sdk.tracing import ensure_context_propagation

@ensure_context_propagation
def function_with_context():
    # Context is properly propagated
    span = trace.get_current_span()
    print(f"Current span: {span.name}")
```

**3. Performance Impact**

```python
# Monitor tracing overhead
from rizk.sdk.tracing import get_tracing_metrics

metrics = get_tracing_metrics()
print(f"Tracing overhead: {metrics.overhead_percentage:.2f}%")
print(f"Spans per second: {metrics.spans_per_second}")

# Optimize sampling if overhead is too high
if metrics.overhead_percentage > 5.0:
    rizk.update_config(trace_sampling_rate=0.05)
```

## Best Practices

### 1. Meaningful Span Names

Use descriptive, hierarchical span names:

```python
# âœ… Good span names
@workflow(name="order_fulfillment")
def fulfill_order(): pass

@task(name="inventory_reservation")  
def reserve_inventory(): pass

@agent(name="shipping_coordinator")
def coordinate_shipping(): pass

# âŒ Poor span names
@workflow(name="process")
def process(): pass

@task(name="do_stuff")
def do_stuff(): pass
```

### 2. Appropriate Attribute Usage

Add meaningful attributes without overwhelming:

```python
# âœ… Useful attributes
span.set_attribute("user.id", user_id)
span.set_attribute("order.value", order_total)
span.set_attribute("payment.method", payment_type)
span.set_attribute("operation.success", True)

# âŒ Too many attributes
span.set_attribute("timestamp", datetime.now().isoformat())  # Redundant
span.set_attribute("random_id", uuid.uuid4().hex)  # Not useful
span.set_attribute("debug_info", large_debug_object)  # Too much data
```

### 3. Error Handling

Properly handle and trace errors:

```python
# âœ… Proper error tracing
@workflow(name="payment_processing")
def process_payment(payment_data: dict) -> dict:
    span = trace.get_current_span()
    
    try:
        result = charge_payment(payment_data)
        span.set_attribute("payment.success", True)
        return result
        
    except PaymentError as e:
        span.record_exception(e)
        span.set_status(Status(StatusCode.ERROR, str(e)))
        span.set_attribute("payment.success", False)
        span.set_attribute("error.type", "payment_failed")
        raise
        
    except Exception as e:
        span.record_exception(e)
        span.set_status(Status(StatusCode.ERROR, "Unexpected error"))
        span.set_attribute("payment.success", False)
        span.set_attribute("error.type", "unexpected")
        raise
```

### 4. Performance Considerations

Balance observability with performance:

```python
# âœ… Performance-conscious tracing
@workflow(name="high_throughput_processing")
def process_high_volume_data(data_batch: list) -> list:
    # Create one span for the batch, not per item
    span = trace.get_current_span()
    span.set_attribute("batch.size", len(data_batch))
    
    results = []
    for item in data_batch:
        # Process without creating individual spans
        result = process_item(item)
        results.append(result)
    
    span.set_attribute("batch.processed", len(results))
    return results
```

## Next Steps

1. **[Analytics](analytics.md)** - Set up basic analytics and event tracking
2. **[Streaming Observability](streaming-observability.md)** - Monitor real-time LLM streaming

---

Distributed tracing provides invaluable insights into your LLM application's behavior. Start with basic tracing and gradually add more sophisticated monitoring as your application grows.

