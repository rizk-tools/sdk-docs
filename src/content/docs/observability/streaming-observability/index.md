---
title: "Streaming LLM Observability"
description: "Streaming LLM Observability"
---

# Streaming LLM Observability

Rizk SDK provides comprehensive observability for streaming LLM interactions, enabling real-time monitoring, guardrails enforcement, and performance optimization for streaming responses.

## Overview

Streaming observability in Rizk enables you to:

- **Real-time Monitoring**: Track streaming LLM responses as they generate
- **Live Guardrails**: Apply policy enforcement to streaming content
- **Performance Analytics**: Monitor streaming latency, throughput, and backpressure
- **Content Validation**: Validate partial responses in real-time
- **Cache Analytics**: Track streaming cache hits and performance

## Quick Start

### Basic Streaming Setup

Enable streaming observability with minimal configuration:

```python
from rizk.sdk import Rizk
from rizk.sdk.streaming import StreamProcessor, StreamConfig
from rizk.sdk.decorators import workflow

# Initialize Rizk with streaming support
rizk = Rizk.init(
    app_name="StreamingLLMApp",
    api_key="your-rizk-api-key",
    enabled=True
)

# Configure streaming processor
stream_config = StreamConfig(
    enable_guardrails=True,
    enable_caching=True,
    enable_metrics=True,
    realtime_validation=True,
    buffer_size=1000,
    validation_interval=0.1  # 100ms validation intervals
)

processor = StreamProcessor(config=stream_config)

@workflow(name="streaming_chat", organization_id="acme", project_id="chat")
async def stream_chat_response(prompt: str):
    """Stream LLM response with real-time observability."""
    
    # Your streaming LLM call (OpenAI, Anthropic, etc.)
    async def llm_stream():
        # Example with OpenAI streaming
        async for chunk in openai_stream_response(prompt):
            yield chunk
    
    # Process stream with observability
    async for event in processor.process_stream(
        input_stream=llm_stream(),
        initial_prompt=prompt
    ):
        if event.event_type == "chunk":
            # Stream chunk with guardrails applied
            yield event.data["content"]
        elif event.event_type == "guardrail_block":
            # Content was blocked by guardrails
            yield "[Content filtered by policy]"
        elif event.event_type == "cache_hit":
            # Response served from cache
            yield event.data["content"]

# Use streaming function
async for response_chunk in stream_chat_response("Tell me about AI safety"):
    print(response_chunk, end="", flush=True)
```

## Stream Events and Monitoring

### Stream Event Types

Rizk tracks comprehensive streaming events:

```python
from rizk.sdk.streaming import StreamEventType

# Core streaming events
StreamEventType.STREAM_START     # Stream initiated
StreamEventType.CHUNK            # Content chunk processed
StreamEventType.STREAM_END       # Stream completed
StreamEventType.STREAM_ERROR     # Stream error occurred

# Guardrails events
StreamEventType.GUARDRAIL_CHECK  # Guardrail validation performed
StreamEventType.GUARDRAIL_BLOCK  # Content blocked by policy
StreamEventType.GUARDRAIL_MODIFY # Content modified by policy

# Performance events
StreamEventType.CACHE_HIT        # Response served from cache
StreamEventType.CACHE_MISS       # Cache miss, generating new response
StreamEventType.BACKPRESSURE     # Backpressure detected
StreamEventType.BUFFER_FULL      # Stream buffer reached capacity

# Metrics events
StreamEventType.METRICS_UPDATE   # Performance metrics updated
```

### Real-time Event Handling

Monitor streaming events in real-time:

```python
from rizk.sdk.streaming import StreamProcessor, StreamEvent

def handle_stream_event(event: StreamEvent):
    """Handle streaming events for monitoring."""
    
    if event.event_type == StreamEventType.GUARDRAIL_BLOCK:
        # Alert on content blocking
        print(f"ðŸš« Content blocked: {event.data['reasons']}")
        
    elif event.event_type == StreamEventType.BACKPRESSURE:
        # Monitor performance issues
        print(f"âš ï¸ Backpressure detected: {event.data['buffer_utilization']}%")
        
    elif event.event_type == StreamEventType.CACHE_HIT:
        # Track cache performance
        print(f"ðŸ’¾ Cache hit: {event.data['cache_key']}")
        
    elif event.event_type == StreamEventType.METRICS_UPDATE:
        # Real-time performance metrics
        metrics = event.data['metrics']
        print(f"ðŸ“Š Throughput: {metrics.chunks_per_second:.2f} chunks/s")

# Add event handler to processor
processor.add_event_handler(handle_stream_event)
```

## Streaming Performance Metrics

### Comprehensive Metrics Collection

Track detailed streaming performance:

```python
# Get streaming metrics for active streams
active_streams = processor.get_active_streams()

for stream_id, metrics in active_streams.items():
    print(f"Stream {stream_id}:")
    print(f"  Duration: {metrics.duration_seconds:.2f}s")
    print(f"  Chunks processed: {metrics.total_chunks}")
    print(f"  Throughput: {metrics.chunks_per_second:.2f} chunks/s")
    print(f"  Latency (avg): {metrics.average_chunk_latency:.3f}s")
    print(f"  Guardrail checks: {metrics.guardrail_checks}")
    print(f"  Content blocked: {metrics.guardrail_blocks}")
    print(f"  Cache hits: {metrics.cache_hits}")
    print(f"  Buffer utilization: {metrics.buffer_utilization:.1f}%")
```

### Performance Monitoring Dashboard

Create real-time monitoring dashboard:

```python
import asyncio
from rizk.sdk.streaming import StreamProcessor

async def streaming_dashboard():
    """Real-time streaming performance dashboard."""
    
    while True:
        # Get current metrics
        active_streams = processor.get_active_streams()
        
        # Calculate aggregate metrics
        total_streams = len(active_streams)
        total_throughput = sum(m.chunks_per_second for m in active_streams.values())
        avg_latency = sum(m.average_chunk_latency for m in active_streams.values()) / max(total_streams, 1)
        total_guardrail_blocks = sum(m.guardrail_blocks for m in active_streams.values())
        
        # Display dashboard
        print(f"\nðŸ”´ LIVE STREAMING DASHBOARD")
        print(f"Active Streams: {total_streams}")
        print(f"Total Throughput: {total_throughput:.2f} chunks/s")
        print(f"Average Latency: {avg_latency:.3f}s")
        print(f"Guardrail Blocks: {total_guardrail_blocks}")
        
        # Wait before next update
        await asyncio.sleep(1)

# Run dashboard
asyncio.create_task(streaming_dashboard())
```

## Streaming Guardrails Observability

### Real-time Policy Enforcement

Monitor guardrails in streaming contexts:

```python
from rizk.sdk.streaming import StreamGuardrailsProcessor

# Configure streaming guardrails
guardrails_processor = StreamGuardrailsProcessor(
    validation_interval=0.1,  # Validate every 100ms
    realtime_validation=True,
    buffer_validation=True    # Validate buffered content
)

async def monitor_streaming_guardrails(stream_id: str, content_stream):
    """Monitor guardrails enforcement in real-time."""
    
    full_content = ""
    violation_count = 0
    
    async for chunk in content_stream:
        # Validate chunk with guardrails
        validation_result = await guardrails_processor.validate_chunk(
            stream_id, chunk, full_content
        )
        
        if not validation_result.is_valid:
            violation_count += 1
            print(f"ðŸš« Policy violation #{violation_count}:")
            print(f"   Content: {validation_result.blocked_content}")
            print(f"   Reasons: {validation_result.violation_reasons}")
            print(f"   Confidence: {validation_result.confidence:.2f}")
            
        if validation_result.modified_content:
            print(f"âœï¸ Content modified by policy:")
            print(f"   Original: {chunk.content}")
            print(f"   Modified: {validation_result.modified_content}")
            
        full_content += chunk.content
```

### Policy Performance Analytics

Track guardrails performance impact:

```python
# Guardrails performance metrics
guardrails_metrics = {
    "total_validations": 0,
    "validation_latency": [],
    "violation_rate": 0.0,
    "policy_hit_rate": {},
    "modification_rate": 0.0
}

async def track_guardrails_performance(stream_processor):
    """Track guardrails performance impact on streaming."""
    
    def on_guardrail_event(event: StreamEvent):
        if event.event_type == StreamEventType.GUARDRAIL_CHECK:
            guardrails_metrics["total_validations"] += 1
            guardrails_metrics["validation_latency"].append(
                event.data.get("validation_time_ms", 0)
            )
            
        elif event.event_type == StreamEventType.GUARDRAIL_BLOCK:
            policy_id = event.data.get("policy_id")
            if policy_id:
                guardrails_metrics["policy_hit_rate"][policy_id] = (
                    guardrails_metrics["policy_hit_rate"].get(policy_id, 0) + 1
                )
                
        elif event.event_type == StreamEventType.GUARDRAIL_MODIFY:
            guardrails_metrics["modification_rate"] += 1
    
    stream_processor.add_event_handler(on_guardrail_event)
    
    # Periodic reporting
    while True:
        await asyncio.sleep(10)  # Report every 10 seconds
        
        if guardrails_metrics["total_validations"] > 0:
            avg_latency = sum(guardrails_metrics["validation_latency"]) / len(guardrails_metrics["validation_latency"])
            violation_rate = sum(guardrails_metrics["policy_hit_rate"].values()) / guardrails_metrics["total_validations"]
            
            print(f"\nðŸ“‹ GUARDRAILS PERFORMANCE REPORT")
            print(f"Total Validations: {guardrails_metrics['total_validations']}")
            print(f"Average Latency: {avg_latency:.2f}ms")
            print(f"Violation Rate: {violation_rate:.1%}")
            print(f"Most Triggered Policies: {sorted(guardrails_metrics['policy_hit_rate'].items(), key=lambda x: x[1], reverse=True)[:3]}")
```

## Streaming Cache Analytics

### Cache Performance Monitoring

Monitor streaming cache effectiveness:

```python
from rizk.sdk.streaming import StreamCache

# Initialize streaming cache with analytics
stream_cache = StreamCache(
    ttl_seconds=300,  # 5 minute TTL
    enable_partial_caching=True,
    enable_analytics=True
)

async def monitor_cache_performance():
    """Monitor streaming cache performance."""
    
    cache_metrics = stream_cache.get_metrics()
    
    print(f"ðŸ’¾ STREAMING CACHE ANALYTICS")
    print(f"Cache Hit Rate: {cache_metrics.hit_rate:.1%}")
    print(f"Partial Cache Hits: {cache_metrics.partial_hits}")
    print(f"Cache Size: {cache_metrics.current_size} items")
    print(f"Memory Usage: {cache_metrics.memory_usage_mb:.1f} MB")
    print(f"Average Response Time: {cache_metrics.avg_response_time_ms:.2f}ms")
    
    # Cache efficiency by content type
    for content_type, stats in cache_metrics.content_type_stats.items():
        print(f"  {content_type}: {stats.hit_rate:.1%} hit rate")
```

### Intelligent Cache Warming

Implement cache warming based on streaming patterns:

```python
async def intelligent_cache_warming(stream_processor):
    """Warm cache based on streaming usage patterns."""
    
    # Analyze streaming patterns
    frequent_prompts = await analyze_streaming_patterns()
    
    for prompt in frequent_prompts:
        # Pre-generate and cache responses for common prompts
        if not await stream_cache.exists(prompt):
            print(f"ðŸ”¥ Warming cache for: {prompt[:50]}...")
            
            # Generate response and cache it
            response = await generate_streaming_response(prompt)
            await stream_cache.store(prompt, response)

async def analyze_streaming_patterns():
    """Analyze streaming usage to identify cache warming opportunities."""
    
    # Get recent streaming metrics
    recent_streams = processor.get_recent_stream_history(hours=24)
    
    # Find frequently requested prompts
    prompt_frequency = {}
    for stream in recent_streams:
        prompt = stream.initial_prompt
        prompt_frequency[prompt] = prompt_frequency.get(prompt, 0) + 1
    
    # Return top prompts for cache warming
    return sorted(prompt_frequency.items(), key=lambda x: x[1], reverse=True)[:10]
```

## Integration with External Systems

### Streaming Metrics Export

Export streaming metrics to external monitoring systems:

```python
import json
from datetime import datetime

async def export_streaming_metrics():
    """Export streaming metrics to external systems."""
    
    # Collect comprehensive metrics
    metrics_data = {
        "timestamp": datetime.utcnow().isoformat(),
        "active_streams": len(processor.get_active_streams()),
        "total_throughput": sum(m.chunks_per_second for m in processor.get_active_streams().values()),
        "guardrails_performance": guardrails_metrics,
        "cache_performance": stream_cache.get_metrics().to_dict(),
        "system_health": {
            "memory_usage": get_memory_usage(),
            "cpu_usage": get_cpu_usage(),
            "buffer_utilization": get_average_buffer_utilization()
        }
    }
    
    # Export to different systems
    await export_to_datadog(metrics_data)
    await export_to_prometheus(metrics_data)
    await export_to_custom_dashboard(metrics_data)

async def export_to_datadog(metrics):
    """Export to DataDog."""
    # Implementation for DataDog API
    pass

async def export_to_prometheus(metrics):
    """Export to Prometheus."""
    # Implementation for Prometheus metrics
    pass
```

## Best Practices

### Performance Optimization

1. **Buffer Size Tuning**: Optimize buffer sizes based on content velocity
2. **Validation Intervals**: Balance real-time validation with performance
3. **Cache Strategy**: Use intelligent caching for frequently accessed content
4. **Backpressure Handling**: Implement proper backpressure management

### Monitoring Strategy

1. **Real-time Dashboards**: Create live monitoring dashboards
2. **Alert Thresholds**: Set up alerts for performance degradation
3. **Trend Analysis**: Monitor long-term streaming performance trends
4. **Capacity Planning**: Use metrics for infrastructure scaling decisions

### Security and Compliance

1. **Content Monitoring**: Track all content flowing through streams
2. **Policy Enforcement**: Ensure real-time guardrails compliance
3. **Audit Trails**: Maintain comprehensive audit logs for streaming content
4. **Data Retention**: Implement appropriate data retention policies

## Troubleshooting

### Common Issues

**High Latency in Streaming**
```python
# Check buffer utilization and validation intervals
metrics = processor.get_stream_metrics(stream_id)
if metrics.buffer_utilization > 80:
    # Increase buffer size or reduce validation frequency
    config.buffer_size *= 2
    config.validation_interval *= 1.5
```

**Guardrails Performance Impact**
```python
# Monitor guardrails latency
if avg_guardrail_latency > 100:  # ms
    # Optimize policy evaluation or reduce validation frequency
    config.realtime_validation = False
    config.validation_interval = 0.5  # Reduce to 500ms
```

**Cache Miss Rate Too High**
```python
# Analyze cache performance
cache_metrics = stream_cache.get_metrics()
if cache_metrics.hit_rate < 0.3:  # Less than 30%
    # Implement cache warming or increase TTL
    await intelligent_cache_warming(processor)
```

---

**Next Steps**: [Cache Analytics](cache-analytics.md) - Monitor distributed caching performance 

