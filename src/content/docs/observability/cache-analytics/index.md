---
title: "Cache Analytics and Performance"
description: "Cache Analytics and Performance"
---

# Cache Analytics and Performance

Rizk SDK provides enterprise-grade distributed caching with comprehensive analytics, Redis integration, and intelligent cache hierarchy management for optimal LLM application performance.

## Overview

Cache analytics in Rizk enables you to:

- **Monitor Cache Performance**: Track hit rates, latency, and throughput
- **Distributed Caching**: Redis-backed caching for enterprise scale
- **Cache Hierarchy**: Multi-level caching with intelligent promotion
- **Performance Optimization**: Automatic cache warming and eviction strategies
- **Cost Analytics**: Monitor cache-related cost savings and efficiency

## Quick Start

### Basic Cache Setup

Enable cache analytics with minimal configuration:

```python
from rizk.sdk import Rizk
from rizk.sdk.cache import CacheHierarchy, CacheHierarchyConfig, RedisAdapter

# Initialize Rizk with caching
rizk = Rizk.init(
    app_name="CachedLLMApp",
    api_key="your-rizk-api-key",
    enabled=True
)

# Configure cache hierarchy
cache_config = CacheHierarchyConfig(
    enable_redis=True,
    redis_url="redis://localhost:6379",
    enable_analytics=True,
    enable_metrics=True,
    cache_ttl_seconds=3600,  # 1 hour TTL
    max_memory_mb=512
)

cache_hierarchy = CacheHierarchy(config=cache_config)

@workflow(name="cached_chat", organization_id="acme", project_id="chat")
def cached_chat_response(prompt: str) -> str:
    """Chat with intelligent caching and analytics."""
    
    # Check cache first
    cache_key = f"chat:{hash(prompt)}"
    cached_response = cache_hierarchy.get(cache_key)
    
    if cached_response:
        # Cache hit - track analytics
        cache_hierarchy.track_hit(cache_key, "chat_response")
        return cached_response
    
    # Cache miss - generate response
    response = generate_llm_response(prompt)
    
    # Store in cache with analytics
    cache_hierarchy.set(
        cache_key, 
        response, 
        ttl=3600,
        content_type="chat_response",
        metadata={"prompt_length": len(prompt)}
    )
    
    return response
```

## Cache Performance Metrics

### Real-time Cache Analytics

Monitor cache performance in real-time:

```python
from rizk.sdk.cache import CacheHierarchy

def monitor_cache_performance():
    """Monitor real-time cache performance."""
    
    # Get comprehensive cache metrics
    metrics = cache_hierarchy.get_metrics()
    
    print(f"ðŸ“Š CACHE PERFORMANCE DASHBOARD")
    print(f"Overall Hit Rate: {metrics.hit_rate:.1%}")
    print(f"Total Requests: {metrics.total_requests:,}")
    print(f"Cache Hits: {metrics.cache_hits:,}")
    print(f"Cache Misses: {metrics.cache_misses:,}")
    print(f"Average Response Time: {metrics.avg_response_time_ms:.2f}ms")
    print(f"Memory Usage: {metrics.memory_usage_mb:.1f} MB")
    print(f"Cache Size: {metrics.total_keys:,} keys")
    
    # Performance by cache level
    for level, level_metrics in metrics.level_metrics.items():
        print(f"\n{level.upper()} Cache:")
        print(f"  Hit Rate: {level_metrics.hit_rate:.1%}")
        print(f"  Latency: {level_metrics.avg_latency_ms:.2f}ms")
        print(f"  Size: {level_metrics.size_mb:.1f} MB")
    
    # Performance by content type
    for content_type, type_metrics in metrics.content_type_metrics.items():
        print(f"\n{content_type}:")
        print(f"  Hit Rate: {type_metrics.hit_rate:.1%}")
        print(f"  Requests: {type_metrics.total_requests:,}")
        print(f"  Avg Size: {type_metrics.avg_size_kb:.1f} KB")
```

### Cache Efficiency Analysis

Analyze cache efficiency patterns:

```python
async def analyze_cache_efficiency():
    """Analyze cache efficiency and optimization opportunities."""
    
    # Get detailed analytics
    analytics = cache_hierarchy.get_analytics()
    
    # Identify hot keys
    hot_keys = analytics.get_hot_keys(limit=10)
    print("ðŸ”¥ HOTTEST CACHE KEYS:")
    for key, stats in hot_keys.items():
        print(f"  {key}: {stats.hit_count:,} hits, {stats.hit_rate:.1%} rate")
    
    # Identify cold keys (candidates for eviction)
    cold_keys = analytics.get_cold_keys(limit=10)
    print("\nâ„ï¸ COLDEST CACHE KEYS:")
    for key, stats in cold_keys.items():
        print(f"  {key}: {stats.last_access} ago, {stats.hit_count} hits")
    
    # Cache efficiency by time of day
    hourly_stats = analytics.get_hourly_performance()
    print("\nâ° HOURLY CACHE PERFORMANCE:")
    for hour, stats in hourly_stats.items():
        print(f"  {hour:02d}:00 - Hit Rate: {stats.hit_rate:.1%}, Requests: {stats.requests:,}")
    
    # Memory efficiency analysis
    memory_analysis = analytics.get_memory_efficiency()
    print(f"\nðŸ’¾ MEMORY EFFICIENCY:")
    print(f"Memory Utilization: {memory_analysis.utilization:.1%}")
    print(f"Fragmentation: {memory_analysis.fragmentation:.1%}")
    print(f"Eviction Rate: {memory_analysis.eviction_rate:.2f}/min")
```

## Redis Integration Analytics

### Redis Performance Monitoring

Monitor Redis-specific performance:

```python
from rizk.sdk.cache import RedisAdapter

# Initialize Redis adapter with analytics
redis_adapter = RedisAdapter(
    redis_url="redis://localhost:6379",
    enable_analytics=True,
    enable_cluster_analytics=True
)

async def monitor_redis_performance():
    """Monitor Redis cache performance."""
    
    # Get Redis-specific metrics
    redis_metrics = await redis_adapter.get_metrics()
    
    print(f"ðŸ”´ REDIS CACHE ANALYTICS")
    print(f"Connection Pool: {redis_metrics.active_connections}/{redis_metrics.max_connections}")
    print(f"Memory Usage: {redis_metrics.used_memory_mb:.1f} MB")
    print(f"Peak Memory: {redis_metrics.peak_memory_mb:.1f} MB")
    print(f"Key Count: {redis_metrics.total_keys:,}")
    print(f"Operations/sec: {redis_metrics.ops_per_second:,.0f}")
    print(f"Network I/O: {redis_metrics.network_io_mbps:.2f} Mbps")
    
    # Redis command statistics
    command_stats = redis_metrics.command_stats
    print(f"\nðŸ“‹ REDIS COMMAND STATS:")
    for command, stats in sorted(command_stats.items(), key=lambda x: x[1].count, reverse=True)[:5]:
        print(f"  {command}: {stats.count:,} calls, {stats.avg_latency_ms:.2f}ms avg")
    
    # Cluster performance (if using Redis Cluster)
    if redis_metrics.cluster_metrics:
        cluster = redis_metrics.cluster_metrics
        print(f"\nðŸ”— REDIS CLUSTER METRICS:")
        print(f"Nodes: {cluster.total_nodes} ({cluster.healthy_nodes} healthy)")
        print(f"Slots Coverage: {cluster.slots_coverage:.1%}")
        print(f"Cross-slot Operations: {cluster.cross_slot_ops:,}")
```

### Redis Cost Optimization

Optimize Redis costs with analytics:

```python
async def optimize_redis_costs():
    """Analyze and optimize Redis costs."""
    
    cost_analytics = await redis_adapter.get_cost_analytics()
    
    print(f"ðŸ’° REDIS COST OPTIMIZATION")
    print(f"Current Memory Cost: ${cost_analytics.memory_cost_per_hour:.2f}/hour")
    print(f"Network Cost: ${cost_analytics.network_cost_per_hour:.2f}/hour")
    print(f"Compute Cost: ${cost_analytics.compute_cost_per_hour:.2f}/hour")
    
    # Optimization recommendations
    recommendations = cost_analytics.get_optimization_recommendations()
    print(f"\nðŸ’¡ COST OPTIMIZATION RECOMMENDATIONS:")
    for rec in recommendations:
        print(f"  â€¢ {rec.description}")
        print(f"    Potential Savings: ${rec.potential_savings_per_month:.2f}/month")
        print(f"    Implementation: {rec.implementation_effort}")
    
    # Memory usage breakdown
    memory_breakdown = cost_analytics.get_memory_breakdown()
    print(f"\nðŸ§  MEMORY USAGE BREAKDOWN:")
    for category, usage in memory_breakdown.items():
        print(f"  {category}: {usage.size_mb:.1f} MB ({usage.percentage:.1%})")
```

## Cache Hierarchy Analytics

### Multi-Level Cache Performance

Monitor cache hierarchy performance:

```python
from rizk.sdk.cache import CacheLevel, CacheStrategy

async def monitor_cache_hierarchy():
    """Monitor multi-level cache hierarchy performance."""
    
    hierarchy_metrics = cache_hierarchy.get_hierarchy_metrics()
    
    print(f"ðŸ—ï¸ CACHE HIERARCHY PERFORMANCE")
    
    # Performance by cache level
    for level in [CacheLevel.L1, CacheLevel.L2, CacheLevel.L3]:
        level_metrics = hierarchy_metrics.get_level_metrics(level)
        
        print(f"\n{level.name} Cache:")
        print(f"  Hit Rate: {level_metrics.hit_rate:.1%}")
        print(f"  Average Latency: {level_metrics.avg_latency_ms:.2f}ms")
        print(f"  Throughput: {level_metrics.ops_per_second:,.0f} ops/s")
        print(f"  Size: {level_metrics.current_size_mb:.1f} MB")
        print(f"  Evictions: {level_metrics.evictions_per_hour:.0f}/hour")
    
    # Cache promotion analytics
    promotion_stats = hierarchy_metrics.promotion_stats
    print(f"\nðŸ“ˆ CACHE PROMOTION ANALYTICS:")
    print(f"L2â†’L1 Promotions: {promotion_stats.l2_to_l1:,}")
    print(f"L3â†’L2 Promotions: {promotion_stats.l3_to_l2:,}")
    print(f"Promotion Success Rate: {promotion_stats.success_rate:.1%}")
    print(f"Avg Promotion Latency: {promotion_stats.avg_latency_ms:.2f}ms")
    
    # Cache coherence metrics
    coherence_metrics = hierarchy_metrics.coherence_metrics
    print(f"\nðŸ”„ CACHE COHERENCE METRICS:")
    print(f"Invalidations: {coherence_metrics.invalidations_per_hour:.0f}/hour")
    print(f"Consistency Checks: {coherence_metrics.consistency_checks:,}")
    print(f"Coherence Violations: {coherence_metrics.violations}")
```

### Intelligent Cache Warming

Implement intelligent cache warming based on analytics:

```python
async def intelligent_cache_warming():
    """Implement intelligent cache warming based on usage analytics."""
    
    # Analyze usage patterns
    usage_analytics = cache_hierarchy.get_usage_analytics()
    
    # Identify warming candidates
    warming_candidates = usage_analytics.get_warming_candidates(
        min_hit_rate=0.7,  # Only warm keys with >70% hit rate
        min_frequency=10,  # Accessed at least 10 times
        time_window_hours=24
    )
    
    print(f"ðŸ”¥ CACHE WARMING ANALYSIS")
    print(f"Warming Candidates: {len(warming_candidates)}")
    
    for candidate in warming_candidates[:10]:  # Top 10 candidates
        print(f"  Key: {candidate.key}")
        print(f"    Hit Rate: {candidate.hit_rate:.1%}")
        print(f"    Frequency: {candidate.frequency}/day")
        print(f"    Avg Response Time: {candidate.avg_response_time_ms:.2f}ms")
        print(f"    Predicted Benefit: {candidate.predicted_benefit_score:.2f}")
        
        # Warm the cache
        if candidate.predicted_benefit_score > 0.8:
            await warm_cache_key(candidate.key, candidate.predicted_value)
            print(f"    âœ… Warmed cache for {candidate.key}")

async def warm_cache_key(key: str, predicted_value: str):
    """Warm specific cache key."""
    try:
        # Pre-compute and cache the value
        computed_value = await compute_cache_value(key)
        await cache_hierarchy.set(key, computed_value, ttl=3600)
        
        # Track warming success
        cache_hierarchy.track_warming_success(key)
        
    except Exception as e:
        cache_hierarchy.track_warming_failure(key, str(e))
```

## Cache Analytics Integration

### Export to External Systems

Export cache analytics to external monitoring systems:

```python
import json
from datetime import datetime

async def export_cache_analytics():
    """Export cache analytics to external systems."""
    
    # Collect comprehensive cache analytics
    analytics_data = {
        "timestamp": datetime.utcnow().isoformat(),
        "cache_hierarchy": {
            "overall_hit_rate": cache_hierarchy.get_hit_rate(),
            "total_requests": cache_hierarchy.get_total_requests(),
            "memory_usage_mb": cache_hierarchy.get_memory_usage(),
            "level_performance": cache_hierarchy.get_level_performance()
        },
        "redis_metrics": await redis_adapter.get_metrics() if redis_adapter else None,
        "cost_analytics": {
            "estimated_savings": calculate_cache_savings(),
            "cost_per_request": calculate_cost_per_request(),
            "roi": calculate_cache_roi()
        },
        "optimization_opportunities": get_optimization_opportunities()
    }
    
    # Export to different monitoring systems
    await export_to_datadog(analytics_data)
    await export_to_prometheus(analytics_data)
    await export_to_grafana(analytics_data)

async def export_to_datadog(data):
    """Export cache metrics to DataDog."""
    # DataDog API integration
    datadog_metrics = [
        {
            "metric": "rizk.cache.hit_rate",
            "points": [[time.time(), data["cache_hierarchy"]["overall_hit_rate"]]],
            "tags": ["service:rizk", "component:cache"]
        },
        {
            "metric": "rizk.cache.memory_usage",
            "points": [[time.time(), data["cache_hierarchy"]["memory_usage_mb"]]],
            "tags": ["service:rizk", "component:cache"]
        }
    ]
    
    # Send to DataDog API
    await send_datadog_metrics(datadog_metrics)

async def export_to_prometheus(data):
    """Export cache metrics to Prometheus."""
    # Prometheus metrics format
    prometheus_metrics = f"""
# HELP rizk_cache_hit_rate Cache hit rate percentage
# TYPE rizk_cache_hit_rate gauge
rizk_cache_hit_rate{{service="rizk",component="cache"}} {data["cache_hierarchy"]["overall_hit_rate"]}

# HELP rizk_cache_memory_usage_mb Cache memory usage in MB
# TYPE rizk_cache_memory_usage_mb gauge
rizk_cache_memory_usage_mb{{service="rizk",component="cache"}} {data["cache_hierarchy"]["memory_usage_mb"]}
"""
    
    # Push to Prometheus pushgateway
    await push_to_prometheus(prometheus_metrics)
```

## Cache Cost Analytics

### ROI Analysis

Calculate cache return on investment:

```python
def calculate_cache_roi():
    """Calculate cache ROI and cost savings."""
    
    # Get cache performance metrics
    metrics = cache_hierarchy.get_metrics()
    
    # Calculate costs
    cache_infrastructure_cost = calculate_infrastructure_cost()
    llm_api_cost_savings = calculate_llm_savings(metrics.cache_hits)
    latency_improvement_value = calculate_latency_value(metrics.avg_response_time_ms)
    
    # ROI calculation
    total_savings = llm_api_cost_savings + latency_improvement_value
    roi_percentage = ((total_savings - cache_infrastructure_cost) / cache_infrastructure_cost) * 100
    
    print(f"ðŸ’° CACHE ROI ANALYSIS")
    print(f"Infrastructure Cost: ${cache_infrastructure_cost:.2f}/month")
    print(f"LLM API Savings: ${llm_api_cost_savings:.2f}/month")
    print(f"Latency Value: ${latency_improvement_value:.2f}/month")
    print(f"Total Savings: ${total_savings:.2f}/month")
    print(f"ROI: {roi_percentage:.1f}%")
    
    return {
        "infrastructure_cost": cache_infrastructure_cost,
        "total_savings": total_savings,
        "roi_percentage": roi_percentage,
        "payback_period_months": cache_infrastructure_cost / (total_savings / 12) if total_savings > 0 else float('inf')
    }

def calculate_llm_savings(cache_hits: int):
    """Calculate savings from avoided LLM API calls."""
    avg_llm_cost_per_request = 0.002  # $0.002 per request
    return cache_hits * avg_llm_cost_per_request

def calculate_latency_value(avg_response_time_ms: float):
    """Calculate business value of latency improvements."""
    baseline_latency_ms = 2000  # 2 seconds baseline
    improvement_ms = max(0, baseline_latency_ms - avg_response_time_ms)
    
    # Value per millisecond improvement (based on user engagement studies)
    value_per_ms = 0.001  # $0.001 per ms improvement per request
    
    total_requests = cache_hierarchy.get_total_requests()
    return (improvement_ms * value_per_ms * total_requests) / 30  # Monthly value
```

## Best Practices

### Cache Strategy Optimization

1. **TTL Tuning**: Optimize TTL based on content freshness requirements
2. **Memory Management**: Monitor memory usage and implement intelligent eviction
3. **Key Design**: Use consistent, hierarchical key naming conventions
4. **Monitoring**: Set up comprehensive monitoring and alerting

### Performance Optimization

1. **Cache Warming**: Implement predictive cache warming for hot content
2. **Compression**: Use compression for large cached values
3. **Partitioning**: Distribute cache load across multiple Redis instances
4. **Connection Pooling**: Optimize Redis connection management

### Cost Optimization

1. **Right-sizing**: Monitor usage patterns to right-size Redis instances
2. **Compression**: Reduce memory costs with intelligent compression
3. **Eviction Policies**: Implement cost-aware eviction strategies
4. **Regional Deployment**: Deploy caches close to users to reduce latency costs

## Troubleshooting

### Common Cache Issues

**Low Hit Rate**
```python
# Analyze cache miss patterns
miss_analysis = cache_hierarchy.analyze_misses()
if miss_analysis.ttl_expiry_rate > 0.5:
    # Increase TTL for stable content
    cache_hierarchy.update_ttl_policy(default_ttl=7200)
```

**High Memory Usage**
```python
# Implement memory optimization
memory_usage = cache_hierarchy.get_memory_usage()
if memory_usage.utilization > 0.8:
    # Enable compression and optimize eviction
    cache_hierarchy.enable_compression()
    cache_hierarchy.set_eviction_policy("lru")
```

**Redis Connection Issues**
```python
# Monitor Redis connection health
redis_health = await redis_adapter.check_health()
if not redis_health.is_healthy:
    # Implement connection recovery
    await redis_adapter.recover_connections()
```

---

**Next Steps**: [Workflow Telemetry](workflow-telemetry.md) - Monitor decorator-based workflow performance 

---

**Note**: This demonstrates the enterprise-grade caching capabilities available in Rizk SDK for high-performance LLM applications. 

