---
title: "Performance Tuning Guide"
description: "Comprehensive performance optimization strategies for Rizk SDK, based on real-world deployments."
---

# Performance Tuning Guide

Comprehensive performance optimization strategies for Rizk SDK, based on real-world deployments.

## Performance Overview

Rizk SDK provides multiple layers of performance optimization:

- **Multi-layer Caching**: L1 (memory) + L2 (Redis) + L3 (future)
- **Framework Detection Caching**: Compiled regex patterns with LRU cache
- **Streaming Optimization**: Real-time processing with backpressure handling
- **Connection Pooling**: Redis connection pools for high throughput
- **Lazy Loading**: On-demand adapter registration
- **Batch Processing**: Efficient analytics and telemetry batching

## Cache Optimization

### 1. Cache Hierarchy Configuration

```python
from rizk.sdk.cache.cache_hierarchy import CacheHierarchy, CacheHierarchyConfig
from rizk.sdk.cache.redis_adapter import RedisConfig

# High-performance cache configuration
cache_config = CacheHierarchyConfig(
    # L1: Local memory cache (fastest)
    l1_enabled=True,
    l1_max_size=50000,  # Larger for high-traffic apps
    l1_ttl_seconds=600,  # 10 minutes
    
    # L2: Distributed Redis cache
    l2_enabled=True,
    l2_ttl_seconds=3600,  # 1 hour
    l2_fallback_on_error=True,
    
    # Performance optimizations
    async_write_behind=True,  # Non-blocking writes
    promotion_threshold=1,    # Aggressive promotion
    
    # Monitoring
    metrics_enabled=True,
    health_check_interval=30
)

# Redis configuration for maximum performance
redis_config = RedisConfig(
    url="redis://redis-cluster:6379",
    max_connections=200,  # High connection pool
    socket_timeout=1.0,   # Fast timeouts
    socket_connect_timeout=2.0,
    retry_on_timeout=True,
    retry_attempts=2,     # Quick retries
    enable_cluster=True,  # Distributed Redis
    key_prefix="perf:rizk:",
    default_ttl=1800
)

cache_hierarchy = CacheHierarchy(cache_config)
```

### 2. Cache Performance Monitoring

```python
import time
from typing import Dict, Any

class CachePerformanceMonitor:
    """Monitor and optimize cache performance."""
    
    def __init__(self, cache_hierarchy: CacheHierarchy):
        self.cache = cache_hierarchy
        self.metrics_history = []
        
    def collect_metrics(self) -> Dict[str, Any]:
        """Collect current cache performance metrics."""
        stats = self.cache.get_stats()
        
        metrics = {
            "timestamp": time.time(),
            "l1_hit_rate": stats.get("l1_hit_rate", 0),
            "l2_hit_rate": stats.get("l2_hit_rate", 0),
            "overall_hit_rate": stats.get("overall_hit_rate", 0),
            "avg_latency_ms": stats.get("avg_latency_ms", 0),
            "l1_size": stats.get("l1_size", 0),
            "total_requests": stats.get("total_requests", 0)
        }
        
        self.metrics_history.append(metrics)
        return metrics
    
    def analyze_performance(self) -> Dict[str, Any]:
        """Analyze cache performance and provide recommendations."""
        if not self.metrics_history:
            return {"status": "no_data"}
        
        latest = self.metrics_history[-1]
        recommendations = []
        
        # Hit rate analysis
        if latest["overall_hit_rate"] < 70:
            recommendations.append({
                "issue": "Low cache hit rate",
                "current": f"{latest['overall_hit_rate']:.1f}%",
                "target": ">80%",
                "action": "Increase cache size or TTL"
            })
        
        # Latency analysis
        if latest["avg_latency_ms"] > 50:
            recommendations.append({
                "issue": "High cache latency",
                "current": f"{latest['avg_latency_ms']:.1f}ms",
                "target": "<20ms",
                "action": "Check Redis connection or reduce cache size"
            })
        
        # Memory usage analysis
        if latest["l1_size"] > 40000:  # 80% of 50k max
            recommendations.append({
                "issue": "L1 cache near capacity",
                "current": f"{latest['l1_size']} items",
                "target": "<40k items",
                "action": "Increase l1_max_size or reduce TTL"
            })
        
        return {
            "status": "healthy" if not recommendations else "needs_attention",
            "current_metrics": latest,
            "recommendations": recommendations
        }

# Usage
monitor = CachePerformanceMonitor(cache_hierarchy)

# Periodic monitoring
def monitor_cache_performance():
    metrics = monitor.collect_metrics()
    analysis = monitor.analyze_performance()
    
    print(f"Cache Performance: {metrics['overall_hit_rate']:.1f}% hit rate, "
          f"{metrics['avg_latency_ms']:.1f}ms latency")
    
    if analysis["recommendations"]:
        print("Performance recommendations:")
        for rec in analysis["recommendations"]:
            print(f"  - {rec['issue']}: {rec['action']}")
```

## Framework Detection Optimization

### 1. Optimized Framework Detection Configuration

```python
from rizk.sdk.config import RizkConfig

# High-performance framework detection
config = RizkConfig(
    framework_detection_cache_size=20000,  # Large cache
    lazy_loading=True,  # Load adapters on demand
    debug_mode=False,   # Disable debug overhead
    verbose=False       # Disable verbose logging
)
```

### 2. Framework Detection Performance Monitoring

```python
from rizk.sdk.utils.framework_detection import detect_framework
from rizk.sdk.performance import performance_instrumented
import time
from functools import lru_cache

@lru_cache(maxsize=10000)
def cached_framework_detection(context_hash: str) -> str:
    """Cached framework detection for repeated contexts."""
    return detect_framework()

@performance_instrumented("framework_detection", "detect")
def optimized_framework_detection() -> str:
    """Optimized framework detection with caching."""
    # Create context hash for caching
    context_hash = hash(str(globals().keys()))
    return cached_framework_detection(str(context_hash))

# Performance testing
def benchmark_framework_detection(iterations: int = 1000):
    """Benchmark framework detection performance."""
    
    # Warm up
    for _ in range(10):
        optimized_framework_detection()
    
    # Benchmark
    start_time = time.time()
    for _ in range(iterations):
        optimized_framework_detection()
    end_time = time.time()
    
    avg_time_ms = (end_time - start_time) / iterations * 1000
    print(f"Framework detection: {avg_time_ms:.2f}ms average over {iterations} iterations")
    
    return avg_time_ms

# Run benchmark
benchmark_framework_detection()
```

## Streaming Performance Optimization

### 1. High-Performance Streaming Configuration

```python
from rizk.sdk.streaming.types import StreamConfig

# Optimized streaming configuration
stream_config = StreamConfig(
    # Performance settings
    max_chunk_size=4096,    # Larger chunks for efficiency
    buffer_size=50,         # Large buffer for throughput
    timeout_seconds=120.0,  # Longer timeout for complex operations
    
    # Guardrail optimization
    enable_guardrails=True,
    realtime_validation=True,
    validation_interval=3,  # Validate every 3 chunks for performance
    
    # Caching optimization
    enable_caching=True,
    cache_partial_responses=True,
    cache_ttl_seconds=900,  # 15 minutes
    
    # Monitoring optimization
    enable_metrics=True,
    metrics_interval=10.0,  # Less frequent metrics for performance
    
    # Framework-specific optimizations
    framework_specific={
        "openai": {
            "stream_options": {
                "include_usage": True,
                "parallel_tool_calls": True
            }
        },
        "anthropic": {
            "max_tokens": 8192,
            "stream": True
        },
        "langchain": {
            "streaming": True,
            "chunk_size": 4096
        }
    }
)
```

## Performance Best Practices

### 1. Configuration Checklist

```python
# High-performance production configuration checklist
PERFORMANCE_CONFIG_CHECKLIST = {
    "cache": {
        "l1_max_size": 50000,  # âœ… Large L1 cache
        "l2_enabled": True,    # âœ… Redis distributed cache
        "async_write_behind": True,  # âœ… Non-blocking writes
        "promotion_threshold": 1,    # âœ… Aggressive promotion
    },
    "framework": {
        "lazy_loading": True,  # âœ… On-demand loading
        "cache_size": 20000,   # âœ… Large detection cache
        "debug_mode": False,   # âœ… No debug overhead
    },
    "streaming": {
        "max_chunk_size": 4096,  # âœ… Large chunks
        "buffer_size": 50,       # âœ… Large buffer
        "validation_interval": 3, # âœ… Reduced validation
    },
    "redis": {
        "max_connections": 200,   # âœ… High connection pool
        "socket_timeout": 1.0,    # âœ… Fast timeouts
        "enable_cluster": True,   # âœ… Distributed Redis
    },
    "analytics": {
        "batch_size": 200,        # âœ… Large batches
        "flush_interval": 10,     # âœ… Frequent flushes
        "keep_raw_data": False,   # âœ… Memory optimization
    }
}

def validate_performance_config(config: RizkConfig) -> List[str]:
    """Validate configuration against performance best practices."""
    issues = []
    
    if config.framework_detection_cache_size < 10000:
        issues.append("Framework cache size too small for high performance")
    
    if config.debug_mode:
        issues.append("Debug mode enabled - disable for production performance")
    
    if config.verbose:
        issues.append("Verbose logging enabled - disable for production performance")
    
    return issues
```

### 2. Monitoring and Alerting

```python
class PerformanceMonitor:
    """Comprehensive performance monitoring."""
    
    def __init__(self):
        self.thresholds = {
            "cache_hit_rate": 80,      # Minimum 80% hit rate
            "avg_latency_ms": 50,      # Maximum 50ms latency
            "error_rate": 1,           # Maximum 1% error rate
            "memory_usage_mb": 2000,   # Maximum 2GB memory
            "redis_response_ms": 10    # Maximum 10ms Redis response
        }
        
    def check_performance_health(self) -> Dict[str, any]:
        """Check overall performance health."""
        health_status = {
            "overall": "healthy",
            "components": {},
            "alerts": []
        }
        
        # Check cache performance
        cache_stats = cache_hierarchy.get_stats()
        if cache_stats["overall_hit_rate"] < self.thresholds["cache_hit_rate"]:
            health_status["overall"] = "degraded"
            health_status["alerts"].append({
                "component": "cache",
                "issue": f"Hit rate {cache_stats['overall_hit_rate']:.1f}% below threshold {self.thresholds['cache_hit_rate']}%"
            })
        
        return health_status

# Usage
monitor = PerformanceMonitor()

def check_system_health():
    health = monitor.check_performance_health()
    
    if health["overall"] == "healthy":
        print("âœ… System performance: HEALTHY")
    else:
        print("âš ï¸ Performance issues detected:")
        for alert in health["alerts"]:
            print(f"  - {alert['component']}: {alert['issue']}")
```

## Next Steps

1. **[Security Best Practices](security.md)** - Secure your high-performance setup
2. **[Scaling Guide](scaling.md)** - Scale your optimized configuration
3. **[Production Setup](production-setup.md)** - Deploy your performance-tuned system

---

**Performance Optimization Checklist**

âœ… Cache hierarchy configured for workload  
âœ… Redis connection pool optimized  
âœ… Framework detection cached  
âœ… Streaming configuration tuned  
âœ… Analytics batching optimized  
âœ… Performance monitoring enabled  
âœ… Benchmarks established  
âœ… Health checks implemented  

*Maximum performance for enterprise LLM governance* 
