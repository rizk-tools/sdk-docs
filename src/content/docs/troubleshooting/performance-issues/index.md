---
title: "Performance Issues"
description: "Performance Issues"
---

# Performance Issues

This guide helps you identify, diagnose, and resolve performance problems when using Rizk SDK.

## ðŸš€ Performance Overview

Rizk SDK is designed for minimal overhead, but performance can be affected by:

- **Framework Detection**: Pattern matching and caching
- **Guardrails Evaluation**: Policy processing and LLM calls
- **Adapter Registration**: Framework and LLM client patching
- **Telemetry Collection**: Tracing and metrics overhead
- **Cache Management**: Memory and Redis operations

## ðŸ“Š Performance Monitoring

### Built-in Performance Instrumentation

Rizk SDK includes performance monitoring capabilities:

```python
from rizk.sdk.performance import performance_instrumented
from rizk.sdk.analytics.processors import PerformanceMonitoringProcessor

# Enable performance monitoring
@performance_instrumented("my_operation")
def my_function():
    return "result"

# Check performance stats
processor = PerformanceMonitoringProcessor()
stats = processor.get_performance_stats()
print(f"Average latency: {stats.get('avg_latency_ms', 0):.2f}ms")
print(f"P95 latency: {stats.get('p95_latency_ms', 0):.2f}ms")
print(f"Total operations: {stats.get('total_operations', 0)}")
```

### Custom Performance Profiler

Create a detailed profiler for SDK operations:

```python
import time
import cProfile
import pstats
import io
from functools import wraps
from typing import Callable, Dict, List
from contextlib import contextmanager

class RizkPerformanceProfiler:
    """Performance profiler for Rizk SDK operations."""
    
    def __init__(self):
        self.operation_times: Dict[str, List[float]] = {}
        self.total_calls: Dict[str, int] = {}
    
    @contextmanager
    def profile_operation(self, operation_name: str):
        """Context manager to profile an operation."""
        start_time = time.time()
        try:
            yield
        finally:
            elapsed = time.time() - start_time
            
            if operation_name not in self.operation_times:
                self.operation_times[operation_name] = []
                self.total_calls[operation_name] = 0
            
            self.operation_times[operation_name].append(elapsed * 1000)  # Convert to ms
            self.total_calls[operation_name] += 1
    
    def get_stats(self) -> Dict[str, Dict[str, float]]:
        """Get performance statistics."""
        stats = {}
        
        for operation, times in self.operation_times.items():
            if times:
                stats[operation] = {
                    "total_calls": self.total_calls[operation],
                    "avg_time_ms": sum(times) / len(times),
                    "min_time_ms": min(times),
                    "max_time_ms": max(times),
                    "total_time_ms": sum(times),
                    "p95_time_ms": sorted(times)[int(len(times) * 0.95)] if len(times) > 1 else times[0],
                    "p99_time_ms": sorted(times)[int(len(times) * 0.99)] if len(times) > 1 else times[0]
                }
        
        return stats
    
    def print_report(self):
        """Print a formatted performance report."""
        stats = self.get_stats()
        
        print("\nðŸ“Š Rizk SDK Performance Report")
        print("=" * 80)
        
        if not stats:
            print("No performance data collected.")
            return
        
        # Sort by total time
        sorted_stats = sorted(stats.items(), key=lambda x: x[1]["total_time_ms"], reverse=True)
        
        print(f"{'Operation':<30} {'Calls':<8} {'Avg(ms)':<10} {'Total(ms)':<12} {'P95(ms)':<10}")
        print("-" * 80)
        
        for operation, data in sorted_stats:
            operation_short = operation[-28:] if len(operation) > 30 else operation
            print(f"{operation_short:<30} {data['total_calls']:<8} "
                  f"{data['avg_time_ms']:<10.2f} {data['total_time_ms']:<12.2f} "
                  f"{data['p95_time_ms']:<10.2f}")
        
        # Find performance issues
        print("\nâš ï¸ Performance Issues:")
        issues_found = False
        
        for operation, data in stats.items():
            if data["avg_time_ms"] > 100:  # More than 100ms average
                print(f"  â€¢ {operation}: High average latency ({data['avg_time_ms']:.1f}ms)")
                issues_found = True
            
            if data["p95_time_ms"] > 500:  # P95 more than 500ms
                print(f"  â€¢ {operation}: High P95 latency ({data['p95_time_ms']:.1f}ms)")
                issues_found = True
            
            if data["total_calls"] > 1000:  # More than 1000 calls
                print(f"  â€¢ {operation}: High call frequency ({data['total_calls']} calls)")
                issues_found = True
        
        if not issues_found:
            print("  No significant performance issues detected.")

# Global profiler instance
profiler = RizkPerformanceProfiler()
```

## ðŸ”§ Common Performance Issues

### 1. Slow Framework Detection

**Symptoms**: 
- High latency on first decorator calls
- Repeated framework detection overhead

**Diagnosis**:
```python
from rizk.sdk.utils.framework_detection import get_detection_cache_stats
import time

# Measure detection time
start = time.time()
from rizk.sdk.utils.framework_detection import detect_framework
framework = detect_framework()
detection_time = (time.time() - start) * 1000

print(f"Framework detection took: {detection_time:.2f}ms")
print(f"Detected framework: {framework}")

# Check cache performance
cache_stats = get_detection_cache_stats()
print(f"Cache hit rate: {cache_stats['hit_rate']:.1f}%")
print(f"Cache size: {cache_stats['size']}")
```

**Solutions**:
```python
import os

# 1. Increase cache size
os.environ["RIZK_FRAMEWORK_CACHE_SIZE"] = "5000"  # Default: 1000

# 2. Enable lazy loading
os.environ["RIZK_LAZY_LOADING"] = "true"

# 3. Explicit framework specification (bypasses detection)
from rizk.sdk.decorators import workflow

@workflow(name="my_workflow", framework="langchain")  # Explicit
def my_function():
    pass

# 4. Pre-warm the cache during initialization
from rizk.sdk import Rizk
from rizk.sdk.utils.framework_detection import detect_framework_cached

Rizk.init(app_name="OptimizedApp")

# Pre-warm detection cache
for _ in range(10):
    detect_framework_cached()
```

### 2. Guardrails Performance Issues

**Symptoms**:
- High latency on message processing
- Blocking during policy evaluation
- Memory growth with large message volumes

**Diagnosis**:
```python
from rizk.sdk.guardrails.engine import GuardrailsEngine
import time

# Test guardrails performance
engine = GuardrailsEngine.get_instance()

test_messages = [
    "Short message",
    "A much longer message with more content to evaluate for policy violations",
    "SSN: 123-45-6789",  # Should trigger policies
    "Multiple sensitive items: SSN 123-45-6789, Credit Card 1234-5678-9012-3456"
]

for message in test_messages:
    start = time.time()
    result = engine.evaluate(message)
    latency = (time.time() - start) * 1000
    
    print(f"Message length: {len(message):3d} | "
          f"Latency: {latency:6.2f}ms | "
          f"Decision: {'BLOCKED' if not result.allowed else 'ALLOWED'}")
```

**Solutions**:
```python
# 1. Optimize guardrails configuration
from rizk.sdk import Rizk

Rizk.init(
    app_name="OptimizedApp",
    # Disable features you don't need
    telemetry_enabled=False,
    disable_batch=True,  # Reduce latency for individual calls
    
    # Optimize guardrails settings
    llm_cache_size=10000,  # Larger LLM cache
    state_ttl_seconds=300,  # Faster state cleanup
)

# 2. Use async evaluation for high throughput
import asyncio

async def process_messages_async(messages):
    engine = GuardrailsEngine.get_instance()
    tasks = [engine.process_message(msg) for msg in messages]
    results = await asyncio.gather(*tasks)
    return results

# 3. Batch processing for multiple messages
async def batch_process():
    messages = ["msg1", "msg2", "msg3", "msg4", "msg5"]
    results = await process_messages_async(messages)
    return results

# 4. Confidence threshold tuning
from rizk.sdk.decorators import guardrails

@guardrails(confidence_threshold=0.8)  # Higher threshold = fewer evaluations
def optimized_function(message):
    return f"Processed: {message}"
```

### 3. Memory Performance Issues

**Symptoms**:
- Growing memory usage over time
- Slow performance with large conversation histories
- Cache bloat

**Diagnosis**:
```python
import psutil
import gc
from rizk.sdk.utils.framework_detection import clear_detection_cache

def memory_usage_mb():
    """Get current memory usage in MB."""
    return psutil.Process().memory_info().rss / 1024 / 1024

print(f"Initial memory: {memory_usage_mb():.1f} MB")

# Run your operations
for i in range(1000):
    # Your Rizk operations here
    pass

print(f"After operations: {memory_usage_mb():.1f} MB")

# Check for cache sizes
try:
    from rizk.sdk.utils.framework_detection import get_detection_cache_stats
    cache_stats = get_detection_cache_stats()
    print(f"Detection cache size: {cache_stats['size']}")
except:
    pass

# Manual cleanup
gc.collect()
clear_detection_cache()
print(f"After cleanup: {memory_usage_mb():.1f} MB")
```

**Solutions**:
```python
import os

# 1. Configure cache limits
os.environ["RIZK_FRAMEWORK_CACHE_SIZE"] = "2000"  # Limit cache size
os.environ["RIZK_LLM_CACHE_SIZE"] = "5000"

# 2. Enable automatic cleanup
os.environ["RIZK_STATE_TTL_SECONDS"] = "300"  # 5 minutes

# 3. Regular cache maintenance
import schedule
import time

def cleanup_caches():
    """Periodic cache cleanup."""
    from rizk.sdk.utils.framework_detection import clear_detection_cache
    clear_detection_cache()
    gc.collect()
    print(f"Cache cleanup completed. Memory: {memory_usage_mb():.1f} MB")

# Schedule cleanup every hour
schedule.every().hour.do(cleanup_caches)

# 4. Use context managers for temporary operations
from contextlib import contextmanager

@contextmanager
def temporary_rizk_operation():
    """Context manager that cleans up after operations."""
    try:
        yield
    finally:
        gc.collect()

with temporary_rizk_operation():
    # Your operations here
    pass
```

### 4. Adapter Registration Overhead

**Symptoms**:
- Slow SDK initialization
- Import time delays
- Multiple adapter registrations

**Diagnosis**:
```python
import time
from rizk.sdk import Rizk

# Measure initialization time
start = time.time()
Rizk.init(app_name="PerfTest")
init_time = (time.time() - start) * 1000

print(f"SDK initialization took: {init_time:.2f}ms")

# Check registered adapters
from rizk.sdk.utils.framework_registry import FrameworkRegistry, LLMClientRegistry

framework_adapters = FrameworkRegistry.get_all_framework_names()
llm_adapters = LLMClientRegistry.get_all_adapter_instances()

print(f"Framework adapters: {len(framework_adapters)}")
print(f"LLM adapters: {len(llm_adapters)}")
```

**Solutions**:
```python
# 1. Lazy initialization
import os
os.environ["RIZK_LAZY_LOADING"] = "true"

# 2. Initialize only once per application
# Store the Rizk instance and reuse it
_rizk_instance = None

def get_rizk_instance():
    global _rizk_instance
    if _rizk_instance is None:
        _rizk_instance = Rizk.init(app_name="MyApp")
    return _rizk_instance

# 3. Disable unused features during initialization
Rizk.init(
    app_name="FastApp",
    telemetry_enabled=False,  # Skip telemetry setup
    disable_batch=True,       # Skip batch processing setup
)

# 4. Profile initialization components
import cProfile

def profile_initialization():
    profiler = cProfile.Profile()
    profiler.enable()
    
    Rizk.init(app_name="ProfiledApp")
    
    profiler.disable()
    
    # Print top time consumers
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    stats.print_stats(10)

profile_initialization()
```

## âš¡ Performance Optimization Techniques

### 1. Caching Strategies

```python
# Framework detection caching
from rizk.sdk.utils.framework_detection import detect_framework_cached

# Use cached version for repeated calls
framework = detect_framework_cached()  # Much faster after first call

# Custom LRU cache for your functions
from functools import lru_cache

@lru_cache(maxsize=1000)
def expensive_operation(input_data):
    # Your expensive computation
    return result

# Redis caching for distributed systems
from rizk.sdk.cache.redis import RedisCacheAdapter

cache = RedisCacheAdapter()
cache.set("key", "value", ttl=3600)  # 1 hour TTL
```

### 2. Async Processing

```python
import asyncio
from rizk.sdk.decorators import workflow

@workflow(name="async_workflow")
async def async_function(data):
    # Async processing
    await asyncio.sleep(0.1)  # Simulate async work
    return f"Processed: {data}"

# Batch async processing
async def process_batch(items):
    tasks = [async_function(item) for item in items]
    results = await asyncio.gather(*tasks)
    return results

# Run async batch
items = ["item1", "item2", "item3"]
results = asyncio.run(process_batch(items))
```

### 3. Configuration Tuning

```python
import os

# Performance-optimized configuration
performance_config = {
    # Framework detection
    "RIZK_FRAMEWORK_CACHE_SIZE": "5000",
    "RIZK_LAZY_LOADING": "true",
    
    # Guardrails
    "RIZK_LLM_CACHE_SIZE": "10000",
    "RIZK_STATE_TTL_SECONDS": "300",
    
    # Telemetry (disable for max performance)
    "RIZK_TRACING_ENABLED": "false",
    "RIZK_METRICS_ENABLED": "false",
    "RIZK_TELEMETRY": "false",
    
    # Memory management
    "RIZK_DEBUG": "false",  # Disable in production
    "RIZK_VERBOSE": "false",
}

# Apply configuration
for key, value in performance_config.items():
    os.environ[key] = value

from rizk.sdk import Rizk
Rizk.init(
    app_name="HighPerformanceApp",
    disable_batch=True,  # Reduce latency
)
```

## ðŸ“ˆ Performance Benchmarking

### Benchmark Suite

```python
import time
import statistics
from typing import List, Callable

class RizkBenchmark:
    """Benchmark suite for Rizk SDK operations."""
    
    def __init__(self, warmup_runs: int = 10, benchmark_runs: int = 100):
        self.warmup_runs = warmup_runs
        self.benchmark_runs = benchmark_runs
    
    def benchmark_function(self, func: Callable, *args, **kwargs) -> dict:
        """Benchmark a function and return statistics."""
        
        # Warmup runs
        for _ in range(self.warmup_runs):
            func(*args, **kwargs)
        
        # Benchmark runs
        times = []
        for _ in range(self.benchmark_runs):
            start = time.time()
            func(*args, **kwargs)
            times.append((time.time() - start) * 1000)  # Convert to ms
        
        return {
            "mean_ms": statistics.mean(times),
            "median_ms": statistics.median(times),
            "std_dev_ms": statistics.stdev(times) if len(times) > 1 else 0,
            "min_ms": min(times),
            "max_ms": max(times),
            "p95_ms": sorted(times)[int(len(times) * 0.95)],
            "p99_ms": sorted(times)[int(len(times) * 0.99)],
            "total_runs": len(times)
        }
    
    def run_comprehensive_benchmark(self):
        """Run comprehensive Rizk SDK benchmark."""
        
        print("ðŸš€ Rizk SDK Performance Benchmark")
        print("=" * 50)
        
        from rizk.sdk.utils.framework_detection import detect_framework, detect_framework_cached
        from rizk.sdk.decorators import workflow
        from rizk.sdk.guardrails.engine import GuardrailsEngine
        
        # 1. Framework Detection Benchmark
        print("\n1. Framework Detection:")
        
        # Uncached detection
        uncached_stats = self.benchmark_function(detect_framework)
        print(f"  Uncached: {uncached_stats['mean_ms']:.2f}ms Â± {uncached_stats['std_dev_ms']:.2f}ms")
        
        # Cached detection
        cached_stats = self.benchmark_function(detect_framework_cached)
        print(f"  Cached:   {cached_stats['mean_ms']:.2f}ms Â± {cached_stats['std_dev_ms']:.2f}ms")
        print(f"  Speedup:  {uncached_stats['mean_ms'] / cached_stats['mean_ms']:.1f}x")
        
        # 2. Decorator Application Benchmark
        print("\n2. Decorator Application:")
        
        @workflow(name="benchmark_workflow")
        def simple_function():
            return "result"
        
        decorator_stats = self.benchmark_function(simple_function)
        print(f"  Decorated function: {decorator_stats['mean_ms']:.2f}ms Â± {decorator_stats['std_dev_ms']:.2f}ms")
        
        # 3. Guardrails Benchmark
        print("\n3. Guardrails Evaluation:")
        
        try:
            engine = GuardrailsEngine.get_instance()
            
            # Simple message
            simple_stats = self.benchmark_function(engine.evaluate, "Hello world")
            print(f"  Simple message: {simple_stats['mean_ms']:.2f}ms Â± {simple_stats['std_dev_ms']:.2f}ms")
            
            # Complex message
            complex_msg = "This is a longer message with more content to evaluate"
            complex_stats = self.benchmark_function(engine.evaluate, complex_msg)
            print(f"  Complex message: {complex_stats['mean_ms']:.2f}ms Â± {complex_stats['std_dev_ms']:.2f}ms")
            
        except Exception as e:
            print(f"  Guardrails benchmark failed: {e}")
        
        print("\nðŸ“Š Performance Summary:")
        print(f"  Framework detection is {uncached_stats['mean_ms'] / cached_stats['mean_ms']:.1f}x faster when cached")
        print(f"  Decorator overhead: ~{decorator_stats['mean_ms']:.2f}ms per call")

# Run benchmark
benchmark = RizkBenchmark()
benchmark.run_comprehensive_benchmark()
```

## ðŸŽ¯ Performance Best Practices

### 1. Application Startup
- Initialize Rizk SDK once during application startup
- Use lazy loading for better startup times
- Pre-warm caches during initialization

### 2. Framework Detection
- Use explicit framework specification when possible
- Enable caching for repeated detection calls
- Clear detection cache periodically in long-running applications

### 3. Guardrails Optimization
- Tune confidence thresholds to reduce unnecessary evaluations
- Use async processing for high-throughput scenarios
- Implement proper caching for repeated policy evaluations

### 4. Memory Management
- Monitor memory usage in production
- Implement periodic cache cleanup
- Use appropriate TTL values for cached data

### 5. Configuration
- Disable unused features in production
- Use environment variables for performance tuning
- Profile your specific use case to identify bottlenecks

By following these guidelines and using the provided tools, you can optimize Rizk SDK performance for your specific use case and requirements.


