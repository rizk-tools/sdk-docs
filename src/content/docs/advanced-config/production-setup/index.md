---
title: "Production Setup Guide"
description: "Production Setup Guide"
---

# Production Setup Guide

This guide covers enterprise-grade deployment and configuration of Rizk SDK for production environments, based on proven patterns from production deployments.

## Quick Production Checklist

âœ… **Security**: API keys via environment variables, secure endpoints  
âœ… **Performance**: Redis caching, optimized batch sizes, connection pooling  
âœ… **Monitoring**: Distributed tracing, performance metrics, error tracking  
âœ… **Reliability**: Retry logic, fallback mechanisms, health checks  
âœ… **Scalability**: Multi-instance deployment, load balancing  

## Core Production Configuration

### 1. Environment-Based Configuration

```python
# production_config.py
import os
from rizk.sdk import Rizk
from rizk.sdk.config import RizkConfig

def create_production_config() -> RizkConfig:
    """Create production-ready configuration."""
    return RizkConfig(
        # Core settings
        app_name=os.getenv("RIZK_APP_NAME", "ProductionApp"),
        api_key=os.getenv("RIZK_API_KEY"),  # Required
        
        # OpenTelemetry settings (uses https://api.rizk.tools by default)
        tracing_enabled=True,
        trace_content=False,  # Disable for privacy in production
        metrics_enabled=True,
        
        # Performance settings
        lazy_loading=True,
        framework_detection_cache_size=5000,  # Larger cache for production
        
        # Security settings
        policy_enforcement=True,
        policies_path=os.getenv("RIZK_POLICIES_PATH", "/app/policies"),
        
        # Debugging (disabled in production)
        debug_mode=False,
        verbose=False,
        logging_enabled=False,  # Use structured logging instead
    )

# Initialize with production config
config = create_production_config()
rizk = Rizk.init(**config.to_dict())

> **Note**: To use a custom OpenTelemetry endpoint instead of Rizk's default, add `opentelemetry_endpoint=os.getenv("RIZK_OPENTELEMETRY_ENDPOINT", "https://your-otlp-endpoint.com")` to the RizkConfig.
```

### 2. Environment Variables

Set these environment variables in your production environment:

```bash
# Core Configuration
export RIZK_API_KEY="rizk_prod_your_api_key_here"
export RIZK_APP_NAME="YourApp-Production"

# OpenTelemetry (uses https://api.rizk.tools by default)
export RIZK_TRACING_ENABLED="true"
export RIZK_TRACE_CONTENT="false"  # Privacy in production
export RIZK_METRICS_ENABLED="true"

# Performance
export RIZK_LAZY_LOADING="true"
export RIZK_FRAMEWORK_CACHE_SIZE="5000"

# Policies
export RIZK_POLICY_ENFORCEMENT="true"
export RIZK_POLICIES_PATH="/app/policies"

# Security
export RIZK_DEBUG="false"
export RIZK_VERBOSE="false"
export RIZK_LOGGING_ENABLED="false"
```

> **Note**: To use a custom OTLP endpoint instead of Rizk's default, add:
> ```bash
> export RIZK_OPENTELEMETRY_ENDPOINT="https://your-otlp-endpoint.com"
> ```

## High-Performance Caching Setup

### Redis Configuration for Production

```python
from rizk.sdk.cache.redis_adapter import RedisAdapter, RedisConfig
from rizk.sdk.cache.cache_hierarchy import CacheHierarchy, CacheHierarchyConfig

# Production Redis configuration
redis_config = RedisConfig(
    url=os.getenv("REDIS_URL", "redis://redis-cluster:6379"),
    max_connections=50,  # Higher for production load
    socket_timeout=2.0,  # Faster timeout
    socket_connect_timeout=3.0,
    retry_on_timeout=True,
    retry_attempts=3,
    enable_cluster=True,  # For high availability
    cluster_nodes=[
        "redis-node1:6379",
        "redis-node2:6379", 
        "redis-node3:6379"
    ],
    key_prefix="prod:rizk:",
    default_ttl=1800  # 30 minutes
)

# Multi-layer cache hierarchy
cache_config = CacheHierarchyConfig(
    # L1: Local memory cache
    l1_enabled=True,
    l1_max_size=20000,  # Larger for production
    l1_ttl_seconds=300,
    
    # L2: Distributed Redis cache
    l2_enabled=True,
    l2_redis_url=redis_config.url,
    l2_ttl_seconds=1800,
    l2_fallback_on_error=True,  # Critical for reliability
    
    # Performance optimizations
    async_write_behind=True,
    promotion_threshold=2,  # Faster promotion
    
    # Monitoring
    metrics_enabled=True,
    health_check_interval=30  # More frequent checks
)

# Initialize cache hierarchy
cache_hierarchy = CacheHierarchy(cache_config)
```

### Cache Performance Monitoring

```python
def monitor_cache_performance():
    """Monitor cache performance in production."""
    stats = cache_hierarchy.get_stats()
    
    # Key metrics to track
    l1_hit_rate = stats["l1_hit_rate"]
    l2_hit_rate = stats["l2_hit_rate"] 
    overall_hit_rate = stats["overall_hit_rate"]
    avg_latency = stats["avg_latency_ms"]
    
    # Alert thresholds
    if overall_hit_rate < 80:
        logger.warning(f"Cache hit rate low: {overall_hit_rate}%")
    
    if avg_latency > 50:
        logger.warning(f"Cache latency high: {avg_latency}ms")
    
    # Log metrics
    logger.info(f"Cache performance: {overall_hit_rate}% hit rate, {avg_latency}ms avg latency")
```

## Streaming Configuration for Production

```python
from rizk.sdk.streaming.types import StreamConfig

# Production streaming configuration
stream_config = StreamConfig(
    # Performance settings
    max_chunk_size=2048,  # Larger chunks for efficiency
    buffer_size=20,  # Larger buffer
    timeout_seconds=60.0,  # Longer timeout for production
    
    # Guardrail settings
    enable_guardrails=True,
    realtime_validation=True,
    validation_interval=2,  # Every 2 chunks for performance
    
    # Caching settings
    enable_caching=True,
    cache_partial_responses=True,
    cache_ttl_seconds=600,  # 10 minutes
    
    # Monitoring
    enable_metrics=True,
    metrics_interval=5.0,  # Every 5 seconds
    
    # Framework-specific optimizations
    framework_specific={
        "openai": {
            "stream_options": {"include_usage": True}
        },
        "anthropic": {
            "max_tokens": 4096
        }
    }
)
```

## Analytics and Monitoring Setup

### Production Analytics Configuration

```python
from rizk.sdk.analytics.processors import (
    RizkHubProcessor,
    FileAnalyticsProcessor,
    PerformanceMonitoringProcessor,
    MetricsAggregationProcessor
)

# Configure analytics processors for production
def setup_production_analytics():
    processors = []
    
    # 1. Rizk Hub processor (primary)
    rizk_hub = RizkHubProcessor(
        api_key=os.getenv("RIZK_API_KEY"),
        api_endpoint="https://api.rizk.tools",
        batch_size=100,  # Larger batches for efficiency
        flush_interval_seconds=15,  # More frequent flushes
        max_retries=5,
        timeout_seconds=30,
        include_blocked_messages=True,
        include_policy_violations=True,
        include_performance_metrics=True
    )
    processors.append(rizk_hub)
    
    # 2. Local file backup (failsafe)
    file_processor = FileAnalyticsProcessor(
        output_dir="/var/log/rizk",
        buffer_size=2000,
        auto_flush_interval=30,
        create_daily_files=True  # Better for log rotation
    )
    processors.append(file_processor)
    
    # 3. Performance monitoring with alerts
    perf_monitor = PerformanceMonitoringProcessor(
        latency_threshold_ms=3000,  # 3 second threshold
        error_rate_threshold=0.05,  # 5% error rate
        memory_threshold_mb=2000,   # 2GB threshold
        monitoring_window_seconds=300
    )
    
    # Add alerting callbacks
    def alert_callback(alert_data):
        # Send to your monitoring system (PagerDuty, Slack, etc.)
        logger.critical(f"RIZK ALERT: {alert_data}")
        # send_to_monitoring_system(alert_data)
    
    perf_monitor.add_alert_callback(alert_callback)
    processors.append(perf_monitor)
    
    # 4. Metrics aggregation
    metrics_agg = MetricsAggregationProcessor(
        aggregation_window_seconds=300,  # 5 minute windows
        keep_raw_data=False,  # Save memory in production
        export_interval_seconds=60
    )
    processors.append(metrics_agg)
    
    return processors
```

## Security Best Practices

### 1. API Key Management

```python
# âŒ Never do this in production
rizk = Rizk.init(
    app_name="MyApp",
    api_key="rizk_hardcoded_key_bad"  # Never hardcode!
)

# âœ… Always use environment variables
rizk = Rizk.init(
    app_name="MyApp",
    api_key=os.getenv("RIZK_API_KEY")  # Secure
)

# âœ… Even better: Use secrets management
import boto3

def get_api_key():
    """Get API key from AWS Secrets Manager."""
    client = boto3.client('secretsmanager')
    response = client.get_secret_value(SecretId='rizk/api-key')
    return response['SecretString']

rizk = Rizk.init(
    app_name="MyApp",
    api_key=get_api_key()
)
```

### 2. Content Privacy

```python
# Production configuration for content privacy
config = RizkConfig(
    trace_content=False,  # Never trace content in production
    debug_mode=False,     # Disable debug logging
    verbose=False,        # Disable verbose logging
    logging_enabled=False # Use structured logging instead
)
```

### 3. Network Security

```python
# Secure OTLP endpoint configuration
config = RizkConfig(
    opentelemetry_endpoint="https://secure-otlp.company.com",
    # Add custom headers for authentication
)

# Initialize with secure headers
rizk = Rizk.init(
    **config.to_dict(),
    headers={
        "Authorization": f"Bearer {os.getenv('OTLP_TOKEN')}",
        "X-Company-ID": os.getenv("COMPANY_ID")
    }
)
```

## Multi-Instance Deployment

### Load Balancer Configuration

```nginx
# nginx.conf for Rizk-enabled applications
upstream rizk_app {
    server app1:8000;
    server app2:8000;  
    server app3:8000;
    
    # Health checks
    keepalive 32;
}

server {
    listen 80;
    server_name your-app.com;
    
    location / {
        proxy_pass http://rizk_app;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        
        # Important for streaming
        proxy_buffering off;
        proxy_cache off;
        
        # Timeouts for long-running LLM requests
        proxy_read_timeout 300s;
        proxy_send_timeout 300s;
    }
}
```

### Container Deployment

```dockerfile
# Dockerfile for production Rizk app
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Production configuration
ENV RIZK_TRACING_ENABLED=true
ENV RIZK_TRACE_CONTENT=false
ENV RIZK_DEBUG=false
ENV RIZK_VERBOSE=false

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "from rizk.sdk import Rizk; print('healthy')"

CMD ["python", "app.py"]
```

### Kubernetes Deployment

```yaml
# k8s-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: rizk-app
spec:
  replicas: 3
  selector:
    matchLabels:
      app: rizk-app
  template:
    metadata:
      labels:
        app: rizk-app
    spec:
      containers:
      - name: app
        image: your-company/rizk-app:latest
        env:
        - name: RIZK_API_KEY
          valueFrom:
            secretKeyRef:
              name: rizk-secrets
              key: api-key
        - name: REDIS_URL
          value: "redis://redis-service:6379"
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: rizk-app-service
spec:
  selector:
    app: rizk-app
  ports:
  - port: 80
    targetPort: 8000
  type: LoadBalancer
```

## Performance Optimization

### 1. Framework Detection Caching

```python
# Optimize framework detection for production
config = RizkConfig(
    framework_detection_cache_size=10000,  # Large cache
    lazy_loading=True  # Load adapters on demand
)
```

### 2. Batch Processing Configuration

```python
# Optimize batch processing
rizk = Rizk.init(
    app_name="ProductionApp",
    disable_batch=False,  # Enable batching for efficiency
    headers={
        "User-Agent": "YourApp/1.0 (Rizk-SDK)"
    }
)
```

### 3. Connection Pooling

```python
# Redis connection pooling for high throughput
redis_config = RedisConfig(
    max_connections=100,  # High connection pool
    socket_timeout=1.0,   # Fast timeouts
    retry_attempts=2      # Quick retries
)
```

## Monitoring and Alerting

### Health Check Endpoints

```python
from flask import Flask, jsonify
from rizk.sdk.cache.cache_hierarchy import CacheHierarchy

app = Flask(__name__)

@app.route('/health')
def health_check():
    """Basic health check."""
    try:
        # Test Rizk SDK health
        config = get_config()
        if not config.api_key:
            return jsonify({"status": "unhealthy", "reason": "No API key"}), 500
        
        return jsonify({"status": "healthy"}), 200
    except Exception as e:
        return jsonify({"status": "unhealthy", "error": str(e)}), 500

@app.route('/ready')
def readiness_check():
    """Readiness check including dependencies."""
    try:
        # Check cache health
        cache_health = cache_hierarchy.health_check()
        if not cache_health["healthy"]:
            return jsonify({
                "status": "not_ready", 
                "cache": cache_health
            }), 503
        
        # Check Redis connectivity
        redis_stats = cache_hierarchy.get_stats()
        
        return jsonify({
            "status": "ready",
            "cache": cache_health,
            "stats": redis_stats
        }), 200
    except Exception as e:
        return jsonify({"status": "not_ready", "error": str(e)}), 503
```

### Production Metrics

```python
def log_production_metrics():
    """Log key production metrics."""
    # Cache performance
    cache_stats = cache_hierarchy.get_stats()
    logger.info(f"Cache hit rate: {cache_stats['overall_hit_rate']}%")
    
    # Performance metrics
    perf_stats = perf_monitor.get_performance_stats()
    logger.info(f"Avg latency: {perf_stats['avg_latency_ms']}ms")
    logger.info(f"Error rate: {perf_stats['error_rate']}%")
    
    # Analytics stats
    analytics_stats = rizk_hub.get_stats()
    logger.info(f"Events sent: {analytics_stats['events_sent']}")
    logger.info(f"Send errors: {analytics_stats['send_errors']}")
```

## Troubleshooting Production Issues

### Common Production Issues

1. **High Latency**
   ```python
   # Check cache performance
   cache_stats = cache_hierarchy.get_stats()
   if cache_stats['overall_hit_rate'] < 70:
       logger.warning("Low cache hit rate - consider cache tuning")
   ```

2. **Memory Usage**
   ```python
   # Monitor cache memory usage
   if cache_stats['l1_size'] > 15000:
       logger.warning("L1 cache approaching limit")
   ```

3. **Redis Connection Issues**
   ```python
   # Check Redis health
   redis_health = cache_hierarchy.health_check()
   if not redis_health['redis_available']:
       logger.error("Redis unavailable - running in degraded mode")
   ```

### Performance Tuning Checklist

- [ ] **Cache hit rate > 80%**
- [ ] **Average latency < 100ms**
- [ ] **Error rate < 1%**
- [ ] **Memory usage stable**
- [ ] **Redis connections healthy**
- [ ] **Analytics data flowing**

## Next Steps

1. **[Performance Tuning](performance-tuning.md)** - Detailed optimization strategies
2. **[Security Best Practices](security.md)** - Comprehensive security guide
3. **[Environment Variables](environment-variables.md)** - Complete configuration reference
4. **[Scaling Guide](scaling.md)** - Multi-region deployment patterns

---

**Production Deployment Checklist**

âœ… Environment variables configured  
âœ… Redis cluster deployed  
âœ… Monitoring and alerting setup  
âœ… Health checks implemented  
âœ… Load balancer configured  
âœ… Container images built  
âœ… Kubernetes manifests deployed  
âœ… Performance baselines established  

*Ready for enterprise-scale LLM governance* 

