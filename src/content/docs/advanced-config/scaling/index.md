---
title: "Scaling Guide"
description: "Scaling Guide"
---

# Scaling Guide

Comprehensive guide for scaling Rizk SDK deployments from single instance to enterprise-scale distributed systems, based on real-world patterns from production deployments.

## Scaling Overview

Rizk SDK supports multiple scaling patterns:

- **Horizontal Scaling**: Multiple instances with shared Redis cache
- **Vertical Scaling**: Single instance with optimized resource allocation
- **Multi-Region Deployment**: Distributed deployments across regions
- **Auto-Scaling**: Dynamic scaling based on load
- **Load Balancing**: Intelligent request distribution
- **Cache Distribution**: Multi-tier distributed caching

## Single Instance to Multi-Instance

### 1. Shared Cache Configuration

```python
from rizk.sdk.cache.cache_hierarchy import CacheHierarchy, CacheHierarchyConfig
from rizk.sdk.cache.redis_adapter import RedisConfig

# Multi-instance shared cache configuration
def create_shared_cache_config(instance_id: str) -> CacheHierarchyConfig:
    """Create cache configuration for multi-instance deployment."""
    
    redis_config = RedisConfig(
        url=os.getenv("REDIS_CLUSTER_URL", "redis://redis-cluster:6379"),
        max_connections=100,  # Per instance
        socket_timeout=1.0,
        socket_connect_timeout=2.0,
        retry_on_timeout=True,
        retry_attempts=3,
        enable_cluster=True,  # Critical for multi-instance
        cluster_nodes=[
            "redis-node1:6379",
            "redis-node2:6379", 
            "redis-node3:6379"
        ],
        key_prefix=f"rizk:{instance_id}:",  # Instance-specific prefix
        default_ttl=1800
    )
    
    return CacheHierarchyConfig(
        # L1: Instance-local cache
        l1_enabled=True,
        l1_max_size=10000,  # Smaller per instance
        l1_ttl_seconds=300,
        
        # L2: Shared Redis cluster
        l2_enabled=True,
        l2_redis_config=redis_config,
        l2_ttl_seconds=3600,
        l2_fallback_on_error=True,
        
        # Multi-instance optimizations
        async_write_behind=True,
        promotion_threshold=2,  # Less aggressive promotion
        
        # Instance coordination
        instance_id=instance_id,
        enable_instance_coordination=True,
        coordination_interval=30
    )

# Initialize with instance-specific configuration
instance_id = os.getenv("INSTANCE_ID", f"instance-{os.getpid()}")
cache_config = create_shared_cache_config(instance_id)
cache_hierarchy = CacheHierarchy(cache_config)
```

### 2. Load Balancer Configuration

```nginx
# nginx.conf for Rizk SDK load balancing
upstream rizk_backend {
    # Health check enabled
    server rizk-instance-1:8000 max_fails=3 fail_timeout=30s;
    server rizk-instance-2:8000 max_fails=3 fail_timeout=30s;
    server rizk-instance-3:8000 max_fails=3 fail_timeout=30s;
    
    # Load balancing method
    least_conn;  # Route to instance with fewest connections
    
    # Session persistence for stateful operations
    ip_hash;  # Optional: sticky sessions
    
    # Health check
    keepalive 32;
}

server {
    listen 80;
    server_name api.yourcompany.com;
    
    # Rate limiting
    limit_req_zone $binary_remote_addr zone=rizk_rate_limit:10m rate=100r/s;
    limit_req zone=rizk_rate_limit burst=200 nodelay;
    
    location / {
        proxy_pass http://rizk_backend;
        
        # Load balancing headers
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_set_header X-Instance-ID $upstream_addr;
        
        # Streaming support
        proxy_buffering off;
        proxy_cache off;
        proxy_read_timeout 300s;
        proxy_send_timeout 300s;
        
        # Health check
        proxy_next_upstream error timeout invalid_header http_500 http_502 http_503;
        proxy_next_upstream_tries 3;
        proxy_next_upstream_timeout 10s;
    }
    
    # Health check endpoint
    location /health {
        proxy_pass http://rizk_backend/health;
        access_log off;
    }
    
    # Metrics endpoint (internal only)
    location /metrics {
        allow 10.0.0.0/8;   # Internal network only
        allow 172.16.0.0/12;
        allow 192.168.0.0/16;
        deny all;
        
        proxy_pass http://rizk_backend/metrics;
    }
}
```

## Kubernetes Scaling

### 1. Horizontal Pod Autoscaler

```yaml
# hpa.yaml - Horizontal Pod Autoscaler
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: rizk-app-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: rizk-app
  minReplicas: 3
  maxReplicas: 50
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
  - type: Pods
    pods:
      metric:
        name: rizk_requests_per_second
      target:
        type: AverageValue
        averageValue: "100"
  behavior:
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Percent
        value: 50
        periodSeconds: 60
    scaleUp:
      stabilizationWindowSeconds: 60
      policies:
      - type: Percent
        value: 100
        periodSeconds: 15
      - type: Pods
        value: 5
        periodSeconds: 15
      selectPolicy: Max
```

### 2. Deployment with Resource Optimization

```yaml
# deployment.yaml - Optimized for scaling
apiVersion: apps/v1
kind: Deployment
metadata:
  name: rizk-app
  labels:
    app: rizk-app
spec:
  replicas: 3
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 2
      maxUnavailable: 1
  selector:
    matchLabels:
      app: rizk-app
  template:
    metadata:
      labels:
        app: rizk-app
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "8080"
        prometheus.io/path: "/metrics"
    spec:
      containers:
      - name: rizk-app
        image: your-company/rizk-app:latest
        ports:
        - containerPort: 8000
          name: http
        - containerPort: 8080
          name: metrics
        
        # Resource allocation for scaling
        resources:
          requests:
            memory: "256Mi"
            cpu: "200m"
          limits:
            memory: "1Gi"
            cpu: "1000m"
        
        # Environment configuration
        env:
        - name: RIZK_API_KEY
          valueFrom:
            secretKeyRef:
              name: rizk-secrets
              key: api-key
        - name: REDIS_URL
          value: "redis://redis-cluster:6379"
        - name: INSTANCE_ID
          valueFrom:
            fieldRef:
              fieldPath: metadata.name
        - name: RIZK_FRAMEWORK_CACHE_SIZE
          value: "10000"  # Optimized for multiple instances
        
        # Health checks
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
          timeoutSeconds: 5
          failureThreshold: 3
        
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
          timeoutSeconds: 3
          failureThreshold: 2
        
        # Startup probe for slow initialization
        startupProbe:
          httpGet:
            path: /health
            port: 8000
          failureThreshold: 30
          periodSeconds: 10
      
      # Pod disruption budget
      affinity:
        podAntiAffinity:
          preferredDuringSchedulingIgnoredDuringExecution:
          - weight: 100
            podAffinityTerm:
              labelSelector:
                matchExpressions:
                - key: app
                  operator: In
                  values:
                  - rizk-app
              topologyKey: kubernetes.io/hostname
```

## Auto-Scaling Strategies

### 1. Custom Metrics for Scaling

```python
from typing import Dict, Any
import time
import psutil
from dataclasses import dataclass

@dataclass
class ScalingMetrics:
    """Metrics used for auto-scaling decisions."""
    cpu_percent: float
    memory_percent: float
    requests_per_second: float
    cache_hit_rate: float
    error_rate: float
    response_time_ms: float
    active_connections: int

class AutoScalingController:
    """Control auto-scaling based on Rizk SDK metrics."""
    
    def __init__(self):
        self.metrics_history = []
        self.scaling_thresholds = {
            "scale_up": {
                "cpu_percent": 70,
                "memory_percent": 80,
                "requests_per_second": 100,
                "response_time_ms": 1000,
                "error_rate": 5
            },
            "scale_down": {
                "cpu_percent": 30,
                "memory_percent": 40,
                "requests_per_second": 20,
                "response_time_ms": 200,
                "error_rate": 1
            }
        }
        
    def collect_metrics(self) -> ScalingMetrics:
        """Collect current system and Rizk metrics."""
        # System metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        
        # Rizk-specific metrics (would be collected from SDK)
        cache_stats = cache_hierarchy.get_stats() if 'cache_hierarchy' in globals() else {}
        
        metrics = ScalingMetrics(
            cpu_percent=cpu_percent,
            memory_percent=memory_percent,
            requests_per_second=self._get_requests_per_second(),
            cache_hit_rate=cache_stats.get("overall_hit_rate", 0),
            error_rate=self._get_error_rate(),
            response_time_ms=self._get_avg_response_time(),
            active_connections=self._get_active_connections()
        )
        
        self.metrics_history.append(metrics)
        
        # Keep only recent metrics
        if len(self.metrics_history) > 100:
            self.metrics_history = self.metrics_history[-100:]
        
        return metrics
    
    def should_scale_up(self, metrics: ScalingMetrics) -> bool:
        """Determine if scaling up is needed."""
        thresholds = self.scaling_thresholds["scale_up"]
        
        conditions = [
            metrics.cpu_percent > thresholds["cpu_percent"],
            metrics.memory_percent > thresholds["memory_percent"],
            metrics.requests_per_second > thresholds["requests_per_second"],
            metrics.response_time_ms > thresholds["response_time_ms"],
            metrics.error_rate > thresholds["error_rate"]
        ]
        
        # Scale up if any 2 conditions are met
        return sum(conditions) >= 2
    
    def should_scale_down(self, metrics: ScalingMetrics) -> bool:
        """Determine if scaling down is safe."""
        thresholds = self.scaling_thresholds["scale_down"]
        
        # Only scale down if ALL conditions are met for safety
        conditions = [
            metrics.cpu_percent < thresholds["cpu_percent"],
            metrics.memory_percent < thresholds["memory_percent"],
            metrics.requests_per_second < thresholds["requests_per_second"],
            metrics.response_time_ms < thresholds["response_time_ms"],
            metrics.error_rate < thresholds["error_rate"]
        ]
        
        return all(conditions)
    
    def get_scaling_recommendation(self) -> Dict[str, Any]:
        """Get scaling recommendation based on current metrics."""
        if len(self.metrics_history) < 3:
            return {"action": "wait", "reason": "Insufficient metrics history"}
        
        current_metrics = self.metrics_history[-1]
        
        # Check recent trend
        recent_metrics = self.metrics_history[-3:]
        avg_cpu = sum(m.cpu_percent for m in recent_metrics) / len(recent_metrics)
        avg_memory = sum(m.memory_percent for m in recent_metrics) / len(recent_metrics)
        avg_response_time = sum(m.response_time_ms for m in recent_metrics) / len(recent_metrics)
        
        if self.should_scale_up(current_metrics):
            return {
                "action": "scale_up",
                "reason": f"High resource usage: CPU {avg_cpu:.1f}%, Memory {avg_memory:.1f}%, Response Time {avg_response_time:.1f}ms",
                "recommended_replicas": self._calculate_scale_up_replicas(current_metrics)
            }
        elif self.should_scale_down(current_metrics):
            return {
                "action": "scale_down", 
                "reason": f"Low resource usage: CPU {avg_cpu:.1f}%, Memory {avg_memory:.1f}%",
                "recommended_replicas": self._calculate_scale_down_replicas(current_metrics)
            }
        else:
            return {
                "action": "maintain",
                "reason": "Metrics within acceptable range"
            }
    
    def _calculate_scale_up_replicas(self, metrics: ScalingMetrics) -> int:
        """Calculate how many replicas to add."""
        # Simple calculation based on CPU usage
        if metrics.cpu_percent > 90:
            return 3  # Aggressive scaling for very high CPU
        elif metrics.cpu_percent > 80:
            return 2
        else:
            return 1
    
    def _calculate_scale_down_replicas(self, metrics: ScalingMetrics) -> int:
        """Calculate how many replicas to remove."""
        # Conservative scaling down
        return 1  # Remove one at a time
    
    def _get_requests_per_second(self) -> float:
        """Get current requests per second (implementation specific)."""
        # This would typically come from your web framework metrics
        return 50.0  # Placeholder
    
    def _get_error_rate(self) -> float:
        """Get current error rate percentage."""
        # This would typically come from your application metrics
        return 2.0  # Placeholder
    
    def _get_avg_response_time(self) -> float:
        """Get average response time in milliseconds."""
        # This would typically come from your application metrics
        return 250.0  # Placeholder
    
    def _get_active_connections(self) -> int:
        """Get number of active connections."""
        # This would typically come from your web server metrics
        return 50  # Placeholder

# Usage
scaling_controller = AutoScalingController()

def monitor_and_scale():
    """Monitor metrics and provide scaling recommendations."""
    metrics = scaling_controller.collect_metrics()
    recommendation = scaling_controller.get_scaling_recommendation()
    
    print(f"Current Metrics:")
    print(f"  CPU: {metrics.cpu_percent:.1f}%")
    print(f"  Memory: {metrics.memory_percent:.1f}%")
    print(f"  RPS: {metrics.requests_per_second:.1f}")
    print(f"  Response Time: {metrics.response_time_ms:.1f}ms")
    print(f"  Cache Hit Rate: {metrics.cache_hit_rate:.1f}%")
    
    print(f"\nScaling Recommendation: {recommendation['action']}")
    print(f"Reason: {recommendation['reason']}")
    
    if "recommended_replicas" in recommendation:
        print(f"Recommended Replica Change: {recommendation['recommended_replicas']}")
    
    return recommendation
```

## Scaling Best Practices

### 1. Scaling Checklist

```python
SCALING_CHECKLIST = {
    "infrastructure": [
        "âœ… Redis cluster configured for high availability",
        "âœ… Load balancer configured with health checks",
        "âœ… Auto-scaling policies defined",
        "âœ… Resource limits and requests configured",
        "âœ… Pod disruption budgets set"
    ],
    "configuration": [
        "âœ… Instance-specific cache prefixes",
        "âœ… Shared cache configuration",
        "âœ… Regional endpoints configured", 
        "âœ… Cross-region sync enabled",
        "âœ… Monitoring and alerting setup"
    ],
    "performance": [
        "âœ… Cache hit rates > 80%",
        "âœ… Response times < 500ms",
        "âœ… Error rates < 1%",
        "âœ… CPU utilization 60-80%",
        "âœ… Memory utilization < 85%"
    ],
    "reliability": [
        "âœ… Multi-AZ deployment",
        "âœ… Graceful shutdown handling",
        "âœ… Circuit breakers implemented",
        "âœ… Retry logic configured",
        "âœ… Backup and recovery tested"
    ]
}

def validate_scaling_readiness() -> Dict[str, Any]:
    """Validate readiness for scaling deployment."""
    print("ðŸš€ Validating scaling readiness...")
    
    # This would include actual checks
    validation_results = {
        "ready": True,
        "warnings": [],
        "errors": []
    }
    
    # Example checks
    redis_health = check_redis_cluster_health()
    if not redis_health["healthy"]:
        validation_results["errors"].append("Redis cluster not healthy")
        validation_results["ready"] = False
    
    cache_hit_rate = get_current_cache_hit_rate()
    if cache_hit_rate < 70:
        validation_results["warnings"].append(f"Cache hit rate low: {cache_hit_rate:.1f}%")
    
    return validation_results

def check_redis_cluster_health() -> Dict[str, Any]:
    """Check Redis cluster health (placeholder)."""
    return {"healthy": True, "nodes": 3, "status": "ok"}

def get_current_cache_hit_rate() -> float:
    """Get current cache hit rate (placeholder)."""
    return 85.0
```

### 2. Deployment Strategy

```bash
#!/bin/bash
# deploy-scaled.sh - Deploy scaled Rizk SDK application

set -e

echo "ðŸš€ Starting scaled deployment..."

# 1. Validate prerequisites
echo "ðŸ“‹ Validating prerequisites..."
kubectl cluster-info
kubectl get nodes
kubectl get pv  # Check persistent volumes

# 2. Deploy Redis cluster first
echo "ðŸ”§ Deploying Redis cluster..."
kubectl apply -f redis-cluster.yaml
kubectl wait --for=condition=ready pod -l app=redis-cluster --timeout=300s

# 3. Deploy application with minimal replicas
echo "ðŸš€ Deploying application (minimal replicas)..."
kubectl apply -f deployment.yaml
kubectl wait --for=condition=available deployment/rizk-app --timeout=300s

# 4. Run health checks
echo "ðŸ¥ Running health checks..."
kubectl get pods -l app=rizk-app
kubectl exec deployment/rizk-app -- curl -f http://localhost:8000/health

# 5. Deploy auto-scaling
echo "ðŸ“ˆ Enabling auto-scaling..."
kubectl apply -f hpa.yaml
kubectl apply -f pdb.yaml

# 6. Configure monitoring
echo "ðŸ“Š Setting up monitoring..."
kubectl apply -f monitoring.yaml

# 7. Run load test to verify scaling
echo "ðŸ§ª Running scaling verification..."
kubectl run load-test --image=busybox --rm -it --restart=Never -- \
    sh -c 'for i in $(seq 1 100); do wget -q -O- http://rizk-app-service/health; done'

# 8. Monitor scaling behavior
echo "ðŸ‘€ Monitoring scaling behavior..."
kubectl get hpa rizk-app-hpa --watch

echo "âœ… Scaled deployment completed successfully!"
```

## Next Steps

1. **[Production Setup](production-setup.md)** - Deploy your scaled architecture
2. **[Performance Tuning](performance-tuning.md)** - Optimize for scale
3. **[Security Best Practices](security.md)** - Secure your scaled deployment

---

**Scaling Implementation Checklist**

âœ… Multi-instance configuration ready  
âœ… Shared cache infrastructure deployed  
âœ… Load balancing configured  
âœ… Auto-scaling policies defined  
âœ… Multi-region strategy planned  
âœ… Monitoring and alerting setup  
âœ… Performance benchmarks established  
âœ… Disaster recovery tested  

*Enterprise-scale LLM governance architecture* 

