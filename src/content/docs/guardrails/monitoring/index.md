---
title: "Guardrails Monitoring"
description: "Guardrails Monitoring"
---

# Guardrails Monitoring

This guide covers monitoring and analyzing your guardrails system to ensure optimal performance, effectiveness, and user experience. Proper monitoring helps you understand policy impact, identify issues, and optimize your configuration.

## Overview

Rizk provides comprehensive monitoring capabilities for guardrails:

- **Real-time Metrics**: Track performance and decisions as they happen
- **Policy Analytics**: Understand which policies are most effective
- **Performance Monitoring**: Monitor latency, cache hit rates, and resource usage
- **Compliance Reporting**: Generate reports for regulatory requirements
- **Alerting**: Get notified when thresholds are exceeded

## Key Metrics

### Decision Metrics

Track guardrail decisions and their outcomes:

```python
from rizk.sdk.analytics import GuardrailAnalytics

analytics = GuardrailAnalytics()

# Get decision metrics for the last 24 hours
metrics = analytics.get_decision_metrics(time_range="24h")

print(f"Total decisions: {metrics.total_decisions}")
print(f"Allowed: {metrics.allowed_count} ({metrics.allowed_percentage:.1f}%)")
print(f"Blocked: {metrics.blocked_count} ({metrics.blocked_percentage:.1f}%)")
print(f"Average confidence: {metrics.avg_confidence:.2f}")
print(f"Decisions per minute: {metrics.decisions_per_minute:.1f}")
```

### Performance Metrics

Monitor system performance and resource usage:

```python
# Get performance metrics
perf_metrics = analytics.get_performance_metrics(time_range="24h")

print(f"Average latency: {perf_metrics.avg_latency_ms:.1f}ms")
print(f"95th percentile latency: {perf_metrics.p95_latency_ms:.1f}ms")
print(f"99th percentile latency: {perf_metrics.p99_latency_ms:.1f}ms")
print(f"Cache hit rate: {perf_metrics.cache_hit_rate:.1f}%")
print(f"Error rate: {perf_metrics.error_rate:.2f}%")
print(f"Throughput: {perf_metrics.requests_per_second:.1f} req/s")
```

### Policy Effectiveness

Measure how well your policies are working:

```python
# Get policy effectiveness metrics
policy_metrics = analytics.get_policy_effectiveness(
    policy_id="content_moderation",
    time_range="7d"
)

print(f"Policy triggers: {policy_metrics.trigger_count}")
print(f"True positives: {policy_metrics.true_positives}")
print(f"False positives: {policy_metrics.false_positives}")
print(f"Precision: {policy_metrics.precision:.2f}")
print(f"Recall: {policy_metrics.recall:.2f}")
print(f"F1 Score: {policy_metrics.f1_score:.2f}")
```

## Real-Time Monitoring

### Dashboard Setup

Set up a real-time monitoring dashboard:

```python
from rizk.sdk.monitoring import GuardrailDashboard

# Create dashboard instance
dashboard = GuardrailDashboard()

# Configure dashboard widgets
dashboard.add_widget("decision_rate", {
    "title": "Decisions per Minute",
    "type": "line_chart",
    "metric": "decisions_per_minute",
    "time_range": "1h"
})

dashboard.add_widget("block_rate", {
    "title": "Block Rate",
    "type": "gauge",
    "metric": "block_percentage",
    "time_range": "5m",
    "alert_threshold": 20  # Alert if block rate > 20%
})

dashboard.add_widget("latency", {
    "title": "Response Latency",
    "type": "histogram",
    "metric": "latency_distribution",
    "time_range": "1h"
})

dashboard.add_widget("top_policies", {
    "title": "Most Active Policies",
    "type": "table",
    "metric": "policy_trigger_counts",
    "time_range": "24h",
    "limit": 10
})

# Start dashboard server
dashboard.start(port=8080)
```

## Best Practices

### 1. Monitor Key Metrics

Focus on the most important metrics:

```python
# âœ… Essential metrics to monitor
essential_metrics = [
    "decision_rate",      # Throughput
    "block_rate",         # Policy effectiveness
    "avg_latency",        # Performance
    "cache_hit_rate",     # Efficiency
    "error_rate",         # Reliability
    "user_satisfaction"   # User experience
]

for metric in essential_metrics:
    analytics.add_to_dashboard(metric, alert_threshold=True)
```

### 2. Set Appropriate Thresholds

Configure meaningful alert thresholds:

```python
# âœ… Contextual thresholds
thresholds = {
    "block_rate": {
        "warning": 15,    # 15% block rate
        "critical": 30    # 30% block rate
    },
    "avg_latency": {
        "warning": 200,   # 200ms average latency
        "critical": 500   # 500ms average latency
    },
    "cache_hit_rate": {
        "warning": 70,    # 70% cache hit rate
        "critical": 50    # 50% cache hit rate
    }
}
```

### 3. Regular Review and Analysis

Establish regular monitoring reviews:

```python
# âœ… Weekly monitoring review
def weekly_monitoring_review():
    """Perform weekly analysis of guardrail performance."""
    
    # Get weekly metrics
    metrics = analytics.get_metrics(time_range="7d")
    
    # Check for trends
    trends = analytics.get_trends(time_range="30d")
    
    # Generate insights
    insights = analytics.generate_insights(metrics, trends)
    
    # Create review report
    report = {
        "period": "week",
        "key_metrics": metrics,
        "trends": trends,
        "insights": insights,
        "recommendations": analytics.get_recommendations(insights)
    }
    
    return report

# Schedule weekly reviews
import schedule
schedule.every().monday.at("09:00").do(weekly_monitoring_review)
```

## Next Steps

1. **[Configuration](configuration.md)** - Optimize your guardrails configuration
2. **[Policy Enforcement](policy-enforcement.md)** - Understand policy decisions
3. **[Using Guardrails](using-guardrails.md)** - Implement guardrails effectively
4. **[Overview](overview.md)** - Understand the guardrails system

---

Effective monitoring is essential for maintaining optimal guardrail performance. Regular analysis and proactive optimization ensure your guardrails continue to provide value while maintaining excellent user experience.


