---
title: "Guardrails Configuration"
description: "Guardrails Configuration"
---

# Guardrails Configuration

This guide covers all configuration options for Rizk's guardrails system, from basic setup to advanced customization for enterprise deployments.

## Basic Configuration

### Environment Variables

The simplest way to configure guardrails is through environment variables:

```bash
# Core guardrails settings
export RIZK_GUARDRAILS_ENABLED="true"
export RIZK_POLICY_ENFORCEMENT="moderate"  # strict, moderate, lenient

# Policy configuration
export RIZK_POLICIES_PATH="/path/to/custom/policies"
export RIZK_USE_BUILTIN_POLICIES="true"

# Performance settings
export RIZK_GUARDRAILS_CACHE_SIZE="10000"
export RIZK_GUARDRAILS_CACHE_TTL="3600"

# LLM service for complex evaluations
export RIZK_LLM_SERVICE="openai"
export RIZK_LLM_MODEL="gpt-4"

# Logging and debugging
export RIZK_TRACE_POLICY_DECISIONS="false"
export RIZK_GUARDRAILS_VERBOSE="false"
```

### Programmatic Configuration

Configure guardrails programmatically during initialization:

```python
from rizk.sdk import Rizk

rizk = Rizk.init(
    app_name="MyApp",
    api_key="your-api-key",
    
    # Basic guardrails settings
    guardrails_enabled=True,
    policy_enforcement="moderate",
    
    # Performance optimization
    llm_cache_size=10000,
    fast_rules_enabled=True,
    
    # Custom LLM for evaluations
    llm_service="openai",
    llm_model="gpt-4",
    
    # Observability
    trace_content=True,
    metrics_enabled=True
)
```

## Enforcement Levels

### Strict Enforcement

Maximum security with aggressive blocking:

```python
rizk = Rizk.init(
    app_name="HighSecurityApp",
    policy_enforcement="strict",
    
    # Strict-specific settings
    confidence_threshold=0.95,
    block_threshold=0.85,
    allow_edge_cases=False,
    
    # Enable all security layers
    fast_rules_enabled=True,
    llm_fallback_enabled=True,
    pattern_matching_aggressive=True
)
```

**Use Cases:**
- Financial services applications
- Healthcare systems with PHI
- Legal document processing
- Government/defense applications

### Moderate Enforcement (Default)

Balanced security and usability:

```python
rizk = Rizk.init(
    app_name="StandardApp",
    policy_enforcement="moderate",
    
    # Balanced settings
    confidence_threshold=0.8,
    block_threshold=0.9,
    allow_edge_cases=True,
    
    # Standard security layers
    fast_rules_enabled=True,
    llm_fallback_enabled=True,
    context_aware_matching=True
)
```

**Use Cases:**
- Customer service bots
- Content generation tools
- Educational applications
- General business applications

### Lenient Enforcement

Minimal blocking with focus on guidance:

```python
rizk = Rizk.init(
    app_name="CreativeApp",
    policy_enforcement="lenient",
    
    # Lenient settings
    confidence_threshold=0.7,
    block_threshold=0.95,
    allow_edge_cases=True,
    
    # Focus on guidance over blocking
    fast_rules_enabled=False,
    llm_fallback_enabled=True,
    guidance_over_blocking=True
)
```

**Use Cases:**
- Creative writing tools
- Internal development tools
- Research applications
- Brainstorming assistants

## Performance Configuration

### Caching Settings

Optimize performance through intelligent caching:

```python
rizk = Rizk.init(
    app_name="HighPerformanceApp",
    
    # Cache configuration
    llm_cache_size=50000,           # Number of cached evaluations
    cache_ttl_seconds=3600,         # Cache expiration (1 hour)
    cache_hit_optimization=True,    # Optimize for cache hits
    
    # Memory management
    max_memory_usage_mb=512,        # Maximum memory for caches
    cache_cleanup_interval=300,     # Cleanup every 5 minutes
    
    # Distributed caching (enterprise)
    redis_cache_enabled=False,
    redis_cache_url="redis://localhost:6379"
)
```

### Fast Rules Optimization

Configure fast rules for immediate pattern matching:

```python
rizk = Rizk.init(
    app_name="FastApp",
    
    # Fast rules settings
    fast_rules_enabled=True,
    fast_rules_cache_size=5000,
    fast_rules_timeout_ms=10,
    
    # Pattern matching optimization
    regex_compilation_cache=True,
    pattern_matching_threads=4,
    
    # Skip slow evaluations for obvious cases
    skip_llm_for_obvious_blocks=True,
    obvious_block_confidence=0.99
)
```

### LLM Service Configuration

Configure LLM services for complex evaluations:

```python
rizk = Rizk.init(
    app_name="LLMOptimizedApp",
    
    # Primary LLM service
    llm_service="openai",
    llm_model="gpt-4",
    llm_api_key="your-openai-key",
    
    # Fallback LLM service
    fallback_llm_service="anthropic",
    fallback_llm_model="claude-3-sonnet",
    
    # LLM performance settings
    llm_timeout_seconds=30,
    llm_max_retries=3,
    llm_batch_size=10,
    
    # Cost optimization
    llm_cache_aggressive=True,
    use_cheaper_model_for_simple_cases=True,
    simple_case_model="gpt-3.5-turbo"
)
```

## Advanced Configuration

### Multi-Environment Setup

Configure different settings for different environments:

```python
import os

def get_guardrails_config():
    env = os.getenv("ENVIRONMENT", "development")
    
    if env == "production":
        return {
            "policy_enforcement": "strict",
            "confidence_threshold": 0.95,
            "llm_cache_size": 100000,
            "trace_content": False,
            "metrics_enabled": True
        }
    elif env == "staging":
        return {
            "policy_enforcement": "moderate",
            "confidence_threshold": 0.8,
            "llm_cache_size": 10000,
            "trace_content": True,
            "metrics_enabled": True
        }
    else:  # development
        return {
            "policy_enforcement": "lenient",
            "confidence_threshold": 0.7,
            "llm_cache_size": 1000,
            "trace_content": True,
            "verbose": True
        }

# Apply environment-specific configuration
config = get_guardrails_config()
rizk = Rizk.init(app_name="MyApp", **config)
```

### Function-Level Configuration

Override global settings for specific functions:

```python
from rizk.sdk.decorators import guardrails

# Strict enforcement for sensitive functions
@guardrails(
    enforcement_level="strict",
    confidence_threshold=0.95,
    policies=["financial_compliance", "data_privacy"],
    timeout_seconds=60
)
def process_financial_data(data: str) -> str:
    return handle_sensitive_financial_data(data)

# Lenient enforcement for creative functions
@guardrails(
    enforcement_level="lenient",
    confidence_threshold=0.6,
    policies=["basic_content_moderation"],
    allow_creative_content=True
)
def generate_creative_content(prompt: str) -> str:
    return create_artistic_content(prompt)

# Custom configuration for specific use cases
@guardrails(
    input_validation=True,
    output_validation=False,  # Skip output validation
    prompt_augmentation=True,
    custom_evaluator="business_rules_engine"
)
def business_process(input_data: str) -> str:
    return process_business_logic(input_data)
```

### Dynamic Configuration

Adjust configuration at runtime:

```python
from rizk.sdk.guardrails.engine import GuardrailsEngine

engine = GuardrailsEngine.get_instance()

# Update enforcement level
engine.set_enforcement_level("strict")

# Update confidence thresholds
engine.set_confidence_threshold(0.95)
engine.set_block_threshold(0.9)

# Enable/disable specific features
engine.enable_fast_rules()
engine.disable_llm_fallback()

# Update LLM settings
engine.set_llm_service("anthropic")
engine.set_llm_model("claude-3-opus")

# Update cache settings
engine.set_cache_size(20000)
engine.set_cache_ttl(7200)  # 2 hours
```

## Best Practices

### 1. Environment-Specific Configuration

Use different configurations for different environments:

```python
# âœ… Environment-specific settings
configs = {
    "development": {
        "policy_enforcement": "lenient",
        "trace_content": True,
        "verbose": True
    },
    "staging": {
        "policy_enforcement": "moderate", 
        "trace_content": True,
        "metrics_enabled": True
    },
    "production": {
        "policy_enforcement": "strict",
        "trace_content": False,
        "metrics_enabled": True,
        "audit_all_decisions": True
    }
}

env = os.getenv("ENVIRONMENT", "development")
rizk = Rizk.init(app_name="MyApp", **configs[env])
```

### 2. Gradual Configuration Changes

Make configuration changes gradually:

```python
# âœ… Gradual enforcement level increase
def upgrade_enforcement_gradually():
    # Week 1: Start with lenient
    rizk = Rizk.init(app_name="MyApp", policy_enforcement="lenient")
    
    # Week 2: Move to moderate (after monitoring)
    # rizk.update_config(policy_enforcement="moderate")
    
    # Week 3: Move to strict (after validation)
    # rizk.update_config(policy_enforcement="strict")
```

### 3. Monitor Configuration Impact

Monitor the impact of configuration changes:

```python
# âœ… Monitor configuration changes
def monitor_config_impact():
    from rizk.sdk.analytics import ConfigAnalytics
    
    analytics = ConfigAnalytics()
    
    # Get metrics before and after config changes
    before_metrics = analytics.get_metrics(time_range="24h")
    
    # Apply configuration change
    engine = GuardrailsEngine.get_instance()
    engine.set_enforcement_level("strict")
    
    # Wait and measure impact
    time.sleep(3600)  # Wait 1 hour
    after_metrics = analytics.get_metrics(time_range="1h")
    
    # Compare metrics
    impact = analytics.compare_metrics(before_metrics, after_metrics)
    
    if impact.user_satisfaction_drop > 0.1:
        print("Warning: User satisfaction dropped significantly")
    
    if impact.error_rate_increase > 0.05:
        print("Warning: Error rate increased significantly")
```

## Next Steps

1. **[Monitoring](monitoring.md)** - Track guardrail performance and effectiveness
2. **[Policy Enforcement](policy-enforcement.md)** - Understand how policies work
3. **[Using Guardrails](using-guardrails.md)** - Practical implementation guide
4. **[Overview](overview.md)** - Understanding the guardrails system

---

Proper configuration is essential for optimal guardrail performance. Start with conservative settings and adjust based on monitoring data and user feedback.


