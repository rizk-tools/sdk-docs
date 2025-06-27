---
title: "Environment Variables Reference"
description: "Complete reference for all Rizk SDK environment variables for production deployments."
---

# Environment Variables Reference

Complete reference for all Rizk SDK environment variables for production deployments.

## Core Configuration

### Required Variables

| Variable | Description | Example | Required |
|----------|-------------|---------|----------|
| `RIZK_API_KEY` | Your Rizk API key for authentication | `rizk_prod_abc123...` | âœ… Yes |

### Application Settings

| Variable | Default | Description | Example |
|----------|---------|-------------|---------|
| `RIZK_APP_NAME` | `"RizkApp"` | Application name for identification | `"MyApp-Production"` |

## OpenTelemetry Configuration

### Tracing Settings

| Variable | Default | Description | Example |
|----------|---------|-------------|---------|
| `RIZK_OPENTELEMETRY_ENDPOINT` | Uses `https://api.rizk.tools` when `RIZK_API_KEY` is set | Custom OTLP endpoint (overrides default) | `"https://otlp.company.com"` |
| `RIZK_TRACING_ENABLED` | `"true"` | Enable/disable distributed tracing | `"true"` |
| `RIZK_TRACE_CONTENT` | `"true"` | Include content in traces (disable for privacy) | `"false"` |
| `RIZK_METRICS_ENABLED` | `"true"` | Enable/disable metrics collection | `"true"` |

**Note**: When you set `RIZK_API_KEY`, traces are automatically sent to `https://api.rizk.tools`. Set `RIZK_OPENTELEMETRY_ENDPOINT` only if you want to use a different endpoint.

### Example Configuration

```bash
# Using Rizk's default endpoint (recommended)
export RIZK_API_KEY="rizk_prod_your_key_here"
# Traces automatically sent to https://api.rizk.tools

# Advanced: Disable content tracing for privacy
export RIZK_TRACING_ENABLED="true"
export RIZK_TRACE_CONTENT="false"  # Privacy in production
export RIZK_METRICS_ENABLED="true"
```

> **Note**: To use a custom OTLP collector instead of Rizk's default endpoint:
> ```bash
> export RIZK_API_KEY=""  # Clear API key when using custom endpoint
> export RIZK_OPENTELEMETRY_ENDPOINT="https://otel-collector.company.com:4317"
> ```

## Performance Settings

### Framework Detection

| Variable | Default | Description | Example |
|----------|---------|-------------|---------|
| `RIZK_LAZY_LOADING` | `"true"` | Enable lazy loading of adapters | `"true"` |
| `RIZK_FRAMEWORK_CACHE_SIZE` | `"1000"` | Framework detection cache size | `"5000"` |

### Example Configuration

```bash
# High-performance production settings
export RIZK_LAZY_LOADING="true"
export RIZK_FRAMEWORK_CACHE_SIZE="10000"
```

## Guardrails Configuration

### Policy Settings

| Variable | Default | Description | Example |
|----------|---------|-------------|---------|
| `RIZK_POLICY_ENFORCEMENT` | `"true"` | Enable/disable policy enforcement | `"true"` |
| `RIZK_POLICIES_PATH` | Auto-detected | Path to custom policies directory | `"/app/policies"` |

### Example Configuration

```bash
# Guardrails configuration
export RIZK_POLICY_ENFORCEMENT="true"
export RIZK_POLICIES_PATH="/opt/rizk/policies"
```

## Security and Privacy

### Debug and Logging

| Variable | Default | Description | Example |
|----------|---------|-------------|---------|
| `RIZK_DEBUG` | `"false"` | Enable debug mode | `"false"` |
| `RIZK_VERBOSE` | `"false"` | Enable verbose logging | `"false"` |
| `RIZK_LOGGING_ENABLED` | `"false"` | Enable SDK internal logging | `"false"` |

### Telemetry

| Variable | Default | Description | Example |
|----------|---------|-------------|---------|
| `RIZK_TELEMETRY` | `"false"` | Enable anonymous telemetry | `"false"` |

### Production Security Example

```bash
# Secure production configuration
export RIZK_DEBUG="false"
export RIZK_VERBOSE="false"
export RIZK_LOGGING_ENABLED="false"
export RIZK_TELEMETRY="false"
export RIZK_TRACE_CONTENT="false"
```

## Cache Configuration

### Redis Settings

Redis caching is configured through the SDK's cache initialization, not environment variables. Redis connection is handled automatically when available.

**Redis Installation:**
```bash
# Install Redis support (optional)
pip install redis
```

**Redis Configuration in Code:**
```python
from rizk.sdk.cache.redis_adapter import RedisAdapter, RedisConfig

# Custom Redis configuration
redis_config = RedisConfig(
    url="redis://redis-cluster:6379",
    max_connections=50,
    socket_timeout=2.0,
    retry_attempts=3
)

adapter = RedisAdapter(redis_config)
```

> **Note**: Rizk SDK gracefully handles missing Redis dependencies. If Redis is not available, the SDK uses in-memory caching as fallback.

## Analytics Configuration

### Rizk Hub Integration

Analytics data is sent to Rizk Hub using your existing API key. No additional configuration required.

**Built-in Analytics:**
```python
from rizk.sdk import Rizk

# Analytics automatically enabled with API key
rizk = Rizk.init(
    app_name="MyApp",
    api_key="rizk_your_key_here"  # Enables Rizk Hub analytics
)
```

> **Note**: Analytics data includes blocked messages, policy violations, and performance metrics. All data is sent to `https://api.rizk.tools` using your Rizk API key.

## Streaming Configuration

### Performance Settings

Streaming is configured through the SDK's StreamConfig class, not environment variables.

**Stream Configuration in Code:**
```python
from rizk.sdk.streaming import StreamProcessor, StreamConfig

# Custom streaming configuration
stream_config = StreamConfig(
    buffer_size=20,
    validation_interval=2,
    timeout_seconds=60.0,
    enable_guardrails=True,
    enable_caching=True
)

processor = StreamProcessor(stream_config)
```

**Available StreamConfig Options:**
- `buffer_size`: Chunk buffer size (default: 10)
- `validation_interval`: Validate every N chunks (default: 1)  
- `timeout_seconds`: Stream timeout (default: 30.0)
- `enable_guardrails`: Enable real-time guardrails (default: True)
- `enable_caching`: Enable partial response caching (default: True)

## Environment-Specific Configurations

### Development Environment

```bash
# development.env
export RIZK_API_KEY="rizk_dev_your_key_here"
export RIZK_DEBUG="true"
export RIZK_VERBOSE="true"
export RIZK_TRACE_CONTENT="true"
export RIZK_FRAMEWORK_CACHE_SIZE="1000"
```

### Staging Environment

```bash
# staging.env
export RIZK_API_KEY="rizk_staging_your_key_here"
export RIZK_DEBUG="false"
export RIZK_VERBOSE="false"
export RIZK_TRACE_CONTENT="false"
export RIZK_FRAMEWORK_CACHE_SIZE="3000"
```

### Production Environment

```bash
# production.env
export RIZK_API_KEY="rizk_prod_your_key_here"
export RIZK_DEBUG="false"
export RIZK_VERBOSE="false"
export RIZK_LOGGING_ENABLED="false"
export RIZK_TRACE_CONTENT="false"
export RIZK_TELEMETRY="false"
export RIZK_FRAMEWORK_CACHE_SIZE="10000"
export RIZK_POLICIES_PATH="/app/policies"
```

> **Note**: To use custom OTLP endpoints in staging/production instead of Rizk's default:
> ```bash
> # Staging with custom endpoint
> export RIZK_OPENTELEMETRY_ENDPOINT="https://otlp-staging.company.com"
> 
> # Production with custom endpoint  
> export RIZK_OPENTELEMETRY_ENDPOINT="https://otlp.company.com"
> ```

## Container and Kubernetes Configuration

### Docker Compose

```yaml
# docker-compose.yml
version: '3.8'
services:
  app:
    image: your-app:latest
    environment:
      - RIZK_API_KEY=${RIZK_API_KEY}
      - RIZK_TRACING_ENABLED=true
      - RIZK_TRACE_CONTENT=false
      - RIZK_DEBUG=false
      - RIZK_VERBOSE=false
    depends_on:
      - redis
  
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
```

### Kubernetes ConfigMap

```yaml
# configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: rizk-config
data:
  RIZK_TRACING_ENABLED: "true"
  RIZK_TRACE_CONTENT: "false"
  RIZK_METRICS_ENABLED: "true"
  RIZK_DEBUG: "false"
  RIZK_VERBOSE: "false"
  RIZK_FRAMEWORK_CACHE_SIZE: "5000"
---
apiVersion: v1
kind: Secret
metadata:
  name: rizk-secrets
type: Opaque
stringData:
  RIZK_API_KEY: "rizk_prod_your_key_here"
```

## Configuration Validation

### Python Validation Script

```python
#!/usr/bin/env python3
"""Validate Rizk SDK environment configuration."""

import os
import sys
from typing import List, Tuple

def validate_config() -> List[Tuple[str, str]]:
    """Validate environment configuration."""
    errors = []
    
    # Required variables
    if not os.getenv("RIZK_API_KEY"):
        errors.append(("RIZK_API_KEY", "Required but not set"))
    elif not os.getenv("RIZK_API_KEY", "").startswith("rizk_"):
        errors.append(("RIZK_API_KEY", "Must start with 'rizk_'"))
    
    # Boolean validations
    bool_vars = [
        "RIZK_TRACING_ENABLED", "RIZK_TRACE_CONTENT", "RIZK_METRICS_ENABLED",
        "RIZK_DEBUG", "RIZK_VERBOSE", "RIZK_LOGGING_ENABLED", "RIZK_TELEMETRY",
        "RIZK_POLICY_ENFORCEMENT", "RIZK_LAZY_LOADING"
    ]
    
    for var in bool_vars:
        value = os.getenv(var)
        if value and value.lower() not in ["true", "false"]:
            errors.append((var, f"Must be 'true' or 'false', got '{value}'"))
    
    # Numeric validations
    numeric_vars = {
        "RIZK_FRAMEWORK_CACHE_SIZE": (1, 100000),
        "REDIS_MAX_CONNECTIONS": (1, 1000),
        "REDIS_SOCKET_TIMEOUT": (0.1, 60.0),
        "REDIS_RETRY_ATTEMPTS": (1, 10),
        "RIZK_HUB_BATCH_SIZE": (1, 1000),
        "RIZK_HUB_FLUSH_INTERVAL": (1, 3600),
        "RIZK_STREAM_MAX_CHUNK_SIZE": (1, 10000),
        "RIZK_STREAM_BUFFER_SIZE": (1, 100),
    }
    
    for var, (min_val, max_val) in numeric_vars.items():
        value = os.getenv(var)
        if value:
            try:
                num_val = float(value)
                if not (min_val <= num_val <= max_val):
                    errors.append((var, f"Must be between {min_val} and {max_val}"))
            except ValueError:
                errors.append((var, f"Must be a number, got '{value}'"))
    
    # URL validations
    url_vars = ["RIZK_OPENTELEMETRY_ENDPOINT", "REDIS_URL", "RIZK_HUB_ENDPOINT"]
    for var in url_vars:
        value = os.getenv(var)
        if value and not (value.startswith("http://") or value.startswith("https://") or value.startswith("redis://")):
            errors.append((var, f"Must be a valid URL, got '{value}'"))
    
    # Path validations
    policies_path = os.getenv("RIZK_POLICIES_PATH")
    if policies_path and not os.path.exists(policies_path):
        errors.append(("RIZK_POLICIES_PATH", f"Path does not exist: {policies_path}"))
    
    return errors

if __name__ == "__main__":
    errors = validate_config()
    
    if errors:
        print("âŒ Configuration validation failed:")
        for var, error in errors:
            print(f"  {var}: {error}")
        sys.exit(1)
    else:
        print("âœ… Configuration validation passed")
        
        # Print current configuration
        print("\nðŸ“‹ Current configuration:")
        config_vars = [
            "RIZK_API_KEY", "RIZK_APP_NAME", "RIZK_TRACING_ENABLED",
            "RIZK_TRACE_CONTENT", "RIZK_DEBUG", "RIZK_FRAMEWORK_CACHE_SIZE"
        ]
        
        for var in config_vars:
            value = os.getenv(var, "Not set")
            if var == "RIZK_API_KEY" and value != "Not set":
                value = f"{value[:10]}..." if len(value) > 10 else value
            print(f"  {var}: {value}")
```

### Shell Validation Script

```bash
#!/bin/bash
# validate_config.sh - Validate Rizk SDK configuration

set -e

echo "ðŸ” Validating Rizk SDK configuration..."

# Check required variables
if [ -z "$RIZK_API_KEY" ]; then
    echo "âŒ RIZK_API_KEY is required but not set"
    exit 1
fi

if [[ ! "$RIZK_API_KEY" =~ ^rizk_ ]]; then
    echo "âŒ RIZK_API_KEY must start with 'rizk_'"
    exit 1
fi

# Check boolean variables
check_boolean() {
    local var_name=$1
    local var_value=${!var_name}
    
    if [ -n "$var_value" ] && [ "$var_value" != "true" ] && [ "$var_value" != "false" ]; then
        echo "âŒ $var_name must be 'true' or 'false', got '$var_value'"
        exit 1
    fi
}

check_boolean RIZK_TRACING_ENABLED
check_boolean RIZK_TRACE_CONTENT
check_boolean RIZK_DEBUG
check_boolean RIZK_VERBOSE

# Check numeric variables
check_numeric() {
    local var_name=$1
    local var_value=${!var_name}
    local min_val=$2
    local max_val=$3
    
    if [ -n "$var_value" ]; then
        if ! [[ "$var_value" =~ ^[0-9]+$ ]] || [ "$var_value" -lt "$min_val" ] || [ "$var_value" -gt "$max_val" ]; then
            echo "âŒ $var_name must be between $min_val and $max_val, got '$var_value'"
            exit 1
        fi
    fi
}

check_numeric RIZK_FRAMEWORK_CACHE_SIZE 1 100000
check_numeric REDIS_MAX_CONNECTIONS 1 1000

# Check paths
if [ -n "$RIZK_POLICIES_PATH" ] && [ ! -d "$RIZK_POLICIES_PATH" ]; then
    echo "âŒ RIZK_POLICIES_PATH directory does not exist: $RIZK_POLICIES_PATH"
    exit 1
fi

echo "âœ… Configuration validation passed"

# Display current configuration
echo ""
echo "ðŸ“‹ Current configuration:"
echo "  RIZK_API_KEY: ${RIZK_API_KEY:0:10}..."
echo "  RIZK_APP_NAME: ${RIZK_APP_NAME:-Not set}"
echo "  RIZK_TRACING_ENABLED: ${RIZK_TRACING_ENABLED:-Not set}"
echo "  RIZK_DEBUG: ${RIZK_DEBUG:-Not set}"
echo "  REDIS_URL: ${REDIS_URL:-Not set}"
```

## Best Practices

### 1. Environment Separation

```bash
# Use different prefixes for different environments
# Development
export RIZK_API_KEY="rizk_dev_..."

# Staging  
export RIZK_API_KEY="rizk_staging_..."

# Production
export RIZK_API_KEY="rizk_prod_..."
```

### 2. Secrets Management

```bash
# âŒ Never commit secrets to version control
echo "RIZK_API_KEY=rizk_secret_key" >> .env

# âœ… Use secrets management systems
# AWS Secrets Manager
export RIZK_API_KEY=$(aws secretsmanager get-secret-value --secret-id rizk/api-key --query SecretString --output text)

# HashiCorp Vault
export RIZK_API_KEY=$(vault kv get -field=api_key secret/rizk)

# Kubernetes Secrets
kubectl create secret generic rizk-secrets --from-literal=api-key=rizk_your_key_here
```

### 3. Configuration Templates

```bash
# config-template.env
RIZK_API_KEY=__REPLACE_WITH_ACTUAL_KEY__
RIZK_APP_NAME=__REPLACE_WITH_APP_NAME__
RIZK_OPENTELEMETRY_ENDPOINT=__REPLACE_WITH_OTLP_ENDPOINT__
REDIS_URL=__REPLACE_WITH_REDIS_URL__

# Use with envsubst or similar tools
envsubst < config-template.env > production.env
```

## Troubleshooting

### Common Issues

1. **API Key Format Error**
   ```
   Error: API key must start with 'rizk_'
   Solution: Ensure your API key has the correct format
   ```

2. **Boolean Value Error**
   ```
   Error: RIZK_DEBUG must be 'true' or 'false'
   Solution: Use lowercase 'true' or 'false', not 'True'/'False'
   ```

3. **Numeric Range Error**
   ```
   Error: RIZK_FRAMEWORK_CACHE_SIZE must be between 1 and 100000
   Solution: Use a reasonable cache size value
   ```

### Debugging Configuration

```python
from rizk.sdk.config import get_config

# Print current configuration
config = get_config()
print("Current Rizk configuration:")
for key, value in config.to_dict().items():
    print(f"  {key}: {value}")

# Validate configuration
errors = config.validate()
if errors:
    print("Configuration errors:")
    for error in errors:
        print(f"  - {error}")
```

## Next Steps

1. **[Production Setup](production-setup.md)** - Complete production deployment guide
2. **[Performance Tuning](performance-tuning.md)** - Optimize configuration for performance
3. **[Security Best Practices](security.md)** - Secure your configuration

---

*Complete environment variable reference for enterprise Rizk SDK deployments* 
