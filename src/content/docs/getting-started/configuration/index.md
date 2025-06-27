---
title: "Configuration Guide"
description: "Configuration Guide"
---

# Configuration Guide

Essential configuration patterns for Rizk SDK across development, staging, and production environments. This guide covers environment variables, configuration files, and deployment-specific settings for enterprise-grade LLM observability.

## Configuration Overview

Rizk SDK uses a hierarchical configuration system that prioritizes:

1. **Environment Variables** (highest priority)
2. **Configuration Files** (.env, config files)
3. **Programmatic Configuration** (runtime settings)
4. **Default Values** (lowest priority)

This approach ensures secure, flexible configuration management across different deployment scenarios.

## Core Configuration

### Required Settings

```bash
# Essential configuration
RIZK_API_KEY="rizk_prod_your-api-key-here"
RIZK_APP_NAME="your-application-name"
```

### Basic Initialization

```python
from rizk.sdk import Rizk

# Minimal configuration
rizk = Rizk.init(
    app_name="my-app",
    api_key="your-api-key",  # Better to use environment variable
    enabled=True
)
```

## Environment Variables Reference

### Core Settings

| Variable | Description | Default | Example |
|----------|-------------|---------|---------|
| `RIZK_API_KEY` | Authentication key from rizk.tools | Required | `rizk_prod_abc123...` |
| `RIZK_APP_NAME` | Application identifier | `"rizk-app"` | `"customer-support-bot"` |
| `RIZK_ENABLED` | Enable/disable SDK globally | `true` | `false` |
| `RIZK_ENVIRONMENT` | Deployment environment | `"development"` | `"production"` |

### Observability Settings

| Variable | Description | Default | Example |
|----------|-------------|---------|---------|
| `RIZK_TRACING_ENABLED` | Enable distributed tracing | `true` | `false` |
| `RIZK_TRACE_CONTENT` | Include content in traces | `false` | `true` |
| `RIZK_METRICS_ENABLED` | Enable metrics collection | `true` | `false` |
| `RIZK_OPENTELEMETRY_ENDPOINT` | Custom OTLP endpoint | Rizk Cloud | `https://otel.company.com` |

### Guardrails Settings

| Variable | Description | Default | Example |
|----------|-------------|---------|---------|
| `RIZK_POLICY_ENFORCEMENT` | Enable policy enforcement | `true` | `false` |
| `RIZK_POLICIES_PATH` | Custom policies directory | `./policies` | `/app/config/policies` |
| `RIZK_GUARDRAILS_TIMEOUT` | Policy evaluation timeout (ms) | `5000` | `10000` |

### Performance Settings

| Variable | Description | Default | Example |
|----------|-------------|---------|---------|
| `RIZK_CACHE_ENABLED` | Enable response caching | `true` | `false` |
| `RIZK_CACHE_TTL` | Cache TTL in seconds | `300` | `3600` |
| `RIZK_MAX_CONCURRENT_REQUESTS` | Concurrent request limit | `100` | `500` |
| `RIZK_REQUEST_TIMEOUT` | Request timeout in seconds | `30` | `60` |

### Logging Settings

| Variable | Description | Default | Example |
|----------|-------------|---------|---------|
| `RIZK_LOG_LEVEL` | Logging level | `INFO` | `DEBUG` |
| `RIZK_LOG_FORMAT` | Log format | `structured` | `json` |
| `RIZK_LOG_FILE` | Log file path | None | `/var/log/rizk.log` |

## Environment-Specific Configurations

### Development Environment

```bash
# .env.development
RIZK_API_KEY="rizk_dev_your-dev-key"
RIZK_APP_NAME="myapp-dev"
RIZK_ENVIRONMENT="development"
RIZK_ENABLED=true
RIZK_TRACE_CONTENT=true
RIZK_LOG_LEVEL=DEBUG
RIZK_POLICY_ENFORCEMENT=false
```

```python
# Development configuration
rizk = Rizk.init(
    app_name="myapp-dev",
    enabled=True,
    verbose=True,  # Detailed logging
    trace_content=True,  # Include content in traces for debugging
    policy_enforcement=False  # Disable policies during development
)
```

### Staging Environment

```bash
# .env.staging
RIZK_API_KEY="rizk_staging_your-staging-key"
RIZK_APP_NAME="myapp-staging"
RIZK_ENVIRONMENT="staging"
RIZK_ENABLED=true
RIZK_TRACE_CONTENT=false
RIZK_LOG_LEVEL=INFO
RIZK_POLICY_ENFORCEMENT=true
RIZK_POLICIES_PATH="/app/policies"
```

```python
# Staging configuration
rizk = Rizk.init(
    app_name="myapp-staging",
    enabled=True,
    trace_content=False,  # Exclude sensitive content
    policy_enforcement=True,  # Test policies in staging
    policies_path="/app/policies"
)
```

### Production Environment

```bash
# .env.production
RIZK_API_KEY="rizk_prod_your-production-key"
RIZK_APP_NAME="myapp-prod"
RIZK_ENVIRONMENT="production"
RIZK_ENABLED=true
RIZK_TRACE_CONTENT=false
RIZK_LOG_LEVEL=WARN
RIZK_POLICY_ENFORCEMENT=true
RIZK_POLICIES_PATH="/app/config/policies"
RIZK_CACHE_ENABLED=true
RIZK_CACHE_TTL=3600
RIZK_MAX_CONCURRENT_REQUESTS=1000
```

```python
# Production configuration
rizk = Rizk.init(
    app_name="myapp-prod",
    enabled=True,
    trace_content=False,  # Security: no content in traces
    policy_enforcement=True,  # Full governance
    cache_enabled=True,  # Performance optimization
    max_concurrent_requests=1000  # Scale for production load
)
```

## Configuration Files

### .env File Structure

```env
# .env
# Core Configuration
RIZK_API_KEY=your-api-key-here
RIZK_APP_NAME=my-application
RIZK_ENVIRONMENT=production

# Observability
RIZK_TRACING_ENABLED=true
RIZK_TRACE_CONTENT=false
RIZK_METRICS_ENABLED=true
RIZK_OPENTELEMETRY_ENDPOINT=https://your-otlp-endpoint.com

# Guardrails
RIZK_POLICY_ENFORCEMENT=true
RIZK_POLICIES_PATH=./policies
RIZK_GUARDRAILS_TIMEOUT=5000

# Performance
RIZK_CACHE_ENABLED=true
RIZK_CACHE_TTL=300
RIZK_MAX_CONCURRENT_REQUESTS=100

# Logging
RIZK_LOG_LEVEL=INFO
RIZK_LOG_FORMAT=json
RIZK_LOG_FILE=/var/log/rizk.log
```

### YAML Configuration File

Create `rizk.config.yaml`:

```yaml
# rizk.config.yaml
app:
  name: "customer-support-bot"
  environment: "production"
  version: "1.0.0"

rizk:
  api_key: "${RIZK_API_KEY}"  # Reference environment variable
  enabled: true

observability:
  tracing:
    enabled: true
    content_tracing: false
    endpoint: "https://otel.company.com"
  metrics:
    enabled: true
    export_interval: 60
  logging:
    level: "INFO"
    format: "json"
    file: "/var/log/rizk.log"

guardrails:
  enabled: true
  policies_path: "./policies"
  timeout_ms: 5000
  cache_policies: true

performance:
  cache:
    enabled: true
    ttl_seconds: 300
    max_size: 1000
  concurrency:
    max_requests: 100
    timeout_seconds: 30

# Framework-specific settings
frameworks:
  langchain:
    callback_enabled: true
  crewai:
    process_monitoring: true
  openai_agents:
    function_tracing: true
```

Load YAML configuration:

```python
import yaml
from rizk.sdk import Rizk

def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    # Expand environment variables
    def expand_env_vars(obj):
        if isinstance(obj, dict):
            return {k: expand_env_vars(v) for k, v in obj.items()}
        elif isinstance(obj, str) and obj.startswith('${') and obj.endswith('}'):
            env_var = obj[2:-1]
            return os.getenv(env_var, obj)
        return obj
    
    return expand_env_vars(config)

# Initialize with YAML config
config = load_config('rizk.config.yaml')
rizk = Rizk.init(
    app_name=config['app']['name'],
    api_key=config['rizk']['api_key'],
    enabled=config['rizk']['enabled'],
    # ... other configuration options
)
```

## Framework-Specific Configuration

### OpenAI Agents Configuration

```python
from rizk.sdk import Rizk

rizk = Rizk.init(
    app_name="openai-agents-app",
    enabled=True,
    # OpenAI Agents specific settings
    framework_config={
        "openai_agents": {
            "trace_function_calls": True,
            "include_tool_results": True,
            "monitor_streaming": True
        }
    }
)
```

### LangChain Configuration

```python
rizk = Rizk.init(
    app_name="langchain-app",
    enabled=True,
    framework_config={
        "langchain": {
            "callback_handler_enabled": True,
            "trace_chains": True,
            "trace_tools": True,
            "include_prompts": False  # Security: don't log prompts
        }
    }
)
```

### CrewAI Configuration

```python
rizk = Rizk.init(
    app_name="crewai-app",
    enabled=True,
    framework_config={
        "crewai": {
            "trace_crew_execution": True,
            "trace_individual_agents": True,
            "trace_task_delegation": True,
            "include_agent_reasoning": False
        }
    }
)
```

## Advanced Configuration Patterns

### Configuration Validation

```python
from dataclasses import dataclass
from typing import Optional, List
import os

@dataclass
class RizkConfig:
    api_key: str
    app_name: str
    enabled: bool = True
    environment: str = "development"
    trace_content: bool = False
    policy_enforcement: bool = True
    policies_path: str = "./policies"
    log_level: str = "INFO"
    
    @classmethod
    def from_env(cls) -> 'RizkConfig':
        return cls(
            api_key=os.getenv("RIZK_API_KEY", ""),
            app_name=os.getenv("RIZK_APP_NAME", "rizk-app"),
            enabled=os.getenv("RIZK_ENABLED", "true").lower() == "true",
            environment=os.getenv("RIZK_ENVIRONMENT", "development"),
            trace_content=os.getenv("RIZK_TRACE_CONTENT", "false").lower() == "true",
            policy_enforcement=os.getenv("RIZK_POLICY_ENFORCEMENT", "true").lower() == "true",
            policies_path=os.getenv("RIZK_POLICIES_PATH", "./policies"),
            log_level=os.getenv("RIZK_LOG_LEVEL", "INFO")
        )
    
    def validate(self) -> List[str]:
        """Validate configuration and return list of errors."""
        errors = []
        
        if not self.api_key:
            errors.append("RIZK_API_KEY is required")
        elif not self.api_key.startswith("rizk_"):
            errors.append("RIZK_API_KEY must start with 'rizk_'")
        
        if not self.app_name:
            errors.append("RIZK_APP_NAME is required")
        
        if self.log_level not in ["DEBUG", "INFO", "WARN", "ERROR"]:
            errors.append("RIZK_LOG_LEVEL must be one of: DEBUG, INFO, WARN, ERROR")
        
        if not os.path.exists(self.policies_path):
            errors.append(f"Policies path does not exist: {self.policies_path}")
        
        return errors

# Usage
config = RizkConfig.from_env()
errors = config.validate()

if errors:
    raise ValueError(f"Configuration errors: {', '.join(errors)}")

rizk = Rizk.init(
    app_name=config.app_name,
    api_key=config.api_key,
    enabled=config.enabled,
    trace_content=config.trace_content,
    policy_enforcement=config.policy_enforcement,
    policies_path=config.policies_path
)
```

### Dynamic Configuration Updates

```python
from rizk.sdk import Rizk

# Initialize with base configuration
rizk = Rizk.init(app_name="dynamic-app", enabled=True)

# Update configuration at runtime
def update_config_for_user(user_role: str):
    if user_role == "admin":
        # Admins get full tracing
        rizk.update_config(trace_content=True, log_level="DEBUG")
    elif user_role == "developer":
        # Developers get standard tracing
        rizk.update_config(trace_content=False, log_level="INFO")
    else:
        # Regular users get minimal tracing
        rizk.update_config(trace_content=False, log_level="WARN")

# Usage in request handler
@workflow(name="handle_request")
def handle_user_request(user_role: str, request: str):
    update_config_for_user(user_role)
    # Process request with appropriate configuration
    return process_request(request)
```

### Multi-Tenant Configuration

```python
from rizk.sdk import Rizk
from typing import Dict, Any

class MultiTenantRizk:
    def __init__(self):
        self._instances: Dict[str, Rizk] = {}
    
    def get_instance(self, tenant_id: str) -> Rizk:
        if tenant_id not in self._instances:
            self._instances[tenant_id] = self._create_tenant_instance(tenant_id)
        return self._instances[tenant_id]
    
    def _create_tenant_instance(self, tenant_id: str) -> Rizk:
        # Load tenant-specific configuration
        config = self._load_tenant_config(tenant_id)
        
        return Rizk.init(
            app_name=f"app-{tenant_id}",
            api_key=config["api_key"],
            enabled=config.get("enabled", True),
            policies_path=config.get("policies_path", f"./policies/{tenant_id}"),
            organization_id=tenant_id
        )
    
    def _load_tenant_config(self, tenant_id: str) -> Dict[str, Any]:
        # Load from database, file, or environment
        return {
            "api_key": os.getenv(f"RIZK_API_KEY_{tenant_id.upper()}"),
            "enabled": True,
            "policies_path": f"./policies/{tenant_id}"
        }

# Usage
multi_tenant_rizk = MultiTenantRizk()

@workflow(name="tenant_request")
def handle_tenant_request(tenant_id: str, request: str):
    # Get tenant-specific Rizk instance
    rizk = multi_tenant_rizk.get_instance(tenant_id)
    
    # Process with tenant-specific configuration
    return process_request(request)
```

## Security Configuration

### API Key Management

```bash
# Use different keys for different environments
RIZK_API_KEY_DEV="rizk_dev_development-key"
RIZK_API_KEY_STAGING="rizk_staging_staging-key"
RIZK_API_KEY_PROD="rizk_prod_production-key"

# Runtime key selection
RIZK_API_KEY="${RIZK_API_KEY_PROD}"
```

### Content Security

```python
# Production security settings
rizk = Rizk.init(
    app_name="secure-app",
    enabled=True,
    
    # Security: Disable content tracing in production
    trace_content=False,
    
    # Security: Enable policy enforcement
    policy_enforcement=True,
    
    # Security: Use secure policies path
    policies_path="/app/secure/policies",
    
    # Security: Minimal logging
    log_level="WARN",
    
    # Security: Secure headers
    custom_headers={
        "X-Rizk-Environment": "production",
        "X-Rizk-Security-Level": "high"
    }
)
```

### Network Security

```python
# Configure for secure network environments
rizk = Rizk.init(
    app_name="network-secure-app",
    enabled=True,
    
    # Use internal OTLP endpoint
    opentelemetry_endpoint="https://internal-otel.company.com",
    
    # Configure TLS
    tls_config={
        "cert_file": "/app/certs/client.crt",
        "key_file": "/app/certs/client.key",
        "ca_file": "/app/certs/ca.crt"
    },
    
    # Network timeouts
    request_timeout=10,
    connection_timeout=5
)
```

## Troubleshooting Configuration

### Configuration Validation

```python
def validate_rizk_config():
    """Validate Rizk configuration and report issues."""
    issues = []
    
    # Check required environment variables
    required_vars = ["RIZK_API_KEY", "RIZK_APP_NAME"]
    for var in required_vars:
        if not os.getenv(var):
            issues.append(f"Missing required environment variable: {var}")
    
    # Check API key format
    api_key = os.getenv("RIZK_API_KEY", "")
    if api_key and not api_key.startswith("rizk_"):
        issues.append("RIZK_API_KEY should start with 'rizk_'")
    
    # Check policies path
    policies_path = os.getenv("RIZK_POLICIES_PATH", "./policies")
    if not os.path.exists(policies_path):
        issues.append(f"Policies path does not exist: {policies_path}")
    
    # Check log level
    log_level = os.getenv("RIZK_LOG_LEVEL", "INFO")
    if log_level not in ["DEBUG", "INFO", "WARN", "ERROR"]:
        issues.append(f"Invalid log level: {log_level}")
    
    return issues

# Usage
issues = validate_rizk_config()
if issues:
    print("Configuration issues found:")
    for issue in issues:
        print(f"  - {issue}")
    exit(1)
```

### Configuration Debug Information

```python
def print_rizk_config():
    """Print current Rizk configuration for debugging."""
    print("Rizk SDK Configuration:")
    print(f"  API Key: {'***' + os.getenv('RIZK_API_KEY', '')[-4:] if os.getenv('RIZK_API_KEY') else 'Not set'}")
    print(f"  App Name: {os.getenv('RIZK_APP_NAME', 'Not set')}")
    print(f"  Environment: {os.getenv('RIZK_ENVIRONMENT', 'development')}")
    print(f"  Enabled: {os.getenv('RIZK_ENABLED', 'true')}")
    print(f"  Tracing: {os.getenv('RIZK_TRACING_ENABLED', 'true')}")
    print(f"  Policy Enforcement: {os.getenv('RIZK_POLICY_ENFORCEMENT', 'true')}")
    print(f"  Policies Path: {os.getenv('RIZK_POLICIES_PATH', './policies')}")
    print(f"  Log Level: {os.getenv('RIZK_LOG_LEVEL', 'INFO')}")

# Usage
if __name__ == "__main__":
    print_rizk_config()
```

## Next Steps

### Production Deployment
- **[Production Setup Guide](../advanced-config/production-setup.md)** - Complete production deployment
- **[Security Best Practices](../advanced-config/security.md)** - Secure your deployment
- **[Performance Tuning](../advanced-config/performance-tuning.md)** - Optimize for scale

### Advanced Configuration
- **[Custom Policies](../guardrails/creating-policies.md)** - Create custom governance rules
- **[Observability Setup](../observability/)** - Advanced monitoring configuration
- **[Framework Integration](../framework-integration/)** - Framework-specific settings

---

**Configuration is critical for production success.** Take time to properly configure Rizk SDK for your environment and security requirements.

*Need help with configuration? Check our [troubleshooting guide](../troubleshooting/) or contact [support](mailto:hello@rizk.tools).* 

