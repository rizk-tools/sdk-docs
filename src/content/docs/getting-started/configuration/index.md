---
title: "Configuration"
description: "Documentation for Configuration"
---

This guide covers all configuration options available in the Rizk SDK.

## SDK Initialization

The Rizk SDK can be initialized with various configuration options:

```python
from rizk.sdk import Rizk

client = Rizk.init(
    # Required
    app_name="my_app",
    api_key="your_api_key_here",
    
    # Optional
    opentelemetry_endpoint="https://api.rizk.tools",
    enabled=True,
    telemetry_enabled=True,
    headers={"custom-header": "value"},
    disable_batch=False,
    resource_attributes={
        "service.name": "my_service",
        "service.version": "1.0.0",
        "deployment.environment": "production"
    },
    policies_path="/path/to/policies",
    llm_service=my_custom_llm_service
)
```

## Configuration Options

### Required Options

| Option | Type | Description |
|--------|------|-------------|
| `app_name` | `str` | Name of your application |
| `api_key` | `str` | Your Rizk API key |

### Optional Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `opentelemetry_endpoint` | `str` | `None` | OpenTelemetry collector endpoint |
| `enabled` | `bool` | `True` | Whether tracing is enabled |
| `telemetry_enabled` | `bool` | `True` | Whether telemetry is enabled |
| `headers` | `Dict[str, str]` | `{}` | Custom headers for API requests |
| `disable_batch` | `bool` | `False` | Whether to disable batch processing |
| `resource_attributes` | `Dict[str, str]` | `{}` | Additional resource attributes |
| `policies_path` | `str` | `None` | Path to custom policies directory |
| `llm_service` | `Any` | `None` | Custom LLM service implementation |

## Environment Variables

You can also configure the SDK using environment variables:

```env
RIZK_API_KEY=your_api_key_here
RIZK_OPENTELEMETRY_ENDPOINT=https://api.rizk.tools
RIZK_ENABLED=true
RIZK_TELEMETRY_ENABLED=true
RIZK_DISABLE_BATCH=false
RIZK_POLICIES_PATH=/path/to/policies
```

## Resource Attributes

Resource attributes help identify your application in telemetry data:

```python
resource_attributes = {
    "service.name": "my_service",
    "service.version": "1.0.0",
    "deployment.environment": "production",
    "custom.attribute": "value"
}

client = Rizk.init(
    app_name="my_app",
    api_key="your_api_key",
    resource_attributes=resource_attributes
)
```

## Custom Policies

You can specify a custom path for your policies:

```python
client = Rizk.init(
    app_name="my_app",
    api_key="your_api_key",
    policies_path="/path/to/your/policies"
)
```

The policies directory should contain YAML files with policy definitions.

## Custom LLM Service

You can provide your own LLM service implementation:

```python
class MyCustomLLMService:
    async def generate(self, prompt: str, **kwargs):
        # Your custom LLM implementation
        pass

client = Rizk.init(
    app_name="my_app",
    api_key="your_api_key",
    llm_service=MyCustomLLMService()
)
```

## Next Steps

- [Guardrails Documentation](../core-concepts/guardrails.md)
- [Telemetry Guide](../guides/using-telemetry.md)
- [Policy Management](../guides/policy-management.md)
- [API Reference](../api/rizk.md) 