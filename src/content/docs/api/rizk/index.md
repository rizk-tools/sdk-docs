---
title: "Rizk SDK"
description: "Documentation for Rizk SDK"
---

The Rizk SDK provides a comprehensive solution for implementing AI guardrails and policy enforcement in your applications.

## Class Definition

```python
class Rizk:
    """Main SDK class for Rizk functionality."""
```

## Static Methods

### init

```python
@staticmethod
def init(
    api_key: str,
    app_name: str,
    opentelemetry_endpoint: Optional[str] = None,
    policies_path: Optional[str] = None,
    **kwargs
) -> None
```

Initialize the Rizk SDK.

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `api_key` | `str` | - | Rizk API key |
| `app_name` | `str` | - | Name of your application |
| `opentelemetry_endpoint` | `Optional[str]` | `None` | OpenTelemetry endpoint URL |
| `policies_path` | `Optional[str]` | `None` | Path to policy definitions |
| `**kwargs` | - | - | Additional configuration options |

#### Example

```python
Rizk.init(
    api_key="your_api_key",
    app_name="my_app",
    opentelemetry_endpoint="http://localhost:4317",
    policies_path="/path/to/policies"
)
```

### set_association_properties

```python
@staticmethod
def set_association_properties(
    organization_id: str,
    project_id: str,
    conversation_id: Optional[str] = None,
    **kwargs
) -> None
```

Set association properties for telemetry and tracking.

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `organization_id` | `str` | - | Organization identifier |
| `project_id` | `str` | - | Project identifier |
| `conversation_id` | `Optional[str]` | `None` | Conversation identifier |
| `**kwargs` | - | - | Additional properties |

#### Example

```python
Rizk.set_association_properties(
    organization_id="org123",
    project_id="proj456",
    conversation_id="conv789"
)
```

### get_association_properties

```python
@staticmethod
def get_association_properties() -> Dict[str, Any]
```

Get current association properties.

#### Returns

- `Dict[str, Any]`: Current association properties

#### Example

```python
props = Rizk.get_association_properties()
print(f"Organization: {props['organization_id']}")
```

### clear_association_properties

```python
@staticmethod
def clear_association_properties() -> None
```

Clear all association properties.

#### Example

```python
Rizk.clear_association_properties()
```

### get_guardrails

```python
@staticmethod
def get_guardrails() -> GuardrailsEngine
```

Get the guardrails engine instance.

#### Returns

- `GuardrailsEngine`: The guardrails engine instance

#### Example

```python
guardrails = Rizk.get_guardrails()
result = await guardrails.process_message("User message")
```

### get_telemetry

```python
@staticmethod
def get_telemetry() -> Telemetry
```

Get the telemetry instance.

#### Returns

- `Telemetry`: The telemetry instance

#### Example

```python
telemetry = Rizk.get_telemetry()
telemetry.capture_event("policy_violation", {
    "policy_id": "content_moderation",
    "severity": "high"
})
```

### get_client

```python
@staticmethod
def get_client() -> Client
```

Get the Rizk client instance.

#### Returns

- `Client`: The Rizk client instance

#### Example

```python
client = Rizk.get_client()
await client.send_telemetry({
    "event_type": "policy_check",
    "status": "success"
})
```

## Decorators

### with_guardrails

```python
@staticmethod
def with_guardrails(
    policies: Optional[List[str]] = None,
    **kwargs
) -> Callable
```

Decorator to apply guardrails to a function.

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `policies` | `Optional[List[str]]` | `None` | List of policy IDs to apply |
| `**kwargs` | - | - | Additional configuration options |

#### Returns

- `Callable`: Decorated function

#### Example

```python
@Rizk.with_guardrails(policies=["content_moderation"])
async def process_message(message: str) -> str:
    # Process message with guardrails
    return "Processed message"
```

## Properties

| Property | Type | Description |
|----------|------|-------------|
| `client` | `Client` | The Rizk client instance |
| `guardrails` | `GuardrailsEngine` | The guardrails engine instance |
| `telemetry` | `Telemetry` | The telemetry instance |

## Error Handling

The SDK handles various types of errors:

```python
try:
    await Rizk.get_guardrails().process_message("User message")
except PolicyViolationError as e:
    print(f"Policy violation: {e.message}")
except ConfigurationError as e:
    print(f"Configuration error: {e.message}")
except APIError as e:
    print(f"API error: {e.message}")
```

## Related Documentation

- [Guardrails Guide](../core-concepts/guardrails.md)
- [Policy Enforcement Guide](../core-concepts/policy-enforcement.md)
- [Examples](../examples/advanced-guardrails.md)
- [API Reference](../api/client.md) 