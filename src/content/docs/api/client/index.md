---
title: "Client Class"
description: "Documentation for Client Class"
---

The `Client` class provides the core functionality for interacting with the Rizk API. It extends Traceloop's client with Rizk-specific features.

## Class Definition

```python
class Client:
    """Client for interacting with the Rizk API."""
```

## Instance Methods

### __init__

```python
def __init__(
    api_key: str,
    app_name: str,
    opentelemetry_endpoint: Optional[str] = None,
    **kwargs
) -> None
```

Initialize the Rizk client.

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `api_key` | `str` | - | Rizk API key |
| `app_name` | `str` | - | Name of the application |
| `opentelemetry_endpoint` | `Optional[str]` | `None` | OpenTelemetry endpoint URL |
| `**kwargs` | - | - | Additional configuration options |

#### Example

```python
client = Client(
    api_key="your_api_key",
    app_name="my_app",
    opentelemetry_endpoint="http://localhost:4317"
)
```

### set_association_properties

```python
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
client.set_association_properties(
    organization_id="org123",
    project_id="proj456",
    conversation_id="conv789",
    user_id="user123"
)
```

### get_association_properties

```python
def get_association_properties() -> Dict[str, Any]
```

Get current association properties.

#### Returns

- `Dict[str, Any]`: Current association properties

#### Example

```python
props = client.get_association_properties()
print(f"Organization: {props['organization_id']}")
```

### clear_association_properties

```python
def clear_association_properties() -> None
```

Clear all association properties.

#### Example

```python
client.clear_association_properties()
```

### send_telemetry

```python
async def send_telemetry(
    data: Dict[str, Any],
    context: Optional[Dict[str, Any]] = None
) -> None
```

Send telemetry data to the Rizk API.

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data` | `Dict[str, Any]` | - | Telemetry data to send |
| `context` | `Optional[Dict[str, Any]]` | `None` | Additional context |

#### Example

```python
await client.send_telemetry({
    "event_type": "policy_violation",
    "policy_id": "content_moderation",
    "severity": "high"
})
```

### get_tracer

```python
def get_tracer(name: str) -> Tracer
```

Get an OpenTelemetry tracer.

#### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `name` | `str` | Name of the tracer |

#### Returns

- `Tracer`: OpenTelemetry tracer instance

#### Example

```python
tracer = client.get_tracer("my_component")
```

### get_metrics

```python
def get_metrics() -> Metrics
```

Get metrics collection instance.

#### Returns

- `Metrics`: Metrics collection instance

#### Example

```python
metrics = client.get_metrics()
metrics.counter("policy_checks").add(1)
```

## Properties

| Property | Type | Description |
|----------|------|-------------|
| `api_key` | `str` | Rizk API key |
| `app_name` | `str` | Application name |
| `opentelemetry_endpoint` | `Optional[str]` | OpenTelemetry endpoint URL |
| `association_properties` | `Dict[str, Any]` | Current association properties |
| `tracer_provider` | `TracerProvider` | OpenTelemetry tracer provider |
| `metrics_provider` | `MetricsProvider` | OpenTelemetry metrics provider |

## Error Handling

The client handles various types of errors:

```python
try:
    await client.send_telemetry(data)
except APIError as e:
    print(f"API error: {e.message}")
except ConnectionError as e:
    print(f"Connection error: {e.message}")
except ValidationError as e:
    print(f"Validation error: {e.message}")
```

## Related Documentation

- [Guardrails Guide](../core-concepts/guardrails.md)
- [Policy Enforcement Guide](../core-concepts/policy-enforcement.md)
- [Examples](../examples/advanced-guardrails.md)
- [API Reference](../api/rizk.md) 