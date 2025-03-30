---
title: "Telemetry Class"
description: "Documentation for Telemetry Class"
---

The `Telemetry` class provides telemetry and observability functionality for the Rizk SDK. It wraps Traceloop's telemetry with Rizk-specific functionality.

## Class Definition

```python
class Telemetry:
    """Wraps Traceloop's telemetry with Rizk-specific functionality."""
```

## Instance Methods

### capture_event

```python
def capture_event(
    event_name: str,
    properties: Optional[Dict[str, Any]] = None,
    context: Optional[Dict[str, Any]] = None
) -> None
```

Capture a telemetry event.

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `event_name` | `str` | - | Name of the event |
| `properties` | `Optional[Dict[str, Any]]` | `None` | Event properties |
| `context` | `Optional[Dict[str, Any]]` | `None` | Additional context |

#### Example

```python
telemetry.capture_event(
    event_name="policy_violation",
    properties={
        "policy_id": "content_moderation",
        "severity": "high",
        "message": "Inappropriate content detected"
    }
)
```

### log_exception

```python
def log_exception(
    exception: Exception,
    properties: Optional[Dict[str, Any]] = None,
    context: Optional[Dict[str, Any]] = None
) -> None
```

Log an exception with telemetry.

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `exception` | `Exception` | - | The exception to log |
| `properties` | `Optional[Dict[str, Any]]` | `None` | Exception properties |
| `context` | `Optional[Dict[str, Any]]` | `None` | Additional context |

#### Example

```python
try:
    # Some code that might raise an exception
    pass
except Exception as e:
    telemetry.log_exception(
        exception=e,
        properties={
            "component": "guardrails",
            "operation": "policy_check"
        }
    )
```

### is_feature_enabled

```python
def is_feature_enabled(
    feature_name: str,
    context: Optional[Dict[str, Any]] = None
) -> bool
```

Check if a feature is enabled.

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `feature_name` | `str` | - | Name of the feature |
| `context` | `Optional[Dict[str, Any]]` | `None` | Additional context |

#### Returns

- `bool`: Whether the feature is enabled

#### Example

```python
if telemetry.is_feature_enabled("advanced_policy_checking"):
    # Use advanced policy checking
    pass
```

### get_feature_value

```python
def get_feature_value(
    feature_name: str,
    default: Any = None,
    context: Optional[Dict[str, Any]] = None
) -> Any
```

Get the value of a feature flag.

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `feature_name` | `str` | - | Name of the feature |
| `default` | `Any` | `None` | Default value if feature is not found |
| `context` | `Optional[Dict[str, Any]]` | `None` | Additional context |

#### Returns

- `Any`: The feature value or default

#### Example

```python
max_retries = telemetry.get_feature_value(
    feature_name="max_policy_retries",
    default=3
)
```

### set_context

```python
def set_context(
    key: str,
    value: Any,
    context: Optional[Dict[str, Any]] = None
) -> None
```

Set a context value for telemetry.

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `key` | `str` | - | Context key |
| `value` | `Any` | - | Context value |
| `context` | `Optional[Dict[str, Any]]` | `None` | Additional context |

#### Example

```python
telemetry.set_context(
    key="organization_id",
    value="org123"
)
```

## Properties

| Property | Type | Description |
|----------|------|-------------|
| `client` | `Client` | The Traceloop client |
| `tracer` | `Tracer` | The OpenTelemetry tracer |

## Event Types

The following event types are supported:

- `policy_violation`: Recorded when a policy is violated
- `policy_check`: Recorded when a policy check is performed
- `feature_flag`: Recorded when a feature flag is accessed
- `exception`: Recorded when an exception occurs
- `metric`: Recorded for metric collection

## Related Documentation

- [Guardrails Guide](../core-concepts/guardrails.md)
- [Policy Enforcement Guide](../core-concepts/policy-enforcement.md)
- [Examples](../examples/advanced-guardrails.md)
- [API Reference](../api/rizk.md) 