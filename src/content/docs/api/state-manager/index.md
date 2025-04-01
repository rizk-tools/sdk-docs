---
title: "StateManager Class"
description: "Documentation for StateManager Class"
---

The `StateManager` class manages conversation state and policy violation history for the guardrails system. It provides methods for storing and retrieving state information.

## Class Definition

```python
class StateManager:
    """Manages conversation state and policy violation history."""
```

## Instance Methods

### get_state

```python
def get_state(
    conversation_id: str,
    context: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]
```

Get the state for a conversation.

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `conversation_id` | `str` | - | Unique identifier for the conversation |
| `context` | `Optional[Dict[str, Any]]` | `None` | Additional context |

#### Returns

- `Dict[str, Any]`: Conversation state

#### Example

```python
state = state_manager.get_state(
    conversation_id="conv123",
    context={"user_id": "user123"}
)

if "policy_violations" in state:
    print(f"Violations: {state['policy_violations']}")
```

### update_state

```python
def update_state(
    conversation_id: str,
    updates: Dict[str, Any],
    context: Optional[Dict[str, Any]] = None
) -> None
```

Update the state for a conversation.

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `conversation_id` | `str` | - | Unique identifier for the conversation |
| `updates` | `Dict[str, Any]` | - | State updates to apply |
| `context` | `Optional[Dict[str, Any]]` | `None` | Additional context |

#### Example

```python
state_manager.update_state(
    conversation_id="conv123",
    updates={
        "policy_violations": [
            {
                "policy_id": "content_moderation",
                "timestamp": "2024-03-22T12:00:00Z",
                "details": "Inappropriate content detected"
            }
        ]
    }
)
```

### add_violation

```python
def add_violation(
    conversation_id: str,
    policy_id: str,
    details: Dict[str, Any],
    context: Optional[Dict[str, Any]] = None
) -> None
```

Add a policy violation to the conversation state.

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `conversation_id` | `str` | - | Unique identifier for the conversation |
| `policy_id` | `str` | - | ID of the violated policy |
| `details` | `Dict[str, Any]` | - | Violation details |
| `context` | `Optional[Dict[str, Any]]` | `None` | Additional context |

#### Example

```python
state_manager.add_violation(
    conversation_id="conv123",
    policy_id="content_moderation",
    details={
        "message": "Inappropriate content",
        "severity": "high",
        "timestamp": "2024-03-22T12:00:00Z"
    }
)
```

### get_violations

```python
def get_violations(
    conversation_id: str,
    context: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]
```

Get all policy violations for a conversation.

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `conversation_id` | `str` | - | Unique identifier for the conversation |
| `context` | `Optional[Dict[str, Any]]` | `None` | Additional context |

#### Returns

- `List[Dict[str, Any]]`: List of policy violations

#### Example

```python
violations = state_manager.get_violations("conv123")
for violation in violations:
    print(f"Policy {violation['policy_id']} violated at {violation['timestamp']}")
```

### clear_state

```python
def clear_state(
    conversation_id: str,
    context: Optional[Dict[str, Any]] = None
) -> None
```

Clear the state for a conversation.

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `conversation_id` | `str` | - | Unique identifier for the conversation |
| `context` | `Optional[Dict[str, Any]]` | `None` | Additional context |

#### Example

```python
state_manager.clear_state("conv123")
```

## Properties

| Property | Type | Description |
|----------|------|-------------|
| `states` | `Dict[str, Dict[str, Any]]` | Dictionary of conversation states |
| `tracer` | `Tracer` | The OpenTelemetry tracer |

## State Structure

The conversation state is a dictionary with the following structure:

```python
{
    "conversation_id": str,           # Unique conversation identifier
    "created_at": str,               # ISO timestamp of creation
    "updated_at": str,               # ISO timestamp of last update
    "policy_violations": [           # List of policy violations
        {
            "policy_id": str,        # ID of violated policy
            "timestamp": str,        # ISO timestamp of violation
            "details": dict,         # Violation details
            "severity": str          # Violation severity
        }
    ],
    "metadata": dict                 # Additional state metadata
}
```

## Related Documentation

- [Guardrails Guide](../core-concepts/guardrails)
- [Policy Enforcement Guide](../core-concepts/policy-enforcement)
<!-- - [Examples](../examples/advanced-guardrails) -->
- [API Reference](../api/rizk)