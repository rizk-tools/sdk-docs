---
title: "LLMFallback Class"
description: "Documentation for LLMFallback Class"
---

The `LLMFallback` class provides advanced policy evaluation using language models when fast rules are insufficient. It handles complex policy checks that require semantic understanding.

## Class Definition

```python
class LLMFallback:
    """Advanced policy evaluation using language models."""
```

## Instance Methods

### evaluate

```python
async def evaluate(
    message: str,
    context: Optional[Dict[str, Any]] = None,
    fast_result: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]
```

Evaluate a message using language models for complex policy checks.

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `message` | `str` | - | The message to evaluate |
| `context` | `Optional[Dict[str, Any]]` | `None` | Additional context |
| `fast_result` | `Optional[Dict[str, Any]]` | `None` | Results from fast rules evaluation |

#### Returns

- `Dict[str, Any]`: Evaluation result with confidence and violation information

#### Example

```python
result = await llm_fallback.evaluate(
    message="User message here",
    context={
        "conversation_id": "unique_id",
        "matched_policies": ["content_moderation"]
    },
    fast_result={"confidence": 0.3}
)

if not result["allowed"]:
    print(f"Complex violation detected: {result['violation_reason']}")
```

### check_policy_compliance

```python
async def check_policy_compliance(
    message: str,
    policy_id: str,
    context: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]
```

Check message compliance with a specific policy using language models.

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `message` | `str` | - | The message to check |
| `policy_id` | `str` | - | ID of the policy to check |
| `context` | `Optional[Dict[str, Any]]` | `None` | Additional context |

#### Returns

- `Dict[str, Any]`: Compliance check result

#### Example

```python
result = await llm_fallback.check_policy_compliance(
    message="User message here",
    policy_id="content_moderation",
    context={"language": "en"}
)

if not result["compliant"]:
    print(f"Policy violation: {result['violation_details']}")
```

### get_policy_description

```python
def get_policy_description(
    policy_id: str,
    context: Optional[Dict[str, Any]] = None
) -> str
```

Get the description of a policy for LLM evaluation.

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `policy_id` | `str` | - | ID of the policy |
| `context` | `Optional[Dict[str, Any]]` | `None` | Additional context |

#### Returns

- `str`: Policy description

#### Example

```python
description = llm_fallback.get_policy_description(
    policy_id="content_moderation",
    context={"language": "en"}
)
```

## Properties

| Property | Type | Description |
|----------|------|-------------|
| `llm_service` | `LLMService` | The language model service |
| `tracer` | `Tracer` | The OpenTelemetry tracer |

## Evaluation Result Structure

The evaluation result is a dictionary with the following structure:

```python
{
    "allowed": bool,           # Whether the message is allowed
    "confidence": float,       # Confidence in the evaluation
    "violation_reason": str,   # Description of any violation
    "violation_details": dict, # Detailed violation information
    "policy_id": str,         # ID of the violated policy
    "metadata": dict          # Additional evaluation metadata
}
```

## Related Documentation

- [Guardrails Guide](../core-concepts/guardrails.md)
- [Policy Enforcement Guide](../core-concepts/policy-enforcement.md)
- [Examples](../examples/advanced-guardrails.md)
- [API Reference](../api/rizk.md) 