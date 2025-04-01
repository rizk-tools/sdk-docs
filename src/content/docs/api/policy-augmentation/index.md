---
title: "PolicyAugmentation Class"
description: "Documentation for PolicyAugmentation Class"
---

The `PolicyAugmentation` class is responsible for enhancing system prompts with policy guidelines and ensuring AI responses align with defined policies.

## Class Definition

```python
class PolicyAugmentation:
    """Handles policy-based system prompt augmentation."""
```

## Instance Methods

### process_message

```python
async def process_message(
    message: str,
    context: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]
```

Process a message and augment the system prompt with relevant policy guidelines.

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `message` | `str` | - | The user message to process |
| `context` | `Optional[Dict[str, Any]]` | `None` | Additional context including matched policies |

#### Returns

- `Dict[str, Any]`: Processing result with augmented prompt and metadata

#### Example

```python
result = await policy_augmentation.process_message(
    message="User query here",
    context={
        "matched_policies": ["content_moderation", "data_privacy"],
        "conversation_id": "unique_id"
    }
)

if result["augmented"]:
    print(f"Augmented prompt: {result['augmented_prompt']}")
```

### augment_prompt

```python
async def augment_prompt(
    system_prompt: str,
    matched_policies: List[str],
    context: Optional[Dict[str, Any]] = None
) -> str
```

Augment a system prompt with policy guidelines.

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `system_prompt` | `str` | - | The original system prompt |
| `matched_policies` | `List[str]` | - | List of policy IDs to apply |
| `context` | `Optional[Dict[str, Any]]` | `None` | Additional context |

#### Returns

- `str`: The augmented system prompt

#### Example

```python
augmented_prompt = await policy_augmentation.augment_prompt(
    system_prompt="Original prompt",
    matched_policies=["content_moderation", "data_privacy"],
    context={"user_id": "user123"}
)
```

### get_policy_guidelines

```python
def get_policy_guidelines(
    policy_id: str,
    context: Optional[Dict[str, Any]] = None
) -> str
```

Get the guidelines for a specific policy.

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `policy_id` | `str` | - | The ID of the policy |
| `context` | `Optional[Dict[str, Any]]` | `None` | Additional context |

#### Returns

- `str`: The policy guidelines

#### Example

```python
guidelines = policy_augmentation.get_policy_guidelines(
    policy_id="content_moderation",
    context={"language": "en"}
)
```

## Properties

| Property | Type | Description |
|----------|------|-------------|
| `policies_path` | `str` | Path to the policies directory |
| `tracer` | `Tracer` | The OpenTelemetry tracer |

## Related Documentation

- [Guardrails Guide](../core-concepts/guardrails)
- [Policy Enforcement Guide](../core-concepts/policy-enforcement)
<!-- - [Examples](../examples/advanced-guardrails) -->
- [API Reference](../api/rizk)