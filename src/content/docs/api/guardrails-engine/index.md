---
title: "GuardrailsEngine Class"
description: "Documentation for GuardrailsEngine Class"
---

The `GuardrailsEngine` class is responsible for enforcing policies and ensuring safe AI interactions. It provides methods for processing messages and checking outputs against defined policies.

## Class Definition

```python
class GuardrailsEngine:
    """Main guardrails engine for policy enforcement."""
```

## Static Methods

### get_instance

```python
@staticmethod
def get_instance(config=None) -> GuardrailsEngine
```

Get or create the singleton instance of the GuardrailsEngine.

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `config` | `dict` | `None` | Configuration dictionary |

#### Returns

- `GuardrailsEngine`: The guardrails engine instance

#### Example

```python
guardrails = GuardrailsEngine.get_instance({
    "policies_path": "/path/to/policies",
    "llm_service": my_llm_service
})
```

## Instance Methods

### process_message

```python
async def process_message(
    message: str,
    context: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]
```

Process a message through the guardrails system.

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `message` | `str` | - | The user message to evaluate |
| `context` | `Optional[Dict[str, Any]]` | `None` | Additional context |

#### Returns

- `Dict[str, Any]`: Processing result with response and policy information

#### Example

```python
result = await guardrails.process_message(
    message="User query here",
    context={
        "conversation_id": "unique_id",
        "user_id": "user123"
    }
)

if not result["allowed"]:
    print(f"Message blocked: {result.get('blocked_reason')}")
```

### check_output

```python
async def check_output(
    ai_response: str,
    context: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]
```

Check AI-generated output for policy compliance.

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `ai_response` | `str` | - | The AI-generated response to check |
| `context` | `Optional[Dict[str, Any]]` | `None` | Additional context |

#### Returns

- `Dict[str, Any]`: Evaluation result with compliance information

#### Example

```python
result = await guardrails.check_output(
    ai_response="AI generated response",
    context={"conversation_id": "unique_id"}
)

if not result["allowed"]:
    print("Response blocked: Policy violation")
```

### augment_system_prompt

```python
async def augment_system_prompt(
    system_prompt: str,
    context: Optional[Dict[str, Any]] = None
) -> str
```

Augment a system prompt with policy guidelines.

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `system_prompt` | `str` | - | The original system prompt |
| `context` | `Optional[Dict[str, Any]]` | `None` | Additional context |

#### Returns

- `str`: The augmented system prompt

#### Example

```python
augmented_prompt = await guardrails.augment_system_prompt(
    system_prompt="Original prompt",
    context={"matched_policies": ["policy1", "policy2"]}
)
```

## Components

### Fast Rules Engine

The fast rules engine provides quick policy checks:

```python
fast_result = guardrails.fast_rules.evaluate(message, context)
if fast_result["confidence"] > 0.8 and not fast_result["allowed"]:
    return handle_violation(fast_result)
```

### Policy Augmentation

Policy augmentation enhances system prompts:

```python
augmentation_result = await guardrails.policy_augmentation.process_message(
    message, context
)
if augmentation_result["augmented"]:
    return handle_augmented_response(augmentation_result)
```

### LLM Fallback

The LLM fallback provides advanced policy checking:

```python
if fast_result["confidence"] < 0.4:
    llm_result = await guardrails.llm_fallback.evaluate(
        message, context, fast_result
    )
    return handle_llm_result(llm_result)
```

### State Manager

Manage conversation state:

```python
state = guardrails.state_manager.get_state("conv123")
guardrails.state_manager.update_state("conv123", {
    "policy_violations": [
        {"policy_id": "content_moderation", "timestamp": "2024-03-22T12:00:00Z"}
    ]
})
```

## Properties

| Property | Type | Description |
|----------|------|-------------|
| `fast_rules` | `FastRulesEngine` | The fast rules engine |
| `policy_augmentation` | `PolicyAugmentation` | The policy augmentation component |
| `llm_fallback` | `LLMFallback` | The LLM fallback component |
| `state_manager` | `StateManager` | The state manager |
| `tracer` | `Tracer` | The OpenTelemetry tracer |

## Related Documentation

- [Guardrails Guide](../core-concepts/guardrails.md)
- [Policy Enforcement Guide](../core-concepts/policy-enforcement.md)
- [Examples](../examples/advanced-guardrails.md)
- [API Reference](../api/rizk.md) 