---
title: "Policy Enforcement"
description: "Documentation for Policy Enforcement"
---

Policy enforcement is a core feature of the Rizk SDK that helps ensure your AI applications adhere to defined rules and guidelines. This guide explains how to use and configure policy enforcement.

## Overview

Policy enforcement in the Rizk SDK works through multiple layers:

1. **Fast Rules**: Quick pattern-based checks
2. **Policy Augmentation**: Context-aware policy application
3. **LLM Fallback**: Advanced language model-based evaluation

## Basic Usage

### Using Policy Decorators

The simplest way to apply policies is through decorators:

```python
from rizk.sdk import Rizk
from rizk.sdk.decorators import agent, add_policies

@agent
@add_policies(["content_moderation", "safety"])
async def my_agent(query: str):
    # Your agent logic here
    pass
```

### Manual Policy Enforcement

You can also enforce policies manually:

```python
from rizk.sdk import Rizk

# Initialize SDK
client = Rizk.init(app_name="my_app", api_key="your_api_key")

# Get guardrails instance
guardrails = Rizk.get_guardrails()

# Process a message with policies
result = await guardrails.process_message(
    message="User query here",
    context={
        "conversation_id": "unique_id",
        "user_id": "user123"
    }
)

if not result["allowed"]:
    print(f"Policy violation: {result.get('blocked_reason')}")
```

## Policy Types

### 1. Content Moderation

Basic content filtering:

```yaml
policies:
  - id: content_moderation
    description: Basic content moderation
    rules:
      - pattern: "harmful|dangerous"
        action: block
        confidence: 0.9
```

### 2. Safety Policies

Safety-focused rules:

```yaml
policies:
  - id: safety
    description: Safety guidelines
    rules:
      - pattern: "illegal|harmful|dangerous"
        action: block
        confidence: 0.95
```

### 3. Custom Policies

Define your own policies:

```yaml
policies:
  - id: custom_policy
    description: Custom policy rules
    rules:
      - pattern: "your_pattern"
        action: block
        confidence: 0.8
```

## Policy Configuration

### Policy File Structure

Policies are defined in YAML files:

```yaml
policies:
  - id: policy_id
    description: Policy description
    rules:
      - pattern: regex_pattern
        action: block|warn|allow
        confidence: 0.0-1.0
```

### Policy Actions

Available policy actions:

- `block`: Prevent the action
- `warn`: Allow but log warning
- `allow`: Explicitly allow

## Context and State

### Setting Context

Provide context for better policy evaluation:

```python
# Set context for the current conversation
Rizk.set_association_properties({
    "organization_id": "org123",
    "project_id": "project456",
    "agent_id": "agent789"
})

# Process message with context
result = await guardrails.process_message(
    message="User query",
    context={
        "conversation_id": "conv123",
        "recent_messages": [
            {"role": "user", "content": "Previous message"}
        ]
    }
)
```

### State Management

Policies can maintain state across interactions:

```python
# Get state for a conversation
state = guardrails.state_manager.get_state("conv123")

# Update state
guardrails.state_manager.update_state("conv123", {
    "policy_violations": [
        {"policy_id": "content_moderation", "timestamp": "2024-03-22T12:00:00Z"}
    ]
})
```

## Policy Evaluation Flow

1. **Fast Rules Check**:
   ```python
   fast_result = guardrails.fast_rules.evaluate(message, context)
   if fast_result["confidence"] > 0.8 and not fast_result["allowed"]:
       return handle_violation(fast_result)
   ```

2. **Policy Augmentation**:
   ```python
   augmentation_result = await guardrails.policy_augmentation.process_message(
       message, context
   )
   if augmentation_result["augmented"]:
       return handle_augmented_response(augmentation_result)
   ```

3. **LLM Fallback**:
   ```python
   if fast_result["confidence"] < 0.4:
       llm_result = await guardrails.llm_fallback.evaluate(
           message, context, fast_result
       )
       return handle_llm_result(llm_result)
   ```

## Best Practices

1. **Use Appropriate Confidence Levels**:
   ```yaml
   rules:
     - pattern: "critical_pattern"
       action: block
       confidence: 0.95  # High confidence for critical rules
     - pattern: "warning_pattern"
       action: warn
       confidence: 0.7   # Lower confidence for warnings
   ```

2. **Handle Policy Violations**:
   ```python
   if not result["allowed"]:
       logging.warning(f"Policy violation: {result.get('blocked_reason')}")
       span.set_attribute("policy.violation", True)
       span.set_attribute("policy.violation_reason", result.get('blocked_reason'))
   ```

3. **Monitor Policy Performance**:
   ```python
   if result["decision_layer"] == "fast_rules":
       logging.info("Using fast rules for policy check")
   elif result["decision_layer"] == "llm_fallback":
       logging.info("Using LLM for policy check")
   ```

## Next Steps

- [Policy Management Guide](../guides/policy-management.md)
- [Advanced Guardrails Examples](../examples/advanced-guardrails.md)
- [Custom Policies Guide](../examples/custom-policies.md)
- [API Reference](../api/guardrails-engine.md) 