---
title: "Guardrails"
description: "Documentation for Guardrails"
---

Guardrails are a powerful feature of the Rizk SDK that helps ensure your AI applications operate safely and ethically. This guide explains how guardrails work and how to use them effectively.

## Overview

Guardrails provide multiple layers of protection:

1. **Fast Rules**: Quick, rule-based checks for immediate policy violations
2. **Policy Augmentation**: Context-aware policy enforcement
3. **LLM Fallback**: Advanced language model-based policy checking

## Basic Usage

### Using Guardrails with Decorators

The simplest way to use guardrails is through decorators:

```python
from rizk.sdk import Rizk
from rizk.sdk.decorators import agent, add_policies

@agent
@add_policies(["content_moderation", "safety"])
async def my_agent(query: str):
    # Your agent logic here
    pass
```

### Manual Guardrails Processing

You can also process messages manually through the guardrails engine:

```python
from rizk.sdk import Rizk

# Initialize SDK
client = Rizk.init(app_name="my_app", api_key="your_api_key")

# Get guardrails instance
guardrails = Rizk.get_guardrails()

# Process a message
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

## Guardrails Components

### 1. Fast Rules Engine

Fast rules provide immediate policy checks using predefined rules:

```python
# Example fast rule in YAML
- id: no_harmful_content
  description: Block harmful content
  rules:
    - pattern: "harmful|dangerous|illegal"
      action: block
      confidence: 0.9
```

### 2. Policy Augmentation

Policy augmentation enhances the system prompt with policy guidelines:

```python
# Example policy augmentation
@add_policies(["content_moderation"])
async def my_agent(query: str):
    # The system prompt will be automatically augmented with policy guidelines
    response = await process_query(query)
    return response
```

### 3. LLM Fallback

The LLM fallback provides advanced policy checking:

```python
# The LLM fallback is automatically used when fast rules have low confidence
result = await guardrails.process_message(
    message="Complex query requiring deep analysis",
    context={"conversation_id": "unique_id"}
)

if result["decision_layer"] == "llm_fallback":
    print("Using LLM for policy evaluation")
```

## Policy Configuration

### Default Policies

The SDK includes default policies in `default_policies.yaml`:

```yaml
policies:
  - id: content_moderation
    description: Basic content moderation
    rules:
      - pattern: "harmful|dangerous"
        action: block
        confidence: 0.9
```

### Custom Policies

Create your own policies in YAML format:

```yaml
policies:
  - id: custom_policy
    description: Custom policy rules
    rules:
      - pattern: "your_pattern"
        action: block
        confidence: 0.8
```

## Context and State Management

Guardrails maintain conversation state for better context:

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

## Output Checking

Guardrails can also check AI-generated responses:

```python
# Generate response
response = "AI generated response"

# Check output
output_check = await guardrails.check_output(
    ai_response=response,
    context={"conversation_id": "conv123"}
)

if not output_check["allowed"]:
    print("Response blocked: Policy violation")
```

## Best Practices

1. **Always Use Context**:
   ```python
   context = {
       "conversation_id": str(uuid.uuid4()),
       "user_id": "user123",
       "timestamp": datetime.now().isoformat()
   }
   ```

2. **Handle Policy Violations**:
   ```python
   result = await guardrails.process_message(message, context)
   if not result["allowed"]:
       logging.warning(f"Policy violation: {result.get('blocked_reason')}")
       return handle_violation(result)
   ```

3. **Monitor Decision Layers**:
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