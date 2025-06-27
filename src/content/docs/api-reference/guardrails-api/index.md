---
title: "GuardrailsEngine API Reference"
description: "GuardrailsEngine API Reference"
---

# GuardrailsEngine API Reference

The `GuardrailsEngine` is the core component for policy enforcement and evaluation in Rizk SDK. It orchestrates multiple guardrail components to process messages, check outputs, and apply policies.

## Class Overview

```python
from rizk.sdk import Rizk
from rizk.sdk.guardrails.engine import GuardrailsEngine

# Get guardrails engine instance
guardrails = Rizk.get_guardrails()

# Process a message
result = await guardrails.process_message("Hello world")
print(f"Allowed: {result['allowed']}")
```

## Class Structure

```python
class GuardrailsEngine:
    """Main engine for Rizk guardrails policy enforcement."""
    
    fast_rules: Optional[FastRulesEngine]
    policy_augmentation: Optional[PolicyAugmentation]
    llm_fallback: Optional[LLMFallback]
    state_manager: Optional[StateManager]
    tracer: Optional[Tracer]
```

## Initialization

### `GuardrailsEngine.get_instance()`

**Static method to get or create the singleton GuardrailsEngine instance.**

```python
@classmethod
def get_instance(
    cls, 
    config: Optional[Dict[str, Any]] = None
) -> "GuardrailsEngine"
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `config` | `Optional[Dict[str, Any]]` | `None` | Configuration dictionary for initialization |

#### Configuration Keys

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `policies_path` | `str` | Auto-detected | Path to policy files directory |
| `llm_service` | `LLMServiceProtocol` | `DefaultLLMService()` | LLM service for policy evaluation |
| `state_ttl_seconds` | `int` | `3600` | TTL for conversation state |
| `state_cleanup_interval` | `int` | `300` | State cleanup interval in seconds |
| `llm_cache_size` | `int` | `1000` | LLM fallback cache size |

#### Returns

- `GuardrailsEngine`: The singleton guardrails engine instance

#### Example

```python
# Basic initialization
guardrails = GuardrailsEngine.get_instance()

# With custom configuration
config = {
    "policies_path": "/app/custom-policies",
    "state_ttl_seconds": 7200,
    "llm_cache_size": 5000
}
guardrails = GuardrailsEngine.get_instance(config)
```

---

### `GuardrailsEngine.lazy_initialize()`

**Static method to initialize components if not already initialized.**

```python
@classmethod
def lazy_initialize(cls) -> None
```

#### Example

```python
# Get instance and ensure initialization
guardrails = GuardrailsEngine.get_instance()
GuardrailsEngine.lazy_initialize()
```

## Core Methods

### `process_message()`

**Process a user message using the configured guardrail decision logic.**

```python
async def process_message(
    self, 
    message: str, 
    context: Optional[Dict[str, Any]] = None
) -> GuardrailProcessingResult
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `message` | `str` | - | The user message to evaluate |
| `context` | `Optional[Dict[str, Any]]` | `None` | Optional context dictionary |

#### Context Keys

| Key | Type | Description |
|-----|------|-------------|
| `conversation_id` | `str` | Conversation identifier |
| `organization_id` | `str` | Organization identifier |
| `project_id` | `str` | Project identifier |
| `user_id` | `str` | User identifier |
| `policy_ids` | `List[str]` | Specific policies to apply |

#### Returns

- `GuardrailProcessingResult`: Dictionary with evaluation results

#### GuardrailProcessingResult Structure

| Field | Type | Description |
|-------|------|-------------|
| `allowed` | `bool` | Whether the message is allowed |
| `confidence` | `float` | Confidence score (0.0-1.0) |
| `decision_layer` | `str` | Layer that made the decision |
| `violated_policies` | `Optional[List[str]]` | IDs of violated policies |
| `applied_policies` | `Optional[List[str]]` | IDs of applied policies |
| `response` | `Optional[str]` | Alternative response if blocked |
| `blocked_reason` | `Optional[str]` | Reason for blocking |
| `error` | `Optional[str]` | Error description if occurred |

#### Examples

**Basic message processing:**
```python
result = await guardrails.process_message("Can you help me with my account?")
print(f"Allowed: {result['allowed']}")
print(f"Confidence: {result['confidence']}")
print(f"Decision layer: {result['decision_layer']}")
```

**With context:**
```python
context = {
    "conversation_id": "conv_123",
    "organization_id": "acme_corp",
    "project_id": "customer_service",
    "user_id": "user_456"
}

result = await guardrails.process_message(
    "I need help accessing sensitive data", 
    context
)

if not result['allowed']:
    print(f"Blocked: {result['blocked_reason']}")
    print(f"Violated policies: {result['violated_policies']}")
```

**Error handling:**
```python
try:
    result = await guardrails.process_message(message, context)
    if result['error']:
        print(f"Guardrails error: {result['error']}")
    elif result['allowed']:
        # Process the allowed message
        response = process_user_message(message)
    else:
        # Handle blocked message
        response = result.get('response', 'Request blocked by policy')
except Exception as e:
    print(f"Unexpected error: {e}")
    # Implement fallback behavior
```

---

### `check_output()`

**Evaluate an AI-generated response using guardrail policies.**

```python
async def check_output(
    self, 
    ai_response: str, 
    context: Optional[Dict[str, Any]] = None
) -> GuardrailOutputCheckResult
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `ai_response` | `str` | - | The AI-generated response to evaluate |
| `context` | `Optional[Dict[str, Any]]` | `None` | Optional context dictionary |

#### Returns

- `GuardrailOutputCheckResult`: Dictionary with output evaluation results

#### GuardrailOutputCheckResult Structure

| Field | Type | Description |
|-------|------|-------------|
| `allowed` | `bool` | Whether the output is allowed |
| `confidence` | `float` | Confidence score (0.0-1.0) |
| `decision_layer` | `str` | Layer that made the decision |
| `violated_policies` | `Optional[List[str]]` | IDs of violated policies |
| `blocked_reason` | `Optional[str]` | Reason for blocking |
| `transformed_response` | `Optional[str]` | Redacted/modified response |
| `error` | `Optional[str]` | Error description if occurred |

#### Examples

**Basic output checking:**
```python
ai_response = "Here's your account balance: $1,234.56"
result = await guardrails.check_output(ai_response)

if result['allowed']:
    print("Output approved for delivery")
else:
    print(f"Output blocked: {result['blocked_reason']}")
    if result['transformed_response']:
        print(f"Suggested alternative: {result['transformed_response']}")
```

**With conversation context:**
```python
context = {
    "conversation_id": "conv_123",
    "organization_id": "finance_org",
    "user_id": "user_456"
}

result = await guardrails.check_output(
    "Your SSN is 123-45-6789",
    context
)

if not result['allowed']:
    # Use transformed response if available
    safe_response = result.get(
        'transformed_response', 
        'Information has been redacted for privacy'
    )
    print(f"Safe response: {safe_response}")
```

---

### `evaluate()`

**Simplified evaluation interface for framework adapters.**

```python
async def evaluate(
    self, 
    input_text: str, 
    context: Optional[Dict[str, Any]] = None,
    direction: str = "inbound"
) -> Decision
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `input_text` | `str` | - | Text to evaluate against policies |
| `context` | `Optional[Dict[str, Any]]` | `None` | Context information |
| `direction` | `str` | `"inbound"` | Evaluation direction: `"inbound"` or `"outbound"` |

#### Returns

- `Decision`: Simple decision object

#### Decision Structure

| Field | Type | Description |
|-------|------|-------------|
| `allowed` | `bool` | Whether the input is allowed |
| `reason` | `Optional[str]` | Reason for blocking if not allowed |
| `policy_id` | `Optional[str]` | ID of the policy that triggered |
| `confidence` | `Optional[float]` | Confidence score |

#### Examples

**Inbound evaluation:**
```python
decision = await guardrails.evaluate(
    "Can you show me user passwords?",
    context={"user_id": "admin_001"},
    direction="inbound"
)

if not decision.allowed:
    print(f"Request blocked: {decision.reason}")
    print(f"Policy: {decision.policy_id}")
```

**Outbound evaluation:**
```python
decision = await guardrails.evaluate(
    "Password: secret123",
    direction="outbound"
)

if not decision.allowed:
    print("Output contains sensitive information")
```

## Utility Methods

### `augment_system_prompt()`

**Augment a system prompt with policy guidelines.**

```python
def augment_system_prompt(
    self, 
    original_prompt: str, 
    message: str
) -> str
```

#### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `original_prompt` | `str` | The original system prompt |
| `message` | `str` | User message for context |

#### Returns

- `str`: Augmented system prompt with guidelines

#### Example

```python
original = "You are a helpful assistant."
user_message = "Help me with financial advice"

augmented = guardrails.augment_system_prompt(original, user_message)
print(augmented)
# Output: "You are a helpful assistant.\n\nIMPORTANT GUIDELINES:\n- Never provide specific investment advice..."
```

---

### Context Management

### `set_current_guidelines()`

**Set guidelines in the current OpenTelemetry context.**

```python
@classmethod
def set_current_guidelines(cls, guidelines: List[str]) -> None
```

#### Example

```python
guidelines = [
    "Always verify user identity before sharing account information",
    "Never provide passwords or sensitive authentication details"
]
GuardrailsEngine.set_current_guidelines(guidelines)
```

---

### `clear_current_guidelines()`

**Clear guidelines from the current context.**

```python
@classmethod
def clear_current_guidelines(cls) -> None
```

#### Example

```python
# Clear guidelines after processing
GuardrailsEngine.clear_current_guidelines()
```

## Error Handling

The GuardrailsEngine uses the `@handle_errors` decorator for robust error handling:

### Error Response Structure

When errors occur, methods return structured error responses:

```python
# For process_message errors
error_result = {
    "allowed": False,
    "confidence": 1.0,
    "decision_layer": "error",
    "error": "Guardrails processing failed",
    "violated_policies": [],
    "blocked_reason": "Guardrails internal error"
}

# For check_output errors  
error_result = {
    "allowed": False,
    "confidence": 1.0,
    "decision_layer": "error",
    "error": "Guardrails output check failed",
    "violated_policies": [],
    "blocked_reason": "Guardrails internal error"
}
```

### Handling Errors

```python
result = await guardrails.process_message(message)

if result.get('error'):
    # Handle guardrails system error
    print(f"System error: {result['error']}")
    # Implement fallback logic
    fallback_result = handle_without_guardrails(message)
elif not result['allowed']:
    # Handle policy violation
    print(f"Policy violation: {result['blocked_reason']}")
    response = result.get('response', 'Request blocked')
else:
    # Message approved
    response = process_approved_message(message)
```

## Thread Safety

The GuardrailsEngine implements thread-safe patterns:

- **Singleton Pattern**: Thread-safe instance creation with double-checked locking
- **Lazy Initialization**: Thread-safe component initialization using threading events
- **State Management**: Thread-safe conversation state with reentrant locks

```python
import threading
import asyncio

# Safe to use across multiple threads
async def worker(worker_id: int):
    guardrails = GuardrailsEngine.get_instance()
    
    for i in range(10):
        message = f"Message {i} from worker {worker_id}"
        result = await guardrails.process_message(message)
        print(f"Worker {worker_id}: {result['allowed']}")

# Run multiple workers
async def main():
    tasks = [worker(i) for i in range(5)]
    await asyncio.gather(*tasks)
```

## Performance Considerations

### Caching

The GuardrailsEngine implements intelligent caching:

- **LLM Cache**: Caches LLM evaluation results (configurable size)
- **Policy Cache**: Caches policy matching results
- **Framework Detection Cache**: Caches framework detection results

### Batch Processing

For high-throughput scenarios:

```python
# Process multiple messages
messages = ["message1", "message2", "message3"]
context = {"conversation_id": "conv_123"}

results = await asyncio.gather(*[
    guardrails.process_message(msg, context) 
    for msg in messages
])

for i, result in enumerate(results):
    print(f"Message {i}: {result['allowed']}")
```

### Performance Monitoring

```python
import time

# Monitor processing time
start_time = time.time()
result = await guardrails.process_message(message)
processing_time = time.time() - start_time

print(f"Processing took {processing_time:.3f} seconds")
print(f"Decision layer: {result['decision_layer']}")
```

## Integration Examples

### With Framework Adapters

```python
# Framework adapter integration
class MyFrameworkAdapter:
    def __init__(self):
        self.guardrails = GuardrailsEngine.get_instance()
    
    async def apply_input_guardrails(self, user_input: str, context: dict):
        result = await self.guardrails.process_message(user_input, context)
        
        if not result['allowed']:
            raise PolicyViolationError(result['blocked_reason'])
        
        return result.get('response', user_input)
    
    async def apply_output_guardrails(self, ai_output: str, context: dict):
        result = await self.guardrails.check_output(ai_output, context)
        
        if not result['allowed']:
            return result.get(
                'transformed_response',
                'Response blocked by policy'
            )
        
        return ai_output
```

### With Custom LLM Services

```python
from rizk.sdk.guardrails.llm_service import LLMServiceProtocol

class CustomLLMService(LLMServiceProtocol):
    async def evaluate_policy(self, message: str, policy: dict) -> dict:
        # Custom LLM evaluation logic
        return {
            "allowed": True,
            "confidence": 0.9,
            "reasoning": "Content appears safe"
        }

# Use custom LLM service
config = {"llm_service": CustomLLMService()}
guardrails = GuardrailsEngine.get_instance(config)
```

## Best Practices

### 1. Context Management

Always provide context for better policy evaluation:

```python
context = {
    "conversation_id": conversation_id,
    "organization_id": "my_org",
    "project_id": "my_project",
    "user_id": user_id,
    "user_role": user_role
}

result = await guardrails.process_message(message, context)
```

### 2. Error Handling

Implement comprehensive error handling:

```python
async def safe_process_message(message: str, context: dict) -> str:
    try:
        result = await guardrails.process_message(message, context)
        
        if result.get('error'):
            # Log system error and use fallback
            logger.error(f"Guardrails error: {result['error']}")
            return fallback_process(message)
        
        if not result['allowed']:
            # Handle policy violation
            return result.get('response', 'Request blocked')
        
        # Process approved message
        return process_message(message)
        
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return "I'm sorry, I cannot process that request right now."
```

### 3. Performance Optimization

Monitor and optimize performance:

```python
# Use async/await properly
async def batch_process(messages: List[str], context: dict):
    # Process in parallel for better performance
    tasks = [
        guardrails.process_message(msg, context) 
        for msg in messages
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    return [
        r if not isinstance(r, Exception) else {"allowed": False, "error": str(r)}
        for r in results
    ]
```

### 4. Resource Management

Properly manage resources in long-running applications:

```python
# Initialize once, use many times
guardrails = GuardrailsEngine.get_instance({
    "llm_cache_size": 10000,  # Larger cache for production
    "state_ttl_seconds": 7200  # 2-hour conversation TTL
})

# Use throughout application lifecycle
async def handle_request(message: str, context: dict):
    result = await guardrails.process_message(message, context)
    return process_result(result)
```

## Related APIs

- **[Rizk Class API](./rizk-class.md)** - SDK initialization and configuration
- **[Decorators API](./decorators-api.md)** - Function and class decorators
- **[Types API](./types.md)** - Type definitions and protocols
- **[Configuration API](./configuration-api.md)** - Configuration management 

