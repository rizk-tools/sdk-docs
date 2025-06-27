---
title: "OpenAI Completions Adapter"
description: "OpenAI Completions Adapter"
---

# OpenAI Completions Adapter

The OpenAI Completions Adapter provides direct integration with OpenAI's Chat Completions API by patching the `openai.chat.completions.create` method. This adapter automatically injects policy guidelines into system prompts and applies outbound guardrails to responses.

## Overview

Unlike the OpenAI Responses Adapter which processes response objects, the Completions Adapter intercepts the actual API calls to:

- **Inject Policy Guidelines**: Automatically augments system prompts with relevant policy directives
- **Apply Outbound Guardrails**: Evaluates and potentially blocks responses based on content policies
- **Support Async Operations**: Patches both sync and async completion methods
- **Maintain Compatibility**: Works transparently with existing OpenAI client code

## Installation

```bash
pip install rizk[openai]
# or
pip install rizk openai
```

## How It Works

The adapter patches the OpenAI client at runtime:

```python
# Before patching
response = openai.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": "Hello"}]
)

# After Rizk initialization - same code, enhanced behavior
rizk = Rizk.init(app_name="MyApp", enabled=True)
# Now includes automatic policy injection and response filtering
response = openai.chat.completions.create(
    model="gpt-3.5-turbo", 
    messages=[{"role": "user", "content": "Hello"}]
)
```

## Basic Usage

### Automatic Integration

The adapter activates automatically when Rizk SDK is initialized:

```python
import openai
from rizk.sdk import Rizk

# Initialize Rizk - this patches OpenAI automatically
rizk = Rizk.init(
    app_name="OpenAI-App",
    api_key="your-rizk-api-key",
    enabled=True
)

# Your existing OpenAI code works unchanged
client = openai.OpenAI(api_key="your-openai-key")

response = client.chat.completions.create(
    model="gpt-4",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain quantum computing"}
    ]
)

print(response.choices[0].message.content)
```

### With Custom Policies

```python
import openai
from rizk.sdk import Rizk

# Initialize with custom policies
rizk = Rizk.init(
    app_name="SecureChat",
    policies_path="./policies",
    enabled=True
)

# Policy guidelines are automatically injected
response = openai.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "You are a customer service assistant."},
        {"role": "user", "content": "Can you help me with my account?"}
    ]
)

# Response is automatically evaluated against policies
print(response.choices[0].message.content)
```

## Policy Injection

The adapter automatically injects policy guidelines into system prompts:

### Before Injection
```python
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Tell me about AI safety"}
]
```

### After Injection
```python
messages = [
    {
        "role": "system", 
        "content": "You are a helpful assistant.\n\nIMPORTANT POLICY DIRECTIVES:\nâ€¢ Ensure all AI safety discussions are balanced and factual\nâ€¢ Avoid speculation about future AI capabilities\nâ€¢ Focus on current best practices and research"
    },
    {"role": "user", "content": "Tell me about AI safety"}
]
```

## Outbound Guardrails

The adapter evaluates responses and can block inappropriate content:

```python
import openai
from rizk.sdk import Rizk

rizk = Rizk.init(
    app_name="ContentFilter",
    enabled=True
)

# This response might be blocked if it violates policies
response = openai.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "user", "content": "Tell me something inappropriate"}
    ]
)

# If blocked, you'll get a modified response
if hasattr(response, '_rizk_blocked'):
    print("Response was blocked by policy")
    print(f"Reason: {response._rizk_block_reason}")
```

## Async Support

The adapter automatically patches async methods:

```python
import asyncio
import openai
from rizk.sdk import Rizk

async def async_chat():
    rizk = Rizk.init(app_name="AsyncApp", enabled=True)
    
    # Async calls are also patched
    response = await openai.chat.completions.acreate(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": "Hello async world!"}
        ]
    )
    
    return response.choices[0].message.content

# Run async function
result = asyncio.run(async_chat())
print(result)
```

## Framework Integration

### With Decorators

```python
import openai
from rizk.sdk import Rizk
from rizk.sdk.decorators import workflow, guardrails

rizk = Rizk.init(app_name="DecoratedApp", enabled=True)

@workflow(name="chat_completion", organization_id="demo", project_id="openai")
@guardrails()
def chat_with_openai(prompt: str, model: str = "gpt-3.5-turbo") -> str:
    """Chat with OpenAI with full monitoring and governance."""
    
    response = openai.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=150
    )
    
    return response.choices[0].message.content

# Usage
result = chat_with_openai("Explain machine learning in simple terms")
print(result)
```

### Error Handling

```python
import openai
from rizk.sdk import Rizk

rizk = Rizk.init(app_name="RobustApp", enabled=True)

def safe_chat(prompt: str) -> str:
    try:
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=100
        )
        
        # Check if response was blocked
        if hasattr(response, '_rizk_blocked') and response._rizk_blocked:
            return f"Response blocked: {response._rizk_block_reason}"
        
        return response.choices[0].message.content
        
    except openai.RateLimitError:
        return "Rate limit exceeded. Please try again later."
    except openai.APIError as e:
        return f"OpenAI API error: {str(e)}"
    except Exception as e:
        return f"Unexpected error: {str(e)}"

# Usage
result = safe_chat("Tell me a joke")
print(result)
```

## Configuration

### Custom Policy Paths

```python
from rizk.sdk import Rizk

# Load policies from custom directory
rizk = Rizk.init(
    app_name="CustomPolicies",
    policies_path="/path/to/policies",
    enabled=True
)
```

### Disable Specific Features

```python
from rizk.sdk import Rizk

# Disable outbound guardrails but keep policy injection
rizk = Rizk.init(
    app_name="PolicyOnly",
    enabled=True,
    # Custom configuration would go here
)
```

## Monitoring and Observability

The adapter automatically creates spans for all OpenAI API calls:

```python
import openai
from rizk.sdk import Rizk

# Enable detailed tracing
rizk = Rizk.init(
    app_name="TracedApp",
    enabled=True,
    trace_content=True  # Include request/response content in traces
)

# This call will be fully traced
response = openai.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": "Hello"}]
)

# Traces include:
# - Request parameters (model, messages, etc.)
# - Response metadata (tokens used, finish reason, etc.)
# - Policy decisions and injections
# - Performance metrics
```

## Best Practices

### 1. System Prompt Design

Design your system prompts to work well with policy injection:

```python
# Good - Clear separation of concerns
system_prompt = """You are a financial advisor assistant.

Your role is to provide general financial guidance and education.
You should always remind users to consult with qualified professionals.

Guidelines will be automatically added below this section."""

# The adapter will append policy guidelines after your content
```

### 2. Error Handling

Always check for blocked responses:

```python
def handle_openai_response(response):
    # Check if response was blocked
    if hasattr(response, '_rizk_blocked') and response._rizk_blocked:
        # Handle blocked response
        return {
            "blocked": True,
            "reason": response._rizk_block_reason,
            "content": None
        }
    
    # Normal response
    return {
        "blocked": False,
        "reason": None,
        "content": response.choices[0].message.content
    }
```

### 3. Performance Considerations

```python
# For high-throughput applications
rizk = Rizk.init(
    app_name="HighThroughput",
    enabled=True,
    # Configure for performance
    disable_batch=True,  # Reduce latency
    llm_cache_size=10000  # Larger cache for repeated queries
)
```

## Troubleshooting

### Common Issues

**1. Policies Not Being Injected**
```python
# Check if guidelines are available
from rizk.sdk.guardrails.engine import GuardrailsEngine

engine = GuardrailsEngine.get_instance()
guidelines = engine.get_current_guidelines()
print(f"Available guidelines: {guidelines}")
```

**2. Response Blocking Issues**
```python
# Debug response evaluation
response = openai.chat.completions.create(...)

if hasattr(response, '_rizk_blocked'):
    print(f"Blocked: {response._rizk_blocked}")
    print(f"Reason: {response._rizk_block_reason}")
    print(f"Original content: {response._rizk_original_content}")
```

**3. Import Errors**
```bash
# Ensure OpenAI is installed
pip install openai>=1.0.0

# Check Rizk installation
pip show rizk
```

### Debug Mode

Enable debug logging to see adapter behavior:

```python
import logging
from rizk.sdk import Rizk

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("rizk.adapters.openai_completion")
logger.setLevel(logging.DEBUG)

rizk = Rizk.init(app_name="DebugApp", enabled=True)
```

## Advanced Usage

### Custom Policy Evaluation

```python
from rizk.sdk import Rizk
from rizk.sdk.guardrails.types import PolicySet, Policy

# Create custom policy
custom_policy = Policy(
    id="openai_custom",
    name="OpenAI Custom Policy",
    description="Custom rules for OpenAI interactions",
    action="allow",
    guidelines=[
        "Always provide sources for factual claims",
        "Limit responses to 200 words maximum",
        "Use professional tone for business queries"
    ]
)

policy_set = PolicySet(policies=[custom_policy])

# Initialize with custom policies
rizk = Rizk.init(
    app_name="CustomPolicyApp",
    enabled=True
)

# Your OpenAI calls will now use these custom policies
```

### Integration with Other Systems

```python
import openai
from rizk.sdk import Rizk

# Initialize with custom telemetry endpoint
rizk = Rizk.init(
    app_name="IntegratedApp",
    opentelemetry_endpoint="https://your-otlp-collector.com",
    enabled=True
)

# All OpenAI calls will send telemetry to your custom endpoint
response = openai.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Hello"}]
)
```

The OpenAI Completions Adapter provides seamless integration with minimal code changes, ensuring your OpenAI applications are automatically governed and monitored according to your organization's policies. 

