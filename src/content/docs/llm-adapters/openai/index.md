---
title: "OpenAI Integration"
description: "OpenAI Integration"
---

# OpenAI Integration

The Rizk SDK provides seamless integration with OpenAI's APIs, automatically adding observability, guardrails, and policy enforcement to your OpenAI-powered applications. This integration supports both the Completions API and Chat Completions API with automatic prompt augmentation and response monitoring.

## Overview

The OpenAI adapter automatically patches OpenAI API calls to:

- **Inject Guidelines**: Automatically enhance prompts with relevant policy guidelines
- **Apply Guardrails**: Evaluate inputs and outputs against organizational policies
- **Monitor Performance**: Track latency, token usage, and costs
- **Ensure Compliance**: Enforce content safety and regulatory requirements
- **Provide Observability**: Generate detailed traces and metrics

## Quick Start

### Basic Setup

```python
import openai
from rizk.sdk import Rizk
from rizk.sdk.decorators import workflow, guardrails

# Initialize Rizk SDK
rizk = Rizk.init(
    app_name="OpenAI-App",
    enabled=True
)

# Set your OpenAI API key
openai.api_key = "your-openai-api-key-here"  # Better to use environment variable

@workflow(name="openai_chat", organization_id="acme", project_id="ai_assistant")
@guardrails(enforcement_level="strict")
def chat_completion(user_message: str) -> str:
    """Create a chat completion with automatic monitoring and governance."""
    
    response = openai.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": user_message}
        ],
        max_tokens=150,
        temperature=0.7
    )
    
    return response.choices[0].message.content

# Usage
result = chat_completion("Explain quantum computing in simple terms")
print(result)
```

### Environment Setup

Set your OpenAI API key as an environment variable:

```bash
# Windows PowerShell
$env:OPENAI_API_KEY="sk-your-api-key-here"

# Linux/macOS
export OPENAI_API_KEY="sk-your-api-key-here"
```

```python
import os
import openai

# Use environment variable
openai.api_key = os.getenv("OPENAI_API_KEY")
```

## Supported OpenAI APIs

### Chat Completions API (Recommended)

The Chat Completions API is the modern interface for GPT models:

```python
from rizk.sdk import Rizk
from rizk.sdk.decorators import workflow, guardrails
import openai

rizk = Rizk.init(app_name="ChatApp", enabled=True)

@workflow(name="advanced_chat", organization_id="company", project_id="chatbot")
@guardrails(enforcement_level="moderate")
def advanced_chat_completion(
    messages: list,
    model: str = "gpt-4",
    temperature: float = 0.7,
    max_tokens: int = 500
) -> dict:
    """Advanced chat completion with full parameter control."""
    
    response = openai.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0
    )
    
    return {
        "content": response.choices[0].message.content,
        "model": response.model,
        "usage": response.usage.dict() if response.usage else None,
        "finish_reason": response.choices[0].finish_reason
    }

# Multi-turn conversation
conversation = [
    {"role": "system", "content": "You are an expert Python developer."},
    {"role": "user", "content": "How do I implement a binary search algorithm?"},
]

result = advanced_chat_completion(conversation, model="gpt-4", temperature=0.3)
print(f"Response: {result['content']}")
print(f"Tokens used: {result['usage']}")
```

### Completions API (Legacy)

For older models or specific use cases:

```python
@workflow(name="text_completion", organization_id="company", project_id="legacy")
@guardrails(enforcement_level="strict")
def text_completion(prompt: str, model: str = "gpt-3.5-turbo-instruct") -> str:
    """Text completion using the legacy Completions API."""
    
    response = openai.completions.create(
        model=model,
        prompt=prompt,
        max_tokens=200,
        temperature=0.8,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0
    )
    
    return response.choices[0].text.strip()

# Usage
result = text_completion("Write a haiku about artificial intelligence:")
print(result)
```

## Authentication and Configuration

### API Key Management

```python
import os
from rizk.sdk import Rizk
import openai

# Method 1: Environment variable (recommended)
openai.api_key = os.getenv("OPENAI_API_KEY")

# Method 2: Direct assignment (not recommended for production)
openai.api_key = "sk-your-api-key-here"

# Method 3: Using OpenAI client configuration
from openai import OpenAI

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    organization="org-your-organization-id",  # Optional
    project="proj-your-project-id"            # Optional
)

@workflow(name="client_based_chat")
def chat_with_client(message: str) -> str:
    """Chat using configured OpenAI client."""
    
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": message}]
    )
    
    return response.choices[0].message.content
```

### Organization and Project Settings

```python
# Set organization and project for billing and access control
openai.organization = "org-your-organization-id"
openai.project = "proj-your-project-id"

# Or use the client approach
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    organization="org-your-organization-id",
    project="proj-your-project-id"
)
```

## Rate Limiting and Error Handling

### Built-in Rate Limiting

The OpenAI adapter automatically handles rate limiting:

```python
import time
from openai import RateLimitError, APITimeoutError, APIError

@workflow(name="robust_chat")
@guardrails(enforcement_level="moderate")
def robust_chat_completion(message: str, max_retries: int = 3) -> str:
    """Chat completion with robust error handling."""
    
    for attempt in range(max_retries):
        try:
            response = openai.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": message}],
                timeout=30.0  # 30 second timeout
            )
            return response.choices[0].message.content
            
        except RateLimitError as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # Exponential backoff
                print(f"Rate limit hit, waiting {wait_time} seconds...")
                time.sleep(wait_time)
                continue
            raise
            
        except APITimeoutError:
            if attempt < max_retries - 1:
                print(f"Timeout on attempt {attempt + 1}, retrying...")
                continue
            raise
            
        except APIError as e:
            print(f"OpenAI API error: {e}")
            raise

# Usage with automatic retry
result = robust_chat_completion("What is machine learning?")
print(result)
```

## Model Configuration and Optimization

### Model Selection

```python
@workflow(name="multi_model_chat")
@guardrails(enforcement_level="moderate")
def multi_model_chat(message: str, use_case: str = "general") -> str:
    """Select optimal model based on use case."""
    
    # Model selection based on use case
    model_config = {
        "general": {"model": "gpt-4", "temperature": 0.7, "max_tokens": 500},
        "creative": {"model": "gpt-4", "temperature": 1.0, "max_tokens": 800},
        "analytical": {"model": "gpt-4", "temperature": 0.2, "max_tokens": 1000},
        "code": {"model": "gpt-4", "temperature": 0.1, "max_tokens": 1500},
        "fast": {"model": "gpt-3.5-turbo", "temperature": 0.7, "max_tokens": 300}
    }
    
    config = model_config.get(use_case, model_config["general"])
    
    response = openai.chat.completions.create(
        model=config["model"],
        messages=[{"role": "user", "content": message}],
        temperature=config["temperature"],
        max_tokens=config["max_tokens"]
    )
    
    return response.choices[0].message.content

# Usage examples
general_response = multi_model_chat("What is AI?", "general")
creative_response = multi_model_chat("Write a story about robots", "creative")
code_response = multi_model_chat("Write a Python function to sort a list", "code")
```

## Best Practices

### 1. **API Key Security**

```python
import os

# Use environment variables
openai.api_key = os.getenv("OPENAI_API_KEY")

# Validate API key format
def validate_api_key(api_key: str) -> bool:
    """Validate OpenAI API key format."""
    return api_key and api_key.startswith("sk-") and len(api_key) > 20
```

### 2. **Cost Management**

```python
@workflow(name="cost_managed_chat")
@guardrails(enforcement_level="moderate")
def cost_managed_chat(message: str, daily_budget: float = 10.0) -> dict:
    """Chat with daily cost limits."""
    
    # Check daily usage (implement your tracking)
    daily_usage = get_daily_usage()  # Your implementation
    
    if daily_usage >= daily_budget:
        return {
            "success": False,
            "error": "daily_budget_exceeded",
            "message": f"Daily budget of ${daily_budget} exceeded"
        }
    
    # Proceed with request
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",  # Use cost-effective model
        messages=[{"role": "user", "content": message}],
        max_tokens=200  # Limit output tokens
    )
    
    return {
        "success": True,
        "content": response.choices[0].message.content
    }
```

## Related Documentation

- **[Anthropic Integration](anthropic.md)** - Claude API integration
- **[Gemini Integration](gemini.md)** - Google Gemini integration
- **[Custom LLM Providers](custom-llm.md)** - Adding new LLM support
- **[Guardrails](../guardrails/policy-system.md)** - Policy enforcement system
- **[Observability](../observability/tracing.md)** - Monitoring and tracing

---

The OpenAI integration provides enterprise-grade observability, governance, and monitoring for your OpenAI-powered applications. With automatic prompt augmentation, response evaluation, and comprehensive error handling, you can build robust, compliant AI systems with confidence. 

