---
title: "Anthropic Claude Integration"
description: "Anthropic Claude Integration"
---

# Anthropic Claude Integration

The Rizk SDK provides seamless integration with Anthropic's Claude API, automatically adding observability, guardrails, and policy enforcement to your Claude-powered applications.

## Quick Start

```python
import anthropic
from rizk.sdk import Rizk
from rizk.sdk.decorators import workflow, guardrails

# Initialize Rizk SDK
rizk = Rizk.init(app_name="Claude-App", enabled=True)

# Initialize Claude client
client = anthropic.Anthropic(api_key="your-anthropic-api-key")

@workflow(name="claude_chat", organization_id="acme", project_id="ai_assistant")
@guardrails(enforcement_level="strict")
def claude_completion(user_message: str) -> str:
    """Create a Claude completion with monitoring and governance."""
    
    response = client.messages.create(
        model="claude-3-sonnet-20240229",
        max_tokens=150,
        messages=[{"role": "user", "content": user_message}]
    )
    
    return response.content[0].text

# Usage
result = claude_completion("Explain quantum computing in simple terms")
print(result)
```

## Supported Claude Models

### Claude 3 Models

```python
models = {
    "haiku": "claude-3-haiku-20240307",      # Fastest, most cost-effective
    "sonnet": "claude-3-sonnet-20240229",   # Balanced performance and speed
    "opus": "claude-3-opus-20240229"        # Most capable, highest cost
}

@workflow(name="claude3_chat")
@guardrails(enforcement_level="moderate")
def claude3_completion(message: str, model: str = "claude-3-sonnet-20240229") -> dict:
    """Claude 3 completion with full parameter control."""
    
    response = client.messages.create(
        model=model,
        max_tokens=500,
        temperature=0.7,
        messages=[{"role": "user", "content": message}]
    )
    
    return {
        "content": response.content[0].text,
        "model": response.model,
        "usage": {
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens
        }
    }
```

## Authentication

```python
import os
import anthropic

# Environment variable (recommended)
client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# With configuration
client = anthropic.Anthropic(
    api_key=os.getenv("ANTHROPIC_API_KEY"),
    timeout=30.0,
    max_retries=3
)
```

## Error Handling

```python
from anthropic import RateLimitError, APITimeoutError

@workflow(name="robust_claude")
@guardrails(enforcement_level="moderate")
def robust_claude_completion(message: str, max_retries: int = 3) -> str:
    """Claude completion with error handling."""
    
    for attempt in range(max_retries):
        try:
            response = client.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=500,
                messages=[{"role": "user", "content": message}]
            )
            return response.content[0].text
            
        except RateLimitError as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                time.sleep(wait_time)
                continue
            raise
        except APITimeoutError:
            if attempt < max_retries - 1:
                continue
            raise
```

## Streaming

```python
@workflow(name="claude_streaming")
@guardrails(enforcement_level="moderate")
def claude_streaming_completion(message: str) -> str:
    """Claude completion with streaming."""
    
    response_chunks = []
    
    with client.messages.stream(
        model="claude-3-sonnet-20240229",
        max_tokens=500,
        messages=[{"role": "user", "content": message}]
    ) as stream:
        for chunk in stream:
            if chunk.type == "content_block_delta":
                content = chunk.delta.text
                response_chunks.append(content)
                print(content, end="", flush=True)
    
    print()
    return "".join(response_chunks)
```

## Best Practices

### API Key Security
```python
# Use environment variables
client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# Validate key format
def validate_anthropic_key(api_key: str) -> bool:
    return api_key and api_key.startswith("sk-ant-") and len(api_key) > 20
```

### Cost Management
```python
@workflow(name="cost_optimized_claude")
def cost_optimized_claude(message: str, complexity: str = "medium") -> dict:
    """Cost-optimized Claude completion."""
    
    models = {
        "simple": "claude-3-haiku-20240307",
        "medium": "claude-3-sonnet-20240229", 
        "complex": "claude-3-opus-20240229"
    }
    
    response = client.messages.create(
        model=models.get(complexity, models["medium"]),
        max_tokens=500,
        messages=[{"role": "user", "content": message}]
    )
    
    return {"content": response.content[0].text}
```

## Related Documentation

- **[OpenAI Integration](openai.md)** - OpenAI API integration
- **[Gemini Integration](gemini.md)** - Google Gemini integration
- **[Custom LLM Providers](custom-llm.md)** - Adding new LLM support

---

The Anthropic Claude integration provides enterprise-grade observability, governance, and monitoring for your Claude-powered applications. With automatic prompt augmentation, response evaluation, and comprehensive error handling, you can build robust, compliant AI systems with confidence. 

