---
title: "Google Gemini Integration"
description: "Google Gemini Integration"
---

# Google Gemini Integration

The Rizk SDK provides seamless integration with Google's Gemini API, automatically adding observability, guardrails, and policy enforcement to your Gemini-powered applications.

## Quick Start

```python
import google.generativeai as genai
from rizk.sdk import Rizk
from rizk.sdk.decorators import workflow, guardrails

# Initialize Rizk SDK
rizk = Rizk.init(app_name="Gemini-App", enabled=True)

# Configure Gemini
genai.configure(api_key="your-google-api-key")

@workflow(name="gemini_chat", organization_id="acme", project_id="ai_assistant")
@guardrails(enforcement_level="strict")
def gemini_completion(user_message: str) -> str:
    """Create a Gemini completion with monitoring and governance."""
    
    model = genai.GenerativeModel('gemini-pro')
    response = model.generate_content(user_message)
    
    return response.text

# Usage
result = gemini_completion("Explain quantum computing in simple terms")
print(result)
```

## Environment Setup

```bash
# Windows PowerShell
$env:GOOGLE_API_KEY="your-google-api-key-here"

# Linux/macOS
export GOOGLE_API_KEY="your-google-api-key-here"
```

```python
import os
import google.generativeai as genai

# Use environment variable
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
```

## Supported Models

```python
models = {
    "pro": "gemini-pro",
    "pro_vision": "gemini-pro-vision",  # For multimodal content
    "flash": "gemini-1.5-flash",       # Faster, more efficient
    "pro_15": "gemini-1.5-pro"         # Most capable
}

@workflow(name="gemini_pro_chat")
@guardrails(enforcement_level="moderate")
def gemini_pro_completion(message: str, model_name: str = "gemini-pro") -> dict:
    """Gemini Pro completion with monitoring."""
    
    model = genai.GenerativeModel(model_name)
    
    response = model.generate_content(
        message,
        generation_config=genai.types.GenerationConfig(
            candidate_count=1,
            max_output_tokens=500,
            temperature=0.7
        )
    )
    
    return {
        "content": response.text,
        "model": model_name
    }
```

## Configuration

```python
@workflow(name="configured_gemini")
@guardrails(enforcement_level="moderate")
def configured_gemini_completion(message: str, creativity: str = "balanced") -> str:
    """Gemini completion with custom configuration."""
    
    configs = {
        "creative": genai.types.GenerationConfig(
            temperature=1.0,
            top_p=0.95,
            max_output_tokens=800
        ),
        "balanced": genai.types.GenerationConfig(
            temperature=0.7,
            top_p=0.8,
            max_output_tokens=500
        ),
        "precise": genai.types.GenerationConfig(
            temperature=0.1,
            top_p=0.1,
            max_output_tokens=300
        )
    }
    
    model = genai.GenerativeModel('gemini-pro')
    config = configs.get(creativity, configs["balanced"])
    
    response = model.generate_content(message, generation_config=config)
    
    return response.text
```

## Safety Settings

```python
@workflow(name="safe_gemini")
@guardrails(enforcement_level="strict")
def safe_gemini_completion(message: str) -> dict:
    """Gemini completion with safety controls."""
    
    safety_settings = [
        {
            "category": genai.types.HarmCategory.HARM_CATEGORY_HARASSMENT,
            "threshold": genai.types.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE
        },
        {
            "category": genai.types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
            "threshold": genai.types.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE
        }
    ]
    
    model = genai.GenerativeModel('gemini-pro', safety_settings=safety_settings)
    
    try:
        response = model.generate_content(message)
        return {
            "success": True,
            "content": response.text
        }
    except genai.types.BlockedPromptException:
        return {
            "success": False,
            "error": "Content was blocked by safety filters"
        }
```

## Streaming

```python
@workflow(name="gemini_streaming")
@guardrails(enforcement_level="moderate")
def gemini_streaming_completion(message: str) -> str:
    """Gemini completion with streaming."""
    
    model = genai.GenerativeModel('gemini-pro')
    response_chunks = []
    
    response = model.generate_content(message, stream=True)
    
    for chunk in response:
        if chunk.text:
            response_chunks.append(chunk.text)
            print(chunk.text, end="", flush=True)
    
    print()
    return "".join(response_chunks)
```

## Multimodal (Vision)

```python
import PIL.Image

@workflow(name="gemini_vision")
@guardrails(enforcement_level="moderate")
def gemini_vision_analysis(image_path: str, prompt: str) -> str:
    """Analyze images with Gemini Pro Vision."""
    
    image = PIL.Image.open(image_path)
    model = genai.GenerativeModel('gemini-pro-vision')
    
    response = model.generate_content([prompt, image])
    
    return response.text
```

## Best Practices

### API Key Security
```python
import os

# Use environment variables
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def validate_google_key(api_key: str) -> bool:
    return api_key and len(api_key) > 20
```

### Error Handling
```python
from google.api_core import exceptions

@workflow(name="robust_gemini")
def robust_gemini_completion(message: str, max_retries: int = 3) -> str:
    """Gemini completion with error handling."""
    
    model = genai.GenerativeModel('gemini-pro')
    
    for attempt in range(max_retries):
        try:
            response = model.generate_content(message)
            return response.text
        except exceptions.ResourceExhausted:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
                continue
            raise
        except genai.types.BlockedPromptException:
            return "Content was blocked by safety filters"
```

## Related Documentation

- **[OpenAI Integration](openai.md)** - OpenAI API integration
- **[Anthropic Integration](anthropic.md)** - Claude API integration
- **[Custom LLM Providers](custom-llm.md)** - Adding new LLM support 

