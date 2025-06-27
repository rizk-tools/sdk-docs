---
title: "Ollama Local Model Integration"
description: "Ollama Local Model Integration"
---

# Ollama Local Model Integration

The Rizk SDK provides seamless integration with Ollama for running local LLM models, automatically adding observability, guardrails, and policy enforcement to your locally-hosted AI applications.

## Quick Start

```python
import ollama
from rizk.sdk import Rizk
from rizk.sdk.decorators import workflow, guardrails

# Initialize Rizk SDK
rizk = Rizk.init(app_name="Ollama-App", enabled=True)

@workflow(name="ollama_chat", organization_id="acme", project_id="local_ai")
@guardrails(enforcement_level="strict")
def ollama_completion(user_message: str, model: str = "llama2") -> str:
    """Create an Ollama completion with monitoring and governance."""
    
    response = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": user_message}]
    )
    
    return response['message']['content']

# Usage
result = ollama_completion("Explain quantum computing in simple terms")
print(result)
```

## Installation and Setup

### Installing Ollama

```bash
# Windows (PowerShell as Administrator)
# Download from https://ollama.ai and run installer

# Linux
curl -fsSL https://ollama.ai/install.sh | sh

# macOS
brew install ollama
```

### Installing Python Client

```bash
pip install ollama
```

### Pulling Models

```bash
# Pull popular models
ollama pull llama2
ollama pull codellama
ollama pull mistral
ollama pull phi

# List available models
ollama list
```

## Supported Models

```python
@workflow(name="ollama_multi_model")
@guardrails(enforcement_level="moderate")
def ollama_multi_model_completion(message: str, use_case: str = "general") -> dict:
    """Ollama completion with model selection based on use case."""
    
    models = {
        "general": "llama2",           # General purpose conversations
        "code": "codellama",           # Code generation and analysis
        "math": "mistral",             # Mathematical reasoning
        "small": "phi",                # Lightweight, fast responses
        "creative": "llama2:13b",      # Creative writing (if available)
        "instruct": "llama2:7b-chat"   # Instruction following
    }
    
    selected_model = models.get(use_case, models["general"])
    
    response = ollama.chat(
        model=selected_model,
        messages=[{"role": "user", "content": message}]
    )
    
    return {
        "content": response['message']['content'],
        "model": selected_model,
        "use_case": use_case
    }
```

## Configuration

```python
@workflow(name="configured_ollama")
@guardrails(enforcement_level="moderate")
def configured_ollama_completion(
    message: str, 
    model: str = "llama2",
    temperature: float = 0.7,
    max_tokens: int = 500
) -> str:
    """Ollama completion with custom configuration."""
    
    response = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": message}],
        options={
            "temperature": temperature,
            "num_predict": max_tokens,
            "top_p": 0.9,
            "top_k": 40
        }
    )
    
    return response['message']['content']
```

## Streaming

```python
@workflow(name="ollama_streaming")
@guardrails(enforcement_level="moderate")
def ollama_streaming_completion(message: str, model: str = "llama2") -> str:
    """Ollama completion with streaming response."""
    
    response_chunks = []
    
    stream = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": message}],
        stream=True
    )
    
    for chunk in stream:
        content = chunk['message']['content']
        response_chunks.append(content)
        print(content, end="", flush=True)
    
    print()
    return "".join(response_chunks)
```

## Error Handling

```python
import time
from ollama import ResponseError

@workflow(name="robust_ollama")
@guardrails(enforcement_level="moderate")
def robust_ollama_completion(message: str, model: str = "llama2", max_retries: int = 3) -> str:
    """Ollama completion with robust error handling."""
    
    for attempt in range(max_retries):
        try:
            response = ollama.chat(
                model=model,
                messages=[{"role": "user", "content": message}]
            )
            return response['message']['content']
            
        except ResponseError as e:
            if "model not found" in str(e).lower():
                try:
                    print(f"Pulling model: {model}")
                    ollama.pull(model)
                    continue
                except Exception as pull_error:
                    raise Exception(f"Failed to pull model {model}: {pull_error}")
            
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                time.sleep(wait_time)
                continue
            raise
            
        except ConnectionError:
            if attempt < max_retries - 1:
                time.sleep(2)
                continue
            raise Exception("Could not connect to Ollama server. Is it running?")
```

## Model Management

```python
@workflow(name="ollama_model_management")
def manage_ollama_models() -> dict:
    """Manage Ollama models programmatically."""
    
    models = ollama.list()
    
    def model_exists(model_name: str) -> bool:
        return any(model['name'].startswith(model_name) for model in models['models'])
    
    def ensure_model(model_name: str) -> bool:
        if not model_exists(model_name):
            try:
                ollama.pull(model_name)
                return True
            except Exception as e:
                print(f"Failed to pull model {model_name}: {e}")
                return False
        return True
    
    required_models = ["llama2", "codellama"]
    results = {}
    
    for model in required_models:
        results[model] = ensure_model(model)
    
    return {
        "available_models": [model['name'] for model in models['models']],
        "pull_results": results
    }
```

## Best Practices

### Resource Management
```python
import psutil

@workflow(name="resource_aware_ollama")
def resource_aware_ollama_completion(message: str) -> dict:
    """Ollama completion with resource monitoring."""
    
    cpu_percent = psutil.cpu_percent(interval=1)
    memory_percent = psutil.virtual_memory().percent
    
    if cpu_percent > 90 or memory_percent > 85:
        return {
            "success": False,
            "error": "system_overloaded",
            "message": f"System resources too high: CPU {cpu_percent}%, Memory {memory_percent}%"
        }
    
    response = ollama.chat(
        model="llama2",
        messages=[{"role": "user", "content": message}]
    )
    
    return {
        "success": True,
        "content": response['message']['content']
    }
```

### Server Health Check
```python
import requests

def check_ollama_server() -> bool:
    """Check if Ollama server is running."""
    try:
        response = requests.get("http://localhost:11434/api/version", timeout=5)
        return response.status_code == 200
    except:
        return False
```

## Related Documentation

- **[OpenAI Integration](openai.md)** - OpenAI API integration
- **[Anthropic Integration](anthropic.md)** - Claude API integration
- **[Custom LLM Providers](custom-llm.md)** - Adding new LLM support 

