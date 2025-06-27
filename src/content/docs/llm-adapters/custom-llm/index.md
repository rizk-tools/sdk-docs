---
title: "Custom LLM Providers"
description: "Custom LLM Providers"
---

# Custom LLM Providers

The Rizk SDK provides a flexible framework for integrating custom LLM providers, allowing you to add support for any LLM service while maintaining full observability, guardrails, and policy enforcement capabilities.

## Overview

Adding a custom LLM provider involves:

1. **Creating an Adapter**: Implement the base LLM adapter interface
2. **Registering the Provider**: Register your adapter with the Rizk SDK
3. **Configuration**: Set up authentication and endpoints
4. **Integration**: Use your custom provider with Rizk decorators

## Quick Start

```python
from rizk.sdk import Rizk
from rizk.sdk.adapters.base_llm_adapter import BaseLLMAdapter
from rizk.sdk.decorators import workflow, guardrails
import requests

# Custom LLM Provider Implementation
class CustomLLMAdapter(BaseLLMAdapter):
    """Custom LLM provider adapter."""
    
    def __init__(self, api_key: str, base_url: str):
        self.api_key = api_key
        self.base_url = base_url
        super().__init__()
    
    def generate_completion(self, prompt: str, **kwargs) -> dict:
        """Generate completion using custom LLM API."""
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "prompt": prompt,
            "max_tokens": kwargs.get("max_tokens", 500),
            "temperature": kwargs.get("temperature", 0.7)
        }
        
        response = requests.post(
            f"{self.base_url}/completions",
            headers=headers,
            json=data
        )
        
        response.raise_for_status()
        return response.json()

# Initialize Rizk SDK
rizk = Rizk.init(app_name="Custom-LLM-App", enabled=True)

# Register custom provider
custom_llm = CustomLLMAdapter(
    api_key="your-custom-api-key",
    base_url="https://api.your-llm-provider.com"
)

@workflow(name="custom_llm_chat", organization_id="acme", project_id="custom_ai")
@guardrails(enforcement_level="strict")
def custom_llm_completion(user_message: str) -> str:
    """Create completion using custom LLM provider."""
    
    result = custom_llm.generate_completion(
        prompt=user_message,
        max_tokens=300,
        temperature=0.8
    )
    
    return result.get("text", "")

# Usage
result = custom_llm_completion("Explain quantum computing")
print(result)
```

## Base Adapter Interface

### Required Methods

```python
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List

class BaseLLMAdapter(ABC):
    """Base class for all LLM adapters."""
    
    @abstractmethod
    def generate_completion(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Generate a text completion."""
        pass
    
    @abstractmethod
    def generate_chat_completion(self, messages: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
        """Generate a chat completion."""
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the model."""
        pass
    
    def validate_configuration(self) -> bool:
        """Validate adapter configuration."""
        return True
    
    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Estimate cost for token usage."""
        return 0.0
    
    def get_supported_features(self) -> List[str]:
        """Get list of supported features."""
        return ["completion", "chat"]
```

## Complete Custom Provider Example

### HuggingFace Inference API Adapter

```python
import requests
import time
from typing import Dict, Any, List, Optional
from rizk.sdk.adapters.base_llm_adapter import BaseLLMAdapter
from rizk.sdk import Rizk
from rizk.sdk.decorators import workflow, guardrails

class HuggingFaceAdapter(BaseLLMAdapter):
    """HuggingFace Inference API adapter."""
    
    def __init__(self, api_key: str, model_name: str = "microsoft/DialoGPT-large"):
        self.api_key = api_key
        self.model_name = model_name
        self.base_url = "https://api-inference.huggingface.co/models"
        super().__init__()
    
    def generate_completion(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Generate completion using HuggingFace API."""
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": kwargs.get("max_tokens", 100),
                "temperature": kwargs.get("temperature", 0.7),
                "top_p": kwargs.get("top_p", 0.9),
                "do_sample": True
            }
        }
        
        response = requests.post(
            f"{self.base_url}/{self.model_name}",
            headers=headers,
            json=data
        )
        
        if response.status_code == 503:
            # Model is loading, wait and retry
            time.sleep(10)
            response = requests.post(
                f"{self.base_url}/{self.model_name}",
                headers=headers,
                json=data
            )
        
        response.raise_for_status()
        result = response.json()
        
        if isinstance(result, list) and len(result) > 0:
            generated_text = result[0].get("generated_text", "")
            # Remove the original prompt from the response
            if generated_text.startswith(prompt):
                generated_text = generated_text[len(prompt):].strip()
            
            return {
                "text": generated_text,
                "model": self.model_name,
                "usage": {
                    "prompt_tokens": len(prompt.split()),
                    "completion_tokens": len(generated_text.split()),
                    "total_tokens": len(prompt.split()) + len(generated_text.split())
                }
            }
        
        return {"text": "", "model": self.model_name, "usage": {}}
    
    def generate_chat_completion(self, messages: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
        """Generate chat completion by converting messages to prompt."""
        
        # Convert chat messages to a single prompt
        prompt = ""
        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")
            
            if role == "system":
                prompt += f"System: {content}\n"
            elif role == "user":
                prompt += f"Human: {content}\n"
            elif role == "assistant":
                prompt += f"Assistant: {content}\n"
        
        prompt += "Assistant: "
        
        return self.generate_completion(prompt, **kwargs)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        return {
            "model_name": self.model_name,
            "provider": "HuggingFace",
            "type": "text-generation",
            "max_context_length": 1024,  # Typical for many HF models
            "supports_chat": True,
            "supports_streaming": False
        }
    
    def validate_configuration(self) -> bool:
        """Validate HuggingFace configuration."""
        try:
            headers = {"Authorization": f"Bearer {self.api_key}"}
            response = requests.get(
                f"https://huggingface.co/api/models/{self.model_name}",
                headers=headers,
                timeout=10
            )
            return response.status_code == 200
        except:
            return False
    
    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Estimate cost (HuggingFace Inference API is free for public models)."""
        return 0.0
    
    def get_supported_features(self) -> List[str]:
        """Get supported features."""
        return ["completion", "chat", "batch_processing"]

# Usage example
rizk = Rizk.init(app_name="HuggingFace-App", enabled=True)

# Initialize HuggingFace adapter
hf_adapter = HuggingFaceAdapter(
    api_key="your-huggingface-token",
    model_name="microsoft/DialoGPT-large"
)

@workflow(name="huggingface_chat", organization_id="company", project_id="hf_ai")
@guardrails(enforcement_level="moderate")
def huggingface_completion(user_message: str) -> str:
    """Generate completion using HuggingFace."""
    
    result = hf_adapter.generate_completion(
        prompt=user_message,
        max_tokens=150,
        temperature=0.8
    )
    
    return result.get("text", "")

@workflow(name="huggingface_chat_conversation")
@guardrails(enforcement_level="moderate")
def huggingface_chat_completion(messages: List[Dict[str, str]]) -> str:
    """Generate chat completion using HuggingFace."""
    
    result = hf_adapter.generate_chat_completion(
        messages=messages,
        max_tokens=200
    )
    
    return result.get("text", "")

# Usage examples
simple_result = huggingface_completion("What is artificial intelligence?")
print(f"Simple completion: {simple_result}")

chat_messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Explain machine learning"},
    {"role": "assistant", "content": "Machine learning is a subset of AI..."},
    {"role": "user", "content": "Can you give an example?"}
]

chat_result = huggingface_chat_completion(chat_messages)
print(f"Chat completion: {chat_result}")
```

## Advanced Custom Provider Features

### Streaming Support

```python
import json
from typing import Generator, Iterator

class StreamingLLMAdapter(BaseLLMAdapter):
    """Custom LLM adapter with streaming support."""
    
    def __init__(self, api_key: str, base_url: str):
        self.api_key = api_key
        self.base_url = base_url
        super().__init__()
    
    def generate_completion_stream(self, prompt: str, **kwargs) -> Iterator[Dict[str, Any]]:
        """Generate streaming completion."""
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream"
        }
        
        data = {
            "prompt": prompt,
            "stream": True,
            "max_tokens": kwargs.get("max_tokens", 500),
            "temperature": kwargs.get("temperature", 0.7)
        }
        
        response = requests.post(
            f"{self.base_url}/stream",
            headers=headers,
            json=data,
            stream=True
        )
        
        response.raise_for_status()
        
        for line in response.iter_lines():
            if line:
                line = line.decode('utf-8')
                if line.startswith('data: '):
                    try:
                        data = json.loads(line[6:])
                        yield data
                    except json.JSONDecodeError:
                        continue
    
    def generate_completion(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Generate non-streaming completion."""
        chunks = []
        for chunk in self.generate_completion_stream(prompt, **kwargs):
            if 'text' in chunk:
                chunks.append(chunk['text'])
        
        return {
            "text": "".join(chunks),
            "model": "custom-streaming-model",
            "usage": {"prompt_tokens": 0, "completion_tokens": 0}
        }

# Usage with streaming
streaming_adapter = StreamingLLMAdapter(
    api_key="your-api-key",
    base_url="https://api.your-streaming-provider.com"
)

@workflow(name="streaming_custom_llm")
@guardrails(enforcement_level="moderate")
def streaming_custom_completion(prompt: str) -> str:
    """Generate streaming completion."""
    
    response_chunks = []
    
    for chunk in streaming_adapter.generate_completion_stream(prompt):
        if 'text' in chunk:
            text = chunk['text']
            response_chunks.append(text)
            print(text, end="", flush=True)
    
    print()  # New line
    return "".join(response_chunks)
```

### Multi-Model Provider

```python
class MultiModelAdapter(BaseLLMAdapter):
    """Adapter supporting multiple models from the same provider."""
    
    def __init__(self, api_key: str, base_url: str):
        self.api_key = api_key
        self.base_url = base_url
        self.models = {
            "fast": {"name": "fast-model-v1", "max_tokens": 1000, "cost_per_1k": 0.001},
            "balanced": {"name": "balanced-model-v2", "max_tokens": 2000, "cost_per_1k": 0.005},
            "advanced": {"name": "advanced-model-v3", "max_tokens": 4000, "cost_per_1k": 0.02}
        }
        super().__init__()
    
    def generate_completion(self, prompt: str, model_type: str = "balanced", **kwargs) -> Dict[str, Any]:
        """Generate completion with model selection."""
        
        model_config = self.models.get(model_type, self.models["balanced"])
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": model_config["name"],
            "prompt": prompt,
            "max_tokens": min(kwargs.get("max_tokens", 500), model_config["max_tokens"]),
            "temperature": kwargs.get("temperature", 0.7)
        }
        
        response = requests.post(
            f"{self.base_url}/completions",
            headers=headers,
            json=data
        )
        
        response.raise_for_status()
        result = response.json()
        
        # Calculate cost
        input_tokens = len(prompt.split()) * 1.3  # Rough estimation
        output_tokens = len(result.get("text", "").split()) * 1.3
        cost = ((input_tokens + output_tokens) / 1000) * model_config["cost_per_1k"]
        
        return {
            "text": result.get("text", ""),
            "model": model_config["name"],
            "model_type": model_type,
            "usage": {
                "prompt_tokens": int(input_tokens),
                "completion_tokens": int(output_tokens),
                "total_tokens": int(input_tokens + output_tokens)
            },
            "cost": cost
        }
    
    def get_optimal_model(self, prompt: str, max_cost: float = None) -> str:
        """Select optimal model based on prompt and cost constraints."""
        
        prompt_length = len(prompt.split())
        
        if max_cost and max_cost < 0.01:
            return "fast"
        elif prompt_length > 500:
            return "advanced"
        else:
            return "balanced"

# Usage with model selection
multi_model_adapter = MultiModelAdapter(
    api_key="your-api-key",
    base_url="https://api.your-multi-model-provider.com"
)

@workflow(name="optimized_custom_llm")
@guardrails(enforcement_level="moderate")
def optimized_custom_completion(prompt: str, max_cost: float = None) -> Dict[str, Any]:
    """Generate completion with optimal model selection."""
    
    optimal_model = multi_model_adapter.get_optimal_model(prompt, max_cost)
    
    result = multi_model_adapter.generate_completion(
        prompt=prompt,
        model_type=optimal_model,
        max_tokens=300
    )
    
    return {
        "response": result.get("text", ""),
        "model_used": result.get("model_type", ""),
        "cost": result.get("cost", 0.0),
        "within_budget": result.get("cost", 0.0) <= (max_cost or float('inf'))
    }
```

## Error Handling and Resilience

### Robust Custom Adapter

```python
import time
import random
from typing import Optional

class RobustCustomAdapter(BaseLLMAdapter):
    """Custom adapter with comprehensive error handling."""
    
    def __init__(self, api_key: str, base_url: str, backup_url: Optional[str] = None):
        self.api_key = api_key
        self.base_url = base_url
        self.backup_url = backup_url
        self.max_retries = 3
        super().__init__()
    
    def _make_request(self, url: str, data: dict, attempt: int = 0) -> dict:
        """Make HTTP request with retry logic."""
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        try:
            response = requests.post(url, headers=headers, json=data, timeout=30)
            
            if response.status_code == 429:  # Rate limit
                if attempt < self.max_retries:
                    wait_time = (2 ** attempt) + random.uniform(0, 1)
                    time.sleep(wait_time)
                    return self._make_request(url, data, attempt + 1)
                else:
                    raise Exception("Rate limit exceeded after retries")
            
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.Timeout:
            if attempt < self.max_retries:
                time.sleep(2)
                return self._make_request(url, data, attempt + 1)
            else:
                raise Exception("Request timeout after retries")
                
        except requests.exceptions.ConnectionError:
            # Try backup URL if available
            if self.backup_url and url != self.backup_url:
                return self._make_request(self.backup_url, data, attempt)
            elif attempt < self.max_retries:
                time.sleep(5)
                return self._make_request(url, data, attempt + 1)
            else:
                raise Exception("Connection failed after retries")
    
    def generate_completion(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Generate completion with error handling."""
        
        data = {
            "prompt": prompt,
            "max_tokens": kwargs.get("max_tokens", 500),
            "temperature": kwargs.get("temperature", 0.7)
        }
        
        try:
            result = self._make_request(f"{self.base_url}/completions", data)
            
            return {
                "text": result.get("text", ""),
                "model": result.get("model", "custom-model"),
                "usage": result.get("usage", {}),
                "success": True
            }
            
        except Exception as e:
            return {
                "text": "",
                "model": "custom-model",
                "usage": {},
                "success": False,
                "error": str(e)
            }
    
    def validate_configuration(self) -> bool:
        """Validate configuration with health check."""
        try:
            test_data = {"prompt": "test", "max_tokens": 1}
            result = self._make_request(f"{self.base_url}/health", test_data)
            return result.get("status") == "healthy"
        except:
            return False
```

## Registration and Integration

### Provider Registry

```python
from typing import Dict, Type
from rizk.sdk.adapters.base_llm_adapter import BaseLLMAdapter

class CustomLLMRegistry:
    """Registry for custom LLM providers."""
    
    def __init__(self):
        self._providers: Dict[str, Type[BaseLLMAdapter]] = {}
        self._instances: Dict[str, BaseLLMAdapter] = {}
    
    def register_provider(self, name: str, adapter_class: Type[BaseLLMAdapter]):
        """Register a custom LLM provider."""
        self._providers[name] = adapter_class
        print(f"Registered custom LLM provider: {name}")
    
    def create_instance(self, name: str, **kwargs) -> BaseLLMAdapter:
        """Create instance of registered provider."""
        if name not in self._providers:
            raise ValueError(f"Provider '{name}' not registered")
        
        adapter_class = self._providers[name]
        instance = adapter_class(**kwargs)
        
        if not instance.validate_configuration():
            raise ValueError(f"Invalid configuration for provider '{name}'")
        
        self._instances[name] = instance
        return instance
    
    def get_instance(self, name: str) -> BaseLLMAdapter:
        """Get existing instance of provider."""
        if name not in self._instances:
            raise ValueError(f"No instance found for provider '{name}'")
        return self._instances[name]
    
    def list_providers(self) -> List[str]:
        """List all registered providers."""
        return list(self._providers.keys())

# Global registry
custom_llm_registry = CustomLLMRegistry()

# Register providers
custom_llm_registry.register_provider("huggingface", HuggingFaceAdapter)
custom_llm_registry.register_provider("multi_model", MultiModelAdapter)
custom_llm_registry.register_provider("robust_custom", RobustCustomAdapter)

# Create instances
hf_instance = custom_llm_registry.create_instance(
    "huggingface",
    api_key="your-hf-token",
    model_name="microsoft/DialoGPT-large"
)

multi_instance = custom_llm_registry.create_instance(
    "multi_model",
    api_key="your-api-key",
    base_url="https://api.example.com"
)
```

### Integration with Rizk Decorators

```python
@workflow(name="custom_provider_workflow", organization_id="company", project_id="custom_ai")
@guardrails(enforcement_level="strict")
def unified_custom_completion(prompt: str, provider: str = "huggingface") -> Dict[str, Any]:
    """Use any registered custom provider."""
    
    try:
        adapter = custom_llm_registry.get_instance(provider)
        result = adapter.generate_completion(prompt, max_tokens=200)
        
        return {
            "success": True,
            "provider": provider,
            "response": result.get("text", ""),
            "model": result.get("model", ""),
            "cost": result.get("cost", 0.0)
        }
        
    except Exception as e:
        return {
            "success": False,
            "provider": provider,
            "error": str(e),
            "response": "",
            "model": "",
            "cost": 0.0
        }

# Usage with different providers
hf_result = unified_custom_completion("What is AI?", "huggingface")
multi_result = unified_custom_completion("Explain quantum computing", "multi_model")

print(f"HuggingFace result: {hf_result}")
print(f"Multi-model result: {multi_result}")
```

## Testing Custom Providers

### Unit Testing

```python
import unittest
from unittest.mock import patch, MagicMock

class TestCustomLLMAdapter(unittest.TestCase):
    """Test cases for custom LLM adapters."""
    
    def setUp(self):
        self.adapter = HuggingFaceAdapter(
            api_key="test-key",
            model_name="test-model"
        )
    
    @patch('requests.post')
    def test_generate_completion_success(self, mock_post):
        """Test successful completion generation."""
        
        # Mock response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = [
            {"generated_text": "Test prompt This is a test response"}
        ]
        mock_post.return_value = mock_response
        
        # Test
        result = self.adapter.generate_completion("Test prompt")
        
        # Assertions
        self.assertEqual(result["text"], "This is a test response")
        self.assertEqual(result["model"], "test-model")
        mock_post.assert_called_once()
    
    @patch('requests.post')
    def test_generate_completion_model_loading(self, mock_post):
        """Test handling of model loading state."""
        
        # Mock 503 response first, then success
        mock_response_503 = MagicMock()
        mock_response_503.status_code = 503
        
        mock_response_200 = MagicMock()
        mock_response_200.status_code = 200
        mock_response_200.json.return_value = [
            {"generated_text": "Test prompt Response after loading"}
        ]
        
        mock_post.side_effect = [mock_response_503, mock_response_200]
        
        # Test
        with patch('time.sleep'):  # Mock sleep to speed up test
            result = self.adapter.generate_completion("Test prompt")
        
        # Assertions
        self.assertEqual(result["text"], "Response after loading")
        self.assertEqual(mock_post.call_count, 2)
    
    def test_validate_configuration(self):
        """Test configuration validation."""
        
        with patch('requests.get') as mock_get:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_get.return_value = mock_response
            
            self.assertTrue(self.adapter.validate_configuration())
    
    def test_get_model_info(self):
        """Test model info retrieval."""
        
        info = self.adapter.get_model_info()
        
        self.assertEqual(info["provider"], "HuggingFace")
        self.assertEqual(info["model_name"], "test-model")
        self.assertTrue(info["supports_chat"])

if __name__ == "__main__":
    unittest.main()
```

## Best Practices

### 1. **Configuration Management**

```python
import os
from dataclasses import dataclass
from typing import Optional

@dataclass
class CustomLLMConfig:
    """Configuration for custom LLM providers."""
    
    api_key: str
    base_url: str
    model_name: Optional[str] = None
    max_retries: int = 3
    timeout: int = 30
    backup_url: Optional[str] = None
    
    @classmethod
    def from_env(cls, prefix: str) -> 'CustomLLMConfig':
        """Create config from environment variables."""
        return cls(
            api_key=os.getenv(f"{prefix}_API_KEY", ""),
            base_url=os.getenv(f"{prefix}_BASE_URL", ""),
            model_name=os.getenv(f"{prefix}_MODEL_NAME"),
            max_retries=int(os.getenv(f"{prefix}_MAX_RETRIES", "3")),
            timeout=int(os.getenv(f"{prefix}_TIMEOUT", "30")),
            backup_url=os.getenv(f"{prefix}_BACKUP_URL")
        )
    
    def validate(self) -> List[str]:
        """Validate configuration."""
        errors = []
        
        if not self.api_key:
            errors.append("API key is required")
        
        if not self.base_url:
            errors.append("Base URL is required")
        
        if self.max_retries < 0:
            errors.append("Max retries must be non-negative")
        
        if self.timeout <= 0:
            errors.append("Timeout must be positive")
        
        return errors

# Usage
config = CustomLLMConfig.from_env("CUSTOM_LLM")
errors = config.validate()

if errors:
    raise ValueError(f"Configuration errors: {errors}")

adapter = CustomLLMAdapter(
    api_key=config.api_key,
    base_url=config.base_url
)
```

### 2. **Monitoring and Logging**

```python
import logging
from functools import wraps

logger = logging.getLogger(__name__)

def monitor_llm_calls(func):
    """Decorator to monitor LLM API calls."""
    
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        start_time = time.time()
        
        try:
            result = func(self, *args, **kwargs)
            
            duration = time.time() - start_time
            logger.info(f"LLM call successful: {func.__name__} took {duration:.2f}s")
            
            return result
            
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"LLM call failed: {func.__name__} after {duration:.2f}s - {str(e)}")
            raise
    
    return wrapper

class MonitoredCustomAdapter(BaseLLMAdapter):
    """Custom adapter with monitoring."""
    
    @monitor_llm_calls
    def generate_completion(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Generate completion with monitoring."""
        # Implementation here
        pass
```

## Related Documentation

- **[OpenAI Integration](openai.md)** - OpenAI API integration
- **[Anthropic Integration](anthropic.md)** - Claude API integration  
- **[Gemini Integration](gemini.md)** - Google Gemini integration
- **[Ollama Integration](ollama.md)** - Local model integration

---

The custom LLM provider framework enables you to integrate any LLM service while maintaining the full benefits of Rizk's observability, governance, and policy enforcement capabilities. This ensures consistent monitoring and compliance across all your AI integrations. 

