---
title: "Custom Framework Integration Guide"
description: "Custom Framework Integration Guide"
---

# Custom Framework Integration Guide

This guide shows you how to integrate Rizk SDK with custom LLM frameworks, create your own adapters, and extend Rizk's observability to any AI system. Whether you're building a proprietary framework or integrating with an unsupported library, this guide provides the patterns and tools you need.

## Overview

Rizk SDK's adapter architecture allows you to:

- **Create Custom Adapters**: Build adapters for any LLM framework or library
- **Extend Existing Adapters**: Modify behavior for specific use cases  
- **Implement Standard Patterns**: Follow proven patterns for observability and governance
- **Maintain Compatibility**: Ensure your custom integrations work with Rizk's ecosystem

## Architecture Overview

### Adapter Pattern

Rizk uses the adapter pattern to provide a unified interface across different frameworks:

```
Your Framework â†’ Custom Adapter â†’ Rizk SDK Core â†’ Tracing & Guardrails
```

### Base Adapter Interface

All adapters implement the `BaseAdapter` interface:

```python
from abc import ABC, abstractmethod
from typing import Callable, Any, Union, Type, Tuple, Dict

class BaseAdapter(ABC):
    """Base class for all framework adapters."""
    
    @abstractmethod
    def adapt_workflow(self, func: Callable, name: str = None, **kwargs) -> Callable:
        """Adapt workflow-level functions."""
        pass
    
    @abstractmethod
    def adapt_task(self, func: Callable, name: str = None, **kwargs) -> Callable:
        """Adapt task-level functions."""
        pass
    
    @abstractmethod
    def adapt_agent(self, func: Callable, name: str = None, **kwargs) -> Callable:
        """Adapt agent-level functions."""
        pass
    
    @abstractmethod
    def adapt_tool(self, func_or_class: Union[Callable, Type], name: str = None, **kwargs) -> Union[Callable, Type]:
        """Adapt tool functions or classes."""
        pass
    
    @abstractmethod
    def apply_input_guardrails(self, args: Tuple, kwargs: Dict, func_name: str, strategy: str = "auto") -> Tuple[Tuple, Dict, bool, str]:
        """Apply input guardrails to function arguments."""
        pass
    
    @abstractmethod
    def apply_output_guardrails(self, result: Any, func_name: str) -> Tuple[Any, bool, str]:
        """Apply output guardrails to function results."""
        pass
    
    @abstractmethod
    def apply_augmentation(self, args: Tuple, kwargs: Dict, guidelines: list, func_name: str) -> Tuple[Tuple, Dict]:
        """Apply prompt augmentation to function arguments."""
        pass
```

## Creating Your First Custom Adapter

### Step 1: Basic Adapter Structure

```python
from rizk.sdk.adapters.base import BaseAdapter
from rizk.sdk.utils.span_utils import get_tracer
from opentelemetry.trace import Status, StatusCode
import functools
import logging
from typing import Callable, Any, Union, Type, Tuple, Dict

logger = logging.getLogger("rizk.adapters.custom")

class CustomFrameworkAdapter(BaseAdapter):
    """Adapter for your custom framework."""
    
    FRAMEWORK_NAME = "custom_framework"
    
    def __init__(self):
        """Initialize the custom adapter."""
        super().__init__()
        self.tracer = get_tracer()
        logger.info(f"Initialized {self.FRAMEWORK_NAME} adapter")
    
    def _get_func_name(self, func_or_class: Union[Callable, Type], name: str = None) -> str:
        """Helper to get the best name for a function or class."""
        if hasattr(func_or_class, '__name__'):
            return name or func_or_class.__name__
        return name or "unknown_function"
    
    def _trace_function(self, func: Callable, span_name_prefix: str, name_attribute: str, func_name: str) -> Callable:
        """Generic tracing wrapper for custom framework functions."""
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            span_name = f"{self.FRAMEWORK_NAME}.{span_name_prefix}.{func_name}"
            
            with self.tracer.start_as_current_span(span_name) as span:
                span.set_attribute(name_attribute, func_name)
                span.set_attribute("framework", self.FRAMEWORK_NAME)
                span.set_attribute(f"{span_name_prefix}.input", str(args)[:200])
                
                try:
                    result = func(*args, **kwargs)
                    span.set_attribute(f"{span_name_prefix}.success", True)
                    span.set_status(Status(StatusCode.OK))
                    if result:
                        span.set_attribute(f"{span_name_prefix}.output", str(result)[:500])
                    return result
                except Exception as e:
                    logger.error(f"Error in {func_name}: {e}", exc_info=True)
                    span.set_attribute(f"{span_name_prefix}.error", str(e))
                    span.record_exception(e)
                    span.set_status(Status(StatusCode.ERROR, description=str(e)))
                    raise
        
        return wrapper
    
    def adapt_workflow(self, func: Callable, name: str = None, **kwargs) -> Callable:
        """Adapt workflow-level functions."""
        func_name = self._get_func_name(func, name)
        logger.debug(f"Adapting custom framework workflow: {func_name}")
        return self._trace_function(func, "workflow", "workflow.name", func_name)
    
    def adapt_task(self, func: Callable, name: str = None, **kwargs) -> Callable:
        """Adapt task-level functions."""
        func_name = self._get_func_name(func, name)
        logger.debug(f"Adapting custom framework task: {func_name}")
        return self._trace_function(func, "task", "task.name", func_name)
    
    def adapt_agent(self, func: Callable, name: str = None, **kwargs) -> Callable:
        """Adapt agent-level functions."""
        func_name = self._get_func_name(func, name)
        logger.debug(f"Adapting custom framework agent: {func_name}")
        return self._trace_function(func, "agent", "agent.name", func_name)
    
    def adapt_tool(self, func_or_class: Union[Callable, Type], name: str = None, **kwargs) -> Union[Callable, Type]:
        """Adapt tool functions or classes."""
        tool_name = self._get_func_name(func_or_class, name)
        logger.debug(f"Adapting custom framework tool: {tool_name}")
        
        if callable(func_or_class):
            return self._trace_function(func_or_class, "tool", "tool.name", tool_name)
        else:
            # For classes, return as-is or implement class wrapping
            return func_or_class
    
    def apply_input_guardrails(self, args: Tuple, kwargs: Dict, func_name: str, strategy: str = "auto") -> Tuple[Tuple, Dict, bool, str]:
        """Apply input guardrails to function arguments."""
        # Implement your custom input guardrails logic here
        # For now, return the arguments unchanged
        return args, kwargs, False, ""
    
    def apply_output_guardrails(self, result: Any, func_name: str) -> Tuple[Any, bool, str]:
        """Apply output guardrails to function results."""
        # Implement your custom output guardrails logic here
        # For now, return the result unchanged
        return result, False, ""
    
    def apply_augmentation(self, args: Tuple, kwargs: Dict, guidelines: list, func_name: str) -> Tuple[Tuple, Dict]:
        """Apply prompt augmentation to function arguments."""
        # Implement your custom augmentation logic here
        # For now, return the arguments unchanged
        return args, kwargs
```

### Step 2: Register Your Custom Adapter

```python
from rizk.sdk.utils.framework_registry import FrameworkRegistry

# Register your custom adapter
custom_adapter = CustomFrameworkAdapter()
FrameworkRegistry.register_adapter("custom_framework", custom_adapter)

# Verify registration
print("Registered frameworks:", FrameworkRegistry.get_all_framework_names())
```

### Step 3: Use Your Custom Adapter

```python
from rizk.sdk import Rizk
from rizk.sdk.decorators import workflow, task, agent, tool

# Initialize Rizk
rizk = Rizk.init(
    app_name="Custom-Framework-Demo",
    enabled=True
)

# Your custom framework functions will now be automatically traced
@workflow(name="custom_workflow", organization_id="demo", project_id="custom")
def my_custom_workflow(data: str) -> str:
    """Custom framework workflow with automatic tracing."""
    return f"Custom framework processed: {data}"

@task(name="custom_task", organization_id="demo", project_id="custom")
def my_custom_task(input_data: str) -> str:
    """Custom framework task with automatic tracing."""
    return f"Task result: {input_data}"

# Test the integration
result = my_custom_workflow("test data")
print(result)
```

## Advanced Custom Adapter Examples

### Example 1: Custom ML Pipeline Framework

```python
class MLPipelineAdapter(BaseAdapter):
    """Adapter for custom ML pipeline framework."""
    
    FRAMEWORK_NAME = "ml_pipeline"
    
    def __init__(self):
        super().__init__()
        self.tracer = get_tracer()
    
    def _trace_ml_step(self, func: Callable, step_type: str, step_name: str) -> Callable:
        """Specialized tracing for ML pipeline steps."""
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            span_name = f"{self.FRAMEWORK_NAME}.{step_type}.{step_name}"
            
            with self.tracer.start_as_current_span(span_name) as span:
                # ML-specific attributes
                span.set_attribute("ml.step.type", step_type)
                span.set_attribute("ml.step.name", step_name)
                span.set_attribute("framework", self.FRAMEWORK_NAME)
                
                # Extract ML-specific metadata
                if args and hasattr(args[0], 'shape'):  # Likely a numpy array or tensor
                    span.set_attribute("ml.input.shape", str(args[0].shape))
                
                try:
                    import time
                    start_time = time.time()
                    
                    result = func(*args, **kwargs)
                    
                    end_time = time.time()
                    span.set_attribute("ml.step.duration_ms", round((end_time - start_time) * 1000, 2))
                    
                    # ML result metadata
                    if hasattr(result, 'shape'):
                        span.set_attribute("ml.output.shape", str(result.shape))
                    
                    span.set_attribute("ml.step.success", True)
                    span.set_status(Status(StatusCode.OK))
                    
                    return result
                except Exception as e:
                    span.set_attribute("ml.step.error", str(e))
                    span.record_exception(e)
                    span.set_status(Status(StatusCode.ERROR, description=str(e)))
                    raise
        
        return wrapper
    
    def adapt_workflow(self, func: Callable, name: str = None, **kwargs) -> Callable:
        """Adapt ML pipeline workflows."""
        pipeline_name = name or func.__name__
        return self._trace_ml_step(func, "pipeline", pipeline_name)
    
    def adapt_task(self, func: Callable, name: str = None, **kwargs) -> Callable:
        """Adapt ML pipeline steps."""
        step_name = name or func.__name__
        return self._trace_ml_step(func, "step", step_name)
    
    def adapt_agent(self, func: Callable, name: str = None, **kwargs) -> Callable:
        """Adapt ML models as agents."""
        model_name = name or func.__name__
        return self._trace_ml_step(func, "model", model_name)
    
    def adapt_tool(self, func_or_class: Union[Callable, Type], name: str = None, **kwargs) -> Union[Callable, Type]:
        """Adapt ML utility functions."""
        tool_name = name or getattr(func_or_class, '__name__', 'unknown_tool')
        if callable(func_or_class):
            return self._trace_ml_step(func_or_class, "tool", tool_name)
        return func_or_class
    
    # Implement other required methods...
    def apply_input_guardrails(self, args: Tuple, kwargs: Dict, func_name: str, strategy: str = "auto") -> Tuple[Tuple, Dict, bool, str]:
        return args, kwargs, False, ""
    
    def apply_output_guardrails(self, result: Any, func_name: str) -> Tuple[Any, bool, str]:
        return result, False, ""
    
    def apply_augmentation(self, args: Tuple, kwargs: Dict, guidelines: list, func_name: str) -> Tuple[Tuple, Dict]:
        return args, kwargs

# Usage example
ml_adapter = MLPipelineAdapter()
FrameworkRegistry.register_adapter("ml_pipeline", ml_adapter)

@workflow(name="data_pipeline", organization_id="ml", project_id="training")
def data_preprocessing_pipeline(raw_data):
    """ML data preprocessing pipeline."""
    # Your ML preprocessing logic
    processed_data = raw_data * 2  # Simplified example
    return processed_data

@task(name="feature_extraction", organization_id="ml", project_id="training")
def extract_features(data):
    """Feature extraction step."""
    # Your feature extraction logic
    features = data + 1  # Simplified example
    return features
```

### Example 2: Custom API Framework Adapter

```python
class APIFrameworkAdapter(BaseAdapter):
    """Adapter for custom API framework with HTTP request tracing."""
    
    FRAMEWORK_NAME = "api_framework"
    
    def _trace_api_call(self, func: Callable, endpoint_type: str, endpoint_name: str) -> Callable:
        """Specialized tracing for API calls."""
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            span_name = f"{self.FRAMEWORK_NAME}.{endpoint_type}.{endpoint_name}"
            
            with self.tracer.start_as_current_span(span_name) as span:
                # API-specific attributes
                span.set_attribute("api.endpoint.type", endpoint_type)
                span.set_attribute("api.endpoint.name", endpoint_name)
                span.set_attribute("framework", self.FRAMEWORK_NAME)
                
                # Extract request metadata
                if kwargs.get('method'):
                    span.set_attribute("http.method", kwargs['method'])
                if kwargs.get('url'):
                    span.set_attribute("http.url", kwargs['url'])
                
                try:
                    import time
                    start_time = time.time()
                    
                    result = func(*args, **kwargs)
                    
                    end_time = time.time()
                    response_time = round((end_time - start_time) * 1000, 2)
                    span.set_attribute("api.response_time_ms", response_time)
                    
                    # Extract response metadata
                    if hasattr(result, 'status_code'):
                        span.set_attribute("http.status_code", result.status_code)
                    
                    span.set_attribute("api.call.success", True)
                    span.set_status(Status(StatusCode.OK))
                    
                    return result
                except Exception as e:
                    span.set_attribute("api.call.error", str(e))
                    span.record_exception(e)
                    span.set_status(Status(StatusCode.ERROR, description=str(e)))
                    raise
        
        return wrapper
    
    def adapt_workflow(self, func: Callable, name: str = None, **kwargs) -> Callable:
        """Adapt API workflow orchestrations."""
        workflow_name = name or func.__name__
        return self._trace_api_call(func, "workflow", workflow_name)
    
    def adapt_task(self, func: Callable, name: str = None, **kwargs) -> Callable:
        """Adapt individual API endpoints."""
        endpoint_name = name or func.__name__
        return self._trace_api_call(func, "endpoint", endpoint_name)
    
    def adapt_agent(self, func: Callable, name: str = None, **kwargs) -> Callable:
        """Adapt API agents (e.g., chatbots)."""
        agent_name = name or func.__name__
        return self._trace_api_call(func, "agent", agent_name)
    
    def adapt_tool(self, func_or_class: Union[Callable, Type], name: str = None, **kwargs) -> Union[Callable, Type]:
        """Adapt API utility functions."""
        tool_name = name or getattr(func_or_class, '__name__', 'unknown_tool')
        if callable(func_or_class):
            return self._trace_api_call(func_or_class, "tool", tool_name)
        return func_or_class
    
    # Implement guardrails with API-specific logic
    def apply_input_guardrails(self, args: Tuple, kwargs: Dict, func_name: str, strategy: str = "auto") -> Tuple[Tuple, Dict, bool, str]:
        """Apply API input validation."""
        # Example: Check for required parameters
        if 'url' not in kwargs:
            return args, kwargs, True, "Missing required 'url' parameter"
        
        # Example: Validate URL format
        url = kwargs.get('url', '')
        if not url.startswith(('http://', 'https://')):
            return args, kwargs, True, "Invalid URL format"
        
        return args, kwargs, False, ""
    
    def apply_output_guardrails(self, result: Any, func_name: str) -> Tuple[Any, bool, str]:
        """Apply API output validation."""
        # Example: Check response status
        if hasattr(result, 'status_code') and result.status_code >= 400:
            return result, True, f"API returned error status: {result.status_code}"
        
        return result, False, ""
    
    def apply_augmentation(self, args: Tuple, kwargs: Dict, guidelines: list, func_name: str) -> Tuple[Tuple, Dict]:
        """Apply API request augmentation."""
        if guidelines:
            # Example: Add custom headers based on guidelines
            headers = kwargs.get('headers', {})
            headers['X-Policy-Guidelines'] = '; '.join(guidelines)
            kwargs['headers'] = headers
        
        return args, kwargs
```

## Framework Detection Integration

### Automatic Detection

To make your custom adapter automatically detected, implement detection patterns:

```python
from rizk.sdk.utils.framework_detection import FRAMEWORK_DETECTION_PATTERNS

# Add detection pattern for your framework
FRAMEWORK_DETECTION_PATTERNS["custom_framework"] = {
    "modules": ["custom_framework", "custom_framework.core"],
    "classes": ["CustomWorkflow", "CustomAgent", "CustomPipeline"],
    "import_names": ["custom_framework"],
    "attributes": ["custom_framework_version"]
}

# Register detection function
def is_custom_framework_object(obj):
    """Check if an object belongs to your custom framework."""
    try:
        obj_module = getattr(obj, "__module__", "")
        obj_class = getattr(obj, "__class__", type(obj)).__name__
        
        # Check module patterns
        if obj_module.startswith("custom_framework"):
            return True
        
        # Check class patterns
        if obj_class in ["CustomWorkflow", "CustomAgent", "CustomPipeline"]:
            return True
        
        return False
    except Exception:
        return False

# Add to framework detection
from rizk.sdk.utils.framework_detection import register_framework_detector
register_framework_detector("custom_framework", is_custom_framework_object)
```

## Testing Your Custom Adapter

### Unit Testing Framework

```python
import unittest
from unittest.mock import patch, MagicMock
from rizk.sdk import Rizk
from rizk.sdk.utils.framework_registry import FrameworkRegistry

class TestCustomFrameworkAdapter(unittest.TestCase):
    """Test custom framework adapter integration."""
    
    def setUp(self):
        """Set up test environment."""
        self.adapter = CustomFrameworkAdapter()
        FrameworkRegistry.register_adapter("custom_framework", self.adapter)
        
        self.rizk = Rizk.init(
            app_name="Custom-Framework-Test",
            enabled=True
        )
    
    def test_workflow_adaptation(self):
        """Test workflow function adaptation."""
        
        @workflow(name="test_workflow", organization_id="test", project_id="custom")
        def test_workflow(data: str) -> str:
            return f"Processed: {data}"
        
        result = test_workflow("test input")
        self.assertEqual(result, "Processed: test input")
    
    def test_task_adaptation(self):
        """Test task function adaptation."""
        
        @task(name="test_task", organization_id="test", project_id="custom")
        def test_task(data: str) -> str:
            return f"Task result: {data}"
        
        result = test_task("task input")
        self.assertEqual(result, "Task result: task input")
    
    def test_error_handling(self):
        """Test error handling in adapted functions."""
        
        @task(name="error_task", organization_id="test", project_id="custom")
        def error_task():
            raise ValueError("Test error")
        
        with self.assertRaises(ValueError):
            error_task()
    
    def test_guardrails_integration(self):
        """Test guardrails integration."""
        # Mock guardrails
        with patch.object(self.adapter, 'apply_input_guardrails') as mock_input:
            mock_input.return_value = ((), {}, True, "Blocked by test")
            
            @task(name="guarded_task", organization_id="test", project_id="custom")
            def guarded_task():
                return "Should not execute"
            
            # Test that guardrails are called
            # Implementation depends on your guardrails logic
            pass

if __name__ == "__main__":
    unittest.main()
```

### Integration Testing

```python
def test_custom_framework_integration():
    """Comprehensive integration test."""
    
    # Initialize everything
    custom_adapter = CustomFrameworkAdapter()
    FrameworkRegistry.register_adapter("custom_framework", custom_adapter)
    
    rizk = Rizk.init(
        app_name="Custom-Integration-Test",
        enabled=True
    )
    
    # Test workflow chain
    @workflow(name="integration_workflow", organization_id="test", project_id="integration")
    def integration_workflow(data: str) -> str:
        # Call task from workflow
        processed = integration_task(data)
        return f"Workflow result: {processed}"
    
    @task(name="integration_task", organization_id="test", project_id="integration")
    def integration_task(data: str) -> str:
        # Call tool from task
        enhanced = integration_tool(data)
        return f"Task processed: {enhanced}"
    
    @tool(name="integration_tool", organization_id="test", project_id="integration")
    def integration_tool(data: str) -> str:
        return f"Tool enhanced: {data}"
    
    # Test the full chain
    result = integration_workflow("test data")
    expected = "Workflow result: Task processed: Tool enhanced: test data"
    
    print(f"Integration test result: {result}")
    assert result == expected
    print("âœ… Custom framework integration test passed!")

# Run the test
test_custom_framework_integration()
```

## Best Practices for Custom Adapters

### 1. Error Handling

```python
def _safe_trace_function(self, func: Callable, span_name_prefix: str, func_name: str) -> Callable:
    """Trace function with comprehensive error handling."""
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        span_name = f"{self.FRAMEWORK_NAME}.{span_name_prefix}.{func_name}"
        
        try:
            with self.tracer.start_as_current_span(span_name) as span:
                # Set basic attributes
                span.set_attribute("framework", self.FRAMEWORK_NAME)
                span.set_attribute(f"{span_name_prefix}.name", func_name)
                
                try:
                    # Safely extract input information
                    input_info = self._safe_extract_input_info(args, kwargs)
                    span.set_attribute(f"{span_name_prefix}.input", input_info)
                except Exception as e:
                    logger.debug(f"Could not extract input info: {e}")
                
                # Execute function
                result = func(*args, **kwargs)
                
                try:
                    # Safely extract output information
                    output_info = self._safe_extract_output_info(result)
                    span.set_attribute(f"{span_name_prefix}.output", output_info)
                except Exception as e:
                    logger.debug(f"Could not extract output info: {e}")
                
                span.set_status(Status(StatusCode.OK))
                return result
                
        except Exception as e:
            # Always log errors but don't break the original function
            logger.error(f"Error in {span_name}: {e}", exc_info=True)
            
            # Still try to record the error in tracing if possible
            try:
                with self.tracer.start_as_current_span(span_name) as error_span:
                    error_span.set_attribute("framework", self.FRAMEWORK_NAME)
                    error_span.set_attribute("error", str(e))
                    error_span.record_exception(e)
                    error_span.set_status(Status(StatusCode.ERROR, description=str(e)))
            except Exception:
                pass  # Don't let tracing errors break the original function
            
            # Re-raise the original exception
            raise
    
    return wrapper

def _safe_extract_input_info(self, args: Tuple, kwargs: Dict) -> str:
    """Safely extract input information for tracing."""
    try:
        # Limit string length to prevent huge traces
        max_length = 500
        
        args_str = str(args)[:max_length] if args else ""
        kwargs_str = str(kwargs)[:max_length] if kwargs else ""
        
        return f"args: {args_str}, kwargs: {kwargs_str}"
    except Exception:
        return "Could not extract input info"

def _safe_extract_output_info(self, result: Any) -> str:
    """Safely extract output information for tracing."""
    try:
        max_length = 500
        return str(result)[:max_length]
    except Exception:
        return "Could not extract output info"
```

### 2. Performance Optimization

```python
class OptimizedCustomAdapter(BaseAdapter):
    """Performance-optimized custom adapter."""
    
    def __init__(self):
        super().__init__()
        self._trace_cache = {}  # Cache traced functions
        self._span_pool = []    # Reuse span objects
    
    def _get_cached_tracer(self, func: Callable, span_type: str) -> Callable:
        """Get cached traced function or create new one."""
        cache_key = (id(func), span_type)
        
        if cache_key not in self._trace_cache:
            self._trace_cache[cache_key] = self._trace_function(
                func, span_type, f"{span_type}.name", func.__name__
            )
        
        return self._trace_cache[cache_key]
    
    def _lightweight_trace(self, func: Callable, span_name: str) -> Callable:
        """Lightweight tracing for high-frequency functions."""
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Minimal tracing overhead
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                
                # Only record timing and success
                duration = time.time() - start_time
                logger.debug(f"{span_name} completed in {duration:.3f}s")
                
                return result
            except Exception as e:
                duration = time.time() - start_time
                logger.error(f"{span_name} failed after {duration:.3f}s: {e}")
                raise
        
        return wrapper
```

### 3. Configuration Management

```python
class ConfigurableCustomAdapter(BaseAdapter):
    """Custom adapter with configuration support."""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__()
        self.config = config or {}
        
        # Configuration options
        self.trace_inputs = self.config.get('trace_inputs', True)
        self.trace_outputs = self.config.get('trace_outputs', True)
        self.max_trace_length = self.config.get('max_trace_length', 500)
        self.enable_performance_metrics = self.config.get('enable_performance_metrics', True)
        self.custom_attributes = self.config.get('custom_attributes', {})
    
    def _configure_span(self, span, span_type: str, func_name: str):
        """Configure span with custom attributes."""
        # Standard attributes
        span.set_attribute("framework", self.FRAMEWORK_NAME)
        span.set_attribute(f"{span_type}.name", func_name)
        
        # Custom attributes from configuration
        for key, value in self.custom_attributes.items():
            span.set_attribute(f"custom.{key}", value)
        
        # Environment-specific attributes
        if self.config.get('environment'):
            span.set_attribute("environment", self.config['environment'])

# Usage with configuration
custom_config = {
    'trace_inputs': True,
    'trace_outputs': False,  # Disable output tracing for privacy
    'max_trace_length': 1000,
    'enable_performance_metrics': True,
    'custom_attributes': {
        'team': 'ai-platform',
        'version': '1.0.0'
    },
    'environment': 'production'
}

adapter = ConfigurableCustomAdapter(custom_config)
FrameworkRegistry.register_adapter("custom_framework", adapter)
```

This comprehensive guide provides everything you need to create custom framework adapters that integrate seamlessly with Rizk SDK's observability and governance features. The examples show real-world patterns for ML pipelines, API frameworks, and other custom systems. 

