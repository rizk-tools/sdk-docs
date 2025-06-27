---
title: "Utilities API Reference"
description: "Utilities API Reference"
---

# Utilities API Reference

This document provides a comprehensive reference for utility functions, helper classes, and convenience methods available in the Rizk SDK.

## Framework Detection Utilities

### `detect_framework()`

**Automatically detect the LLM framework being used in a function.**

```python
from rizk.sdk.utils.framework_detection import detect_framework

def detect_framework(func: Callable) -> Optional[str]
```

#### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `func` | `Callable` | Function to analyze for framework detection |

#### Returns

- `Optional[str]`: Framework name if detected, None otherwise

#### Supported Frameworks

- `"agents_sdk"` - OpenAI Agents SDK
- `"langchain"` - LangChain
- `"crewai"` - CrewAI  
- `"llama_index"` - LlamaIndex
- `"custom"` - Custom framework patterns

#### Example

```python
from agents import Agent
from rizk.sdk.utils.framework_detection import detect_framework

def create_openai_agent():
    return Agent(name="Assistant", instructions="Be helpful")

# Detect framework
framework = detect_framework(create_openai_agent)
print(framework)  # Output: "agents_sdk"
```

---

### `get_framework_from_object()`

**Detect framework from a returned object.**

```python
def get_framework_from_object(obj: Any) -> Optional[str]
```

#### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `obj` | `Any` | Object to analyze for framework detection |

#### Returns

- `Optional[str]`: Framework name if detected, None otherwise

#### Example

```python
from langchain.agents import AgentExecutor
from rizk.sdk.utils.framework_detection import get_framework_from_object

agent_executor = AgentExecutor(agent=agent, tools=tools)
framework = get_framework_from_object(agent_executor)
print(framework)  # Output: "langchain"
```

---

### `is_framework_available()`

**Check if a specific framework is available in the current environment.**

```python
def is_framework_available(framework_name: str) -> bool
```

#### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `framework_name` | `str` | Name of the framework to check |

#### Returns

- `bool`: True if framework is available, False otherwise

#### Example

```python
from rizk.sdk.utils.framework_detection import is_framework_available

if is_framework_available("langchain"):
    print("LangChain is available")
    from langchain.agents import create_openai_tools_agent
else:
    print("LangChain not installed")
```

## Registry Utilities

### `FrameworkRegistry`

**Centralized registry for framework adapters.**

```python
from rizk.sdk.utils.framework_registry import FrameworkRegistry

class FrameworkRegistry:
    """Registry for framework adapters."""
    
    @classmethod
    def register(cls, name: str, adapter: Any) -> None:
        """Register a framework adapter."""
        
    @classmethod
    def get(cls, name: str) -> Optional[Any]:
        """Get a registered framework adapter."""
        
    @classmethod
    def list_available(cls) -> List[str]:
        """List all available framework adapters."""
        
    @classmethod
    def is_registered(cls, name: str) -> bool:
        """Check if a framework adapter is registered."""
```

#### Examples

**Register custom adapter:**
```python
from rizk.sdk.utils.framework_registry import FrameworkRegistry
from my_framework import MyCustomAdapter

# Register custom adapter
FrameworkRegistry.register("my_framework", MyCustomAdapter)

# Check if registered
if FrameworkRegistry.is_registered("my_framework"):
    adapter = FrameworkRegistry.get("my_framework")
```

**List available adapters:**
```python
available = FrameworkRegistry.list_available()
print(f"Available adapters: {available}")
# Output: ['agents_sdk', 'langchain', 'crewai', 'llama_index', 'my_framework']
```

---

### `LLMClientRegistry`

**Registry for LLM client adapters.**

```python
from rizk.sdk.utils.framework_registry import LLMClientRegistry

class LLMClientRegistry:
    """Registry for LLM client adapters."""
    
    @classmethod
    def register(cls, name: str, adapter: Any) -> None:
        """Register an LLM client adapter."""
        
    @classmethod
    def get(cls, name: str) -> Optional[Any]:
        """Get a registered LLM client adapter."""
        
    @classmethod
    def list_available(cls) -> List[str]:
        """List all available LLM client adapters."""
```

#### Example

```python
from rizk.sdk.utils.framework_registry import LLMClientRegistry

# Check available LLM adapters
llm_adapters = LLMClientRegistry.list_available()
print(f"Available LLM adapters: {llm_adapters}")
# Output: ['openai_completion', 'openai_responses', 'anthropic', 'agents']
```

## Context Management Utilities

### `set_hierarchy_context()`

**Set hierarchical context for distributed tracing.**

```python
from rizk.sdk.utils.context import set_hierarchy_context

def set_hierarchy_context(
    organization_id: Optional[str] = None,
    project_id: Optional[str] = None,
    agent_id: Optional[str] = None,
    conversation_id: Optional[str] = None,
    user_id: Optional[str] = None,
    **additional_context: Any
) -> None
```

#### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `organization_id` | `Optional[str]` | Organization identifier |
| `project_id` | `Optional[str]` | Project identifier |
| `agent_id` | `Optional[str]` | Agent identifier |
| `conversation_id` | `Optional[str]` | Conversation identifier |
| `user_id` | `Optional[str]` | User identifier |
| `**additional_context` | `Any` | Additional context key-value pairs |

#### Example

```python
from rizk.sdk.utils.context import set_hierarchy_context

# Set context for a customer service conversation
set_hierarchy_context(
    organization_id="acme_corp",
    project_id="customer_service",
    agent_id="support_bot_v2",
    conversation_id="conv_12345",
    user_id="customer_789",
    session_id="session_abc",
    priority="high"
)
```

---

### `get_current_context()`

**Get the current context from OpenTelemetry."""

```python
def get_current_context() -> Dict[str, Any]
```

#### Returns

- `Dict[str, Any]`: Current context dictionary

#### Example

```python
from rizk.sdk.utils.context import get_current_context, set_hierarchy_context

# Set context
set_hierarchy_context(organization_id="test_org", project_id="test_project")

# Get context
context = get_current_context()
print(context)
# Output: {'organization_id': 'test_org', 'project_id': 'test_project'}
```

---

### `clear_context()`

**Clear the current context.**

```python
def clear_context() -> None
```

#### Example

```python
from rizk.sdk.utils.context import clear_context

# Clear all context
clear_context()
```

## Caching Utilities

### `CacheManager`

**Centralized cache management for the SDK.**

```python
from rizk.sdk.utils.cache import CacheManager

class CacheManager:
    """Centralized cache management."""
    
    @classmethod
    def get(cls, key: str, cache_type: str = "default") -> Optional[Any]:
        """Get value from cache."""
        
    @classmethod
    def set(
        cls, 
        key: str, 
        value: Any, 
        ttl: Optional[int] = None,
        cache_type: str = "default"
    ) -> None:
        """Set value in cache."""
        
    @classmethod
    def delete(cls, key: str, cache_type: str = "default") -> None:
        """Delete value from cache."""
        
    @classmethod
    def clear(cls, cache_type: str = "default") -> None:
        """Clear entire cache."""
        
    @classmethod
    def get_stats(cls, cache_type: str = "default") -> Dict[str, Any]:
        """Get cache statistics."""
```

#### Cache Types

- `"default"` - General purpose cache
- `"framework_detection"` - Framework detection results
- `"policy_evaluation"` - Policy evaluation results
- `"llm_fallback"` - LLM fallback results

#### Examples

**Basic caching:**
```python
from rizk.sdk.utils.cache import CacheManager

# Cache a value
CacheManager.set("user_profile_123", user_data, ttl=3600)

# Retrieve cached value
cached_data = CacheManager.get("user_profile_123")
if cached_data:
    print("Found in cache!")
else:
    print("Cache miss")
```

**Framework detection caching:**
```python
# Cache framework detection result
CacheManager.set(
    "func_framework_detection", 
    "langchain", 
    cache_type="framework_detection"
)

# Check cache
framework = CacheManager.get("func_framework_detection", "framework_detection")
```

**Cache statistics:**
```python
stats = CacheManager.get_stats("policy_evaluation")
print(f"Cache hits: {stats['hits']}")
print(f"Cache misses: {stats['misses']}")
print(f"Hit ratio: {stats['hit_ratio']:.2%}")
```

## Performance Utilities

### `@performance_monitor`

**Decorator to monitor function performance.**

```python
from rizk.sdk.utils.performance import performance_monitor

@performance_monitor(log_threshold_ms=100)
def expensive_operation(data: str) -> str:
    # Operation that might be slow
    return process_large_dataset(data)
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `log_threshold_ms` | `int` | `50` | Log execution time if above this threshold |
| `include_args` | `bool` | `False` | Include function arguments in logs |
| `metric_name` | `Optional[str]` | `None` | Custom metric name for telemetry |

#### Example

```python
@performance_monitor(log_threshold_ms=200, include_args=True)
async def async_llm_call(prompt: str) -> str:
    response = await openai.ChatCompletion.acreate(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

# Function automatically logs if execution > 200ms
result = await async_llm_call("Hello, world!")
```

---

### `Timer`

**Context manager for timing operations.**

```python
from rizk.sdk.utils.performance import Timer

class Timer:
    """Context manager for timing operations."""
    
    def __init__(self, name: str = "operation"):
        self.name = name
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
    
    @property
    def elapsed_ms(self) -> float:
        """Get elapsed time in milliseconds."""
        
    def __enter__(self) -> "Timer":
        """Start timing."""
        
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Stop timing and log result."""
```

#### Example

```python
from rizk.sdk.utils.performance import Timer

# Time a block of code
with Timer("policy_evaluation") as timer:
    result = await guardrails.process_message(message)
    
print(f"Policy evaluation took {timer.elapsed_ms:.2f}ms")

# Multiple timing points
with Timer("total_operation") as total_timer:
    with Timer("data_prep") as prep_timer:
        prepared_data = prepare_data(raw_data)
    
    with Timer("processing") as proc_timer:
        result = process_data(prepared_data)

print(f"Data prep: {prep_timer.elapsed_ms:.2f}ms")
print(f"Processing: {proc_timer.elapsed_ms:.2f}ms")
print(f"Total: {total_timer.elapsed_ms:.2f}ms")
```

## Error Handling Utilities

### `@handle_errors`

**Decorator for standardized error handling.**

```python
from rizk.sdk.utils.error_handling import handle_errors

@handle_errors(
    fail_closed: bool = False,
    default_return_on_error: Any = None,
    log_errors: bool = True,
    reraise_on: Optional[List[Type[Exception]]] = None
)
def my_function():
    pass
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `fail_closed` | `bool` | `False` | If True, re-raise exceptions instead of returning default |
| `default_return_on_error` | `Any` | `None` | Default value to return on error |
| `log_errors` | `bool` | `True` | Whether to log errors |
| `reraise_on` | `Optional[List[Type[Exception]]]` | `None` | Exception types to always re-raise |

#### Examples

**Fail-safe operation:**
```python
@handle_errors(fail_closed=False, default_return_on_error={"allowed": True})
async def optional_guardrails_check(message: str) -> Dict[str, Any]:
    # If this fails, we allow the message by default
    return await guardrails.process_message(message)
```

**Critical operation:**
```python
@handle_errors(fail_closed=True, reraise_on=[KeyError, ValueError])
def critical_config_validation(config: dict) -> bool:
    # Must succeed or fail explicitly
    return validate_required_config(config)
```

---

### `RizkErrorReporter`

**Centralized error reporting and logging.**

```python
from rizk.sdk.utils.error_handling import RizkErrorReporter

class RizkErrorReporter:
    """Centralized error reporting."""
    
    @classmethod
    def report_error(
        cls,
        error: Exception,
        context: Optional[Dict[str, Any]] = None,
        severity: str = "error"
    ) -> None:
        """Report an error with context."""
        
    @classmethod
    def report_warning(
        cls,
        message: str,
        context: Optional[Dict[str, Any]] = None
    ) -> None:
        """Report a warning."""
        
    @classmethod
    def get_error_stats(cls) -> Dict[str, Any]:
        """Get error statistics."""
```

#### Example

```python
from rizk.sdk.utils.error_handling import RizkErrorReporter

try:
    result = risky_operation()
except Exception as e:
    RizkErrorReporter.report_error(
        error=e,
        context={
            "operation": "policy_evaluation",
            "user_id": "user_123",
            "message_length": len(message)
        },
        severity="high"
    )
    # Handle gracefully
    result = default_safe_result()
```

## Configuration Utilities

### `validate_environment()`

**Validate environment configuration.**

```python
from rizk.sdk.utils.config import validate_environment

def validate_environment() -> Dict[str, Any]
```

#### Returns

- `Dict[str, Any]`: Validation results with errors and warnings

#### Example

```python
from rizk.sdk.utils.config import validate_environment

validation = validate_environment()

if validation["errors"]:
    print("Configuration errors found:")
    for error in validation["errors"]:
        print(f"  âŒ {error}")

if validation["warnings"]:
    print("Configuration warnings:")
    for warning in validation["warnings"]:
        print(f"  âš ï¸ {warning}")

if validation["is_valid"]:
    print("âœ… Environment configuration is valid")
```

---

### `get_default_policies_path()`

**Get the default policies path with fallback logic.**

```python
def get_default_policies_path() -> str
```

#### Returns

- `str`: Path to policies directory

#### Path Resolution Order

1. `RIZK_POLICIES_PATH` environment variable
2. `./policies` directory if it exists
3. `./rizk_policies` directory if it exists
4. SDK built-in policies

#### Example

```python
from rizk.sdk.utils.config import get_default_policies_path

policies_path = get_default_policies_path()
print(f"Using policies from: {policies_path}")

# Check if custom policies exist
import os
if os.path.exists(os.path.join(policies_path, "custom_policy.yaml")):
    print("Custom policies found")
```

## Logging Utilities

### `get_logger()`

**Get a configured logger for the SDK.**

```python
from rizk.sdk.utils.logging import get_logger

def get_logger(name: str) -> logging.Logger
```

#### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `name` | `str` | Logger name (usually module name) |

#### Returns

- `logging.Logger`: Configured logger instance

#### Example

```python
from rizk.sdk.utils.logging import get_logger

logger = get_logger(__name__)

logger.info("Starting operation")
logger.warning("Deprecated feature used")
logger.error("Operation failed", extra={"user_id": "123"})
```

---

### `setup_logging()`

**Configure logging for the SDK.**

```python
def setup_logging(
    level: str = "INFO",
    format_string: Optional[str] = None,
    include_context: bool = True
) -> None
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `level` | `str` | `"INFO"` | Logging level |
| `format_string` | `Optional[str]` | `None` | Custom format string |
| `include_context` | `bool` | `True` | Include Rizk context in logs |

#### Example

```python
from rizk.sdk.utils.logging import setup_logging

# Setup debug logging
setup_logging(level="DEBUG", include_context=True)

# Custom format
setup_logging(
    level="INFO",
    format_string="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    include_context=False
)
```

## Validation Utilities

### `validate_api_key()`

**Validate Rizk API key format.**

```python
from rizk.sdk.utils.validation import validate_api_key

def validate_api_key(api_key: str) -> bool
```

#### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `api_key` | `str` | API key to validate |

#### Returns

- `bool`: True if valid format, False otherwise

#### Example

```python
from rizk.sdk.utils.validation import validate_api_key

api_key = "rizk_live_abc123def456"
if validate_api_key(api_key):
    print("âœ… Valid API key format")
else:
    print("âŒ Invalid API key format")
```

---

### `validate_url()`

**Validate URL format.**

```python
def validate_url(url: str) -> bool
```

#### Example

```python
from rizk.sdk.utils.validation import validate_url

endpoint = "https://api.rizk.tools"
if validate_url(endpoint):
    print("âœ… Valid URL")
else:
    print("âŒ Invalid URL")
```

## Helper Functions

### `safe_import()`

**Safely import optional dependencies.**

```python
from rizk.sdk.utils.helpers import safe_import

def safe_import(module_name: str) -> Optional[Any]
```

#### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `module_name` | `str` | Name of module to import |

#### Returns

- `Optional[Any]`: Imported module or None if not available

#### Example

```python
from rizk.sdk.utils.helpers import safe_import

# Safely import optional dependency
langchain = safe_import("langchain")
if langchain:
    print("LangChain is available")
    from langchain.agents import create_openai_tools_agent
else:
    print("LangChain not installed")
```

---

### `get_version()`

**Get SDK version information.**

```python
def get_version() -> str
```

#### Returns

- `str`: SDK version string

#### Example

```python
from rizk.sdk.utils.helpers import get_version

version = get_version()
print(f"Rizk SDK version: {version}")
```

---

### `is_async_function()`

**Check if a function is async.**

```python
def is_async_function(func: Callable) -> bool
```

#### Example

```python
from rizk.sdk.utils.helpers import is_async_function

async def async_func():
    pass

def sync_func():
    pass

print(is_async_function(async_func))  # True
print(is_async_function(sync_func))   # False
```

## Debugging Utilities

### `debug_context()`

**Context manager for debug information collection.**

```python
from rizk.sdk.utils.debug import debug_context

with debug_context("operation_name") as debug:
    # Your operation here
    result = some_operation()
    
    # Add debug information
    debug.add_info("step", "data_processing")
    debug.add_metric("items_processed", 100)

# Debug information is automatically logged
```

---

### `collect_diagnostics()`

**Collect diagnostic information about SDK state.**

```python
from rizk.sdk.utils.debug import collect_diagnostics

def collect_diagnostics() -> Dict[str, Any]
```

#### Returns

- `Dict[str, Any]`: Diagnostic information

#### Example

```python
from rizk.sdk.utils.debug import collect_diagnostics

diagnostics = collect_diagnostics()
print(f"SDK version: {diagnostics['version']}")
print(f"Active adapters: {diagnostics['adapters']}")
print(f"Cache stats: {diagnostics['cache_stats']}")
print(f"Configuration: {diagnostics['config']}")
```

## Best Practices

### 1. Use Appropriate Caching

```python
from rizk.sdk.utils.cache import CacheManager

# Cache expensive operations
@performance_monitor()
def expensive_framework_detection(func):
    cache_key = f"framework_{func.__name__}_{hash(func)}"
    
    # Check cache first
    cached_result = CacheManager.get(cache_key, "framework_detection")
    if cached_result:
        return cached_result
    
    # Perform detection
    result = detect_framework(func)
    
    # Cache result
    CacheManager.set(cache_key, result, ttl=3600, cache_type="framework_detection")
    return result
```

### 2. Comprehensive Error Handling

```python
from rizk.sdk.utils.error_handling import handle_errors, RizkErrorReporter

@handle_errors(fail_closed=False, default_return_on_error={"allowed": True})
async def safe_guardrails_operation(message: str):
    try:
        return await guardrails.process_message(message)
    except Exception as e:
        RizkErrorReporter.report_error(
            error=e,
            context={"operation": "guardrails", "message_length": len(message)}
        )
        raise  # Let decorator handle with default return
```

### 3. Performance Monitoring

```python
from rizk.sdk.utils.performance import Timer, performance_monitor

@performance_monitor(log_threshold_ms=500)
async def monitored_operation(data):
    with Timer("preprocessing") as prep_timer:
        processed_data = preprocess(data)
    
    with Timer("main_operation") as main_timer:
        result = await main_operation(processed_data)
    
    # Log detailed timing
    logger.info(f"Operation completed - prep: {prep_timer.elapsed_ms}ms, main: {main_timer.elapsed_ms}ms")
    return result
```

### 4. Context Management

```python
from rizk.sdk.utils.context import set_hierarchy_context, get_current_context

async def process_user_request(user_id: str, message: str):
    # Set context for entire operation
    set_hierarchy_context(
        organization_id="my_org",
        project_id="chat_service",
        user_id=user_id,
        conversation_id=f"conv_{user_id}_{int(time.time())}"
    )
    
    try:
        # All operations inherit this context
        result = await guardrails.process_message(message)
        
        # Context is automatically included in traces
        return await process_approved_message(message)
        
    finally:
        # Context is automatically cleared
        pass
```

## Related APIs

- **[Rizk Class API](./rizk-class.md)** - Main SDK interface
- **[Configuration API](./configuration-api.md)** - Configuration management
- **[Types API](./types.md)** - Type definitions
- **[GuardrailsEngine API](./guardrails-api.md)** - Policy enforcement engine 

