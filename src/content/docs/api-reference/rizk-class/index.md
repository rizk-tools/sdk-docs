---
title: "Rizk Class API Reference"
description: "Rizk Class API Reference"
---

# Rizk Class API Reference

The `Rizk` class is the main entry point for initializing and configuring the Rizk SDK. It provides static methods for SDK initialization, configuration management, and access to core components.

## Class Overview

```python
from rizk.sdk import Rizk

# Initialize the SDK
rizk = Rizk.init(
    app_name="MyApplication",
    api_key="your-api-key",
    enabled=True
)

# Access guardrails engine
guardrails = Rizk.get_guardrails()

# Access client instance
client = Rizk.get()
```

## Methods

### `Rizk.init()`

**Static method to initialize the Rizk SDK with configuration options.**

```python
@staticmethod
def init(
    app_name: str = sys.argv[0],
    api_key: Optional[str] = None,
    opentelemetry_endpoint: Optional[str] = None,
    enabled: bool = True,
    telemetry_enabled: bool = False,
    headers: Dict[str, str] = {},
    disable_batch: bool = False,
    exporter: Optional[SpanExporter] = None,
    processor: Optional[SpanProcessor] = None,
    propagator: Optional[TextMapPropagator] = None,
    instruments: Optional[Set[Any]] = None,
    block_instruments: Optional[Set[Any]] = None,
    resource_attributes: Dict[str, Any] = {},
    policies_path: Optional[str] = None,
    llm_service: Optional["LLMServiceProtocol"] = None,
    verbose: bool = False,
    **kwargs: Any,
) -> Optional[Client]
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `app_name` | `str` | `sys.argv[0]` | Name of your application for identification in traces |
| `api_key` | `Optional[str]` | `None` | Your Rizk API key. If not provided, reads from `RIZK_API_KEY` environment variable |
| `opentelemetry_endpoint` | `Optional[str]` | `None` | Custom OTLP endpoint. Defaults to `https://api.rizk.tools` when `api_key` is set |
| `enabled` | `bool` | `True` | Whether tracing is enabled |
| `telemetry_enabled` | `bool` | `False` | Whether to allow anonymous telemetry collection |
| `headers` | `Dict[str, str]` | `{}` | Custom headers for API requests |
| `disable_batch` | `bool` | `False` | Disable batch processing for traces |
| `exporter` | `Optional[SpanExporter]` | `None` | Custom OpenTelemetry span exporter |
| `processor` | `Optional[SpanProcessor]` | `None` | Custom OpenTelemetry span processor |
| `propagator` | `Optional[TextMapPropagator]` | `None` | Custom OpenTelemetry propagator |
| `instruments` | `Optional[Set[Any]]` | `None` | Specific instruments to enable |
| `block_instruments` | `Optional[Set[Any]]` | `None` | Specific instruments to disable |
| `resource_attributes` | `Dict[str, Any]` | `{}` | Additional OpenTelemetry resource attributes |
| `policies_path` | `Optional[str]` | `None` | Path to custom guardrail policies directory |
| `llm_service` | `Optional["LLMServiceProtocol"]` | `None` | Custom LLM service for guardrails evaluation |
| `verbose` | `bool` | `False` | Enable detailed INFO-level logging |
| `**kwargs` | `Any` | - | Additional arguments passed to Traceloop SDK |

#### Returns

- `Optional[Client]`: Rizk client instance if initialization successful, `None` otherwise

#### Examples

**Basic initialization:**
```python
from rizk.sdk import Rizk

# Basic setup with API key
rizk = Rizk.init(
    app_name="ChatBot",
    api_key="rizk_live_your_key_here",
    enabled=True
)
```

**Production configuration:**
```python
# Production setup with custom policies
rizk = Rizk.init(
    app_name="ProductionApp",
    api_key=os.getenv("RIZK_API_KEY"),
    policies_path="/app/policies",
    verbose=False,
    telemetry_enabled=False
)
```

**Custom endpoint configuration:**
```python
# Using custom OTLP endpoint
rizk = Rizk.init(
    app_name="CustomTelemetry",
    opentelemetry_endpoint="https://otel.company.com:4317",
    enabled=True
)
```

> **Note**: When `api_key` is provided, traces are automatically sent to `https://api.rizk.tools`. Set `opentelemetry_endpoint` only if you want to use a different OTLP collector.

#### Raises

- `Exception`: If SDK is already initialized or configuration validation fails

---

### `Rizk.get()`

**Static method to get the initialized Rizk client instance.**

```python
@staticmethod
def get() -> Client
```

#### Returns

- `Client`: The initialized Rizk client instance

#### Raises

- `Exception`: If SDK not initialized or client not available

#### Example

```python
# Initialize first
Rizk.init(app_name="MyApp", api_key="your-key")

# Get client instance
client = Rizk.get()
```

---

### `Rizk.get_guardrails()`

**Static method to get the initialized GuardrailsEngine instance.**

```python
@staticmethod
def get_guardrails() -> GuardrailsEngine
```

#### Returns

- `GuardrailsEngine`: The initialized guardrails engine instance

#### Raises

- `Exception`: If SDK not initialized or guardrails engine not available

#### Example

```python
# Initialize first
Rizk.init(app_name="MyApp", api_key="your-key")

# Get guardrails engine
guardrails = Rizk.get_guardrails()

# Process a message
result = await guardrails.process_message("Hello world")
print(f"Allowed: {result['allowed']}")
```

---

### `Rizk.set_association_properties()`

**Static method to set association properties for the current trace context.**

```python
@staticmethod
def set_association_properties(properties: Dict[str, Any]) -> None
```

#### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `properties` | `Dict[str, Any]` | Dictionary of properties to associate with current trace |

#### Example

```python
# Set context properties for tracing
Rizk.set_association_properties({
    "user_id": "user_123",
    "conversation_id": "conv_456",
    "organization_id": "org_789",
    "project_id": "proj_abc"
})
```

## Configuration Management

The Rizk class integrates with the centralized configuration system. Configuration is validated on initialization and stored globally.

### Environment Variables

The following environment variables are automatically read during initialization:

| Variable | Default | Description |
|----------|---------|-------------|
| `RIZK_API_KEY` | - | Rizk API key for authentication |
| `RIZK_OPENTELEMETRY_ENDPOINT` | - | Custom OTLP endpoint |
| `RIZK_TRACING_ENABLED` | `"true"` | Enable/disable tracing |
| `RIZK_TRACE_CONTENT` | `"true"` | Include content in traces |
| `RIZK_TELEMETRY` | `"false"` | Enable anonymous telemetry |
| `RIZK_POLICIES_PATH` | - | Custom policies directory |
| `RIZK_VERBOSE` | `"false"` | Enable verbose logging |

### Configuration Validation

The Rizk class validates configuration on initialization:

- **API Key Format**: Must start with `"rizk_"`
- **Endpoint URL**: Must be valid HTTP/HTTPS URL if provided
- **Policies Path**: Must exist if specified
- **App Name**: Cannot be empty

## Error Handling

The Rizk class uses the `@handle_errors` decorator for graceful error handling:

```python
# Initialization errors are handled gracefully
rizk = Rizk.init(
    app_name="MyApp",
    api_key="invalid-key"  # Won't fail initialization
)

# Check if initialization was successful
try:
    client = Rizk.get()
    print("SDK initialized successfully")
except Exception as e:
    print(f"SDK initialization failed: {e}")
```

## Thread Safety

The Rizk class implements thread-safe singleton pattern:

- **Double-checked locking** for instance creation
- **Reentrant locks** for configuration updates
- **Thread-safe lazy initialization** of components

## Integration with Framework Detection

The Rizk class automatically registers and patches framework adapters:

```python
# Automatic framework detection and patching
rizk = Rizk.init(app_name="MyApp", api_key="your-key")

# Framework adapters are automatically registered for:
# - OpenAI Agents SDK
# - LangChain
# - CrewAI  
# - LlamaIndex
# - Custom frameworks via plugins
```

## Best Practices

### 1. Initialize Early

Initialize the SDK as early as possible in your application:

```python
# At the top of your main module
from rizk.sdk import Rizk

# Initialize before importing other modules
rizk = Rizk.init(
    app_name="MyApp",
    api_key=os.getenv("RIZK_API_KEY")
)

# Now import and use other modules
from my_agents import ChatAgent
```

### 2. Use Environment Variables

Store sensitive configuration in environment variables:

```python
# Good - secure configuration
rizk = Rizk.init(
    app_name="ProductionApp",
    api_key=os.getenv("RIZK_API_KEY"),
    policies_path=os.getenv("RIZK_POLICIES_PATH", "./policies")
)
```

### 3. Handle Initialization Errors

Always check if initialization was successful:

```python
# Initialize with error handling
try:
    rizk = Rizk.init(app_name="MyApp")
    client = Rizk.get()
    print("âœ… Rizk SDK initialized successfully")
except Exception as e:
    print(f"âŒ Rizk SDK initialization failed: {e}")
    # Implement fallback behavior
```

### 4. Configure for Environment

Use different configurations for different environments:

```python
import os

# Environment-specific configuration
if os.getenv("ENVIRONMENT") == "production":
    rizk = Rizk.init(
        app_name="MyApp-Prod",
        api_key=os.getenv("RIZK_API_KEY"),
        verbose=False,
        telemetry_enabled=False
    )
else:
    rizk = Rizk.init(
        app_name="MyApp-Dev",
        api_key=os.getenv("RIZK_API_KEY"),
        verbose=True,
        policies_path="./dev-policies"
    )
```

## Related APIs

- **[Configuration API](./configuration-api.md)** - `RizkConfig` class and configuration management
- **[GuardrailsEngine API](./guardrails-api.md)** - Policy enforcement and evaluation
- **[Decorators API](./decorators-api.md)** - Function and class decorators
- **[Client API](./client-api.md)** - Client instance methods and properties 

