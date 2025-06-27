---
title: "Configuration API Reference"
description: "Configuration API Reference"
---

# Configuration API Reference

The Rizk SDK provides centralized configuration management with validation, environment variable parsing, and sensible defaults through the `RizkConfig` class and related utilities.

## Overview

```python
from rizk.sdk.config import RizkConfig, get_config, set_config, reset_config

# Create configuration from environment
config = RizkConfig.from_env()

# Validate configuration
errors = config.validate()
if errors:
    print(f"Configuration errors: {errors}")

# Set as global configuration
set_config(config)
```

## RizkConfig Class

### Class Definition

```python
@dataclass
class RizkConfig:
    """Centralized configuration for the Rizk SDK."""
    
    # Core settings
    app_name: str = "RizkApp"
    api_key: Optional[str] = field(default_factory=lambda: os.getenv("RIZK_API_KEY"))
    
    # OpenTelemetry settings
    opentelemetry_endpoint: Optional[str] = field(default_factory=lambda: os.getenv("RIZK_OPENTELEMETRY_ENDPOINT"))
    tracing_enabled: bool = field(default_factory=lambda: os.getenv("RIZK_TRACING_ENABLED", "true").lower() == "true")
    trace_content: bool = field(default_factory=lambda: os.getenv("RIZK_TRACE_CONTENT", "true").lower() == "true")
    metrics_enabled: bool = field(default_factory=lambda: os.getenv("RIZK_METRICS_ENABLED", "true").lower() == "true")
    
    # Logging settings
    logging_enabled: bool = field(default_factory=lambda: os.getenv("RIZK_LOGGING_ENABLED", "false").lower() == "true")
    
    # Guardrails settings
    policies_path: Optional[str] = field(default=None)
    policy_enforcement: bool = field(default_factory=lambda: os.getenv("RIZK_POLICY_ENFORCEMENT", "true").lower() == "true")
    
    # Telemetry settings
    telemetry_enabled: bool = field(default_factory=lambda: os.getenv("RIZK_TELEMETRY", "false").lower() == "true")
    
    # Performance settings
    lazy_loading: bool = field(default_factory=lambda: os.getenv("RIZK_LAZY_LOADING", "true").lower() == "true")
    framework_detection_cache_size: int = field(default_factory=lambda: int(os.getenv("RIZK_FRAMEWORK_CACHE_SIZE", "1000")))
    
    # Debug settings
    debug_mode: bool = field(default_factory=lambda: os.getenv("RIZK_DEBUG", "false").lower() == "true")
    verbose: bool = field(default_factory=lambda: os.getenv("RIZK_VERBOSE", "false").lower() == "true")
```

### Configuration Fields

#### Core Settings

| Field | Type | Default | Environment Variable | Description |
|-------|------|---------|---------------------|-------------|
| `app_name` | `str` | `"RizkApp"` | - | Application name for identification |
| `api_key` | `Optional[str]` | `None` | `RIZK_API_KEY` | Rizk API key for authentication |

#### OpenTelemetry Settings

| Field | Type | Default | Environment Variable | Description |
|-------|------|---------|---------------------|-------------|
| `opentelemetry_endpoint` | `Optional[str]` | `None` | `RIZK_OPENTELEMETRY_ENDPOINT` | Custom OTLP endpoint |
| `tracing_enabled` | `bool` | `True` | `RIZK_TRACING_ENABLED` | Enable/disable distributed tracing |
| `trace_content` | `bool` | `True` | `RIZK_TRACE_CONTENT` | Include content in traces |
| `metrics_enabled` | `bool` | `True` | `RIZK_METRICS_ENABLED` | Enable/disable metrics collection |

#### Guardrails Settings

| Field | Type | Default | Environment Variable | Description |
|-------|------|---------|---------------------|-------------|
| `policies_path` | `Optional[str]` | Auto-detected | `RIZK_POLICIES_PATH` | Path to custom policies directory |
| `policy_enforcement` | `bool` | `True` | `RIZK_POLICY_ENFORCEMENT` | Enable/disable policy enforcement |

#### Performance Settings

| Field | Type | Default | Environment Variable | Description |
|-------|------|---------|---------------------|-------------|
| `lazy_loading` | `bool` | `True` | `RIZK_LAZY_LOADING` | Enable lazy component loading |
| `framework_detection_cache_size` | `int` | `1000` | `RIZK_FRAMEWORK_CACHE_SIZE` | Cache size for framework detection |

#### Debug Settings

| Field | Type | Default | Environment Variable | Description |
|-------|------|---------|---------------------|-------------|
| `debug_mode` | `bool` | `False` | `RIZK_DEBUG` | Enable debug mode |
| `verbose` | `bool` | `False` | `RIZK_VERBOSE` | Enable verbose logging |
| `logging_enabled` | `bool` | `False` | `RIZK_LOGGING_ENABLED` | Enable SDK internal logging |
| `telemetry_enabled` | `bool` | `False` | `RIZK_TELEMETRY` | Enable anonymous telemetry |

## Class Methods

### `validate()`

**Validate configuration and return list of errors.**

```python
def validate(self) -> List[str]
```

#### Returns

- `List[str]`: List of validation error messages. Empty if valid.

#### Validation Rules

- **API Key Format**: Must start with `"rizk_"` if provided
- **Endpoint URL**: Must be valid HTTP/HTTPS URL if provided
- **Policies Path**: Must exist if specified
- **Cache Size**: Must be non-negative
- **App Name**: Cannot be empty

#### Example

```python
config = RizkConfig(
    app_name="MyApp",
    api_key="invalid-key",
    opentelemetry_endpoint="not-a-url",
    framework_detection_cache_size=-1
)

errors = config.validate()
print(errors)
# Output: [
#     "API key must start with 'rizk_'",
#     "OpenTelemetry endpoint must be a valid HTTP/HTTPS URL",
#     "Framework detection cache size must be non-negative"
# ]
```

---

### `is_valid()`

**Check if configuration is valid.**

```python
def is_valid(self) -> bool
```

#### Returns

- `bool`: True if configuration is valid, False otherwise

#### Example

```python
config = RizkConfig.from_env()
if config.is_valid():
    print("Configuration is valid")
else:
    print("Configuration has errors:")
    for error in config.validate():
        print(f"  - {error}")
```

---

### `to_dict()`

**Convert configuration to dictionary (with sensitive data masked).**

```python
def to_dict(self) -> Dict[str, Any]
```

#### Returns

- `Dict[str, Any]`: Configuration as dictionary with API key masked

#### Example

```python
config = RizkConfig(api_key="rizk_live_secret123")
config_dict = config.to_dict()
print(config_dict["api_key"])  # Output: "***"
```

---

### `from_dict()`

**Create configuration from dictionary.**

```python
@classmethod
def from_dict(cls, config_dict: Dict[str, Any]) -> "RizkConfig"
```

#### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `config_dict` | `Dict[str, Any]` | Configuration dictionary |

#### Returns

- `RizkConfig`: Configuration instance

#### Example

```python
config_dict = {
    "app_name": "ProductionApp",
    "api_key": "rizk_live_key123",
    "tracing_enabled": True,
    "debug_mode": False
}

config = RizkConfig.from_dict(config_dict)
print(config.app_name)  # Output: "ProductionApp"
```

---

### `from_env()`

**Create configuration from environment variables with optional overrides.**

```python
@classmethod
def from_env(cls, **overrides: Any) -> "RizkConfig"
```

#### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `**overrides` | `Any` | Override values for specific configuration fields |

#### Returns

- `RizkConfig`: Configuration instance

#### Example

```python
# Create from environment with overrides
config = RizkConfig.from_env(
    app_name="CustomApp",
    debug_mode=True
)

# Environment variables are read automatically
# RIZK_API_KEY, RIZK_TRACING_ENABLED, etc.
```

## Global Configuration Management

### `get_config()`

**Get the current global configuration.**

```python
def get_config() -> RizkConfig
```

#### Returns

- `RizkConfig`: Current global configuration instance

#### Raises

- `Exception`: If no configuration has been set

#### Example

```python
from rizk.sdk.config import get_config

try:
    config = get_config()
    print(f"App name: {config.app_name}")
except Exception:
    print("No configuration set")
```

---

### `set_config()`

**Set the global configuration.**

```python
def set_config(config: RizkConfig) -> None
```

#### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `config` | `RizkConfig` | Configuration instance to set globally |

#### Example

```python
from rizk.sdk.config import set_config, RizkConfig

# Create and set configuration
config = RizkConfig(
    app_name="GlobalApp",
    api_key="rizk_live_key123"
)
set_config(config)
```

---

### `reset_config()`

**Reset the global configuration (clears current config).**

```python
def reset_config() -> None
```

#### Example

```python
from rizk.sdk.config import reset_config

# Clear global configuration
reset_config()
```

## Utility Functions

### `get_policies_path()`

**Get the policies path with automatic detection.**

```python
def get_policies_path() -> str
```

#### Returns

- `str`: Path to policies directory

#### Path Resolution

1. `RIZK_POLICIES_PATH` environment variable
2. `./policies` directory if it exists
3. `./rizk_policies` directory if it exists
4. SDK default policies

#### Example

```python
from rizk.sdk.config import get_policies_path

policies_path = get_policies_path()
print(f"Using policies from: {policies_path}")
```

---

### `get_api_key()`

**Get the Rizk API key from global configuration.**

```python
def get_api_key() -> Optional[str]
```

#### Returns

- `Optional[str]`: API key if available, None otherwise

#### Example

```python
from rizk.sdk.config import get_api_key

api_key = get_api_key()
if api_key:
    print("API key is configured")
else:
    print("No API key found")
```

## Environment Variable Reference

### Complete Environment Variables List

```bash
# Core Configuration
export RIZK_API_KEY="rizk_live_your_key_here"

# OpenTelemetry Configuration  
export RIZK_OPENTELEMETRY_ENDPOINT="https://custom-otlp.company.com"
export RIZK_TRACING_ENABLED="true"
export RIZK_TRACE_CONTENT="false"  # Disable for privacy
export RIZK_METRICS_ENABLED="true"

# Guardrails Configuration
export RIZK_POLICIES_PATH="/app/custom-policies"
export RIZK_POLICY_ENFORCEMENT="true"

# Performance Configuration
export RIZK_LAZY_LOADING="true"
export RIZK_FRAMEWORK_CACHE_SIZE="5000"

# Debug Configuration
export RIZK_DEBUG="false"
export RIZK_VERBOSE="false"
export RIZK_LOGGING_ENABLED="false"
export RIZK_TELEMETRY="false"
```

### Environment-Specific Examples

#### Development Environment

```bash
# development.env
export RIZK_API_KEY="rizk_dev_your_key_here"
export RIZK_DEBUG="true"
export RIZK_VERBOSE="true"
export RIZK_TRACE_CONTENT="true"
export RIZK_POLICIES_PATH="./dev-policies"
```

#### Staging Environment

```bash
# staging.env
export RIZK_API_KEY="rizk_staging_your_key_here"
export RIZK_DEBUG="false"
export RIZK_VERBOSE="false"
export RIZK_TRACE_CONTENT="false"
export RIZK_FRAMEWORK_CACHE_SIZE="3000"
```

#### Production Environment

```bash
# production.env
export RIZK_API_KEY="rizk_prod_your_key_here"
export RIZK_DEBUG="false"
export RIZK_VERBOSE="false"
export RIZK_LOGGING_ENABLED="false"
export RIZK_TRACE_CONTENT="false"
export RIZK_TELEMETRY="false"
export RIZK_FRAMEWORK_CACHE_SIZE="10000"
export RIZK_POLICIES_PATH="/app/policies"
```

## Configuration Patterns

### Environment-Based Configuration

```python
import os
from rizk.sdk.config import RizkConfig, set_config

def setup_config():
    """Setup configuration based on environment."""
    env = os.getenv("ENVIRONMENT", "development")
    
    if env == "production":
        config = RizkConfig.from_env(
            debug_mode=False,
            verbose=False,
            trace_content=False,
            telemetry_enabled=False
        )
    elif env == "staging":
        config = RizkConfig.from_env(
            debug_mode=False,
            verbose=False,
            trace_content=False
        )
    else:  # development
        config = RizkConfig.from_env(
            debug_mode=True,
            verbose=True,
            trace_content=True
        )
    
    # Validate and set
    errors = config.validate()
    if errors:
        raise ValueError(f"Configuration errors: {errors}")
    
    set_config(config)
    return config
```

### Configuration Validation

```python
from rizk.sdk.config import RizkConfig

def validate_production_config(config: RizkConfig) -> List[str]:
    """Validate production-specific requirements."""
    errors = config.validate()  # Base validation
    
    # Additional production checks
    if not config.api_key:
        errors.append("API key is required in production")
    
    if config.debug_mode:
        errors.append("Debug mode must be disabled in production")
    
    if config.trace_content:
        errors.append("Content tracing should be disabled in production for privacy")
    
    if config.framework_detection_cache_size < 5000:
        errors.append("Cache size should be >= 5000 in production")
    
    return errors

# Usage
config = RizkConfig.from_env()
prod_errors = validate_production_config(config)
if prod_errors:
    print("Production validation failed:")
    for error in prod_errors:
        print(f"  - {error}")
```

### Configuration Templates

```python
from rizk.sdk.config import RizkConfig

class ConfigTemplates:
    """Predefined configuration templates."""
    
    @staticmethod
    def development() -> RizkConfig:
        """Development configuration template."""
        return RizkConfig(
            app_name="MyApp-Dev",
            debug_mode=True,
            verbose=True,
            trace_content=True,
            framework_detection_cache_size=1000,
            policies_path="./dev-policies"
        )
    
    @staticmethod
    def production() -> RizkConfig:
        """Production configuration template."""
        return RizkConfig(
            app_name="MyApp-Prod",
            debug_mode=False,
            verbose=False,
            trace_content=False,
            telemetry_enabled=False,
            framework_detection_cache_size=10000,
            policies_path="/app/policies"
        )
    
    @staticmethod
    def testing() -> RizkConfig:
        """Testing configuration template."""
        return RizkConfig(
            app_name="MyApp-Test",
            tracing_enabled=False,
            policy_enforcement=False,
            debug_mode=True
        )

# Usage
config = ConfigTemplates.production()
config.api_key = os.getenv("RIZK_API_KEY")
set_config(config)
```

## Configuration Monitoring

### Runtime Configuration Changes

```python
from rizk.sdk.config import get_config, set_config

def update_config(**changes):
    """Update configuration at runtime."""
    current_config = get_config()
    
    # Create new config with changes
    new_config = RizkConfig.from_dict({
        **current_config.to_dict(),
        **changes
    })
    
    # Validate before applying
    errors = new_config.validate()
    if errors:
        raise ValueError(f"Invalid configuration changes: {errors}")
    
    set_config(new_config)
    return new_config

# Usage
update_config(
    debug_mode=True,
    verbose=True
)
```

### Configuration Logging

```python
import logging
from rizk.sdk.config import get_config

def log_configuration():
    """Log current configuration (safely)."""
    config = get_config()
    config_dict = config.to_dict()
    
    logger = logging.getLogger("rizk.config")
    logger.info("Current Rizk SDK configuration:")
    for key, value in config_dict.items():
        logger.info(f"  {key}: {value}")

# Usage
log_configuration()
```

## Best Practices

### 1. Environment Variable Management

```python
# Good - use environment variables
config = RizkConfig.from_env()

# Avoid - hardcoded secrets
config = RizkConfig(api_key="rizk_live_secret123")  # Don't do this!
```

### 2. Configuration Validation

```python
def safe_config_setup():
    """Setup configuration with proper validation."""
    try:
        config = RizkConfig.from_env()
        
        # Validate configuration
        errors = config.validate()
        if errors:
            print("Configuration errors found:")
            for error in errors:
                print(f"  - {error}")
            return None
        
        # Set as global configuration
        set_config(config)
        print("âœ… Configuration loaded successfully")
        return config
        
    except Exception as e:
        print(f"âŒ Failed to load configuration: {e}")
        return None
```

### 3. Environment-Specific Settings

```python
import os

def create_environment_config():
    """Create configuration based on deployment environment."""
    env = os.getenv("DEPLOYMENT_ENV", "development")
    
    base_config = RizkConfig.from_env()
    
    if env == "production":
        # Production-specific overrides
        base_config.debug_mode = False
        base_config.verbose = False
        base_config.trace_content = False
        base_config.telemetry_enabled = False
        
    elif env == "staging":
        # Staging-specific overrides
        base_config.debug_mode = False
        base_config.verbose = False
        
    # Development uses defaults
    
    return base_config
```

### 4. Configuration Documentation

```python
def print_config_help():
    """Print help for configuration options."""
    print("""
    Rizk SDK Configuration Environment Variables:
    
    Core Settings:
      RIZK_API_KEY              - Your Rizk API key (required)
      
    OpenTelemetry Settings:
      RIZK_OPENTELEMETRY_ENDPOINT - Custom OTLP endpoint (optional)
      RIZK_TRACING_ENABLED      - Enable tracing (default: true)
      RIZK_TRACE_CONTENT        - Include content in traces (default: true)
      
    Guardrails Settings:
      RIZK_POLICIES_PATH        - Custom policies directory (optional)
      RIZK_POLICY_ENFORCEMENT   - Enable policy enforcement (default: true)
      
    Performance Settings:
      RIZK_FRAMEWORK_CACHE_SIZE - Framework detection cache size (default: 1000)
      
    Debug Settings:
      RIZK_DEBUG                - Enable debug mode (default: false)
      RIZK_VERBOSE              - Enable verbose logging (default: false)
    """)
```

## Related APIs

- **[Rizk Class API](./rizk-class.md)** - SDK initialization using configuration
- **[GuardrailsEngine API](./guardrails-api.md)** - Policy enforcement configuration
- **[Types API](./types.md)** - Configuration type definitions 

