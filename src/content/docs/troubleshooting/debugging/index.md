---
title: "Debugging Guide"
description: "Debugging Guide"
---

# Debugging Guide

This guide provides systematic approaches to debug issues with Rizk SDK, including tools, techniques, and best practices for identifying and resolving problems.

## ðŸ” Debugging Fundamentals

### Enable Debug Logging

The first step in debugging any Rizk SDK issue is to enable comprehensive logging:

```python
import logging
import os

# Set up debug logging for all Rizk components
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # Console output
        logging.FileHandler('rizk_debug.log')  # File output
    ]
)

# Enable verbose mode in Rizk SDK
from rizk.sdk import Rizk
Rizk.init(
    app_name="DebugApp",
    api_key=os.getenv("RIZK_API_KEY"),
    verbose=True,  # Enables detailed internal logging
    debug_mode=True  # Additional debug information
)

# Enable specific logger categories
loggers = [
    "rizk",
    "rizk.sdk",
    "rizk.guardrails",
    "rizk.adapters",
    "rizk.utils.framework_detection",
    "rizk.utils.framework_registry",
]

for logger_name in loggers:
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)
```

### Debug Environment Setup

Create a debugging environment with enhanced diagnostics:

```python
import os
import sys
import traceback
from typing import Any, Dict

def setup_debug_environment() -> Dict[str, Any]:
    """Set up comprehensive debugging environment."""
    
    debug_info = {
        "python_version": sys.version,
        "platform": sys.platform,
        "rizk_version": None,
        "environment_vars": {},
        "installed_packages": [],
    }
    
    # Collect Rizk version
    try:
        from rizk.version import __version__
        debug_info["rizk_version"] = __version__
    except ImportError:
        debug_info["rizk_version"] = "Unknown - check installation"
    
    # Collect relevant environment variables
    for key, value in os.environ.items():
        if key.startswith("RIZK_"):
            # Mask sensitive values
            if "API_KEY" in key or "PASSWORD" in key:
                debug_info["environment_vars"][key] = "***"
            else:
                debug_info["environment_vars"][key] = value
    
    # Check installed packages
    try:
        import pkg_resources
        installed = [d.project_name for d in pkg_resources.working_set]
        relevant_packages = [p for p in installed if any(
            framework in p.lower() for framework in 
            ["rizk", "langchain", "crewai", "llama", "openai", "traceloop"]
        )]
        debug_info["installed_packages"] = relevant_packages
    except ImportError:
        debug_info["installed_packages"] = ["pkg_resources not available"]
    
    return debug_info

# Use the debug setup
debug_info = setup_debug_environment()
print("=== Debug Environment ===")
for key, value in debug_info.items():
    print(f"{key}: {value}")
```

## ðŸ§° Debugging Tools & Utilities

### 1. SDK Health Check

Create a comprehensive health check function:

```python
from typing import Dict, Any
from rizk.sdk import Rizk
from rizk.sdk.config import get_config
from rizk.sdk.guardrails.engine import GuardrailsEngine
from rizk.sdk.utils.framework_detection import detect_framework
from rizk.sdk.utils.framework_registry import FrameworkRegistry, LLMClientRegistry

def comprehensive_health_check() -> Dict[str, Any]:
    """Perform comprehensive SDK health check."""
    
    health_report = {
        "overall_status": "unknown",
        "components": {},
        "configuration": {},
        "errors": [],
        "warnings": []
    }
    
    try:
        # 1. Configuration Check
        print("ðŸ”§ Checking Configuration...")
        config = get_config()
        config_errors = config.validate()
        
        health_report["configuration"] = {
            "valid": len(config_errors) == 0,
            "errors": config_errors,
            "api_key_set": config.api_key is not None,
            "tracing_enabled": config.tracing_enabled,
            "policies_path": config.policies_path
        }
        
        if config_errors:
            health_report["errors"].extend(config_errors)
        
        # 2. SDK Initialization Check
        print("ðŸš€ Checking SDK Initialization...")
        try:
            client = Rizk.get()
            health_report["components"]["sdk_client"] = {
                "status": "initialized",
                "available": True
            }
        except Exception as e:
            health_report["components"]["sdk_client"] = {
                "status": "error",
                "available": False,
                "error": str(e)
            }
            health_report["errors"].append(f"SDK client error: {e}")
        
        # Overall status determination
        if health_report["errors"]:
            health_report["overall_status"] = "error"
        elif health_report["warnings"]:
            health_report["overall_status"] = "warning"
        else:
            health_report["overall_status"] = "healthy"
        
    except Exception as e:
        health_report["overall_status"] = "critical_error"
        health_report["errors"].append(f"Health check failed: {e}")
        print(f"âŒ Health check failed: {e}")
    
    return health_report

# Run health check
print("ðŸ¥ Running Comprehensive Health Check...")
health = comprehensive_health_check()

print(f"\nðŸ“‹ Health Check Results:")
print(f"Overall Status: {health['overall_status'].upper()}")

if health["errors"]:
    print(f"\nâŒ Errors ({len(health['errors'])}):")
    for error in health["errors"]:
        print(f"  â€¢ {error}")

if health["warnings"]:
    print(f"\nâš ï¸ Warnings ({len(health['warnings'])}):")
    for warning in health["warnings"]:
        print(f"  â€¢ {warning}")
```

### 2. Framework Detection Debugger

Debug framework detection issues:

```python
from rizk.sdk.utils.framework_detection import (
    detect_framework, 
    detect_framework_cached,
    clear_detection_cache,
    FRAMEWORK_DETECTION_PATTERNS
)
import sys

def debug_framework_detection(target_object=None):
    """Debug framework detection for a specific object or general environment."""
    
    print("ðŸ” Framework Detection Debug Report")
    print("=" * 50)
    
    # 1. Show available detection patterns
    print("\nðŸ“‹ Available Framework Patterns:")
    for framework, patterns in FRAMEWORK_DETECTION_PATTERNS.items():
        print(f"  {framework}:")
        print(f"    Modules: {patterns['modules']}")
        print(f"    Classes: {patterns['classes']}")
        print(f"    Import Names: {patterns['import_names']}")
    
    # 2. Test general detection
    print("\nðŸŽ¯ General Framework Detection:")
    general_detection = detect_framework()
    print(f"  Result: {general_detection}")
    
    # 3. Test cached detection
    print("\nðŸ’¾ Cached Framework Detection:")
    cached_detection = detect_framework_cached()
    print(f"  Result: {cached_detection}")
    
    # 4. Performance test
    print("\nâš¡ Performance Test:")
    import time
    
    # Test original detection
    start = time.time()
    for _ in range(100):
        detect_framework()
    original_time = time.time() - start
    
    # Test cached detection
    start = time.time()
    for _ in range(100):
        detect_framework_cached()
    cached_time = time.time() - start
    
    print(f"  Original detection (100 calls): {original_time*1000:.2f}ms")
    print(f"  Cached detection (100 calls): {cached_time*1000:.2f}ms")
    print(f"  Speedup: {original_time/cached_time:.1f}x")

# Example usage:
debug_framework_detection()
```

### 3. Policy Debugger

Debug guardrails and policy evaluation:

```python
from rizk.sdk.guardrails.engine import GuardrailsEngine
from rizk.sdk.guardrails.fast_rules import FastRulesEngine
from rizk.sdk.config import get_policies_path

def debug_policy_evaluation(test_message: str, direction: str = "inbound"):
    """Debug policy evaluation step by step."""
    
    print(f"ðŸ›¡ï¸ Policy Evaluation Debug for: '{test_message[:50]}...'")
    print("=" * 60)
    
    try:
        # 1. Check policy loading
        print("\nðŸ“‹ Step 1: Policy Loading")
        policies_path = get_policies_path()
        print(f"Policies path: {policies_path}")
        
        fast_rules = FastRulesEngine(policies_path)
        print(f"âœ… Loaded {len(fast_rules.policies)} policies")
        
        # 2. Fast Rules Evaluation
        print(f"\nâš¡ Step 2: Fast Rules Evaluation")
        fast_result = fast_rules.evaluate(test_message, direction=direction)
        print(f"  Result: {'ðŸš« BLOCKED' if fast_result.blocked else 'âœ… ALLOWED'}")
        print(f"  Confidence: {fast_result.confidence:.2f}")
        print(f"  Reason: {fast_result.reason}")
        
        # 3. Full Guardrails Engine Evaluation
        print(f"\nðŸ”’ Step 3: Full Guardrails Evaluation")
        engine = GuardrailsEngine.get_instance()
        
        context = {
            "conversation_id": "debug_conversation",
            "user_id": "debug_user"
        }
        
        full_result = engine.evaluate(test_message, direction=direction, context=context)
        
        print(f"  Final Result: {'âœ… ALLOWED' if full_result.allowed else 'ðŸš« BLOCKED'}")
        print(f"  Confidence: {full_result.confidence:.2f}")
        print(f"  Decision Layer: {full_result.decision_layer}")
        
    except Exception as e:
        print(f"âŒ Debug process failed: {e}")
        import traceback
        traceback.print_exc()

# Example usage:
test_messages = [
    "Hello, how can I help you?",
    "My SSN is 123-45-6789",
    "Let me think step by step about this problem..."
]

for message in test_messages:
    debug_policy_evaluation(message)
    print("\n" + "="*80)
```

## ðŸ§ª Testing & Validation

### Unit Test Template

Create unit tests for your Rizk SDK integration:

```python
import unittest
import os
from unittest.mock import patch
from rizk.sdk import Rizk
from rizk.sdk.decorators import workflow, guardrails
from rizk.sdk.config import get_config, reset_config

class TestRizkIntegration(unittest.TestCase):
    """Test suite for Rizk SDK integration."""
    
    def setUp(self):
        """Set up test environment."""
        reset_config()
    
    def tearDown(self):
        """Clean up after tests."""
        reset_config()
    
    @patch.dict(os.environ, {"RIZK_API_KEY": "rizk_test_key"})
    def test_sdk_initialization(self):
        """Test basic SDK initialization."""
        client = Rizk.init(
            app_name="TestApp",
            api_key="rizk_test_key",
            enabled=False  # Don't actually send telemetry
        )
        
        # Verify initialization
        self.assertIsNotNone(client)
        
        # Verify configuration
        config = get_config()
        self.assertEqual(config.app_name, "TestApp")
        self.assertEqual(config.api_key, "rizk_test_key")
    
    def test_decorator_application(self):
        """Test that decorators can be applied without errors."""
        
        @workflow(name="test_workflow")
        def test_function(x: int) -> int:
            return x * 2
        
        # Function should work normally
        result = test_function(5)
        self.assertEqual(result, 10)
    
    def test_framework_detection(self):
        """Test framework detection functionality."""
        from rizk.sdk.utils.framework_detection import detect_framework
        
        # Should not crash
        framework = detect_framework()
        self.assertIsInstance(framework, str)

if __name__ == "__main__":
    unittest.main()
```

## ðŸ“‹ Debugging Checklist

Use this checklist when debugging Rizk SDK issues:

### Initial Setup
- [ ] Verify correct package installation (`pip show rizk`)
- [ ] Check Python version compatibility (>=3.10)
- [ ] Confirm API key format (starts with `rizk_`)
- [ ] Test basic SDK initialization
- [ ] Enable debug logging

### Configuration Issues
- [ ] Validate environment variables (`RIZK_*`)
- [ ] Check configuration file syntax (if using)
- [ ] Verify policies path exists and contains valid YAML
- [ ] Test configuration validation
- [ ] Check for conflicting settings

### Framework Integration
- [ ] Verify import order (Rizk before framework)
- [ ] Check framework detection results
- [ ] Confirm adapter registration
- [ ] Test decorator application
- [ ] Verify framework-specific dependencies

### Guardrails Issues
- [ ] Check policy loading and parsing
- [ ] Test individual policy evaluation
- [ ] Verify direction-specific policies (inbound/outbound)
- [ ] Check confidence thresholds
- [ ] Test with known good/bad inputs

### Performance Issues
- [ ] Profile function execution times
- [ ] Check cache hit rates
- [ ] Monitor memory usage
- [ ] Test with different configuration options
- [ ] Identify bottlenecks in call stack

This systematic approach to debugging will help you quickly identify and resolve issues with Rizk SDK.


