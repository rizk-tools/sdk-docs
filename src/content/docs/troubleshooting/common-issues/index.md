---
title: "Common Issues & Solutions"
description: "Common Issues & Solutions"
---

# Common Issues & Solutions

This guide covers the most frequently encountered issues when using Rizk SDK and their solutions.

## ðŸš¨ Installation & Setup Issues

### 1. Package Installation Problems

**Issue**: `pip install rizk` fails or installs wrong package
```bash
ERROR: Could not find a version that satisfies the requirement rizk
```

**Solution**:
```bash
# Ensure you're using the correct package name
pip install rizk  # Not rizk-sdk

# Update pip and try again
pip install --upgrade pip
pip install rizk

# For development/pre-release versions
pip install --pre rizk

# Clear pip cache if corrupted
pip cache purge
pip install rizk
```

### 2. Import Errors

**Issue**: `ImportError: No module named 'rizk'`
```python
from rizk.sdk import Rizk
# ImportError: No module named 'rizk'
```

**Solution**:
```bash
# Verify installation
pip show rizk

# Check Python environment
which python
pip list | grep rizk

# Reinstall if needed
pip uninstall rizk
pip install rizk
```

### 3. API Key Issues

**Issue**: Invalid API key format or authentication failures
```
WARNING: API key validation failed: API key must start with 'rizk_'
```

**Solution**:
```python
# Verify API key format
# âœ… Correct: starts with 'rizk_'
api_key = "rizk_oSOWmTpFjmQRQPvGhETRMgeShhmLdutZsfVLhWvNgfBvndaRNvpQywoKLmvxHLFw"

# âŒ Incorrect: wrong format
api_key = "sk-..." # This is OpenAI format, not Rizk

# Set via environment variable (recommended)
export RIZK_API_KEY="your-rizk-api-key"

# Or programmatically
import os
os.environ["RIZK_API_KEY"] = "your-rizk-api-key"
```

**Get your API key**:
1. Visit [app.rizk.tools](https://app.rizk.tools)
2. Sign up or log in
3. Navigate to API Keys section
4. Copy your key (starts with `rizk_`)

## ðŸ”§ Configuration Issues

### 4. Environment Variables Not Loading

**Issue**: Environment variables are ignored
```python
# Environment variable not being recognized
os.environ["RIZK_TRACING_ENABLED"] = "false"
# Still traces despite setting to false
```

**Solution**:
```python
# Check environment variable names (case-sensitive)
import os
print("Environment variables:")
for key in os.environ:
    if key.startswith("RIZK_"):
        print(f"{key}={os.environ[key]}")

# Correct environment variable names:
export RIZK_API_KEY="rizk_..."
export RIZK_TRACING_ENABLED="true"
export RIZK_TRACE_CONTENT="true" 
export RIZK_METRICS_ENABLED="true"
export RIZK_LOGGING_ENABLED="false"
export RIZK_POLICY_ENFORCEMENT="true"
export RIZK_POLICIES_PATH="/path/to/policies"
export RIZK_OPENTELEMETRY_ENDPOINT="https://your-endpoint.com"

# Verify configuration
from rizk.sdk.config import get_config
config = get_config()
print(config.to_dict())
```

### 5. Invalid Configuration Values

**Issue**: Configuration validation errors during initialization
```
ERROR: Configuration validation failed: ['API key must start with 'rizk_'', 'Framework cache size must be non-negative']
```

**Solution**:
```python
from rizk.sdk.config import RizkConfig

# Create and validate configuration
config = RizkConfig(
    app_name="MyApp",  # âŒ Can't be empty
    api_key="rizk_valid_key",  # âœ… Must start with 'rizk_'
    opentelemetry_endpoint="https://api.rizk.tools",  # âœ… Valid HTTPS URL
    framework_detection_cache_size=1000,  # âœ… Must be positive
)

# Check for validation errors
errors = config.validate()
if errors:
    print("Configuration errors:")
    for error in errors:
        print(f"  - {error}")
else:
    print("âœ… Configuration is valid")
```

## ðŸ” Framework Detection Issues

### 6. Framework Not Detected

**Issue**: SDK defaults to "standard" instead of detecting your framework
```python
from rizk.sdk.utils.framework_detection import detect_framework
print(detect_framework())  # Returns "standard" instead of "langchain"
```

**Solution**:
```python
# Ensure framework is imported before detection
import langchain  # Import your framework first
from rizk.sdk.utils.framework_detection import detect_framework

# Check detection
framework = detect_framework()
print(f"Detected framework: {framework}")

# For explicit framework specification
from rizk.sdk.decorators import workflow

@workflow(name="my_workflow", framework="langchain")  # Explicit
def my_langchain_function():
    pass

# Clear detection cache if needed
from rizk.sdk.utils.framework_detection import clear_detection_cache
clear_detection_cache()
```

### 7. Multiple Frameworks Detected

**Issue**: Conflicting framework detection when multiple frameworks are installed
```
WARNING: Multiple frameworks detected: ['langchain', 'crewai']
```

**Solution**:
```python
# Specify framework explicitly in decorators
from rizk.sdk.decorators import workflow

@workflow(name="langchain_workflow", framework="langchain")
def langchain_function():
    pass

@workflow(name="crewai_workflow", framework="crewai") 
def crewai_function():
    pass

# Or set globally during initialization
from rizk.sdk import Rizk
Rizk.init(
    app_name="MultiFramework", 
    # No global framework setting - specify per decorator
)
```

## ðŸ›¡ï¸ Guardrails Issues

### 8. Policies Not Loading

**Issue**: Custom policies are not being applied
```
WARNING: No policies found in /path/to/policies
```

**Solution**:
```python
# Check policies path exists
import os
policies_path = "/path/to/policies"
if os.path.exists(policies_path):
    print(f"âœ… Policies directory exists: {policies_path}")
    print(f"Files: {os.listdir(policies_path)}")
else:
    print(f"âŒ Policies directory not found: {policies_path}")

# Verify policy file format
cat your_policies.yaml
# Should be valid YAML with this structure:
"""
version: "1.0.0"
policies:
  - id: "my_policy"
    name: "My Custom Policy"
    domains: ["demo"]
    description: "Custom policy description"
    action: "allow"
    guidelines:
      - "Your guideline here"
    patterns:
      - "your_regex_pattern"
"""

# Test policy loading
from rizk.sdk.guardrails.fast_rules import FastRulesEngine
engine = FastRulesEngine(policies_path)
print(f"Loaded {len(engine.policies)} policies")
```

### 9. Guardrails Blocking Legitimate Content

**Issue**: Guardrails are too aggressive and blocking valid responses
```python
from rizk.sdk.decorators import guardrails

@guardrails()
def my_function(query):
    return "This legitimate response gets blocked"
```

**Solution**:
```python
# Option 1: Adjust confidence threshold
@guardrails(confidence_threshold=0.8)  # Higher threshold = less blocking
def my_function(query):
    return "Response"

# Option 2: Use specific policy sets
@guardrails(policy_set="lenient")
def my_function(query):
    return "Response"

# Option 3: Test policy evaluation
from rizk.sdk.guardrails.engine import GuardrailsEngine
engine = GuardrailsEngine.get_instance()

# Test your content
result = engine.evaluate("your test content")
print(f"Allowed: {result.allowed}")
print(f"Reason: {result.reason}")
print(f"Confidence: {result.confidence}")
```

### 10. MCP Guardrails Not Working

**Issue**: Memory leak prevention not blocking sensitive outputs
```python
@mcp_guardrails()
def my_function():
    return "SSN: 123-45-6789"  # Should be blocked but isn't
```

**Solution**:
```python
from rizk.sdk.decorators import mcp_guardrails

# Check violation mode
@mcp_guardrails(on_violation="block")  # Ensure blocking mode
def my_function():
    return "Sensitive content"

# Verify MCP policies are loaded
from rizk.sdk.guardrails.engine import GuardrailsEngine
engine = GuardrailsEngine.get_instance()

# Test outbound evaluation specifically
result = engine.evaluate(
    "SSN: 123-45-6789", 
    direction="outbound"  # Specify outbound direction
)
print(f"Blocked: {not result.allowed}")

# Check policy files include MCP policies
import yaml
with open("path/to/default_policies.yaml") as f:
    policies = yaml.safe_load(f)
    mcp_policies = [p for p in policies.get('policies', []) if 'memory_leak' in p.get('id', '')]
    print(f"Found {len(mcp_policies)} MCP policies")
```

## ðŸ“Š Observability Issues

### 11. No Traces Appearing

**Issue**: Traces not showing up in observability platform
```python
from rizk.sdk import Rizk
from rizk.sdk.decorators import workflow

Rizk.init(app_name="MyApp", api_key="rizk_...")

@workflow(name="test_workflow")
def test_function():
    return "test"

result = test_function()  # No traces appear
```

**Solution**:
```python
# Check if tracing is enabled
from rizk.sdk.config import get_config
config = get_config()
print(f"Tracing enabled: {config.tracing_enabled}")

# Verify API key and endpoint
print(f"API key set: {config.api_key is not None}")
print(f"OTLP endpoint: {config.opentelemetry_endpoint}")

# Check Traceloop integration
try:
    from traceloop.sdk import Traceloop
    print("âœ… Traceloop SDK available")
except ImportError:
    print("âŒ Traceloop SDK not installed")
    print("Install with: pip install traceloop-sdk")

# Enable verbose logging for debugging
import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("rizk")
logger.setLevel(logging.DEBUG)

# Test with explicit initialization
from rizk.sdk import Rizk
Rizk.init(
    app_name="DebugApp",
    api_key="rizk_your_key",
    enabled=True,  # Explicitly enable
    verbose=True,  # Enable verbose logging
)
```

### 12. Performance Issues

**Issue**: SDK adding significant latency to function calls
```python
# Function is much slower with decorators
@workflow(name="slow_function")
def my_function():
    # This now takes much longer
    return "result"
```

**Solution**:
```python
# Check framework detection cache
from rizk.sdk.utils.framework_detection import get_detection_cache_stats
stats = get_detection_cache_stats()
print(f"Cache hit rate: {stats['hit_rate']:.1f}%")

# Optimize cache settings
import os
os.environ["RIZK_FRAMEWORK_CACHE_SIZE"] = "5000"  # Larger cache
os.environ["RIZK_LAZY_LOADING"] = "true"  # Enable lazy loading

# Disable features you don't need
from rizk.sdk import Rizk
Rizk.init(
    app_name="OptimizedApp",
    api_key="rizk_your_key",
    telemetry_enabled=False,  # Disable if not needed
    disable_batch=True,  # Reduce latency
)

# Use performance monitoring
from rizk.sdk.performance import performance_instrumented

@performance_instrumented("my_operation")
def my_function():
    return "result"

# Check performance stats
from rizk.sdk.analytics.processors import PerformanceMonitoringProcessor
processor = PerformanceMonitoringProcessor()
stats = processor.get_performance_stats()
print(f"Average latency: {stats.get('avg_latency_ms', 0):.2f}ms")
```

## ðŸ”Œ Integration Issues

### 13. OpenAI Integration Problems

**Issue**: OpenAI API calls not being traced or governed
```python
import openai
openai.chat.completions.create(...)  # Not traced
```

**Solution**:
```python
# Ensure Rizk is initialized before OpenAI imports
from rizk.sdk import Rizk
Rizk.init(app_name="OpenAI_App", api_key="rizk_...")

# Import OpenAI after Rizk initialization
import openai

# Verify adapter registration
from rizk.sdk.utils.framework_registry import LLMClientRegistry
adapters = LLMClientRegistry.get_all_adapter_instances()
print("Registered LLM adapters:", list(adapters.keys()))

# Test with explicit decoration
from rizk.sdk.decorators import workflow

@workflow(name="openai_chat")
def chat_with_openai(message):
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": message}]
    )
    return response.choices[0].message.content
```

### 14. LangChain Integration Issues

**Issue**: LangChain agents/chains not being monitored
```python
from langchain.agents import AgentExecutor
# Agent execution not traced
```

**Solution**:
```python
# Import order matters - Rizk first
from rizk.sdk import Rizk
Rizk.init(app_name="LangChain_App")

# Then import LangChain
from langchain.agents import AgentExecutor
from langchain_openai import ChatOpenAI

# Use Rizk decorators
from rizk.sdk.decorators import workflow, tool

@tool(name="search_tool")
def search_tool(query: str) -> str:
    return f"Search results for: {query}"

@workflow(name="langchain_agent")
def run_agent(query: str):
    llm = ChatOpenAI(temperature=0)
    # Agent setup and execution
    pass

# Verify LangChain adapter
from rizk.sdk.utils.framework_registry import FrameworkRegistry
adapter = FrameworkRegistry.get_adapter("langchain")
print(f"LangChain adapter: {'âœ…' if adapter else 'âŒ'}")
```

### 15. CrewAI Integration Issues

**Issue**: CrewAI crew execution not being traced
```python
from crewai import Crew, Agent, Task
# Crew execution not monitored
```

**Solution**:
```python
# Correct initialization order
from rizk.sdk import Rizk
Rizk.init(app_name="CrewAI_App")

from crewai import Crew, Agent, Task
from rizk.sdk.decorators import crew, agent, task

@agent(name="research_agent")
def create_researcher():
    return Agent(
        role="Researcher",
        goal="Research information",
        backstory="Expert researcher"
    )

@task(name="research_task")
def create_research_task(agent):
    return Task(
        description="Research the topic",
        agent=agent
    )

@crew(name="research_crew")
def create_crew():
    researcher = create_researcher()
    task = create_research_task(researcher)
    return Crew(agents=[researcher], tasks=[task])

# Test crew creation and execution
my_crew = create_crew()
result = my_crew.kickoff()
```

## ðŸ› Debugging Tips

### General Debugging Steps

1. **Enable verbose logging**:
```python
import logging
logging.basicConfig(level=logging.DEBUG)

from rizk.sdk import Rizk
Rizk.init(app_name="Debug", verbose=True)
```

2. **Check SDK status**:
```python
from rizk.sdk import Rizk
from rizk.sdk.config import get_config

# Check if initialized
try:
    client = Rizk.get()
    print("âœ… SDK initialized")
except Exception as e:
    print(f"âŒ SDK not initialized: {e}")

# Check configuration
config = get_config()
errors = config.validate()
if errors:
    print("Configuration issues:")
    for error in errors:
        print(f"  - {error}")
```

3. **Test individual components**:
```python
# Test framework detection
from rizk.sdk.utils.framework_detection import detect_framework
print(f"Framework: {detect_framework()}")

# Test guardrails
from rizk.sdk.guardrails.engine import GuardrailsEngine
try:
    engine = GuardrailsEngine.get_instance()
    result = engine.evaluate("test message")
    print(f"Guardrails working: {result.allowed}")
except Exception as e:
    print(f"Guardrails error: {e}")

# Test adapters
from rizk.sdk.utils.framework_registry import FrameworkRegistry
adapters = FrameworkRegistry.get_all_framework_names()
print(f"Available adapters: {adapters}")
```

### Common Error Patterns

| Error Pattern | Likely Cause | Solution |
|---|---|---|
| `Module not found: rizk` | Wrong package name | Use `pip install rizk` |
| `API key must start with 'rizk_'` | Wrong API key format | Get key from app.rizk.tools |
| `Framework adapter not found` | Import order issue | Import Rizk before framework |
| `Guardrails engine not initialized` | Initialization failure | Check logs for specific error |
| `No policies found` | Wrong policies path | Verify RIZK_POLICIES_PATH |
| `Traceloop SDK not installed` | Missing dependency | `pip install traceloop-sdk` |

## ðŸ“ž Getting More Help

If you're still experiencing issues:

1. **Check the logs**: Enable debug logging and look for specific error messages
2. **Search documentation**: Use Ctrl+F to search this documentation
3. **GitHub Issues**: Check [GitHub Issues](https://github.com/rizk-ai/rizk-sdk/issues) for similar problems
4. **Community Discord**: Join our [Discord](https://discord.gg/rizk) for community help
5. **Support Email**: Contact [hello@rizk.tools](mailto:hello@rizk.tools) for enterprise support

### Issue Reporting Template

When reporting issues, include:

```
**Environment**:
- OS: Windows/Linux/macOS
- Python version: 3.x
- Rizk SDK version: x.x.x
- Framework: LangChain/CrewAI/etc.

**Issue Description**:
Brief description of the problem

**Code to Reproduce**:
```python
# Minimal code that reproduces the issue
```

**Expected Behavior**:
What you expected to happen

**Actual Behavior**:
What actually happened

**Logs**:
```
Relevant log output with debug enabled
```
```

This template helps us provide faster, more accurate support. 

