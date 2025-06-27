---
title: "API Reference"
description: "API Reference"
---

# API Reference

Complete API reference for the Rizk SDK. This section provides detailed documentation for all classes, methods, types, and utilities available in the SDK.

## Quick Navigation

| Component | Description | Key Features |
|-----------|-------------|--------------|
| **[Rizk Class](./rizk-class.md)** | Main SDK interface | Initialization, configuration, client management |
| **[Decorators](./decorators-api.md)** | Function & class decorators | @workflow, @task, @agent, @tool, @guardrails |
| **[GuardrailsEngine](./guardrails-api.md)** | Policy enforcement engine | Message processing, output checking, policy evaluation |
| **[Configuration](./configuration-api.md)** | Configuration management | Environment variables, validation, global config |
| **[Types](./types.md)** | Type definitions & protocols | Data structures, protocols, enums, exceptions |
| **[Utilities](./utilities.md)** | Helper functions & tools | Framework detection, caching, performance monitoring |

## Core SDK Components

### Rizk Class
The main entry point for the SDK. Handles initialization, configuration, and provides access to core components.

```python
from rizk.sdk import Rizk

# Initialize SDK
rizk = Rizk.init(
    app_name="MyApp",
    api_key="your-api-key"
)

# Access components
guardrails = Rizk.get_guardrails()
client = Rizk.get()
```

**[â†’ Full Rizk Class API Reference](./rizk-class.md)**

---

### Decorators
Universal decorators that automatically adapt to any LLM framework. Add observability, tracing, and governance to your functions.

```python
from rizk.sdk.decorators import workflow, task, agent, tool, guardrails

@workflow(name="chat_workflow", organization_id="my_org")
@guardrails(augment_prompt=True, check_output=True)
def process_chat(user_input: str) -> str:
    # Your workflow logic
    return ai_response(user_input)
```

**[â†’ Full Decorators API Reference](./decorators-api.md)**

---

### GuardrailsEngine
Multi-layer policy enforcement system that processes messages, evaluates policies, and ensures compliance.

```python
from rizk.sdk import Rizk

guardrails = Rizk.get_guardrails()

# Process user input
result = await guardrails.process_message(
    "Can you help me with my account?",
    context={"user_id": "user_123", "organization_id": "acme"}
)

if result['allowed']:
    # Process the approved message
    response = handle_approved_message(message)
else:
    # Handle policy violation
    response = result.get('response', 'Request blocked by policy')
```

**[â†’ Full GuardrailsEngine API Reference](./guardrails-api.md)**

---

### Configuration
Centralized configuration management with environment variable support and validation.

```python
from rizk.sdk.config import RizkConfig, set_config

# Create configuration from environment
config = RizkConfig.from_env()

# Validate configuration
errors = config.validate()
if not errors:
    set_config(config)
```

**[â†’ Full Configuration API Reference](./configuration-api.md)**

## Framework Integration

### Supported Frameworks

The Rizk SDK automatically detects and integrates with multiple LLM frameworks:

| Framework | Package | Automatic Detection | Key Features |
|-----------|---------|---------------------|--------------|
| **OpenAI Agents SDK** | `agents` | âœ… Agent, Runner, function_tool | Native function tools, conversation management |
| **LangChain** | `langchain` | âœ… AgentExecutor, chains, tools | Agent executors, callback handlers, tool chains |
| **CrewAI** | `crewai` | âœ… Agent, Task, Crew | Multi-agent workflows, task management |
| **LlamaIndex** | `llama_index` | âœ… Query engines, chat engines | Document queries, chat interfaces |

### Framework Detection

```python
from rizk.sdk.utils.framework_detection import detect_framework

def create_agent():
    # Framework automatically detected based on return type
    return Agent(name="Assistant", instructions="Be helpful")

framework = detect_framework(create_agent)
print(framework)  # Output: "agents_sdk"
```

**[â†’ Framework Detection Utilities](./utilities.md#framework-detection-utilities)**

## Type System

### Core Types

```python
from rizk.sdk.types import (
    Decision,
    GuardrailProcessingResult,
    GuardrailOutputCheckResult,
    RizkContext,
    Policy,
    ViolationMode
)

# Simple decision
decision = Decision(allowed=True, confidence=0.95)

# Comprehensive result
result = GuardrailProcessingResult(
    allowed=False,
    confidence=0.98,
    decision_layer="fast_rules",
    violated_policies=["profanity_filter"],
    blocked_reason="Inappropriate content detected"
)
```

### Protocols

```python
from rizk.sdk.types import LLMServiceProtocol, FrameworkAdapterProtocol

# Custom LLM service
class MyLLMService:
    async def evaluate_policy(self, message: str, policy: dict) -> dict:
        # Implementation
        return {"allowed": True, "confidence": 0.9}

# Protocol ensures type safety
service: LLMServiceProtocol = MyLLMService()
```

**[â†’ Full Types API Reference](./types.md)**

## Utilities

### Framework Detection
Automatically identify LLM frameworks in use.

### Caching
High-performance caching for framework detection, policy evaluation, and LLM results.

### Performance Monitoring
Built-in performance monitoring and timing utilities.

### Error Handling
Standardized error handling with graceful degradation.

### Context Management
Hierarchical context management for distributed tracing.

```python
from rizk.sdk.utils.context import set_hierarchy_context
from rizk.sdk.utils.performance import Timer, performance_monitor
from rizk.sdk.utils.cache import CacheManager

# Set distributed tracing context
set_hierarchy_context(
    organization_id="my_org",
    project_id="chat_service",
    user_id="user_123"
)

# Monitor performance
@performance_monitor(log_threshold_ms=100)
def expensive_operation():
    with Timer("data_processing"):
        return process_large_dataset()

# Use caching
CacheManager.set("user_profile", user_data, ttl=3600)
cached_profile = CacheManager.get("user_profile")
```

**[â†’ Full Utilities API Reference](./utilities.md)**

## Common Usage Patterns

### Basic SDK Setup

```python
import os
from rizk.sdk import Rizk
from rizk.sdk.decorators import workflow, guardrails

# Initialize SDK
rizk = Rizk.init(
    app_name="MyApplication",
    api_key=os.getenv("RIZK_API_KEY"),
    enabled=True
)

@workflow(name="secure_chat", organization_id="my_org")
@guardrails(augment_prompt=True, check_output=True)
def secure_chat_handler(user_input: str) -> str:
    # Your secure chat logic
    return process_with_llm(user_input)
```

### Advanced Configuration

```python
from rizk.sdk.config import RizkConfig, set_config

# Production configuration
config = RizkConfig.from_env(
    app_name="ProductionApp",
    debug_mode=False,
    trace_content=False,  # Privacy in production
    framework_detection_cache_size=10000
)

# Validate and apply
errors = config.validate()
if not errors:
    set_config(config)
else:
    print(f"Configuration errors: {errors}")
```

### Custom Guardrails Integration

```python
from rizk.sdk import Rizk
from rizk.sdk.types import GuardrailProcessingResult

async def custom_message_handler(message: str, user_id: str) -> str:
    guardrails = Rizk.get_guardrails()
    
    # Process with context
    result: GuardrailProcessingResult = await guardrails.process_message(
        message,
        context={
            "user_id": user_id,
            "organization_id": "my_org",
            "conversation_id": f"conv_{user_id}"
        }
    )
    
    if result['error']:
        # Handle system error
        return "Service temporarily unavailable"
    elif not result['allowed']:
        # Handle policy violation
        return result.get('response', 'Request blocked by policy')
    else:
        # Process approved message
        return await process_approved_message(message)
```

### Framework-Specific Integration

#### OpenAI Agents SDK

```python
from agents import Agent, function_tool
from rizk.sdk.decorators import agent, tool

@tool(name="calculator")
@function_tool()
def calculate(expression: str) -> str:
    return str(eval(expression))

@agent(name="math_assistant")
def create_math_agent():
    return Agent(
        name="MathBot",
        instructions="You are a helpful math assistant",
        tools=[calculate]
    )
```

#### LangChain

```python
from langchain.agents import create_openai_tools_agent, AgentExecutor
from rizk.sdk.decorators import workflow, tool

@tool(name="search_tool")
class SearchTool(BaseTool):
    name = "web_search"
    description = "Search the web"
    
    def _run(self, query: str) -> str:
        return search_web(query)

@workflow(name="langchain_agent")
def create_agent_executor():
    agent = create_openai_tools_agent(llm, [SearchTool()], prompt)
    return AgentExecutor(agent=agent, tools=[SearchTool()])
```

#### CrewAI

```python
from crewai import Agent, Task, Crew, Process
from rizk.sdk.decorators import crew, task

@task(name="research_task")
def create_research_task():
    return Task(
        description="Research the given topic thoroughly",
        expected_output="Comprehensive research report"
    )

@crew(name="research_crew")
def create_research_crew():
    researcher = Agent(role="Researcher", goal="Find information")
    writer = Agent(role="Writer", goal="Write reports")
    
    return Crew(
        agents=[researcher, writer],
        tasks=[create_research_task()],
        process=Process.sequential
    )
```

## Error Handling

### Exception Types

```python
from rizk.sdk.types import (
    RizkSDKError,
    PolicyViolationError,
    ConfigurationError
)

try:
    result = await guardrails.process_message(message)
except PolicyViolationError as e:
    print(f"Policy violation: {e.policy_id}")
except ConfigurationError as e:
    print(f"Configuration error: {e.config_field}")
except RizkSDKError as e:
    print(f"SDK error: {e.error_code}")
```

### Graceful Error Handling

```python
from rizk.sdk.utils.error_handling import handle_errors

@handle_errors(fail_closed=False, default_return_on_error={"allowed": True})
async def safe_guardrails_check(message: str):
    # If guardrails fail, default to allowing the message
    return await guardrails.process_message(message)
```

## Performance Best Practices

### Caching Strategy

```python
from rizk.sdk.utils.cache import CacheManager

# Cache expensive operations
def cached_framework_detection(func):
    cache_key = f"framework_{func.__name__}"
    
    cached_result = CacheManager.get(cache_key, "framework_detection")
    if cached_result:
        return cached_result
    
    result = detect_framework(func)
    CacheManager.set(cache_key, result, ttl=3600, cache_type="framework_detection")
    return result
```

### Async Processing

```python
import asyncio

async def batch_process_messages(messages: List[str]) -> List[dict]:
    guardrails = Rizk.get_guardrails()
    
    # Process messages in parallel
    tasks = [
        guardrails.process_message(msg, context)
        for msg in messages
    ]
    
    results = await asyncio.gather(*tasks, return_exceptions=True)
    return [r if not isinstance(r, Exception) else {"error": str(r)} for r in results]
```

### Performance Monitoring

```python
from rizk.sdk.utils.performance import performance_monitor, Timer

@performance_monitor(log_threshold_ms=200)
async def monitored_workflow(data):
    with Timer("preprocessing") as prep:
        processed = preprocess(data)
    
    with Timer("main_processing") as main:
        result = await main_processing(processed)
    
    logger.info(f"Timing - prep: {prep.elapsed_ms}ms, main: {main.elapsed_ms}ms")
    return result
```

## Environment Configuration

### Development Environment

```bash
# .env.development
export RIZK_API_KEY="rizk_dev_your_key"
export RIZK_DEBUG="true"
export RIZK_VERBOSE="true"
export RIZK_TRACE_CONTENT="true"
export RIZK_POLICIES_PATH="./dev-policies"
```

### Production Environment

```bash
# .env.production
export RIZK_API_KEY="rizk_prod_your_key"
export RIZK_DEBUG="false"
export RIZK_VERBOSE="false"
export RIZK_TRACE_CONTENT="false"
export RIZK_TELEMETRY="false"
export RIZK_FRAMEWORK_CACHE_SIZE="10000"
export RIZK_POLICIES_PATH="/app/policies"
```

**[â†’ Complete Environment Variables Reference](./configuration-api.md#environment-variable-reference)**

## Related Documentation

- **[Getting Started Guide](../getting-started/overview/)** - SDK introduction and basic setup
- **[Framework Integration](../02-framework-integration/overview.md)** - Framework-specific integration guides  
- **[Guardrails Documentation](../guardrails/overview.md)** - Policy enforcement and configuration
- **[Advanced Configuration](../advanced-config/production-setup.md)** - Production deployment and scaling
- **[Observability](../observability/tracing.md)** - Monitoring and analytics

---

**Need help?** Check our [GitHub repository](https://github.com/rizk-sdk) or visit [app.rizk.tools](https://app.rizk.tools) for support. 

