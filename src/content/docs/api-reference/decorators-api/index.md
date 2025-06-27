---
title: "Decorators API Reference"
description: "Decorators API Reference"
---

# Decorators API Reference

Rizk SDK provides a unified set of decorators that automatically adapt to any LLM framework. These decorators add observability, tracing, and governance to your functions and classes.

## Overview

```python
from rizk.sdk.decorators import workflow, task, agent, tool, crew, guardrails, mcp_guardrails, add_policies

# Universal decorators that work with any framework
@workflow(name="chat_workflow", organization_id="my_org", project_id="chatbot")
@guardrails()
def run_chat_workflow(user_input: str) -> str:
    # Your workflow logic here
    pass
```

## Core Decorators

### `@workflow`

**Decorator for high-level processes and workflows.**

```python
def workflow(
    name: Optional[str] = None,
    version: Optional[int] = None,
    organization_id: Optional[str] = None,
    project_id: Optional[str] = None,
    **kwargs: Any
) -> Callable[[Union[F, C]], Union[F, C]]
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | `Optional[str]` | `None` | Name of the workflow. Defaults to function/class name |
| `version` | `Optional[int]` | `None` | Version number of the workflow |
| `organization_id` | `Optional[str]` | `None` | Organization identifier for context |
| `project_id` | `Optional[str]` | `None` | Project identifier for context |
| `**kwargs` | `Any` | - | Additional framework-specific arguments |

#### Returns

- `Callable`: The decorated function or class with workflow tracing

#### Examples

**Basic workflow:**
```python
@workflow(name="user_onboarding", organization_id="acme", project_id="crm")
def onboard_user(user_data: dict) -> dict:
    # Process user onboarding
    return {"status": "success", "user_id": user_data["id"]}
```

**Class-based workflow:**
```python
@workflow(name="data_pipeline", version=2)
class DataProcessingWorkflow:
    def __init__(self, config: dict):
        self.config = config
    
    def execute(self, data: list) -> list:
        # Process data pipeline
        return processed_data
```

**Framework-specific (LangChain):**
```python
@workflow(name="langchain_workflow")
def create_langchain_agent():
    # Returns LangChain AgentExecutor - automatically detected
    return AgentExecutor(agent=agent, tools=tools)
```

---

### `@task`

**Decorator for individual operations and tasks.**

```python
def task(
    name: Optional[str] = None,
    version: Optional[int] = None,
    organization_id: Optional[str] = None,
    project_id: Optional[str] = None,
    **kwargs: Any
) -> Callable[[Union[F, C]], Union[F, C]]
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | `Optional[str]` | `None` | Name of the task. Defaults to function/class name |
| `version` | `Optional[int]` | `None` | Version number of the task |
| `organization_id` | `Optional[str]` | `None` | Organization identifier for context |
| `project_id` | `Optional[str]` | `None` | Project identifier for context |
| `**kwargs` | `Any` | - | Additional framework-specific arguments |

#### Returns

- `Callable`: The decorated function or class with task tracing

#### Examples

**Data processing task:**
```python
@task(name="extract_entities", organization_id="nlp_org", project_id="text_analysis")
def extract_entities(text: str) -> list:
    # Extract named entities from text
    return entities
```

**Async task:**
```python
@task(name="fetch_user_data")
async def fetch_user_data(user_id: str) -> dict:
    # Async data fetching
    async with httpx.AsyncClient() as client:
        response = await client.get(f"/users/{user_id}")
        return response.json()
```

**CrewAI task:**
```python
@task(name="research_task")
def create_research_task(agent):
    # Returns CrewAI Task - automatically detected
    return Task(
        description="Research the given topic",
        agent=agent,
        expected_output="Research report"
    )
```

---

### `@agent`

**Decorator for autonomous components and agents.**

```python
def agent(
    name: Optional[str] = None,
    version: Optional[int] = None,
    organization_id: Optional[str] = None,
    project_id: Optional[str] = None,
    agent_id: Optional[str] = None,
    **kwargs: Any
) -> Callable[[Union[F, C]], Union[F, C]]
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | `Optional[str]` | `None` | Name of the agent. Defaults to function/class name |
| `version` | `Optional[int]` | `None` | Version number of the agent |
| `organization_id` | `Optional[str]` | `None` | Organization identifier for context |
| `project_id` | `Optional[str]` | `None` | Project identifier for context |
| `agent_id` | `Optional[str]` | `None` | Specific agent identifier |
| `**kwargs` | `Any` | - | Additional framework-specific arguments |

#### Returns

- `Callable`: The decorated function or class with agent tracing

#### Examples

**OpenAI Agents SDK:**
```python
@agent(name="customer_service", agent_id="cs_001")
def create_customer_service_agent():
    # Returns OpenAI Agent - automatically detected
    return Agent(
        name="CustomerService",
        instructions="You are a helpful customer service assistant",
        tools=[get_order_status, process_refund]
    )
```

**LangChain agent:**
```python
@agent(name="research_agent", organization_id="research_org")
def create_research_agent():
    # Returns LangChain agent - automatically detected
    return create_openai_tools_agent(llm, tools, prompt)
```

**Custom agent class:**
```python
@agent(name="custom_agent")
class CustomAgent:
    def __init__(self, model: str):
        self.model = model
    
    async def process(self, input_text: str) -> str:
        # Custom agent processing logic
        return response
```

---

### `@tool`

**Decorator for utility functions and tools.**

```python
def tool(
    name: Optional[str] = None,
    version: Optional[int] = None,
    organization_id: Optional[str] = None,
    project_id: Optional[str] = None,
    agent_id: Optional[str] = None,
    tool_id: Optional[str] = None,
    **kwargs: Any
) -> Callable[[Union[F, C]], Union[F, C]]
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | `Optional[str]` | `None` | Name of the tool. Defaults to function/class name |
| `version` | `Optional[int]` | `None` | Version number of the tool |
| `organization_id` | `Optional[str]` | `None` | Organization identifier for context |
| `project_id` | `Optional[str]` | `None` | Project identifier for context |
| `agent_id` | `Optional[str]` | `None` | ID of the agent using the tool |
| `tool_id` | `Optional[str]` | `None` | Specific tool identifier |
| `**kwargs` | `Any` | - | Additional framework-specific arguments |

#### Returns

- `Callable`: The decorated function or class with tool tracing

#### Examples

**Simple tool function:**
```python
@tool(name="calculator", tool_id="calc_001")
def calculate(expression: str) -> str:
    """Calculate a mathematical expression."""
    try:
        result = eval(expression)
        return f"Result: {result}"
    except Exception as e:
        return f"Error: {str(e)}"
```

**OpenAI function tool:**
```python
from agents import function_tool

@tool(name="weather_tool")
@function_tool()  # OpenAI Agents SDK decorator
def get_weather(location: str) -> str:
    """Get weather information for a location."""
    # Weather API logic
    return f"Weather in {location}: Sunny, 72Â°F"
```

**LangChain tool:**
```python
from langchain.tools import BaseTool

@tool(name="search_tool")
class SearchTool(BaseTool):
    name = "web_search"
    description = "Search the web for information"
    
    def _run(self, query: str) -> str:
        # Search implementation
        return search_results
```

---

### `@crew`

**Decorator for CrewAI-specific workflows.**

```python
def crew(
    name: Optional[str] = None,
    version: Optional[int] = None,
    organization_id: Optional[str] = None,
    project_id: Optional[str] = None,
    **kwargs: Any
) -> Callable[[Union[F, C]], Union[F, C]]
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | `Optional[str]` | `None` | Name of the crew. Defaults to function/class name |
| `version` | `Optional[int]` | `None` | Version number of the crew |
| `organization_id` | `Optional[str]` | `None` | Organization identifier for context |
| `project_id` | `Optional[str]` | `None` | Project identifier for context |
| `**kwargs` | `Any` | - | Additional CrewAI-specific arguments |

#### Returns

- `Callable`: The decorated function or class with crew tracing

#### Example

```python
from crewai import Agent, Task, Crew, Process

@crew(name="content_creation_crew", organization_id="media_co")
def create_content_crew():
    # Create agents
    writer = Agent(
        role="Content Writer",
        goal="Write engaging content",
        backstory="Expert content creator"
    )
    
    editor = Agent(
        role="Editor", 
        goal="Edit and improve content",
        backstory="Professional editor"
    )
    
    # Create tasks
    write_task = Task(
        description="Write a blog post about AI",
        agent=writer
    )
    
    edit_task = Task(
        description="Edit the blog post",
        agent=editor
    )
    
    # Create and return crew
    return Crew(
        agents=[writer, editor],
        tasks=[write_task, edit_task],
        process=Process.sequential
    )
```

## Governance Decorators

### `@guardrails`

**Decorator for applying policy enforcement and safety guardrails.**

```python
def guardrails(
    augment_prompt: bool = True,
    check_output: bool = False,
    on_input_violation: str = "alternative",
    on_output_violation: str = "alternative",
    violation_response: str = "I cannot process that request due to policy restrictions.",
    output_violation_response: str = "I cannot provide that information due to policy restrictions."
) -> Callable[[F], F]
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `augment_prompt` | `bool` | `True` | Whether to augment system prompts with policy guidelines |
| `check_output` | `bool` | `False` | Whether to check function output against policies |
| `on_input_violation` | `str` | `"alternative"` | Action on input violation: `"exception"` or `"alternative"` |
| `on_output_violation` | `str` | `"alternative"` | Action on output violation: `"exception"`, `"alternative"`, or `"redact"` |
| `violation_response` | `str` | `"I cannot process..."` | Alternative response for input violations |
| `output_violation_response` | `str` | `"I cannot provide..."` | Alternative response for output violations |

#### Returns

- `Callable[[F], F]`: Function wrapper with guardrails applied

#### Examples

**Basic guardrails:**
```python
@workflow(name="chat_workflow")
@guardrails()
def process_chat(user_input: str) -> str:
    # Function with automatic policy enforcement
    return llm_response(user_input)
```

**Custom violation handling:**
```python
@guardrails(
    augment_prompt=True,
    check_output=True,
    on_input_violation="exception",
    violation_response="Request blocked by content policy"
)
def sensitive_operation(input_data: str) -> str:
    # Sensitive operation with strict policy enforcement
    return process_sensitive_data(input_data)
```

**Output checking:**
```python
@guardrails(
    check_output=True,
    on_output_violation="redact"
)
def generate_response(prompt: str) -> str:
    # Function that checks and potentially redacts output
    return llm.generate(prompt)
```

---

### `@add_policies`

**Decorator for applying custom policy sets.**

```python
def add_policies(
    augment_prompt: bool = False,
    check_output: bool = False,
    on_input_violation: str = "alternative",
    on_output_violation: str = "alternative",
    violation_response: str = "I cannot process that request due to policy restrictions.",
    output_violation_response: str = "I cannot provide that information due to policy restrictions."
) -> Callable[[F], F]
```

#### Parameters

Same as `@guardrails` but with `augment_prompt` defaulting to `False`.

#### Example

```python
@workflow(name="policy_workflow")
@add_policies(
    augment_prompt=False,
    check_output=True,
    on_output_violation="alternative"
)
def custom_policy_workflow(input_text: str) -> str:
    # Workflow with custom policy application
    return process_with_custom_policies(input_text)
```

---

### `@mcp_guardrails`

**Decorator for MCP (Model Context Protocol) guardrails.**

```python
def mcp_guardrails(
    policy_set: Optional[PolicySet] = None,
    on_violation: str = ViolationMode.AUGMENT,
    conversation_id: Optional[str] = None,
    organization_id: Optional[str] = None,
    project_id: Optional[str] = None,
    enabled: bool = True
) -> Callable[[F], F]
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `policy_set` | `Optional[PolicySet]` | `None` | Custom policy set for evaluation |
| `on_violation` | `str` | `ViolationMode.AUGMENT` | Violation handling mode: `"block"`, `"augment"`, or `"warn"` |
| `conversation_id` | `Optional[str]` | `None` | Conversation identifier for context |
| `organization_id` | `Optional[str]` | `None` | Organization identifier |
| `project_id` | `Optional[str]` | `None` | Project identifier |
| `enabled` | `bool` | `True` | Whether guardrails are enabled |

#### Example

```python
from rizk.sdk.decorators import mcp_guardrails, ViolationMode

@mcp_guardrails(on_violation=ViolationMode.BLOCK)
def get_user_data(user_id: str) -> str:
    """Function with MCP guardrail protection."""
    return f"Sensitive user data for {user_id}"

@mcp_guardrails(on_violation=ViolationMode.AUGMENT)
async def process_message(message: str) -> str:
    """Async function with augmentation on violations."""
    return await llm_process(message)
```

## Decorator Composition

### Combining Decorators

Decorators can be combined for comprehensive observability and governance:

```python
@workflow(name="secure_chat", organization_id="security_org", project_id="chat_app")
@guardrails(augment_prompt=True, check_output=True)
@add_policies(on_output_violation="redact")
def secure_chat_workflow(user_input: str, user_id: str) -> str:
    """Secure chat workflow with full observability and governance."""
    
    # Set additional context
    Rizk.set_association_properties({
        "user_id": user_id,
        "conversation_id": f"conv_{user_id}_{int(time.time())}"
    })
    
    # Process with full monitoring and policy enforcement
    return process_chat_securely(user_input)
```

### Order of Decorators

The recommended order for decorator application:

1. **Framework decorators** (`@workflow`, `@task`, `@agent`, `@tool`, `@crew`) - outermost
2. **Governance decorators** (`@guardrails`, `@add_policies`) - middle
3. **Framework-specific decorators** (e.g., `@function_tool`) - innermost

```python
@workflow(name="example")           # 1. Framework decorator (outermost)
@guardrails()                       # 2. Governance decorator
@function_tool()                    # 3. Framework-specific (innermost)
def example_function():
    pass
```

## Framework Detection

All decorators automatically detect the framework being used:

### Supported Frameworks

- **OpenAI Agents SDK**: Automatic detection of `Agent`, `Runner`, `function_tool`
- **LangChain**: Automatic detection of agents, chains, and tools
- **CrewAI**: Automatic detection of `Agent`, `Task`, `Crew`
- **LlamaIndex**: Automatic detection of query engines and chat engines
- **Custom Frameworks**: Plugin-based detection for custom implementations

### Detection Process

1. **Return Type Analysis**: Examines function return types
2. **Decorator Detection**: Identifies framework-specific decorators
3. **Import Detection**: Analyzes imported modules
4. **Adapter Selection**: Chooses appropriate adapter for the detected framework
5. **Fallback**: Uses standard tracing if no framework detected

## Error Handling

All decorators include robust error handling:

```python
@workflow(name="resilient_workflow")
@guardrails()
def resilient_function(input_data: str) -> str:
    # If guardrails fail, function still executes
    # If tracing fails, function still executes
    # Errors are logged but don't break the application
    return process_data(input_data)
```

### Error Recovery

- **Graceful Degradation**: Functions execute even if decorators fail
- **Logging**: Errors are logged for debugging
- **Fallback Tracing**: Standard tracing if framework detection fails
- **Policy Fallback**: Safe defaults if policy evaluation fails

## Performance Considerations

### Async Support

All decorators support both synchronous and asynchronous functions:

```python
@workflow(name="async_workflow")
@guardrails()
async def async_workflow(data: str) -> str:
    # Async function with full decorator support
    result = await async_llm_call(data)
    return result
```

### Caching

Framework detection is cached for performance:

- **Detection Cache**: Framework detection results are cached per function
- **Policy Cache**: Policy evaluation results are cached when appropriate
- **Trace Cache**: Span creation is optimized to reduce overhead

### Lazy Loading

Components are loaded lazily to minimize startup time:

- **Framework Adapters**: Loaded only when needed
- **Guardrails Engine**: Initialized on first use
- **Policy Files**: Loaded when first accessed

## Best Practices

### 1. Use Descriptive Names

```python
# Good - descriptive names
@workflow(name="user_onboarding_workflow", organization_id="hr_dept")
@task(name="validate_user_email")
@tool(name="email_validator")

# Avoid - generic names
@workflow(name="workflow1")
@task(name="task")
@tool(name="tool")
```

### 2. Consistent Context

Use consistent organization and project IDs across related components:

```python
ORG_ID = "my_company"
PROJECT_ID = "customer_service"

@workflow(name="handle_ticket", organization_id=ORG_ID, project_id=PROJECT_ID)
@task(name="categorize_ticket", organization_id=ORG_ID, project_id=PROJECT_ID)
@tool(name="ticket_classifier", organization_id=ORG_ID, project_id=PROJECT_ID)
```

### 3. Appropriate Guardrails

Apply guardrails based on sensitivity:

```python
# High-sensitivity function
@guardrails(
    augment_prompt=True,
    check_output=True,
    on_input_violation="exception",
    on_output_violation="redact"
)
def process_pii(data: str) -> str:
    pass

# Low-sensitivity function
@guardrails(augment_prompt=True, check_output=False)
def general_chat(message: str) -> str:
    pass
```

### 4. Version Your Functions

Use version numbers for tracking changes:

```python
@workflow(name="data_pipeline", version=2)
def data_pipeline_v2(data: list) -> list:
    # Updated pipeline logic
    pass
```

## Related APIs

- **[Rizk Class API](./rizk-class.md)** - SDK initialization and configuration
- **[GuardrailsEngine API](./guardrails-api.md)** - Policy enforcement engine
- **[Framework Adapters API](./adapters-api.md)** - Framework-specific adapters
- **[Configuration API](./configuration-api.md)** - Configuration management 

