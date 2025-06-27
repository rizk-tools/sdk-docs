---
title: "Decorators Overview"
description: "Decorators Overview"
---

# Decorators Overview

Rizk SDK's decorator system provides a unified, framework-agnostic approach to adding observability and governance to your LLM applications. This document explains how the decorator system works, its universal adaptation capabilities, and best practices for use.

## Core Concept

The decorator system automatically adapts to your LLM framework, providing consistent observability and governance regardless of whether you're using OpenAI Agents, LangChain, CrewAI, LlamaIndex, or any other framework.

```python
# Same decorators work across all frameworks
from rizk.sdk.decorators import workflow, task, agent, tool, guardrails

@workflow(name="customer_support")
@guardrails()
def handle_customer_query(query: str) -> str:
    # Framework automatically detected and adapted
    # Observability and governance applied universally
    pass
```

## Available Decorators

### Core Decorators

| Decorator | Purpose | Use Case |
|-----------|---------|----------|
| `@workflow` | High-level processes | Complete user workflows, business processes |
| `@task` | Individual operations | Specific steps within workflows |
| `@agent` | Autonomous components | AI agents, decision-making entities |
| `@tool` | Utility functions | Tools used by agents, helper functions |
| `@crew` | CrewAI workflows | Multi-agent crew orchestration |

### Governance Decorators

| Decorator | Purpose | Use Case |
|-----------|---------|----------|
| `@guardrails` | Policy enforcement | Input/output validation, content filtering |
| `@policies` | Custom policies | Specific compliance rules |
| `@mcp_guardrails` | MCP protocol governance | Model Context Protocol compliance |

## Universal Adaptation

### Automatic Framework Detection

The decorators automatically detect your framework and apply appropriate instrumentation:

```python
# OpenAI Agents - Integrates with Agent.run()
from agents import Agent

@agent(name="openai_assistant")
def create_assistant():
    return Agent(name="assistant", instructions="Help users")

# LangChain - Adds callback handlers
from langchain.agents import AgentExecutor

@workflow(name="langchain_process")
def run_agent_executor(executor: AgentExecutor, query: str):
    return executor.run(query)

# CrewAI - Traces crew execution
from crewai import Crew

@crew(name="research_crew")
def create_research_crew():
    return Crew(agents=[...], tasks=[...])
```

## Decorator Parameters

### Common Parameters

All decorators support these common parameters:

```python
@workflow(
    name="my_workflow",           # Span name (defaults to function name)
    version=1,                    # Version for tracking changes
    organization_id="my_org",     # Hierarchical context
    project_id="my_project",      # Project identification
    **kwargs                      # Framework-specific parameters
)
def my_function():
    pass
```

### Hierarchical Context

Decorators support hierarchical context for enterprise organization:

```python
@workflow(
    name="customer_onboarding",
    organization_id="acme_corp",
    project_id="customer_portal",
    version=2
)
@guardrails()
def onboard_customer(customer_data: dict) -> dict:
    """Complete customer onboarding workflow."""
    
    @task(
        name="validate_data",
        organization_id="acme_corp",     # Inherited context
        project_id="customer_portal",
        task_id="validation"
    )
    def validate_customer_data(data: dict) -> bool:
        # Validation logic
        return True
    
    @agent(
        name="onboarding_agent",
        organization_id="acme_corp",
        project_id="customer_portal", 
        agent_id="onboarding_bot"
    )
    def create_onboarding_agent():
        # Agent creation logic
        pass
    
    # Workflow implementation
    if validate_customer_data(customer_data):
        agent = create_onboarding_agent()
        return {"status": "onboarded"}
    return {"status": "failed"}
```

## Framework-Specific Examples

### OpenAI Agents Integration

```python
from agents import Agent, Task, Workflow
from rizk.sdk.decorators import workflow, agent, tool

@tool(name="web_search", organization_id="demo", project_id="agents")
def web_search(query: str) -> str:
    """Search the web for information."""
    # Implementation
    return f"Search results for: {query}"

@agent(name="research_agent", organization_id="demo", project_id="agents")
def create_research_agent():
    """Create a research agent with tools."""
    return Agent(
        name="Researcher",
        instructions="You are a research assistant",
        functions=[web_search]  # Automatically wrapped with monitoring
    )

@workflow(name="research_workflow", organization_id="demo", project_id="agents")
def run_research_workflow(topic: str):
    """Run a complete research workflow."""
    agent = create_research_agent()
    task = Task(description=f"Research {topic}")
    return agent.run(task)
```

### LangChain Integration

```python
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from rizk.sdk.decorators import workflow, tool, agent

@tool(name="calculator", organization_id="demo", project_id="langchain")
def calculator(expression: str) -> str:
    """Calculate mathematical expressions."""
    try:
        result = eval(expression)  # Note: Use safely in production
        return str(result)
    except Exception as e:
        return f"Error: {str(e)}"

@agent(name="math_agent", organization_id="demo", project_id="langchain")
def create_math_agent():
    """Create a mathematical reasoning agent."""
    llm = ChatOpenAI(temperature=0)
    tools = [calculator]
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful math assistant."),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])
    
    agent = create_openai_tools_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools)

@workflow(name="math_workflow", organization_id="demo", project_id="langchain")
@guardrails()
def solve_math_problem(problem: str) -> str:
    """Solve a mathematical problem with monitoring."""
    agent_executor = create_math_agent()
    result = agent_executor.invoke({"input": problem})
    return result["output"]
```

### CrewAI Integration

```python
from crewai import Agent, Task, Crew, Process
from rizk.sdk.decorators import crew, agent, task

@agent(name="writer_agent", organization_id="demo", project_id="crewai")
def create_writer_agent():
    """Create a content writing agent."""
    return Agent(
        role="Technical Writer",
        goal="Create clear and informative content",
        backstory="Expert technical writer with AI knowledge",
        verbose=True
    )

@agent(name="reviewer_agent", organization_id="demo", project_id="crewai")
def create_reviewer_agent():
    """Create a content review agent."""
    return Agent(
        role="Content Reviewer", 
        goal="Review and improve content quality",
        backstory="Experienced editor and content strategist",
        verbose=True
    )

@task(name="writing_task", organization_id="demo", project_id="crewai")
def create_writing_task(writer: Agent, topic: str):
    """Create a writing task."""
    return Task(
        description=f"Write a comprehensive article about {topic}",
        agent=writer,
        expected_output="Well-structured article with clear explanations"
    )

@task(name="review_task", organization_id="demo", project_id="crewai")
def create_review_task(reviewer: Agent, writer: Agent):
    """Create a review task."""
    return Task(
        description="Review and improve the written content",
        agent=reviewer,
        expected_output="Improved content with suggestions",
        context=[create_writing_task(writer, "AI")]  # Dependency
    )

@crew(name="content_crew", organization_id="demo", project_id="crewai")
@guardrails()
def create_content_crew(topic: str):
    """Create and run a content creation crew."""
    writer = create_writer_agent()
    reviewer = create_reviewer_agent()
    
    writing_task = create_writing_task(writer, topic)
    review_task = create_review_task(reviewer, writer)
    
    crew = Crew(
        agents=[writer, reviewer],
        tasks=[writing_task, review_task],
        process=Process.sequential,
        verbose=True
    )
    
    return crew.kickoff()
```

## Guardrails Integration

### Basic Guardrails

```python
from rizk.sdk.decorators import workflow, guardrails

@workflow(name="content_generation")
@guardrails()
def generate_content(user_prompt: str) -> str:
    """Generate content with automatic policy enforcement."""
    # Input automatically validated against policies
    # Output automatically checked for compliance
    return f"Generated content based on: {user_prompt}"
```

### Custom Policies

```python
from rizk.sdk.decorators import workflow, policies

@workflow(name="financial_analysis")
@policies(["financial_compliance", "data_privacy", "risk_management"])
def analyze_financial_data(data: dict) -> dict:
    """Analyze financial data with specific compliance policies."""
    # Multiple policy layers applied
    return {"analysis": "financial insights"}
```

### MCP Guardrails

```python
from rizk.sdk.decorators import workflow, mcp_guardrails
from rizk.sdk.decorators.mcp_guardrails import ViolationMode

@workflow(name="mcp_compliant_process")
@mcp_guardrails(
    violation_mode=ViolationMode.BLOCK,
    custom_policies=["mcp_compliance"]
)
def mcp_compliant_function(mcp_request: dict) -> dict:
    """Process MCP requests with protocol compliance."""
    # MCP protocol validation applied
    return {"mcp_response": "processed"}
```

## Advanced Patterns

### Conditional Decoration

```python
import os
from rizk.sdk.decorators import workflow, guardrails

def conditional_guardrails(func):
    """Apply guardrails only in production."""
    if os.getenv("ENVIRONMENT") == "production":
        return guardrails()(func)
    return func

@workflow(name="environment_aware")
@conditional_guardrails
def environment_aware_function():
    """Function with conditional governance."""
    pass
```

### Dynamic Parameters

```python
from rizk.sdk.decorators import workflow

def create_workflow_decorator(env: str):
    """Create environment-specific workflow decorator."""
    return workflow(
        name=f"process_{env}",
        organization_id=f"org_{env}",
        project_id=f"project_{env}"
    )

@create_workflow_decorator("production")
@guardrails()
def production_process():
    """Production-specific process."""
    pass
```

### Class-Based Decoration

```python
from rizk.sdk.decorators import agent, tool

@agent(name="service_agent", organization_id="services")
class CustomerServiceAgent:
    """Customer service agent class."""
    
    def __init__(self):
        self.knowledge_base = {}
    
    @tool(name="lookup_customer", agent_id="customer_service")
    def lookup_customer(self, customer_id: str) -> dict:
        """Look up customer information."""
        return self.knowledge_base.get(customer_id, {})
    
    @workflow(name="handle_inquiry")
    def handle_inquiry(self, inquiry: str) -> str:
        """Handle customer inquiry."""
        # Implementation
        return "Response to inquiry"
```

## Error Handling and Debugging

### Decorator Errors

```python
import logging
from rizk.sdk.decorators import workflow

# Enable debug logging
logging.getLogger("rizk.decorators").setLevel(logging.DEBUG)

@workflow(name="debug_process")
def debug_function():
    """Function with debug logging enabled."""
    # Detailed decorator logs will be shown
    pass
```

### Fallback Behavior

```python
# Decorators gracefully degrade if dependencies are missing
@workflow(name="resilient_process")
@guardrails()  # Will work even if guardrails engine fails
def resilient_function():
    """Function that works even with decorator failures."""
    # Core functionality preserved even if monitoring fails
    return "success"
```

### Testing Decorated Functions

```python
import pytest
from rizk.sdk.decorators import workflow, guardrails

@workflow(name="test_workflow")
@guardrails()
def function_to_test(input_data: str) -> str:
    return f"processed: {input_data}"

def test_decorated_function():
    """Test decorated function behavior."""
    # Decorators work transparently in tests
    result = function_to_test("test input")
    assert result == "processed: test input"
    
def test_without_decorators():
    """Test core function without decorators."""
    # Access original function if needed
    original_func = function_to_test.__wrapped__
    result = original_func("test input")
    assert result == "processed: test input"
```

## Performance Considerations

### Decorator Overhead

```python
# Minimal overhead for decorated functions:
# - Framework detection: ~0.1-1ms (cached)
# - Tracing setup: ~0.5-2ms
# - Guardrails check: ~1-10ms (depending on complexity)
# - Total overhead: ~2-13ms per call
```

### Optimization Tips

```python
# 1. Use cached detection for high-frequency functions
from rizk.sdk.utils.framework_detection import detect_framework_cached

@workflow(name="high_frequency_process")
def optimized_function():
    # Framework detection is cached automatically
    pass

# 2. Disable tracing for performance-critical paths
@workflow(name="critical_process", enabled=False)
def performance_critical_function():
    # Monitoring disabled for maximum performance
    pass

# 3. Use lightweight guardrails for high-throughput
@workflow(name="high_throughput")
@guardrails(fast_rules_only=True)  # Skip LLM-based validation
def high_throughput_function():
    pass
```

## Configuration

### Global Configuration

```python
import os
from rizk.sdk import Rizk

# Configure decorator behavior globally
Rizk.init(
    app_name="MyApp",
    enabled=True,                    # Enable/disable all decorators
    trace_content=False,             # Disable content tracing for privacy
    policies_path="./policies",      # Custom policies directory
    verbose=True                     # Enable detailed logging
)
```

### Environment-Specific Settings

```python
# Development environment
if os.getenv("ENVIRONMENT") == "development":
    Rizk.init(
        app_name="MyApp-Dev",
        enabled=True,
        verbose=True,
        policies_path="./dev_policies"
    )

# Production environment  
elif os.getenv("ENVIRONMENT") == "production":
    Rizk.init(
        app_name="MyApp-Prod",
        enabled=True,
        trace_content=False,
        telemetry_enabled=False,
        policies_path="/app/policies"
    )
```

## Best Practices

### 1. Consistent Naming

```python
# âœ… Good - Consistent, descriptive names
@workflow(name="customer_onboarding_v2")
@task(name="validate_customer_data")
@agent(name="onboarding_assistant")

# âŒ Avoid - Inconsistent or unclear names
@workflow(name="proc1")
@task(name="check_stuff")
@agent(name="bot")
```

### 2. Hierarchical Organization

```python
# âœ… Good - Clear hierarchy
@workflow(
    name="order_processing",
    organization_id="ecommerce_platform",
    project_id="order_management"
)
@task(
    name="payment_validation", 
    organization_id="ecommerce_platform",
    project_id="order_management",
    task_id="payment_check"
)

# âŒ Avoid - Flat structure without context
@workflow(name="order_processing")
@task(name="payment_validation")
```

### 3. Appropriate Granularity

```python
# âœ… Good - Appropriate level of detail
@workflow(name="user_registration")      # High-level process
def register_user():
    
    @task(name="validate_email")         # Specific operation
    def validate_email():
        pass
    
    @task(name="create_account")         # Specific operation
    def create_account():
        pass

# âŒ Avoid - Over-decoration
@workflow(name="user_registration")
def register_user():
    
    @task(name="check_email_format")     # Too granular
    @task(name="check_email_domain")     # Too granular
    @task(name="check_email_exists")     # Too granular
    def validate_email():
        pass
```

### 4. Error Recovery

```python
# âœ… Good - Graceful error handling
@workflow(name="resilient_process")
@guardrails()
def resilient_function():
    try:
        # Main logic
        return process_data()
    except Exception as e:
        # Fallback behavior
        logger.error(f"Process failed: {e}")
        return default_response()
```

## Summary

Rizk SDK's decorator system provides:

âœ… **Universal Framework Support** - Works with any LLM framework  
âœ… **Automatic Adaptation** - No manual configuration required  
âœ… **Comprehensive Monitoring** - Full observability out of the box  
âœ… **Policy Enforcement** - Automated governance and compliance  
âœ… **Hierarchical Context** - Enterprise-grade organization  
âœ… **Performance Optimized** - Minimal overhead with caching  
âœ… **Developer Friendly** - Simple, intuitive API  

The decorator system is the primary interface for adding Rizk's capabilities to your LLM applications, providing enterprise-grade observability and governance with minimal code changes. 

