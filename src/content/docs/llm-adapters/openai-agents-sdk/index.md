---
title: "OpenAI Agents SDK Adapter"
description: "OpenAI Agents SDK Adapter"
---

# OpenAI Agents SDK Adapter

The OpenAI Agents SDK Adapter provides specialized integration with OpenAI's Agents SDK, enabling seamless governance and monitoring for agent-based applications. This adapter works as a guardrail function within the Agents SDK framework.

## Overview

The Agents SDK Adapter:

- **Integrates as Guardrail Function**: Works within the Agents SDK's guardrail system
- **Evaluates Agent Inputs**: Applies policies to user inputs before processing
- **Augments Agent Instructions**: Injects policy guidelines into agent instructions
- **Provides Observability**: Creates detailed spans for agent interactions
- **Supports Agent Context**: Converts Agents SDK context to Rizk context

## Installation

```bash
pip install rizk[agents]
# or
pip install rizk agents-sdk
```

## How It Works

The adapter integrates with the OpenAI Agents SDK guardrail system:

```python
from agents import Agent, InputGuardrail
from rizk.sdk.adapters.agents_adapter import add_rizk_guardrails

# Create your agent
agent = Agent(
    name="CustomerService",
    instructions="You are a helpful customer service assistant.",
    model="gpt-4"
)

# Add Rizk guardrails
guardrail_name = add_rizk_guardrails(
    agent=agent,
    organization_id="my_org",
    project_id="customer_service",
    agent_id="cs_agent_1"
)

print(f"Added guardrail: {guardrail_name}")
```

## Basic Usage

### Simple Agent with Guardrails

```python
from agents import Agent, Runner
from rizk.sdk import Rizk
from rizk.sdk.adapters.agents_adapter import add_rizk_guardrails

# Initialize Rizk
rizk = Rizk.init(
    app_name="AgentsDemo",
    api_key="your-rizk-api-key",
    enabled=True
)

# Create agent
agent = Agent(
    name="MathTutor",
    instructions="You are a helpful math tutor. Explain concepts clearly and provide step-by-step solutions.",
    model="gpt-3.5-turbo"
)

# Add Rizk guardrails
add_rizk_guardrails(
    agent=agent,
    organization_id="education",
    project_id="math_tutoring"
)

# Run the agent
runner = Runner(agent=agent)
result = runner.run("Help me solve 2x + 5 = 13")
print(result)
```

### With Custom Policies

```python
from agents import Agent
from rizk.sdk import Rizk
from rizk.sdk.adapters.agents_adapter import add_rizk_guardrails
from rizk.sdk.guardrails.types import PolicySet, Policy

# Create custom policy for educational content
education_policy = Policy(
    id="education_policy",
    name="Educational Content Policy",
    description="Ensures educational content is appropriate and helpful",
    action="allow",
    guidelines=[
        "Always provide step-by-step explanations",
        "Use age-appropriate language and examples",
        "Encourage learning and critical thinking",
        "Avoid providing direct answers without explanation"
    ]
)

policy_set = PolicySet(policies=[education_policy])

# Initialize Rizk with custom policies
rizk = Rizk.init(
    app_name="EducationApp",
    enabled=True
)

# Create agent with custom policies
agent = Agent(
    name="EducationAssistant",
    instructions="You are an educational assistant focused on helping students learn.",
    model="gpt-4"
)

# Add guardrails with custom policies
add_rizk_guardrails(
    agent=agent,
    policies=policy_set,
    organization_id="school_district",
    project_id="ai_tutoring"
)
```

## Advanced Integration

### Multiple Agents with Different Policies

```python
from agents import Agent
from rizk.sdk import Rizk
from rizk.sdk.adapters.agents_adapter import add_rizk_guardrails

rizk = Rizk.init(app_name="MultiAgentSystem", enabled=True)

# Customer service agent
cs_agent = Agent(
    name="CustomerService",
    instructions="You are a customer service representative. Be helpful and professional.",
    model="gpt-3.5-turbo"
)

add_rizk_guardrails(
    agent=cs_agent,
    organization_id="company",
    project_id="customer_support",
    agent_id="cs_agent"
)

# Technical support agent
tech_agent = Agent(
    name="TechnicalSupport", 
    instructions="You are a technical support specialist. Provide detailed technical guidance.",
    model="gpt-4"
)

add_rizk_guardrails(
    agent=tech_agent,
    organization_id="company",
    project_id="technical_support",
    agent_id="tech_agent"
)

# Sales agent
sales_agent = Agent(
    name="SalesAssistant",
    instructions="You are a sales assistant. Help customers find the right products.",
    model="gpt-3.5-turbo"
)

add_rizk_guardrails(
    agent=sales_agent,
    organization_id="company", 
    project_id="sales",
    agent_id="sales_agent"
)
```

### With Agent Tools

```python
from agents import Agent, Tool
from rizk.sdk import Rizk
from rizk.sdk.adapters.agents_adapter import add_rizk_guardrails

rizk = Rizk.init(app_name="ToolAgent", enabled=True)

# Define tools
def get_weather(location: str) -> str:
    """Get weather information for a location."""
    # Mock implementation
    return f"Weather in {location}: Sunny, 72Â°F"

def calculate(expression: str) -> str:
    """Perform mathematical calculations."""
    try:
        result = eval(expression)  # In production, use a safer evaluation method
        return str(result)
    except:
        return "Invalid expression"

# Create tools
weather_tool = Tool(
    name="get_weather",
    description="Get current weather for a location",
    function=get_weather
)

calc_tool = Tool(
    name="calculate", 
    description="Perform mathematical calculations",
    function=calculate
)

# Create agent with tools
agent = Agent(
    name="AssistantWithTools",
    instructions="You are a helpful assistant with access to weather and calculation tools.",
    model="gpt-4",
    tools=[weather_tool, calc_tool]
)

# Add guardrails
add_rizk_guardrails(
    agent=agent,
    organization_id="demo",
    project_id="tool_agent"
)
```

## Context and Observability

### Rich Context Tracking

The adapter automatically tracks rich context information:

```python
from agents import Agent, Runner
from rizk.sdk import Rizk
from rizk.sdk.adapters.agents_adapter import add_rizk_guardrails

rizk = Rizk.init(app_name="ContextDemo", enabled=True)

agent = Agent(
    name="ContextAwareAgent",
    instructions="You are a context-aware assistant.",
    model="gpt-3.5-turbo"
)

add_rizk_guardrails(
    agent=agent,
    organization_id="demo_org",
    project_id="context_project",
    agent_id="context_agent_1"
)

# The adapter automatically tracks:
# - Agent name and configuration
# - Input/output content
# - Policy decisions and reasons
# - Performance metrics
# - Organization/project/agent hierarchy

runner = Runner(agent=agent)
result = runner.run("What can you help me with?")
```

### Custom Context Enhancement

```python
from rizk.sdk.adapters.agents_adapter import AgentsSDKAdapter
from rizk.sdk.guardrails.types import PolicySet

# Create custom adapter with enhanced context
class CustomAgentsAdapter(AgentsSDKAdapter):
    def _convert_agents_context_to_rizk(self, ctx, agent):
        # Call parent method
        rizk_context = super()._convert_agents_context_to_rizk(ctx, agent)
        
        # Add custom context
        rizk_context.update({
            "custom_field": "custom_value",
            "session_id": getattr(ctx, "session_id", None),
            "user_type": getattr(ctx, "user_type", "standard")
        })
        
        return rizk_context

# Use custom adapter
custom_adapter = CustomAgentsAdapter(
    organization_id="custom_org",
    project_id="custom_project"
)

# Apply to agent
agent = Agent(name="CustomAgent", instructions="You are a custom agent.")
guardrail_function = custom_adapter.adapt_to_agents_sdk()

# Add as input guardrail
from agents import InputGuardrail
guardrail = InputGuardrail(guardrail_function=guardrail_function)
agent.input_guardrails.append(guardrail)
```

## Error Handling and Resilience

### Graceful Degradation

```python
from agents import Agent, Runner
from rizk.sdk import Rizk
from rizk.sdk.adapters.agents_adapter import add_rizk_guardrails

try:
    rizk = Rizk.init(app_name="ResilientAgent", enabled=True)
    
    agent = Agent(
        name="ResilientAgent",
        instructions="You are a resilient agent that handles errors gracefully.",
        model="gpt-3.5-turbo"
    )
    
    # Add guardrails with error handling
    guardrail_name = add_rizk_guardrails(
        agent=agent,
        organization_id="resilient_org",
        project_id="error_handling"
    )
    
    print(f"Successfully added guardrail: {guardrail_name}")
    
except Exception as e:
    print(f"Failed to initialize Rizk guardrails: {e}")
    # Agent will still work without Rizk guardrails
    print("Agent will continue without governance features")

# Agent continues to function even if Rizk fails
runner = Runner(agent=agent)
result = runner.run("Hello, how are you?")
print(result)
```

### Input Validation and Blocking

```python
from agents import Agent, Runner, InputGuardrailTripwireTriggered
from rizk.sdk import Rizk
from rizk.sdk.adapters.agents_adapter import add_rizk_guardrails

rizk = Rizk.init(app_name="BlockingDemo", enabled=True)

agent = Agent(
    name="StrictAgent",
    instructions="You are a strict agent that follows all policies.",
    model="gpt-3.5-turbo"
)

add_rizk_guardrails(
    agent=agent,
    organization_id="strict_org",
    project_id="content_filtering"
)

runner = Runner(agent=agent)

try:
    # This might be blocked by policies
    result = runner.run("Tell me something inappropriate")
    print(f"Result: {result}")
except InputGuardrailTripwireTriggered as e:
    print(f"Input was blocked by guardrail: {e}")
except Exception as e:
    print(f"Other error: {e}")
```

## Best Practices

### 1. Agent Instruction Design

Design agent instructions to work well with policy injection:

```python
# Good - Clear structure with space for policy injection
instructions = """You are a customer service agent for TechCorp.

CORE RESPONSIBILITIES:
- Help customers with product questions
- Provide technical support
- Process basic account requests

COMMUNICATION STYLE:
- Professional and friendly
- Clear and concise
- Patient and understanding

IMPORTANT: Additional policy guidelines will be automatically added to ensure compliance with company standards."""

agent = Agent(
    name="CustomerServiceAgent",
    instructions=instructions,
    model="gpt-3.5-turbo"
)
```

### 2. Error Handling Strategy

```python
def create_resilient_agent(name: str, instructions: str) -> Agent:
    """Create an agent with resilient guardrail setup."""
    
    agent = Agent(name=name, instructions=instructions, model="gpt-3.5-turbo")
    
    try:
        guardrail_name = add_rizk_guardrails(
            agent=agent,
            organization_id="resilient_org",
            project_id="production"
        )
        print(f"âœ… Guardrails added: {guardrail_name}")
    except Exception as e:
        print(f"âš ï¸ Guardrails failed to initialize: {e}")
        print("Agent will continue without governance features")
    
    return agent

# Usage
agent = create_resilient_agent(
    name="ProductionAgent",
    instructions="You are a production-ready agent."
)
```

### 3. Testing Strategy

```python
import pytest
from agents import Agent, Runner
from rizk.sdk import Rizk
from rizk.sdk.adapters.agents_adapter import add_rizk_guardrails

class TestAgentWithRizk:
    def setup_method(self):
        """Setup for each test."""
        self.rizk = Rizk.init(
            app_name="TestApp",
            enabled=True
        )
        
        self.agent = Agent(
            name="TestAgent",
            instructions="You are a test agent.",
            model="gpt-3.5-turbo"
        )
        
        add_rizk_guardrails(
            agent=self.agent,
            organization_id="test_org",
            project_id="unit_tests"
        )
        
        self.runner = Runner(agent=self.agent)
    
    def test_normal_interaction(self):
        """Test normal agent interaction."""
        result = self.runner.run("Hello, how are you?")
        assert result is not None
        assert len(result) > 0
    
    def test_policy_compliance(self):
        """Test that agent follows policies."""
        # Test with input that should be allowed
        result = self.runner.run("What is machine learning?")
        assert result is not None
        
        # Test with input that might be blocked
        # (depends on your policies)
        try:
            result = self.runner.run("Tell me something inappropriate")
            # If not blocked, ensure response is appropriate
            assert "inappropriate" not in result.lower()
        except Exception:
            # If blocked, that's expected behavior
            pass

# Run tests
pytest.main([__file__])
```

## Troubleshooting

### Common Issues

**1. Guardrail Not Being Applied**
```python
# Check if the guardrail was added successfully
from agents import Agent
from rizk.sdk.adapters.agents_adapter import add_rizk_guardrails

agent = Agent(name="TestAgent", instructions="Test", model="gpt-3.5-turbo")
guardrail_name = add_rizk_guardrails(agent=agent)

print(f"Guardrail added: {guardrail_name}")
print(f"Agent has {len(agent.input_guardrails)} input guardrails")
```

**2. Import Errors**
```bash
# Ensure Agents SDK is installed
pip install agents-sdk

# Check installation
python -c "import agents; print('Agents SDK available')"
```

**3. Context Issues**
```python
# Debug context conversion
from rizk.sdk.adapters.agents_adapter import AgentsSDKAdapter

adapter = AgentsSDKAdapter()
# Check if context conversion is working
print("Adapter initialized successfully")
```

### Debug Mode

```python
import logging
from rizk.sdk import Rizk

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("rizk.adapters.agents")
logger.setLevel(logging.DEBUG)

rizk = Rizk.init(app_name="DebugAgent", enabled=True)
```

The OpenAI Agents SDK Adapter provides seamless integration with the Agents SDK framework, ensuring your agent applications are automatically governed and monitored while maintaining full compatibility with the Agents SDK's features and capabilities.


