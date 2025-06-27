---
title: "Quick Start Guide"
description: "Quick Start Guide"
---

# Quick Start Guide

Get up and running with Rizk SDK in 5 minutes. This guide will have you monitoring and governing your first LLM application with minimal setup.

## Prerequisites

- Python 3.10 or higher
- Basic familiarity with Python and LLM applications

## Step 1: Install Rizk SDK

```bash
pip install rizk
```

## Step 2: Get Your API Key

1. Sign up at [app.rizk.tools](https://app.rizk.tools) 
2. Navigate to your dashboard and copy your API key
3. Set it as an environment variable:

```bash
# Windows PowerShell
$env:RIZK_API_KEY="your-api-key-here"

# Linux/macOS
export RIZK_API_KEY="your-api-key-here"
```

## Step 3: Your First Monitored Function

Create a file called `quickstart.py`:

```python
from rizk.sdk import Rizk
from rizk.sdk.decorators import workflow, guardrails

# Initialize Rizk SDK
rizk = Rizk.init(
    app_name="QuickStartApp",
    enabled=True
)

# Add monitoring and governance to any function
@workflow(
    name="hello_world",
    organization_id="quickstart_org",
    project_id="demo_project"
)
@guardrails()
def hello_llm(user_input: str) -> str:
    """A simple function that mimics LLM behavior."""
    
    # Your LLM logic would go here
    # For this demo, we'll just return a simple response
    if "weather" in user_input.lower():
        return f"I'd be happy to help with weather information! However, I need your location to provide accurate weather data."
    elif "hello" in user_input.lower():
        return "Hello! How can I assist you today?"
    else:
        return f"You asked: '{user_input}'. I'm a demo function, but in a real app, this would go to an LLM."

# Test the function
if __name__ == "__main__":
    # Test cases
    test_inputs = [
        "Hello there!",
        "What's the weather like?",
        "Tell me about quantum computing"
    ]
    
    print("ðŸš€ Testing Rizk SDK integration...")
    print("=" * 50)
    
    for i, test_input in enumerate(test_inputs, 1):
        print(f"\n{i}. Input: {test_input}")
        result = hello_llm(test_input)
        print(f"   Output: {result}")
    
    print("\nâœ… Success! Your function now has:")
    print("   â€¢ Distributed tracing")
    print("   â€¢ Performance monitoring") 
    print("   â€¢ Policy enforcement")
    print("   â€¢ Automatic instrumentation")
```

## Step 4: Run Your First Example

```bash
python quickstart.py
```

You should see output like:
```
ðŸš€ Testing Rizk SDK integration...
==================================================

1. Input: Hello there!
   Output: Hello! How can I assist you today?

2. Input: What's the weather like?
   Output: I'd be happy to help with weather information! However, I need your location to provide accurate weather data.

3. Input: Tell me about quantum computing
   Output: You asked: 'Tell me about quantum computing'. I'm a demo function, but in a real app, this would go to an LLM.

âœ… Success! Your function now has:
   â€¢ Distributed tracing
   â€¢ Performance monitoring
   â€¢ Policy enforcement
   â€¢ Automatic instrumentation
```

## Step 5: Add Real LLM Integration

Now let's integrate with a real LLM. Here's an example with OpenAI:

```python
import openai
from rizk.sdk import Rizk
from rizk.sdk.decorators import workflow, guardrails

# Initialize Rizk
rizk = Rizk.init(app_name="OpenAI-Demo", enabled=True)

# Set your OpenAI API key
openai.api_key = "your-openai-api-key"  # Better to use env var

@workflow(
    name="openai_chat",
    organization_id="my_org", 
    project_id="openai_project"
)
@guardrails()
def chat_with_openai(user_message: str) -> str:
    """Chat with OpenAI with automatic monitoring and governance."""
    
    try:
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": user_message}
            ],
            max_tokens=150
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {str(e)}"

# Test with real LLM
if __name__ == "__main__":
    test_message = "Explain what observability means in AI systems"
    response = chat_with_openai(test_message)
    print(f"Question: {test_message}")
    print(f"Answer: {response}")
```

## Step 6: Framework-Specific Examples

### With LangChain

```bash
pip install rizk[langchain]
```

```python
from rizk.sdk import Rizk
from rizk.sdk.decorators import workflow, tool
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.prompts import ChatPromptTemplate

# Initialize Rizk
rizk = Rizk.init(app_name="LangChain-Demo", enabled=True)

@tool(name="weather_tool", organization_id="demo", project_id="langchain")
def get_weather(location: str) -> str:
    """Get weather information for a location."""
    # Mock weather function
    return f"Weather in {location}: Sunny, 72Â°F"

@workflow(name="langchain_agent", organization_id="demo", project_id="langchain")
def run_langchain_agent(query: str) -> str:
    """Run a LangChain agent with monitoring."""
    
    llm = ChatOpenAI(temperature=0)
    
    # Create agent with tools
    tools = [get_weather]
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant."),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])
    
    agent = create_openai_tools_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools)
    
    result = agent_executor.invoke({"input": query})
    return result["output"]

# Test
response = run_langchain_agent("What's the weather like in San Francisco?")
print(response)
```

### With CrewAI

```bash
pip install rizk[crewai]
```

```python
from rizk.sdk import Rizk
from rizk.sdk.decorators import crew, agent, task
from crewai import Agent, Task, Crew, Process

# Initialize Rizk
rizk = Rizk.init(app_name="CrewAI-Demo", enabled=True)

@agent(name="writer_agent", organization_id="demo", project_id="crewai")
def create_writer():
    return Agent(
        role="Technical Writer",
        goal="Write clear and informative content",
        backstory="You are an expert technical writer with AI knowledge",
        verbose=True
    )

@task(name="writing_task", organization_id="demo", project_id="crewai")
def create_writing_task(writer_agent: Agent, topic: str):
    return Task(
        description=f"Write a brief explanation of {topic}",
        agent=writer_agent,
        expected_output="A clear, informative paragraph"
    )

@crew(name="writing_crew", organization_id="demo", project_id="crewai")
def create_writing_crew(topic: str):
    writer = create_writer()
    task = create_writing_task(writer, topic)
    
    return Crew(
        agents=[writer],
        tasks=[task],
        process=Process.sequential,
        verbose=True
    )

# Test
crew = create_writing_crew("machine learning")
result = crew.kickoff()
print(result)
```

## Step 7: Custom Policies (Optional)

Add custom governance policies by creating a policy file:

```yaml
# policies/custom_policies.yaml
version: "1.0.0"
policies:
  - id: "demo_policy"
    name: "Demo Content Policy"
    domains: ["demo", "test", "example"]
    description: "Ensures demo content is appropriate"
    action: "allow"
    guidelines:
      - "Keep all demo content professional and appropriate"
      - "Avoid controversial topics in examples"
      - "Focus on technical learning outcomes"
    patterns:
      - "(?i)inappropriate.*demo"
```

Configure Rizk to use your custom policies:

```python
rizk = Rizk.init(
    app_name="CustomPolicyDemo",
    policies_path="./policies",
    enabled=True
)
```

## Step 8: View Your Data

1. **Dashboard**: Visit [app.rizk.tools](https://app.rizk.tools) to see your traces and metrics
2. **Local Logs**: Check your application logs for tracing information
3. **Custom OTLP**: Configure a custom OpenTelemetry endpoint if needed

```python
# Default configuration (uses https://api.rizk.tools automatically)
rizk = Rizk.init(
    app_name="MyApp",
    api_key=os.getenv("RIZK_API_KEY"),
    enabled=True
)
```

> **Note**: To use a custom OpenTelemetry endpoint instead of Rizk's default (https://api.rizk.tools), set `opentelemetry_endpoint="https://your-custom-otlp-endpoint.com"`. This is only needed if you want to send telemetry data to your own OTLP collector rather than using Rizk's built-in telemetry service.

## What You've Accomplished

In just 5 minutes, you've:

âœ… **Installed** Rizk SDK  
âœ… **Instrumented** your first function with monitoring  
âœ… **Added** policy enforcement and guardrails  
âœ… **Integrated** with real LLM services  
âœ… **Tested** framework-specific integrations  
âœ… **Created** custom governance policies  

## Next Steps

### Dive Deeper

1. **[First Example](first-example.md)** - Detailed walkthrough with explanations
2. **[Architecture Overview](../core-concepts/architecture.md)** - Understand how Rizk works
3. **[Creating Custom Policies](../guardrails/creating-policies.md)** - Build advanced governance rules

### Framework-Specific Guides

1. **[OpenAI Agents Integration](../framework-integration/openai-agents.md)**
2. **[LangChain Integration](../framework-integration/langchain.md)**
3. **[CrewAI Integration](../framework-integration/crewai.md)**

### Production Setup

1. **[Production Configuration](../advanced-config/production-setup.md)**
2. **[Performance Tuning](../advanced-config/performance-tuning.md)**
3. **[Security Best Practices](../advanced-config/security.md)**

## Common Issues

### API Key Not Working
```bash
# Verify your API key is set
echo $RIZK_API_KEY  # Linux/macOS
echo $env:RIZK_API_KEY  # Windows PowerShell
```

### Import Errors
```bash
# Ensure Rizk is installed
pip show rizk

# Reinstall if needed
pip install --upgrade rizk
```

### Framework Detection Issues
```python
# Check if your framework is detected
from rizk.sdk.utils.framework_detection import detect_framework
print(detect_framework())  # Should show your framework
```

## Getting Help

- **Documentation**: Browse the full docs in this directory
- **Examples**: Check out the [examples section](../10-examples/)
- **Issues**: Report bugs on [GitHub](https://github.com/rizk-ai/rizk-sdk/issues)
- **Community**: Join our [Discord](https://discord.gg/rizk)
- **Support**: Email [hello@rizk.tools](mailto:hello@rizk.tools)

---

**ðŸŽ‰ Congratulations!** You now have a monitored, governed LLM application. The same patterns work across any framework - just swap out the LLM code while keeping the Rizk decorators. 

