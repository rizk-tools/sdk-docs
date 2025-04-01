---
title: "Quick Start"
description: "Documentation for Quick Start"
---

This guide will help you get started with the Rizk SDK by creating a simple AI agent with guardrails.

## Basic Setup

First, let's create a simple script that uses the Rizk SDK:

```python
import os
from rizk.sdk import Rizk
from rizk.sdk.decorators import agent, add_policies

# Initialize the SDK
client = Rizk.init(
    app_name="my_ai_agent",
    api_key=os.getenv("RIZK_API_KEY"),
    telemetry_enabled=True
)

# Define your AI agent
@agent
@add_policies(["content_moderation", "safety"])
async def my_agent(query: str):
    """
    A simple AI agent that processes user queries with guardrails.
    
    Args:
        query: The user's input query
        
    Returns:
        str: The agent's response
    """
    # Set context for the current conversation
    Rizk.set_association_properties({
        "organization_id": "my_org",
        "project_id": "my_project",
        "agent_id": "my_agent"
    })
    
    # Process the query through guardrails
    guardrails = Rizk.get_guardrails()
    result = await guardrails.process_message(query)
    
    if not result["allowed"]:
        return f"Query blocked: {result.get('blocked_reason', 'Policy violation')}"
    
    # Your AI processing logic here
    response = f"Processed query: {query}"
    
    # Check the output through guardrails
    output_check = await guardrails.check_output(response)
    
    if not output_check["allowed"]:
        return "Response blocked: Policy violation detected"
    
    return response

# Example usage
async def main():
    # Test the agent
    response = await my_agent("Hello, how are you?")
    print(response)
    
    # Test with potentially harmful content
    response = await my_agent("Generate harmful content")
    print(response)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
```

## Understanding the Code

Let's break down the key components:

1. **SDK Initialization**:
   ```python
   client = Rizk.init(
       app_name="my_ai_agent",
       api_key=os.getenv("RIZK_API_KEY"),
       telemetry_enabled=True
   )
   ```
   This initializes the Rizk SDK with your application name and API key.

2. **Agent Decorator**:
   ```python
   @agent
   @add_policies(["content_moderation", "safety"])
   ```
   The `@agent` decorator marks your function as an AI agent, and `@add_policies` applies specific guardrails policies.

3. **Guardrails Processing**:
   ```python
   guardrails = Rizk.get_guardrails()
   result = await guardrails.process_message(query)
   ```
   This processes the input through the guardrails system before your AI logic.

4. **Output Checking**:
   ```python
   output_check = await guardrails.check_output(response)
   ```
   This ensures the AI's response complies with policies.

## Running the Example

1. Save the code in a file (e.g., `my_agent.py`)
2. Set your environment variables:
   ```bash
   export RIZK_API_KEY=your_api_key_here
   ```
3. Run the script:
   ```bash
   python my_agent.py
   ```

## Next Steps

- [Configuration Guide](./configuration)
- [Guardrails Documentation](../core-concepts/guardrails)
<!-- - [Telemetry Guide](../guides/using-telemetry) -->
<!-- - [Advanced Examples](../examples/advanced-guardrails)  -->