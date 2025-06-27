---
title: "Using Guardrails"
description: "Using Guardrails"
---

# Using Guardrails

This guide shows you how to implement and use Rizk's guardrails system in your LLM applications. Guardrails provide automatic governance and policy enforcement with minimal code changes.

## Quick Start

The simplest way to add guardrails to your application:

```python
from rizk.sdk import Rizk
from rizk.sdk.decorators import guardrails

# Initialize Rizk - guardrails are automatically enabled
rizk = Rizk.init(
    app_name="MyApp",
    api_key="your-rizk-api-key",
    enabled=True
)

@guardrails()
def my_llm_function(user_input: str) -> str:
    """Your function now has automatic guardrails."""
    # Your existing LLM code
    return process_with_llm(user_input)

# Use normally - guardrails work automatically
result = my_llm_function("Tell me about AI safety")
```

That's it! Your function now has:
- âœ… Input validation and filtering
- âœ… Output content checking
- âœ… Policy-based prompt enhancement
- âœ… Automatic compliance monitoring

## Integration Methods

### 1. Decorator-Based Integration (Recommended)

The decorator approach is the simplest and most common way to add guardrails:

```python
from rizk.sdk.decorators import workflow, task, agent, guardrails

# Basic guardrails
@guardrails()
def chat_assistant(message: str) -> str:
    return llm_generate(message)

# Combined with workflow tracking
@workflow(name="customer_support", organization_id="acme", project_id="support")
@guardrails()
def customer_support(query: str) -> str:
    return handle_customer_query(query)

# Task-level guardrails
@task(name="content_generation", organization_id="acme", project_id="marketing")
@guardrails()
def generate_content(prompt: str) -> str:
    return generate_marketing_content(prompt)

# Agent-level guardrails
@agent(name="sales_agent", organization_id="acme", project_id="sales")
@guardrails()
def sales_assistant(customer_query: str) -> str:
    return process_sales_inquiry(customer_query)
```

### 2. Framework-Specific Integration

Guardrails work automatically with popular LLM frameworks:

#### OpenAI Integration

```python
import openai
from rizk.sdk import Rizk

# Initialize Rizk - OpenAI calls are automatically protected
rizk = Rizk.init(app_name="OpenAIApp", enabled=True)

# Your existing OpenAI code works unchanged
response = openai.chat.completions.create(
    model="gpt-4",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": user_input}
    ]
)

# Guardrails automatically:
# - Evaluate the user input
# - Enhance the system prompt with relevant guidelines
# - Check the response before returning
```

#### LangChain Integration

```python
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_tools_agent
from rizk.sdk import Rizk

rizk = Rizk.init(app_name="LangChainApp", enabled=True)

# Create your LangChain components normally
llm = ChatOpenAI(temperature=0)
agent = create_openai_tools_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools)

# Guardrails automatically protect all interactions
result = agent_executor.invoke({"input": user_query})
```

#### CrewAI Integration

```python
from crewai import Agent, Task, Crew, Process
from rizk.sdk import Rizk

rizk = Rizk.init(app_name="CrewAIApp", enabled=True)

# Define your crew normally
researcher = Agent(
    role="Research Analyst",
    goal="Gather comprehensive information",
    backstory="Expert researcher with attention to detail"
)

research_task = Task(
    description="Research the given topic thoroughly",
    agent=researcher
)

crew = Crew(
    agents=[researcher],
    tasks=[research_task],
    process=Process.sequential
)

# Guardrails automatically protect all crew interactions
result = crew.kickoff()
```

### 3. Manual Integration

For advanced use cases, you can invoke guardrails manually:

```python
from rizk.sdk.guardrails.engine import GuardrailsEngine

async def advanced_guardrail_usage(user_input: str):
    engine = GuardrailsEngine.get_instance()
    
    # Step 1: Evaluate input
    input_decision = await engine.process_message(user_input)
    
    if not input_decision.allowed:
        return {
            "blocked": True,
            "reason": input_decision.blocked_reason,
            "response": "I can't help with that request."
        }
    
    # Step 2: Use guidelines to enhance prompt
    system_prompt = "You are a helpful assistant."
    if input_decision.guidelines:
        guidelines_text = "\n".join([f"â€¢ {g}" for g in input_decision.guidelines])
        system_prompt += f"\n\nIMPORTANT GUIDELINES:\n{guidelines_text}"
    
    # Step 3: Generate response
    response = await llm_call(system_prompt, user_input)
    
    # Step 4: Evaluate output
    output_decision = await engine.evaluate_response(response)
    
    if not output_decision.allowed:
        return {
            "blocked": True,
            "reason": output_decision.blocked_reason,
            "response": "I need to revise my response to ensure it meets our guidelines."
        }
    
    return {
        "blocked": False,
        "response": response,
        "guidelines_applied": input_decision.guidelines
    }
```

## Configuration Options

### Basic Configuration

```python
from rizk.sdk import Rizk

# Standard configuration
rizk = Rizk.init(
    app_name="MyApp",
    api_key="your-api-key",
    enabled=True
)
```

### Advanced Configuration

```python
# Advanced configuration with custom settings
rizk = Rizk.init(
    app_name="AdvancedApp",
    api_key="your-api-key",
    enabled=True,
    
    # Guardrail settings
    guardrails_enabled=True,
    policy_enforcement="strict",  # strict, moderate, lenient
    
    # Performance settings
    llm_cache_size=10000,
    fast_rules_enabled=True,
    
    # Custom LLM for evaluations
    llm_service="openai",
    llm_model="gpt-4",
    
    # Observability
    trace_content=True,
    metrics_enabled=True
)
```

### Environment-Based Configuration

```bash
# Set via environment variables
export RIZK_API_KEY="your-api-key"
export RIZK_GUARDRAILS_ENABLED="true"
export RIZK_POLICY_ENFORCEMENT="strict"
export RIZK_LLM_SERVICE="openai"
export RIZK_TRACE_CONTENT="true"
```

```python
# Configuration is automatically loaded from environment
rizk = Rizk.init(app_name="MyApp")
```

## Decorator Options

The `@guardrails()` decorator supports various configuration options:

### Basic Usage

```python
@guardrails()
def simple_function(input_text: str) -> str:
    return process_text(input_text)
```

### Custom Configuration

```python
@guardrails(
    input_validation=True,      # Enable input validation (default: True)
    output_validation=True,     # Enable output validation (default: True)
    prompt_augmentation=True,   # Enable prompt enhancement (default: True)
    enforcement_level="strict"  # Override global enforcement level
)
def strict_function(input_text: str) -> str:
    return process_sensitive_content(input_text)
```

### Selective Enforcement

```python
# Only input validation
@guardrails(output_validation=False)
def input_only_validation(user_input: str) -> str:
    return process_user_input(user_input)

# Only output validation
@guardrails(input_validation=False)
def output_only_validation(content: str) -> str:
    return generate_content(content)

# Only prompt augmentation
@guardrails(input_validation=False, output_validation=False)
def prompt_enhancement_only(prompt: str) -> str:
    return llm_generate(prompt)
```

## Common Patterns

### 1. Customer Service Bot

```python
from rizk.sdk.decorators import workflow, guardrails

@workflow(name="customer_service", organization_id="company", project_id="support")
@guardrails()
def customer_service_bot(customer_query: str, customer_id: str = None) -> str:
    """Customer service bot with automatic guardrails."""
    
    # Your business logic
    context = get_customer_context(customer_id) if customer_id else {}
    response = generate_support_response(customer_query, context)
    
    return response

# Usage
response = customer_service_bot(
    customer_query="I'm having trouble with my account",
    customer_id="cust_12345"
)
```

### 2. Content Generation

```python
@task(name="content_generation", organization_id="marketing", project_id="campaigns")
@guardrails()
def generate_marketing_content(topic: str, tone: str = "professional") -> str:
    """Generate marketing content with brand safety guardrails."""
    
    prompt = f"Create marketing content about {topic} with a {tone} tone."
    content = llm_generate(prompt)
    
    return content

# Usage
content = generate_marketing_content(
    topic="new product launch",
    tone="exciting"
)
```

### 3. Educational Assistant

```python
@agent(name="tutor", organization_id="education", project_id="ai_tutoring")
@guardrails()
def educational_assistant(student_question: str, subject: str) -> str:
    """Educational assistant with appropriate content filtering."""
    
    # Customize response based on subject
    if subject.lower() in ["math", "science"]:
        response = generate_technical_explanation(student_question)
    else:
        response = generate_general_explanation(student_question)
    
    return response

# Usage
explanation = educational_assistant(
    student_question="How does photosynthesis work?",
    subject="biology"
)
```

### 4. Financial Advisory (Compliance-Heavy)

```python
@workflow(name="financial_advisory", organization_id="fintech", project_id="advisory")
@guardrails(enforcement_level="strict")
def financial_assistant(query: str, user_profile: dict) -> str:
    """Financial assistant with strict compliance guardrails."""
    
    # Generate response with user context
    response = generate_financial_guidance(query, user_profile)
    
    # Additional business logic
    if requires_human_review(query):
        response += "\n\nThis query has been flagged for human review."
    
    return response

# Usage
advice = financial_assistant(
    query="Should I invest in cryptocurrency?",
    user_profile={"risk_tolerance": "moderate", "age": 35}
)
```

## Error Handling

### Handling Blocked Requests

```python
from rizk.sdk.exceptions import GuardrailBlockedException

@guardrails()
def protected_function(user_input: str) -> str:
    try:
        return process_input(user_input)
    except GuardrailBlockedException as e:
        # Handle blocked requests gracefully
        return f"I can't help with that request. Reason: {e.reason}"

# Alternative: Check return value
@guardrails()
def protected_function_alt(user_input: str) -> dict:
    result = process_input(user_input)
    
    # Check if response was modified by guardrails
    if hasattr(result, '_rizk_blocked'):
        return {
            "success": False,
            "message": "Request was blocked by content policy",
            "reason": result._rizk_block_reason
        }
    
    return {
        "success": True,
        "response": result
    }
```

### Graceful Degradation

```python
@guardrails()
def resilient_function(user_input: str) -> str:
    try:
        # Primary processing
        return advanced_llm_processing(user_input)
    except Exception as e:
        # Fallback to simpler processing if guardrails or LLM fail
        return simple_response_generation(user_input)

# With explicit error handling
def safe_guardrail_function(user_input: str) -> str:
    try:
        # Try with guardrails
        return protected_function(user_input)
    except Exception as e:
        # Log the error and provide fallback
        logger.warning(f"Guardrail error: {e}")
        return "I'm having trouble processing your request. Please try again."
```

## Testing Guardrails

### Unit Testing

```python
import pytest
from rizk.sdk import Rizk
from your_app import your_guardrailed_function

class TestGuardrails:
    def setup_method(self):
        """Setup Rizk for testing."""
        self.rizk = Rizk.init(
            app_name="TestApp",
            enabled=True
        )
    
    def test_normal_input(self):
        """Test that normal inputs work correctly."""
        result = your_guardrailed_function("What is machine learning?")
        assert result is not None
        assert len(result) > 0
    
    def test_inappropriate_input(self):
        """Test that inappropriate inputs are handled."""
        # This should either be blocked or handled gracefully
        result = your_guardrailed_function("inappropriate content")
        
        # Check that function either blocks or provides appropriate response
        assert result is not None
        # Add specific assertions based on your expected behavior
    
    def test_edge_cases(self):
        """Test edge cases."""
        edge_cases = [
            "",  # Empty input
            "a" * 10000,  # Very long input
            "Special chars: @#$%^&*()",  # Special characters
        ]
        
        for case in edge_cases:
            result = your_guardrailed_function(case)
            assert result is not None  # Should handle gracefully
```

### Integration Testing

```python
async def test_guardrail_integration():
    """Test guardrails with real scenarios."""
    
    test_scenarios = [
        {
            "input": "Help me with my homework",
            "expected_allowed": True,
            "description": "Educational query should be allowed"
        },
        {
            "input": "Tell me how to hack systems",
            "expected_allowed": False,
            "description": "Security threat should be blocked"
        },
        {
            "input": "What's the weather like?",
            "expected_allowed": True,
            "description": "Innocent query should be allowed"
        }
    ]
    
    for scenario in test_scenarios:
        result = your_guardrailed_function(scenario["input"])
        
        # Check if result indicates blocking
        is_blocked = (
            result is None or 
            "blocked" in result.lower() or 
            "can't help" in result.lower()
        )
        
        expected_blocked = not scenario["expected_allowed"]
        
        assert is_blocked == expected_blocked, f"Failed: {scenario['description']}"
```

## Performance Optimization

### Caching Configuration

```python
# Configure caching for better performance
rizk = Rizk.init(
    app_name="HighPerformanceApp",
    enabled=True,
    
    # Increase cache size for frequently repeated queries
    llm_cache_size=50000,
    
    # Adjust cache TTL based on your use case
    cache_ttl_seconds=3600,  # 1 hour
    
    # Enable fast rules for immediate pattern matching
    fast_rules_enabled=True
)
```

### Async Usage

```python
import asyncio
from rizk.sdk.decorators import guardrails

@guardrails()
async def async_guardrailed_function(user_input: str) -> str:
    """Async function with guardrails."""
    # Async processing
    result = await async_llm_call(user_input)
    return result

# Usage
async def main():
    tasks = [
        async_guardrailed_function(f"Query {i}")
        for i in range(100)
    ]
    
    results = await asyncio.gather(*tasks)
    return results

# Run async
results = asyncio.run(main())
```

### Batch Processing

```python
@guardrails()
def batch_process(inputs: list[str]) -> list[str]:
    """Process multiple inputs efficiently."""
    
    # Guardrails will evaluate each input individually
    # but can leverage caching for similar inputs
    results = []
    for input_text in inputs:
        result = process_single_input(input_text)
        results.append(result)
    
    return results

# Usage
batch_inputs = [
    "Question 1",
    "Question 2", 
    "Question 1",  # This will hit cache
    "Question 3"
]

batch_results = batch_process(batch_inputs)
```

## Monitoring and Debugging

### Debug Mode

```python
import logging
from rizk.sdk import Rizk

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("rizk.guardrails")
logger.setLevel(logging.DEBUG)

# Initialize with debug mode
rizk = Rizk.init(
    app_name="DebugApp",
    enabled=True,
    verbose=True  # Enable verbose logging
)

@guardrails()
def debug_function(input_text: str) -> str:
    # Debug logs will show:
    # - Which policies were evaluated
    # - Why decisions were made
    # - Performance metrics
    return process_text(input_text)
```

### Performance Monitoring

```python
from rizk.sdk.analytics import GuardrailAnalytics

# Monitor guardrail performance
analytics = GuardrailAnalytics()

# Get metrics for the last 24 hours
metrics = analytics.get_metrics(time_range="24h")

print(f"Total requests: {metrics.total_requests}")
print(f"Blocked requests: {metrics.blocked_requests}")
print(f"Average latency: {metrics.avg_latency}ms")
print(f"Cache hit rate: {metrics.cache_hit_rate}%")

# Get policy-specific metrics
policy_metrics = analytics.get_policy_metrics(
    policy_id="content_moderation",
    time_range="24h"
)

print(f"Policy triggers: {policy_metrics.trigger_count}")
print(f"False positive rate: {policy_metrics.false_positive_rate}")
```

## Best Practices

### 1. Start Simple

```python
# âœ… Start with basic guardrails
@guardrails()
def my_function(input_text: str) -> str:
    return process_text(input_text)

# âŒ Don't over-configure initially
@guardrails(
    input_validation=True,
    output_validation=True,
    custom_policies=["policy1", "policy2", "policy3"],
    enforcement_level="ultra_strict",
    # ... too many options
)
def overly_configured_function(input_text: str) -> str:
    return process_text(input_text)
```

### 2. Test Thoroughly

```python
# âœ… Test with diverse inputs
test_inputs = [
    "Normal question",
    "Edge case with special chars: @#$%",
    "Very long input: " + "a" * 1000,
    "Empty string: ",
    "Potentially problematic content"
]

for test_input in test_inputs:
    result = my_guardrailed_function(test_input)
    assert result is not None, f"Failed on input: {test_input}"
```

### 3. Handle Errors Gracefully

```python
# âœ… Provide meaningful error messages
@guardrails()
def user_facing_function(query: str) -> str:
    try:
        return process_query(query)
    except Exception as e:
        # Don't expose internal errors to users
        return "I'm having trouble with that request. Please try rephrasing."

# âœ… Log errors for debugging
import logging
logger = logging.getLogger(__name__)

@guardrails()
def logged_function(query: str) -> str:
    try:
        return process_query(query)
    except Exception as e:
        logger.error(f"Error processing query: {e}", exc_info=True)
        return "Sorry, I encountered an error processing your request."
```

### 4. Monitor Performance

```python
# âœ… Regular performance checks
def check_guardrail_performance():
    analytics = GuardrailAnalytics()
    metrics = analytics.get_metrics(time_range="1h")
    
    if metrics.avg_latency > 100:  # ms
        logger.warning(f"High guardrail latency: {metrics.avg_latency}ms")
    
    if metrics.cache_hit_rate < 0.8:  # 80%
        logger.warning(f"Low cache hit rate: {metrics.cache_hit_rate}")
    
    return metrics

# Run periodically
performance_metrics = check_guardrail_performance()
```

## Next Steps

1. **[Policy Enforcement](policy-enforcement.md)** - Understanding how policies work
2. **[Configuration](configuration.md)** - Advanced configuration options  
3. **[Monitoring](monitoring.md)** - Tracking guardrail performance
4. **[Troubleshooting](troubleshooting.md)** - Debugging guardrail issues

---

Guardrails provide powerful protection for your LLM applications with minimal integration effort. Start with the basic decorator approach and customize as your needs evolve. 

