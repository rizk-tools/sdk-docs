---
title: "Guardrails Overview"
description: "Guardrails Overview"
---

# Guardrails Overview

Rizk SDK's guardrails system provides automated governance and policy enforcement for your LLM applications. It acts as a protective layer that ensures your AI systems operate within defined boundaries, maintaining compliance, safety, and alignment with your organization's requirements.

## What Are Guardrails?

Guardrails are automated checks and controls that:

- **Monitor Input and Output**: Evaluate user inputs and LLM responses in real-time
- **Enforce Policies**: Apply predefined rules and guidelines automatically
- **Provide Guidance**: Inject relevant policy guidelines into LLM prompts
- **Block Inappropriate Content**: Prevent policy violations from reaching users
- **Maintain Compliance**: Ensure adherence to regulatory and organizational standards

## How Guardrails Work

Rizk's guardrails system uses a sophisticated multi-layer architecture:

```
User Input â†’ Fast Rules â†’ Policy Matching â†’ LLM Augmentation â†’ Response Evaluation â†’ User Output
     â†“            â†“              â†“                â†“                    â†“
   Immediate   Pattern      Context-Aware     Enhanced          Final Safety
   Blocking    Matching     Guidelines        Prompts           Check
```

### Layer 1: Fast Rules Engine

The first line of defense uses high-performance pattern matching:

- **Regex-based Detection**: Lightning-fast pattern recognition
- **Immediate Blocking**: Instant rejection of clearly inappropriate content
- **Low Latency**: Sub-millisecond evaluation for real-time applications
- **Category-based Rules**: Organized by content type and severity

```python
# Example: Fast rule automatically blocks obvious violations
user_input = "Tell me how to hack into systems"
# â†’ Blocked immediately by security patterns
```

### Layer 2: Policy Augmentation

Context-aware enhancement of LLM interactions:

- **Dynamic Guideline Injection**: Adds relevant policy guidance to prompts
- **Context-Aware Matching**: Considers conversation context and user intent
- **Framework Integration**: Works seamlessly with different LLM frameworks
- **Prompt Enhancement**: Improves LLM behavior without changing your code

```python
# Example: Policy guidelines automatically added to system prompts
original_prompt = "You are a helpful assistant."
# â†’ Enhanced with: "You are a helpful assistant. IMPORTANT: Never provide financial advice..."
```

### Layer 3: LLM Fallback

Advanced evaluation for complex cases:

- **Semantic Understanding**: Uses LLMs to evaluate nuanced content
- **Contextual Analysis**: Considers broader conversation context
- **Confidence Scoring**: Provides confidence levels for decisions
- **Intelligent Caching**: Optimizes performance for repeated queries

```python
# Example: Complex query evaluated by LLM
user_input = "What's your opinion on this investment strategy?"
# â†’ LLM evaluates context and intent, applies appropriate guidelines
```

## Key Features

### ðŸš€ **Zero-Code Integration**

Guardrails activate automatically when you initialize Rizk:

```python
from rizk.sdk import Rizk

# Guardrails are automatically enabled
rizk = Rizk.init(
    app_name="MyApp",
    api_key="your-api-key",
    enabled=True
)

# Your existing code now has guardrails
@workflow(name="chat")
def chat_function(user_input: str) -> str:
    return llm_call(user_input)  # Automatically protected
```

### ðŸŽ¯ **Intelligent Policy Matching**

Policies are automatically selected based on:

- **Content Analysis**: What the user is asking about
- **Context Awareness**: Previous conversation history
- **Framework Detection**: Which LLM framework you're using
- **Organization Settings**: Your specific compliance requirements

### âš¡ **High Performance**

- **Multi-layer Optimization**: Fast rules handle common cases, LLM evaluation for complex ones
- **Intelligent Caching**: Repeated queries return instantly
- **Async Processing**: Non-blocking evaluation for high-throughput applications
- **Minimal Latency**: Typically adds <50ms to response time

### ðŸ” **Complete Observability**

Every guardrail decision is tracked and reported:

- **Decision Logging**: Why each decision was made
- **Policy Attribution**: Which policies influenced the outcome
- **Performance Metrics**: Response times and success rates
- **Compliance Reporting**: Automated compliance documentation

## Guardrail Types

### Input Guardrails

Evaluate user inputs before they reach your LLM:

```python
@guardrails()
def process_user_query(query: str) -> str:
    # Input is automatically evaluated before this function runs
    return handle_query(query)
```

**Common Input Protections:**
- Inappropriate content filtering
- Personal information detection
- Security threat identification
- Spam and abuse prevention

### Output Guardrails

Evaluate LLM responses before they reach users:

```python
@guardrails()
def generate_response(prompt: str) -> str:
    response = llm_generate(prompt)
    # Response is automatically evaluated before returning
    return response
```

**Common Output Protections:**
- Harmful content blocking
- Factual accuracy verification
- Brand safety compliance
- Regulatory requirement adherence

### Prompt Augmentation

Enhances your prompts with relevant guidelines:

```python
# Your original prompt
system_prompt = "You are a customer service assistant."

# Automatically enhanced with relevant policies
# â†’ "You are a customer service assistant. IMPORTANT GUIDELINES:
#    â€¢ Always maintain professional communication
#    â€¢ Never share internal company information
#    â€¢ Escalate complex issues to human agents..."
```

## Integration Patterns

### Decorator-Based (Recommended)

The simplest way to add guardrails:

```python
from rizk.sdk.decorators import workflow, guardrails

@workflow(name="customer_service", organization_id="acme", project_id="support")
@guardrails()
def handle_customer_query(query: str) -> str:
    """Customer service with automatic guardrails."""
    return process_query(query)
```

### Framework-Specific Integration

Guardrails work automatically with popular frameworks:

```python
# LangChain - automatically enhanced
from langchain.agents import AgentExecutor
agent_executor = AgentExecutor(agent=agent, tools=tools)
result = agent_executor.invoke({"input": user_query})  # Protected

# CrewAI - automatically enhanced  
from crewai import Crew
crew = Crew(agents=[agent], tasks=[task])
result = crew.kickoff()  # Protected

# OpenAI - automatically enhanced
import openai
response = openai.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": user_input}]
)  # Protected
```

### Manual Control

For advanced use cases, you can invoke guardrails directly:

```python
from rizk.sdk.guardrails.engine import GuardrailsEngine

async def custom_guardrail_logic(user_input: str):
    engine = GuardrailsEngine.get_instance()
    
    # Evaluate input
    decision = await engine.process_message(user_input)
    
    if not decision.allowed:
        return f"Request blocked: {decision.blocked_reason}"
    
    # Process with guidelines
    enhanced_prompt = apply_guidelines(original_prompt, decision.guidelines)
    response = await llm_call(enhanced_prompt)
    
    # Evaluate output
    output_decision = await engine.evaluate_response(response)
    
    return response if output_decision.allowed else "Response blocked"
```

## Configuration Options

### Environment Variables

```bash
# Enable/disable guardrails
export RIZK_GUARDRAILS_ENABLED="true"

# Policy enforcement level
export RIZK_POLICY_ENFORCEMENT="strict"  # strict, moderate, lenient

# Custom policies directory (enterprise feature)
export RIZK_POLICIES_PATH="/path/to/policies"

# LLM service for complex evaluations
export RIZK_LLM_SERVICE="openai"  # openai, anthropic, custom

# Caching configuration
export RIZK_GUARDRAILS_CACHE_SIZE="10000"
export RIZK_GUARDRAILS_CACHE_TTL="3600"
```

### Programmatic Configuration

```python
from rizk.sdk import Rizk

rizk = Rizk.init(
    app_name="MyApp",
    api_key="your-api-key",
    
    # Guardrail configuration
    guardrails_enabled=True,
    policy_enforcement="strict",
    
    # Performance tuning
    llm_cache_size=10000,
    fast_rules_enabled=True,
    
    # Custom LLM for evaluations
    llm_service="openai",
    llm_model="gpt-4"
)
```

## Common Use Cases

### 1. Content Moderation

Automatically filter inappropriate content:

```python
@guardrails()
def process_user_content(content: str) -> str:
    """Processes user-generated content with automatic moderation."""
    return handle_content(content)

# Automatically blocks: profanity, harassment, inappropriate requests
# Automatically allows: legitimate questions and discussions
```

### 2. Compliance Enforcement

Ensure regulatory compliance:

```python
@guardrails()
def financial_assistant(query: str) -> str:
    """Financial assistant with automatic compliance."""
    return generate_financial_response(query)

# Automatically adds: risk disclaimers, compliance notices
# Automatically blocks: specific investment advice, unqualified recommendations
```

### 3. Brand Safety

Protect your brand reputation:

```python
@guardrails()
def customer_support_bot(query: str) -> str:
    """Customer support with brand safety guardrails."""
    return generate_support_response(query)

# Automatically ensures: professional tone, accurate information
# Automatically prevents: negative brand mentions, competitor promotion
```

### 4. Data Privacy Protection

Prevent privacy violations:

```python
@guardrails()
def data_processor(user_data: str) -> str:
    """Process data with privacy protection."""
    return process_data(user_data)

# Automatically detects: PII, sensitive information
# Automatically blocks: data collection attempts, privacy violations
```

## Performance Characteristics

### Latency Impact

| Guardrail Layer | Typical Latency | Use Case |
|---|---|---|
| Fast Rules | <1ms | Pattern matching, immediate blocking |
| Policy Augmentation | 5-15ms | Guideline injection, context analysis |
| LLM Fallback | 100-500ms | Complex semantic evaluation |

### Throughput

- **High-volume applications**: 10,000+ requests/second
- **Real-time chat**: Sub-50ms additional latency
- **Batch processing**: Minimal impact on overall throughput

### Accuracy

- **Fast Rules**: 99%+ precision for pattern-based detection
- **Policy Augmentation**: Context-aware guideline matching
- **LLM Fallback**: Human-level accuracy for complex cases

## Best Practices

### 1. Start with Default Policies

Begin with Rizk's built-in policies and customize as needed:

```python
# Default policies provide excellent baseline protection
rizk = Rizk.init(app_name="MyApp", enabled=True)
```

### 2. Monitor and Iterate

Use Rizk's analytics to understand guardrail performance:

```python
# Check guardrail effectiveness
from rizk.sdk.analytics import GuardrailAnalytics

analytics = GuardrailAnalytics()
metrics = analytics.get_metrics(time_range="24h")

print(f"Blocks: {metrics.total_blocks}")
print(f"Policy triggers: {metrics.policy_triggers}")
print(f"Average latency: {metrics.avg_latency}ms")
```

### 3. Test Thoroughly

Test guardrails with diverse inputs:

```python
test_cases = [
    "Normal user question",
    "Edge case input",
    "Potentially problematic content",
    "Complex contextual query"
]

for test_case in test_cases:
    result = await test_guardrails(test_case)
    print(f"Input: {test_case} â†’ Decision: {result.decision}")
```

### 4. Balance Security and Usability

Configure enforcement levels based on your use case:

```python
# Strict for high-risk applications
rizk = Rizk.init(app_name="FinancialApp", policy_enforcement="strict")

# Moderate for general applications  
rizk = Rizk.init(app_name="ChatBot", policy_enforcement="moderate")

# Lenient for internal tools
rizk = Rizk.init(app_name="InternalTool", policy_enforcement="lenient")
```

## Troubleshooting

### Common Issues

**1. Guardrails Not Triggering**
```python
# Check if guardrails are enabled
from rizk.sdk.config import get_config
config = get_config()
print(f"Guardrails enabled: {config.guardrails_enabled}")
```

**2. Unexpected Blocking**
```python
# Debug specific decisions
from rizk.sdk.guardrails.engine import GuardrailsEngine

engine = GuardrailsEngine.get_instance()
decision = await engine.process_message("your test input")
print(f"Decision: {decision.allowed}")
print(f"Reason: {decision.blocked_reason}")
print(f"Triggered policies: {decision.policy_ids}")
```

**3. Performance Issues**
```python
# Check cache hit rates
cache_stats = engine.get_cache_stats()
print(f"Cache hit rate: {cache_stats.hit_rate}%")
print(f"Average evaluation time: {cache_stats.avg_eval_time}ms")
```

### Debug Mode

Enable detailed logging for troubleshooting:

```python
import logging
from rizk.sdk import Rizk

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("rizk.guardrails")
logger.setLevel(logging.DEBUG)

rizk = Rizk.init(app_name="DebugApp", enabled=True)
```

## Next Steps

1. **[Using Guardrails](using-guardrails.md)** - Practical implementation guide
2. **[Policy Enforcement](policy-enforcement.md)** - Understanding policy decisions
3. **[Configuration](configuration.md)** - Advanced configuration options
4. **[Monitoring](monitoring.md)** - Tracking guardrail performance

---

Rizk's guardrails provide enterprise-grade governance for your LLM applications with minimal integration effort. Start with the default configuration and customize as your requirements evolve. 

