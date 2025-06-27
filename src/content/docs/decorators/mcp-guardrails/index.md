---
title: "MCP Guardrails Decorator"
description: "MCP Guardrails Decorator"
---

# MCP Guardrails Decorator

The `@mcp_guardrails` decorator provides specialized outbound guardrail protection for functions that might leak sensitive information through Model Context Protocol (MCP) communications. This decorator is specifically designed to prevent memory leaks, PII exposure, and context spillage in LLM applications.

## What is MCP (Model Context Protocol)?

Model Context Protocol (MCP) is a communication standard that allows AI agents and LLM applications to share context, data, and function outputs with external systems. While MCP enables powerful integrations, it also creates potential security risks:

- **Memory Leaks**: Previous conversation context bleeding into new sessions
- **PII Exposure**: Personal information included in function outputs
- **Context Spillage**: Internal reasoning or sensitive data exposed to external agents
- **Credential Leakage**: API keys, tokens, or passwords accidentally returned

The `@mcp_guardrails` decorator addresses these risks by evaluating function outputs before they leave your application.

## Installation & Setup

The MCP guardrails are included with the core Rizk SDK installation:

```bash
pip install rizk
```

Initialize Rizk SDK with MCP-specific policies:

```python
from rizk.sdk import Rizk

# Initialize with default MCP policies
rizk = Rizk.init(
    app_name="MCP-SecureApp",
    api_key="your-api-key",
    enabled=True
)
```

## Basic Usage

### Importing the Decorator

```python
from rizk.sdk.decorators import mcp_guardrails, ViolationMode, MCPGuardrailsError
```

### Simple Function Protection

```python
@mcp_guardrails()
def get_user_summary(user_id: str) -> str:
    """Get user summary with automatic PII protection."""
    # This function might accidentally include PII
    user = get_user_data(user_id)
    return f"User {user_id}: {user.name}, Email: {user.email}, SSN: {user.ssn}"

# Usage
try:
    summary = get_user_summary("USER123")
    print(summary)  # May be filtered/redacted if PII detected
except MCPGuardrailsError as e:
    print(f"Output blocked: {e}")
```

### Async Function Protection

```python
@mcp_guardrails(on_violation="augment")
async def process_conversation(message: str) -> str:
    """Process conversation with context leak protection."""
    # Simulate LLM processing that might reference previous conversations
    response = await llm_agent.process(message)
    
    # Automatically filtered if context spillage detected
    return response

# Usage
response = await process_conversation("What did we discuss about pricing?")
# Output may be: "[REDACTED - Context information filtered by MCP guardrails]"
```

## Decorator Parameters

### Core Parameters

```python
@mcp_guardrails(
    policy_set=None,                    # Custom policies
    on_violation="augment",             # Violation handling mode
    conversation_id=None,               # Context tracking
    organization_id=None,               # Telemetry context
    project_id=None,                    # Telemetry context
    enabled=True                        # Enable/disable decorator
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `policy_set` | `Optional[PolicySet]` | `None` | Custom policy set for evaluation |
| `on_violation` | `str` | `"augment"` | How to handle violations: `"block"`, `"augment"`, `"warn"` |
| `conversation_id` | `Optional[str]` | `None` | Conversation ID for context tracking |
| `organization_id` | `Optional[str]` | `None` | Organization ID for telemetry |
| `project_id` | `Optional[str]` | `None` | Project ID for telemetry |
| `enabled` | `bool` | `True` | Whether guardrails are enabled |

## Violation Handling Modes

### 1. Block Mode (`on_violation="block"`)

**Behavior**: Raises `MCPGuardrailsError` when violations are detected
**Use Case**: High-security environments where any policy violation must prevent execution

```python
@mcp_guardrails(on_violation="block")
def get_financial_data(account_id: str) -> str:
    """Strict blocking for financial data."""
    account = get_account(account_id)
    return f"Account {account_id}: Balance ${account.balance}, SSN: {account.ssn}"

try:
    data = get_financial_data("ACC123")
    print(data)
except MCPGuardrailsError as e:
    print(f"Blocked: {e}")
    # Handle security violation appropriately
    log_security_event(e.policy_id, e.confidence)
```

### 2. Augment Mode (`on_violation="augment"`) - Default

**Behavior**: Automatically redacts or filters violating content
**Use Case**: Production environments where functionality should continue with safe content

```python
@mcp_guardrails(on_violation="augment")
def customer_service_response(query: str) -> str:
    """Customer service with automatic content filtering."""
    response = process_customer_query(query)
    # Automatically filtered based on violation type
    return response

# Examples of automatic filtering:
response1 = customer_service_response("What's my SSN?")
# Returns: "[REDACTED - Sensitive information removed by MCP guardrails]"

response2 = customer_service_response("Continue our previous conversation")
# Returns: "[REDACTED - Context information filtered by MCP guardrails]"

response3 = customer_service_response("Normal customer question")
# Returns: "Here's how I can help you..." (original response)
```

### 3. Warn Mode (`on_violation="warn"`)

**Behavior**: Logs warning but returns original content unchanged
**Use Case**: Development/testing environments or monitoring-only scenarios

```python
import logging

@mcp_guardrails(on_violation="warn")
def debug_function(data: str) -> str:
    """Development function with warning-only mode."""
    # Process data that might contain sensitive info
    return f"Debug: Processing {data} with internal token abc123"

# Usage - returns original content but logs warning
result = debug_function("user data")
# Logs: "MCP guardrails warning for debug_function: Policy memory_leak_004 flagged output..."
print(result)  # Shows original content including potential violations
```

## Built-in Protection Policies

The Rizk SDK includes four specialized MCP protection policies:

### 1. PII Outbound Prevention (`memory_leak_001`)

**Purpose**: Prevents personal information from leaking in function outputs
**Patterns Detected**:
- Social Security Numbers (SSN)
- Email addresses in structured formats
- Phone numbers
- API keys and passwords
- Personal names with identifiers

```python
@mcp_guardrails()
def user_profile_summary(user_id: str) -> str:
    user = get_user(user_id)
    # This would be filtered:
    return f"User: {user.name}, SSN: 123-45-6789, Email: user@example.com"
    # Becomes: "[REDACTED - Sensitive information removed by MCP guardrails]"
```

### 2. Context Spillage Prevention (`memory_leak_002`)

**Purpose**: Prevents previous conversation context from leaking to other sessions
**Patterns Detected**:
- References to "previous conversation" or "earlier discussion"
- Context from different user sessions
- Cross-conversation memory references

```python
@mcp_guardrails()
async def chat_response(message: str) -> str:
    # This would be filtered:
    return "As we discussed earlier about your financial situation..."
    # Becomes: "[REDACTED - Context information filtered by MCP guardrails]"
```

### 3. Chain of Thought Revelation (`memory_leak_003`)

**Purpose**: Prevents internal reasoning processes from being exposed
**Patterns Detected**:
- Step-by-step reasoning chains
- Internal analysis processes
- Decision-making logic exposure

```python
@mcp_guardrails()
def analysis_result(data: str) -> str:
    # This would be filtered:
    return "Let me think through this step by step: First, I analyze..."
    # Becomes: "[REDACTED - Context information filtered by MCP guardrails]"
```

### 4. Credential and Secret Prevention (`memory_leak_004`)

**Purpose**: Prevents credentials, tokens, and secrets from being exposed
**Patterns Detected**:
- API keys and tokens
- Authentication credentials
- Secret keys and passwords

```python
@mcp_guardrails()
def system_status() -> str:
    # This would be filtered:
    return f"System connected with API key: sk-abc123xyz789"
    # Becomes: "[REDACTED - Sensitive information removed by MCP guardrails]"
```

## Framework Integration Examples

### OpenAI Agents SDK

```python
from agents import Agent
from rizk.sdk.decorators import mcp_guardrails

@mcp_guardrails(on_violation="augment")
def create_secure_agent():
    """Create OpenAI agent with MCP protection."""
    
    def secure_response(message: str) -> str:
        # Process message through agent
        response = agent.process(message)
        # Automatically filtered for MCP safety
        return response
    
    agent = Agent(
        name="SecureAssistant",
        instructions="You are a helpful assistant",
        functions=[secure_response]
    )
    
    return agent
```

### LangChain Integration

```python
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor

@mcp_guardrails(on_violation="block")
def secure_langchain_tool(query: str) -> str:
    """LangChain tool with MCP protection."""
    llm = ChatOpenAI()
    response = llm.invoke(query)
    # Blocked if sensitive information detected
    return response.content

# Use in LangChain agent
agent_executor = AgentExecutor(
    agent=agent,
    tools=[secure_langchain_tool],
    verbose=True
)
```

### CrewAI Integration

```python
from crewai import Agent, Task

@mcp_guardrails(on_violation="augment")
def secure_crew_task(context: str) -> str:
    """CrewAI task with MCP protection."""
    # Process task that might leak context
    result = perform_analysis(context)
    # Automatically filtered for safety
    return result

# Use in CrewAI workflow
analyst = Agent(
    role="Data Analyst",
    goal="Analyze data securely",
    backstory="Expert analyst with security focus"
)

analysis_task = Task(
    description="Analyze the provided data",
    agent=analyst,
    expected_output="Secure analysis report"
)
```

## Advanced Configuration

### Custom Policy Sets

```python
from rizk.sdk.guardrails.types import PolicySet

# Create custom policy set
custom_policies = PolicySet([
    {
        "id": "custom_mcp_001",
        "name": "Company Secrets Protection",
        "direction": "outbound",
        "action": "block",
        "patterns": [
            "(?i)company\\s+secret",
            "(?i)internal\\s+project\\s+\\w+",
            "(?i)confidential\\s+data"
        ]
    }
])

@mcp_guardrails(
    policy_set=custom_policies,
    on_violation="block"
)
def internal_report() -> str:
    return "This contains company secret information"
    # Raises MCPGuardrailsError due to custom policy
```

### Context-Aware Filtering

```python
@mcp_guardrails(
    conversation_id="conv_123",
    organization_id="acme_corp",
    project_id="customer_service",
    on_violation="augment"
)
async def contextual_response(user_input: str) -> str:
    """Response with full context tracking."""
    response = await process_with_llm(user_input)
    return response  # Automatically filtered with context awareness
```

## Error Handling & Debugging

### Handling MCPGuardrailsError

```python
from rizk.sdk.decorators import MCPGuardrailsError

@mcp_guardrails(on_violation="block")
def risky_function(data: str) -> str:
    return f"Processing: {data} with sensitive info"

try:
    result = risky_function("user data")
    print(result)
except MCPGuardrailsError as e:
    print(f"MCP Violation Detected:")
    print(f"  Policy ID: {e.policy_id}")
    print(f"  Confidence: {e.confidence}")
    print(f"  Reason: {e}")
    
    # Log security event
    security_logger.warning(
        f"MCP guardrails blocked output",
        extra={
            "policy_id": e.policy_id,
            "confidence": e.confidence,
            "function": "risky_function"
        }
    )
```

## Performance Considerations

The `@mcp_guardrails` decorator adds minimal overhead:

- **Fast Rules Evaluation**: ~1-5ms for pattern matching
- **Policy Filtering**: ~0.5ms for direction-aware filtering
- **Content Processing**: ~2-10ms depending on content length
- **Async Support**: No blocking operations, full async/await compatibility

## Security Best Practices

### Environment-Specific Configuration

```python
# Production: Maximum security
if ENVIRONMENT == "production":
    mcp_config = {
        "on_violation": "block",
        "enabled": True
    }
# Development: Monitoring only
else:
    mcp_config = {
        "on_violation": "warn",
        "enabled": True
    }

@mcp_guardrails(**mcp_config)
def adaptive_function(): pass
```

## Monitoring & Telemetry

Key metrics automatically collected:

- **Violation Rate**: Percentage of outputs flagged by policies
- **Policy Distribution**: Which policies are triggered most frequently
- **Performance Impact**: Latency added by MCP evaluation
- **Redaction Effectiveness**: Success rate of content filtering

Access metrics through the Rizk dashboard at [app.rizk.tools](https://app.rizk.tools).

## Next Steps

- **[Using Guardrails](../guardrails/using-guardrails.md)** - Comprehensive guardrails documentation
- **[Policy Enforcement](../guardrails/policy-enforcement.md)** - Understanding policy evaluation
- **[Advanced Configuration](../advanced-config/security.md)** - Security best practices
- **[API Reference](../api-reference/decorators-api.md)** - Complete decorator API documentation

The `@mcp_guardrails` decorator provides enterprise-grade protection for LLM applications using MCP communications. By preventing memory leaks, PII exposure, and context spillage, it ensures your AI applications remain secure while maintaining functionality and performance. 

