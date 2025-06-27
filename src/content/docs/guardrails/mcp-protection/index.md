---
title: "MCP Memory Leak Protection"
description: "MCP Memory Leak Protection"
---

# MCP Memory Leak Protection

Rizk SDK provides specialized protection against memory leaks and information exposure through Model Context Protocol (MCP) communications. This protection system prevents sensitive data, previous conversation context, and internal reasoning from accidentally leaking to external agents or systems.

## Understanding MCP Security Risks

### What is Model Context Protocol (MCP)?

Model Context Protocol (MCP) is a communication standard that enables AI agents to share context, function outputs, and data with external systems. While MCP enables powerful AI integrations, it introduces specific security vulnerabilities:

**1. Memory Leaks**
```python
# DANGEROUS: Context from previous conversation bleeding through
def chat_response(user_input: str) -> str:
    return "As we discussed in our previous conversation about your financial situation..."
    # â†‘ This reveals context from a different user/session
```

**2. PII Exposure**
```python
# DANGEROUS: Personal information in function outputs
def get_user_info(user_id: str) -> str:
    user = database.get_user(user_id)
    return f"User: {user.name}, SSN: {user.ssn}, Email: {user.email}"
    # â†‘ PII being exposed through MCP communication
```

**3. Context Spillage**
```python
# DANGEROUS: Internal reasoning exposed
def analyze_request(request: str) -> str:
    return "Let me think through this step by step: First, I check the user's credit score..."
    # â†‘ Internal decision-making process exposed
```

**4. Credential Leakage**
```python
# DANGEROUS: Secrets accidentally returned
def system_status() -> str:
    return f"System operational. Using API key: {os.getenv('SECRET_KEY')}"
    # â†‘ Credentials exposed in function output
```

## MCP Protection Architecture

### Direction-Aware Policy Evaluation

Rizk's MCP protection uses **direction-aware guardrails** that differentiate between:

- **Inbound**: Messages coming INTO your application
- **Outbound**: Messages going OUT of your application (MCP risk area)

```python
# Standard guardrails protect inbound messages
@guardrails()  # Protects user input
def process_user_message(user_input: str) -> str:
    # user_input is checked for harmful content
    return generate_response(user_input)

# MCP guardrails protect outbound messages  
@mcp_guardrails()  # Protects function output
def get_user_data(user_id: str) -> str:
    data = fetch_user_data(user_id)
    # Return value is checked for PII/context leaks
    return data
```

## MCP Protection Policies

### 1. PII Outbound Prevention (`memory_leak_001`)

**Purpose**: Prevent personal information from leaking through function outputs

**Protected Information**:
- Social Security Numbers (SSN)
- Email addresses with personal identifiers
- Phone numbers in structured formats
- API keys and authentication tokens
- Personal names combined with sensitive data

**Example Protection**:
```python
@mcp_guardrails()
def customer_lookup(customer_id: str) -> str:
    customer = get_customer(customer_id)
    
    # BEFORE: Exposes PII
    # return f"Customer: John Doe, SSN: 123-45-6789, Phone: 555-0123"
    
    # AFTER: Automatically filtered
    # return "[REDACTED - Sensitive information removed by MCP guardrails]"
    
    return customer_summary
```

### 2. Context Spillage Prevention (`memory_leak_002`)

**Purpose**: Prevent previous conversation context from leaking between sessions

**Protected Context**:
- References to "previous conversation" or "earlier discussion"
- Context from different user sessions
- Cross-conversation memory references
- Session state information

**Example Protection**:
```python
@mcp_guardrails()
async def continue_conversation(message: str) -> str:
    # BEFORE: Leaks previous context
    # return "As we discussed earlier about your mortgage application..."
    
    # AFTER: Automatically filtered
    # return "[REDACTED - Context information filtered by MCP guardrails]"
    
    response = await process_message(message)
    return response
```

### 3. Chain of Thought Revelation (`memory_leak_003`)

**Purpose**: Prevent internal reasoning processes from being exposed

**Protected Reasoning**:
- Step-by-step analysis processes
- Internal decision-making logic
- Analytical reasoning chains
- Internal evaluation methods

**Example Protection**:
```python
@mcp_guardrails()
def investment_recommendation(portfolio: dict) -> str:
    # BEFORE: Exposes internal reasoning
    # return "Let me analyze this step by step: First, I calculate risk tolerance..."
    
    # AFTER: Automatically filtered
    # return "[REDACTED - Context information filtered by MCP guardrails]"
    
    recommendation = analyze_portfolio(portfolio)
    return recommendation
```

### 4. Credential and Secret Prevention (`memory_leak_004`)

**Purpose**: Prevent credentials, tokens, and secrets from being exposed

**Protected Secrets**:
- API keys and authentication tokens
- Database connection strings
- Secret keys and passwords
- Internal system identifiers

**Example Protection**:
```python
@mcp_guardrails()
def system_diagnostics() -> str:
    # BEFORE: Exposes credentials
    # return f"Database connected with key: db_secret_abc123"
    
    # AFTER: Automatically filtered
    # return "[REDACTED - Sensitive information removed by MCP guardrails]"
    
    status = check_system_status()
    return status
```

## Implementation Patterns

### Basic MCP Protection

```python
from rizk.sdk.decorators import mcp_guardrails

# Default protection with augmentation
@mcp_guardrails()
def get_customer_summary(customer_id: str) -> str:
    """Get customer summary with automatic PII protection."""
    customer = fetch_customer(customer_id)
    summary = f"Customer {customer_id}: {customer.name}, {customer.details}"
    return summary  # Automatically filtered if PII detected
```

### Strict Security Mode

```python
# High-security environments: block on any violation
@mcp_guardrails(on_violation="block")
def get_financial_report(account_id: str) -> str:
    """Financial reports with strict blocking."""
    try:
        report = generate_financial_report(account_id)
        return report
    except MCPGuardrailsError as e:
        # Log security violation
        security_logger.error(f"MCP violation: {e.policy_id}")
        # Return safe fallback
        return "Report generation blocked due to security policy"
```

### Development/Testing Mode

```python
# Development environments: warnings only
@mcp_guardrails(on_violation="warn")
def debug_user_analysis(user_id: str) -> str:
    """Debug function with warning-only mode."""
    analysis = perform_user_analysis(user_id)
    # Returns original content but logs warnings for violations
    return analysis
```

## Framework Integration

### OpenAI Agents SDK

```python
from agents import Agent, function_tool
from rizk.sdk.decorators import mcp_guardrails

@function_tool
@mcp_guardrails(on_violation="augment")
def get_user_profile(user_id: str) -> str:
    """Protected user profile function for OpenAI agent."""
    profile = database.get_user_profile(user_id)
    return format_profile(profile)  # Automatically filtered

agent = Agent(
    name="SecureCustomerAgent",
    instructions="Help customers while protecting their privacy",
    tools=[get_user_profile]
)
```

### LangChain Integration

```python
from langchain.tools import tool
from rizk.sdk.decorators import mcp_guardrails

@tool
@mcp_guardrails(on_violation="block")
def protected_database_lookup(query: str) -> str:
    """Protected database lookup tool."""
    results = database.query(query)
    return format_results(results)  # Blocked if sensitive data detected

# Use in LangChain agent
agent_executor = AgentExecutor(
    agent=agent,
    tools=[protected_database_lookup],
    verbose=True
)
```

### CrewAI Integration

```python
from crewai import Agent, Task
from rizk.sdk.decorators import mcp_guardrails

@mcp_guardrails(on_violation="augment")
def secure_research_task(topic: str) -> str:
    """Research task with MCP protection."""
    research_data = perform_research(topic)
    return summarize_research(research_data)  # Filtered for safety

researcher = Agent(
    role="Security-Aware Researcher",
    goal="Research topics while protecting sensitive information",
    backstory="Expert researcher with security training"
)

research_task = Task(
    description="Research the given topic securely",
    agent=researcher,
    expected_output="Secure research summary"
)
```

## Custom MCP Policies

### Creating Organization-Specific Policies

```python
from rizk.sdk.guardrails.types import PolicySet

# Define custom MCP policies for your organization
company_mcp_policies = PolicySet([
    {
        "id": "company_internal_001",
        "name": "Internal Project Code Protection",
        "direction": "outbound",
        "action": "block",
        "domains": ["internal", "project", "code"],
        "patterns": [
            "(?i)project\\s+[A-Z]\\d{3,}",  # Project codes like "Project X123"
            "(?i)internal\\s+reference\\s+\\w+",
            "(?i)confidential\\s+(?:data|information)"
        ],
        "guidelines": [
            "Never expose internal project codes",
            "Protect confidential company information",
            "Use generic references instead of specific project names"
        ]
    }
])

# Apply custom policies
@mcp_guardrails(
    policy_set=company_mcp_policies,
    on_violation="block"
)
def internal_system_report() -> str:
    """Generate internal report with company-specific protection."""
    return generate_report()
```

## Performance Impact

### Benchmarking MCP Protection

```python
import time
from rizk.sdk.decorators import mcp_guardrails

# Baseline function (no protection)
def baseline_function(data: str) -> str:
    return f"Processed: {data}"

# MCP protected function
@mcp_guardrails()
def protected_function(data: str) -> str:
    return f"Processed: {data}"

# Performance comparison
def benchmark_mcp_overhead():
    test_data = "Sample customer data for processing"
    iterations = 1000
    
    # Baseline timing
    start = time.time()
    for _ in range(iterations):
        baseline_function(test_data)
    baseline_time = time.time() - start
    
    # Protected timing
    start = time.time()
    for _ in range(iterations):
        protected_function(test_data)
    protected_time = time.time() - start
    
    overhead = ((protected_time - baseline_time) / baseline_time) * 100
    print(f"MCP protection overhead: {overhead:.2f}%")
    # Typical overhead: 5-15% for most content sizes

benchmark_mcp_overhead()
```

## Best Practices

### 1. Layered Security Approach

```python
# Combine multiple protection strategies
@mcp_guardrails(on_violation="augment")  # Primary MCP protection
@guardrails(check_input=True)           # Secondary input protection
def comprehensive_protection(user_input: str) -> str:
    """Function with layered security protection."""
    # Input checked by @guardrails
    # Output checked by @mcp_guardrails
    return process_user_request(user_input)
```

### 2. Graceful Degradation

```python
@mcp_guardrails(on_violation="augment")
def fault_tolerant_function(data: str) -> str:
    """Function that degrades gracefully on policy violations."""
    try:
        result = process_sensitive_data(data)
        
        # Check if content was filtered
        if "[REDACTED" in result:
            # Provide alternative safe response
            return generate_safe_alternative(data)
        
        return result
        
    except Exception as e:
        # Always return safe fallback
        return "Unable to process request due to security constraints"
```

### 3. Testing MCP Protection

```python
import pytest
from rizk.sdk.decorators import mcp_guardrails, MCPGuardrailsError

class TestMCPProtection:
    """Test suite for MCP protection functionality."""
    
    def test_pii_protection(self):
        """Test that PII is properly blocked."""
        
        @mcp_guardrails(on_violation="block")
        def leak_pii():
            return "Customer SSN: 123-45-6789"
        
        with pytest.raises(MCPGuardrailsError):
            leak_pii()
    
    def test_context_spillage_protection(self):
        """Test that context spillage is filtered."""
        
        @mcp_guardrails(on_violation="augment")
        def leak_context():
            return "As we discussed in our previous conversation..."
        
        result = leak_context()
        assert "[REDACTED" in result
        assert "Context information filtered" in result
```

## Next Steps

- **[MCP Guardrails Decorator](../decorators/mcp-guardrails.md)** - Detailed decorator usage guide
- **[Policy Enforcement](./policy-enforcement.md)** - Understanding policy evaluation mechanisms
- **[Configuration](./configuration.md)** - Advanced guardrails configuration
- **[Security Best Practices](../advanced-config/security.md)** - Enterprise security guidelines

MCP protection is essential for any AI application that communicates with external systems. By implementing these protections, you ensure that sensitive information, previous conversation context, and internal reasoning remain secure while maintaining the functionality and performance of your LLM applications.


