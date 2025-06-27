---
title: "@guardrails Decorator"
description: "@guardrails Decorator"
---

# @guardrails Decorator

The `@guardrails` decorator provides automatic policy enforcement and content safety for any function, workflow, or agent. It integrates with Rizk's multi-layer guardrails system to ensure compliance, safety, and governance across your AI applications.

## Overview

**Guardrails** represent automated policy enforcement mechanisms that can validate inputs, monitor outputs, and ensure compliance with organizational policies and safety requirements. The `@guardrails` decorator automatically applies these protections to functions while providing detailed monitoring and reporting.

## Basic Usage

```python
from rizk.sdk import Rizk
from rizk.sdk.decorators import guardrails, workflow

# Initialize Rizk
rizk = Rizk.init(app_name="GuardrailsApp", enabled=True)

@guardrails()
@workflow(name="content_generation")
def generate_content(prompt: str, content_type: str = "blog") -> str:
    """Generate content with automatic policy enforcement."""
    
    # Simulate content generation
    if content_type == "blog":
        content = f"Blog Post: {prompt}\n\nThis is a sample blog post about {prompt}. It includes relevant information and insights."
    elif content_type == "email":
        content = f"Subject: About {prompt}\n\nDear recipient,\n\nI wanted to share some information about {prompt}..."
    else:
        content = f"Content about {prompt}: This is general content covering the topic."
    
    return content

# Usage - guardrails automatically applied
safe_content = generate_content("artificial intelligence benefits", "blog")
print(f"Generated content: {safe_content}")

# This would be blocked by guardrails
try:
    unsafe_content = generate_content("how to hack systems", "blog")
except Exception as e:
    print(f"Blocked by guardrails: {e}")
```

## Parameters Reference

### Core Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `input_validation` | `bool` | `True` | Enable input validation and filtering |
| `output_validation` | `bool` | `True` | Enable output validation and filtering |
| `policy_enforcement` | `str` | `"moderate"` | Enforcement level: `"strict"`, `"moderate"`, `"permissive"` |
| `block_on_violation` | `bool` | `True` | Block execution when policy violations detected |
| `custom_policies` | `list` | `None` | List of custom policy names to apply |

### Advanced Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `audit_logging` | `bool` | Enable detailed audit logging for compliance |
| `content_categories` | `list` | Specific content categories to monitor |
| `risk_tolerance` | `str` | Risk tolerance level: `"low"`, `"medium"`, `"high"` |
| `fallback_response` | `str` | Custom response when content is blocked |

## Enforcement Levels

### Strict Enforcement

```python
@guardrails(
    policy_enforcement="strict",
    block_on_violation=True,
    risk_tolerance="low"
)
@workflow(name="financial_advice")
def provide_financial_advice(query: str) -> str:
    """Provide financial advice with strict policy enforcement."""
    
    # Strict enforcement blocks any potentially risky content
    advice = f"Based on your query about {query}, here are some general considerations..."
    
    # Additional validation for financial content
    if "investment" in query.lower():
        advice += "\n\nDisclaimer: This is not professional financial advice. Consult a financial advisor."
    
    return advice

# Example usage
advice = provide_financial_advice("retirement planning strategies")
print(advice)
```

### Moderate Enforcement

```python
@guardrails(
    policy_enforcement="moderate",
    content_categories=["safety", "compliance"],
    audit_logging=True
)
@workflow(name="customer_support")
def handle_customer_query(query: str, customer_id: str) -> str:
    """Handle customer queries with moderate policy enforcement."""
    
    # Moderate enforcement allows most content with warnings
    response = f"Thank you for your inquiry about {query}. "
    
    if "refund" in query.lower():
        response += "I'll help you with your refund request. Let me check your account details."
    elif "complaint" in query.lower():
        response += "I understand your concern. Let me escalate this to our resolution team."
    else:
        response += "I'll be happy to assist you with your request."
    
    return response

# Example usage
response = handle_customer_query("I want a refund for my order", "customer_123")
print(response)
```

### Permissive Enforcement

```python
@guardrails(
    policy_enforcement="permissive",
    input_validation=True,
    output_validation=False,  # Only validate inputs
    audit_logging=True
)
@workflow(name="creative_writing")
def generate_creative_content(theme: str, style: str = "narrative") -> str:
    """Generate creative content with permissive enforcement."""
    
    # Permissive enforcement allows creative freedom while logging
    if style == "narrative":
        content = f"Once upon a time, in a world where {theme} was the central focus..."
    elif style == "poem":
        content = f"Roses are red,\nViolets are blue,\n{theme} is wonderful,\nAnd so are you."
    else:
        content = f"Creative exploration of {theme} in {style} style."
    
    return content

# Example usage
creative_content = generate_creative_content("space exploration", "poem")
print(creative_content)
```

## Custom Policies

```python
@guardrails(
    custom_policies=["company_confidentiality", "data_privacy", "brand_guidelines"],
    policy_enforcement="strict",
    fallback_response="Content blocked due to policy violation"
)
@workflow(name="marketing_content")
def create_marketing_content(product: str, target_audience: str) -> str:
    """Create marketing content with custom policy enforcement."""
    
    # Custom policies defined in your organization's policy files
    content = f"Introducing {product} - perfect for {target_audience}!"
    
    # Add compliance elements
    content += "\n\n*Terms and conditions apply. See website for details."
    
    return content

# Custom policy configuration (in your policies directory)
"""
# company_confidentiality.yaml
name: "Company Confidentiality"
rules:
  - pattern: "internal.*project"
    action: "block"
    reason: "Internal project information should not be shared externally"
  - pattern: "proprietary.*technology"
    action: "warn"
    reason: "Be careful when discussing proprietary technology"

# data_privacy.yaml  
name: "Data Privacy"
rules:
  - pattern: "\\b\\d{3}-\\d{2}-\\d{4}\\b"  # SSN pattern
    action: "block"
    reason: "Social Security Numbers are not allowed"
  - pattern: "\\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Z|a-z]{2,}\\b"  # Email pattern
    action: "redact"
    reason: "Email addresses should be redacted"
"""

# Example usage
marketing_content = create_marketing_content("CloudSecure Pro", "enterprise customers")
print(marketing_content)
```

## Real-time Monitoring

```python
@guardrails(
    audit_logging=True,
    policy_enforcement="moderate"
)
@workflow(name="chat_moderation")
def moderate_chat_message(message: str, user_id: str, channel: str) -> dict:
    """Moderate chat messages with real-time monitoring."""
    
    # Guardrails automatically check the message
    moderation_result = {
        "original_message": message,
        "user_id": user_id,
        "channel": channel,
        "timestamp": datetime.now().isoformat(),
        "status": "approved",  # Will be updated by guardrails
        "modified_message": message
    }
    
    # Additional custom moderation logic
    if len(message) > 500:
        moderation_result["warnings"] = ["Message is quite long"]
    
    if message.count("!") > 5:
        moderation_result["warnings"] = moderation_result.get("warnings", []) + ["Excessive exclamation marks"]
    
    return moderation_result

# Example usage
chat_result = moderate_chat_message("Hello everyone! How are you doing today?", "user_456", "general")
print(f"Moderation result: {chat_result}")
```

## Integration with Agents and Tools

```python
from rizk.sdk.decorators import agent, tool, guardrails

@tool(name="web_search")
@guardrails(
    content_categories=["misinformation", "harmful_content"],
    policy_enforcement="strict"
)
def search_web(query: str) -> str:
    """Search the web with content safety guardrails."""
    
    # Mock web search with safety checks
    search_results = f"Search results for '{query}':\n"
    search_results += "1. Reliable source about the topic\n"
    search_results += "2. Educational content from trusted institution\n"
    search_results += "3. Recent news from reputable outlet"
    
    return search_results

@agent(name="research_assistant")
@guardrails(
    policy_enforcement="moderate",
    audit_logging=True
)
def create_research_agent() -> dict:
    """Create a research agent with guardrails."""
    
    agent_config = {
        "name": "SafeResearchAssistant",
        "role": "Research Assistant",
        "tools": [search_web],
        "safety_features": ["content_filtering", "source_verification"],
        "compliance_level": "enterprise"
    }
    
    return agent_config

# Example usage
research_agent = create_research_agent()
search_result = search_web("climate change research")
print(f"Safe search result: {search_result}")
```

## Async Guardrails

```python
import asyncio

@guardrails(
    policy_enforcement="strict",
    audit_logging=True
)
async def process_user_content_async(content_batch: list) -> list:
    """Process multiple content items asynchronously with guardrails."""
    
    async def process_single_content(content: str) -> dict:
        # Simulate async content processing
        await asyncio.sleep(0.01)
        
        processed_content = {
            "original": content,
            "processed": f"Processed: {content}",
            "safety_score": 0.95,  # Mock safety score
            "timestamp": datetime.now().isoformat()
        }
        
        return processed_content
    
    # Process all content items concurrently
    tasks = [process_single_content(content) for content in content_batch]
    results = await asyncio.gather(*tasks)
    
    return results

# Example usage
async def run_async_processing():
    content_batch = [
        "Welcome to our platform!",
        "How can I help you today?",
        "Thank you for your feedback."
    ]
    
    results = await process_user_content_async(content_batch)
    for result in results:
        print(f"Processed: {result['processed']}")

# asyncio.run(run_async_processing())
```

## Error Handling and Fallbacks

```python
@guardrails(
    policy_enforcement="moderate",
    fallback_response="Content cannot be processed due to safety concerns",
    block_on_violation=False  # Don't block, just warn
)
@workflow(name="content_processor")
def process_content_with_fallbacks(content: str, content_type: str) -> dict:
    """Process content with comprehensive error handling."""
    
    try:
        # Primary content processing
        if content_type == "article":
            processed = f"Article: {content}\n\nThis article provides insights into the topic."
        elif content_type == "summary":
            processed = f"Summary: {content[:100]}..."
        else:
            processed = f"Content: {content}"
        
        return {
            "status": "success",
            "processed_content": processed,
            "content_type": content_type,
            "safety_checks": "passed"
        }
    
    except Exception as e:
        # Fallback processing
        return {
            "status": "fallback",
            "processed_content": "Generic safe content response",
            "content_type": content_type,
            "safety_checks": "failed",
            "error": str(e)
        }

# Example usage
result = process_content_with_fallbacks("Technology trends in 2024", "article")
print(f"Processing result: {result}")
```

## Monitoring and Analytics

```python
@guardrails(
    audit_logging=True,
    policy_enforcement="moderate"
)
@workflow(name="analytics_content_processor")
def process_with_analytics(content: str) -> dict:
    """Process content with detailed analytics and monitoring."""
    
    # Simulate content processing with metrics
    processing_start = time.time()
    
    # Content analysis
    word_count = len(content.split())
    char_count = len(content)
    
    # Safety analysis (mocked)
    safety_metrics = {
        "toxicity_score": 0.05,  # Low toxicity
        "bias_score": 0.10,      # Low bias
        "factual_score": 0.90,   # High factual accuracy
        "readability_score": 0.85 # Good readability
    }
    
    processing_time = time.time() - processing_start
    
    result = {
        "content": content,
        "metrics": {
            "word_count": word_count,
            "char_count": char_count,
            "processing_time": processing_time,
            "safety_metrics": safety_metrics
        },
        "guardrails_status": "passed",
        "timestamp": datetime.now().isoformat()
    }
    
    return result

# Example usage
analytics_result = process_with_analytics("Artificial intelligence is transforming industries.")
print(f"Analytics result: {analytics_result}")
```

## Testing Guardrails

```python
import pytest
from unittest.mock import patch

def test_guardrails_input_validation():
    """Test input validation functionality."""
    
    @guardrails(policy_enforcement="strict")
    def validate_input_function(text: str) -> str:
        return f"Processed: {text}"
    
    # Test safe input
    safe_result = validate_input_function("Hello, how are you?")
    assert "Processed:" in safe_result
    
    # Test potentially unsafe input (would be caught by actual guardrails)
    # In real implementation, this would be blocked
    result = validate_input_function("This is a test message")
    assert result is not None

def test_guardrails_enforcement_levels():
    """Test different enforcement levels."""
    
    @guardrails(policy_enforcement="permissive")
    def permissive_function(text: str) -> str:
        return f"Permissive: {text}"
    
    @guardrails(policy_enforcement="strict")
    def strict_function(text: str) -> str:
        return f"Strict: {text}"
    
    test_text = "Sample content for testing"
    
    permissive_result = permissive_function(test_text)
    strict_result = strict_function(test_text)
    
    assert "Permissive:" in permissive_result
    assert "Strict:" in strict_result

def test_guardrails_custom_policies():
    """Test custom policy application."""
    
    @guardrails(
        custom_policies=["test_policy"],
        policy_enforcement="moderate"
    )
    def custom_policy_function(text: str) -> str:
        return f"Custom policy applied: {text}"
    
    result = custom_policy_function("Test content")
    assert "Custom policy applied:" in result

@patch('rizk.sdk.guardrails.GuardrailsEngine')
def test_guardrails_monitoring(mock_engine):
    """Test guardrails monitoring and logging."""
    
    mock_engine.return_value.evaluate.return_value = {
        "allowed": True,
        "confidence": 0.95,
        "violations": []
    }
    
    @guardrails(audit_logging=True)
    def monitored_function(text: str) -> str:
        return f"Monitored: {text}"
    
    result = monitored_function("Test monitoring")
    assert "Monitored:" in result
```

## Best Practices

### 1. **Appropriate Enforcement Levels**
```python
# High-risk contexts: Use strict enforcement
@guardrails(policy_enforcement="strict")
def handle_financial_data(data): pass

# Creative contexts: Use permissive enforcement
@guardrails(policy_enforcement="permissive", audit_logging=True)
def generate_creative_content(prompt): pass

# General business: Use moderate enforcement
@guardrails(policy_enforcement="moderate")
def process_business_content(content): pass
```

### 2. **Custom Policy Organization**
```python
# Organize policies by domain
@guardrails(custom_policies=[
    "healthcare_compliance",  # HIPAA, medical guidelines
    "data_privacy",          # GDPR, CCPA compliance
    "company_brand"          # Brand guidelines, tone
])
def healthcare_content_processor(content): pass
```

### 3. **Monitoring and Compliance**
```python
# Enable comprehensive monitoring for compliance
@guardrails(
    audit_logging=True,
    policy_enforcement="strict",
    content_categories=["compliance", "safety", "privacy"]
)
def compliance_critical_function(data): pass
```

### 4. **Graceful Degradation**
```python
# Handle policy violations gracefully
@guardrails(
    block_on_violation=False,
    fallback_response="Content modified for safety",
    policy_enforcement="moderate"
)
def user_facing_content_processor(content): pass
```

## Related Documentation

- **[Guardrails Overview](../core-concepts/guardrails-overview.md)** - Comprehensive guardrails system
- **[Policy Configuration](../05-configuration/policies.md)** - Setting up custom policies
- **[Compliance](../09-compliance/overview.md)** - Enterprise compliance features
- **[Monitoring](../08-monitoring/overview.md)** - Monitoring guardrails performance

---

The `@guardrails` decorator provides automatic policy enforcement and content safety for any function, ensuring compliance and governance while maintaining detailed audit trails and monitoring capabilities.

