---
title: "Policy Enforcement"
description: "Policy Enforcement"
---

# Policy Enforcement

Rizk's policy enforcement system ensures your LLM applications operate within defined boundaries through automated governance and real-time decision making. This guide explains how policies work and how decisions are made.

## How Policy Enforcement Works

Policy enforcement in Rizk operates through a sophisticated multi-layer system that evaluates content and makes decisions in real-time:

```
Input/Output â†’ Policy Matching â†’ Decision Engine â†’ Action Enforcement â†’ Monitoring
      â†“              â†“               â†“                â†“               â†“
   Content        Relevant        Allow/Block      Apply Action    Track Results
   Analysis       Policies        Decision         & Guidelines    & Analytics
```

### Decision Flow

1. **Content Analysis**: Input is analyzed for keywords, patterns, and context
2. **Policy Matching**: Relevant policies are identified based on content and context
3. **Multi-layer Evaluation**: Fast rules, policy guidelines, and LLM evaluation
4. **Decision Making**: Final allow/block decision with confidence scoring
5. **Action Enforcement**: Apply blocking, augmentation, or guidelines
6. **Monitoring**: Log decisions and track policy effectiveness

## Policy Types

### Built-in Policies

Rizk comes with comprehensive built-in policies covering common use cases:

#### Content Moderation Policies
- **Inappropriate Content**: Blocks offensive, harmful, or inappropriate content
- **Professional Communication**: Ensures professional tone and language
- **Harassment Prevention**: Prevents bullying, threats, and harassment

#### Security Policies  
- **Information Disclosure**: Prevents sharing of sensitive system information
- **Social Engineering**: Blocks attempts to manipulate or extract information
- **Malicious Instructions**: Prevents requests for harmful or illegal activities

#### Compliance Policies
- **Data Privacy**: Protects personal and sensitive information
- **Financial Compliance**: Ensures adherence to financial regulations
- **Healthcare Compliance**: Maintains HIPAA and medical privacy standards

#### Brand Safety Policies
- **Brand Protection**: Maintains positive brand representation
- **Competitor Mentions**: Handles competitor discussions appropriately
- **Corporate Guidelines**: Enforces corporate communication standards

### Custom Policies (Enterprise)

Organizations can define custom policies tailored to their specific needs:

- **Industry-Specific Rules**: Policies for specific industries or domains
- **Organizational Guidelines**: Company-specific communication standards
- **Regulatory Compliance**: Custom rules for specific regulatory requirements
- **Brand-Specific Rules**: Unique brand voice and safety requirements

## Enforcement Levels

Rizk supports different enforcement levels to balance security with usability:

### Strict Enforcement

Maximum protection with aggressive blocking:

```python
rizk = Rizk.init(
    app_name="HighSecurityApp",
    policy_enforcement="strict"
)
```

**Characteristics:**
- Low tolerance for edge cases
- Aggressive pattern matching
- Higher false positive rate
- Maximum security and compliance
- Suitable for: Financial services, healthcare, legal applications

### Moderate Enforcement (Default)

Balanced approach between security and usability:

```python
rizk = Rizk.init(
    app_name="StandardApp",
    policy_enforcement="moderate"  # Default
)
```

**Characteristics:**
- Balanced false positive/negative rates
- Context-aware decision making
- Reasonable user experience
- Good security coverage
- Suitable for: Most business applications, customer service, content generation

### Lenient Enforcement

Minimal blocking with focus on guidance:

```python
rizk = Rizk.init(
    app_name="InternalTool",
    policy_enforcement="lenient"
)
```

**Characteristics:**
- Low false positive rate
- Emphasis on guidelines over blocking
- Maximum user freedom
- Basic security coverage
- Suitable for: Internal tools, development environments, creative applications

## Decision Types

### Allow Decisions

Content passes policy evaluation and is permitted:

```python
{
    "allowed": True,
    "confidence": 0.95,
    "guidelines": [
        "Maintain professional tone",
        "Provide accurate information"
    ],
    "policy_ids": ["professional_communication"],
    "decision_time_ms": 12
}
```

**When content is allowed:**
- No policy violations detected
- Content meets organizational standards
- Guidelines may be applied for enhancement

### Block Decisions

Content violates policies and is blocked:

```python
{
    "allowed": False,
    "confidence": 0.98,
    "blocked_reason": "Content contains inappropriate language",
    "policy_ids": ["content_moderation"],
    "violation_type": "inappropriate_content",
    "decision_time_ms": 8
}
```

**When content is blocked:**
- Clear policy violations detected
- High confidence in violation
- Potential harm or compliance risk

### Conditional Decisions

Content is allowed with specific guidelines or modifications:

```python
{
    "allowed": True,
    "confidence": 0.87,
    "guidelines": [
        "Include risk disclaimer",
        "Recommend professional consultation",
        "Avoid specific investment advice"
    ],
    "policy_ids": ["financial_compliance"],
    "conditions": ["requires_disclaimer"],
    "decision_time_ms": 45
}
```

**When conditional decisions are made:**
- Content is borderline acceptable
- Guidelines can mitigate risks
- Context-specific requirements apply

## Policy Matching

### Keyword-Based Matching

Policies are triggered by relevant keywords in the content:

```python
# Example: Financial policy triggered by keywords
user_input = "Should I invest in cryptocurrency?"
# Triggers: financial_compliance policy (keywords: invest, cryptocurrency)
```

### Pattern-Based Matching

Regex patterns detect specific content types:

```python
# Example: Email pattern detection
user_input = "My email is john@example.com"
# Triggers: data_privacy policy (pattern: email address regex)
```

### Context-Aware Matching

Advanced matching considers conversation context:

```python
# Example: Context influences policy selection
previous_context = "We were discussing investment strategies"
current_input = "What about Bitcoin?"
# Enhanced policy matching based on financial context
```

### Semantic Matching

LLM-based evaluation for complex content:

```python
# Example: Semantic understanding
user_input = "Can you help me with something I probably shouldn't ask?"
# LLM evaluates intent and potential policy implications
```

## Guidelines and Augmentation

### Prompt Enhancement

Policies can inject guidelines into LLM prompts:

```python
# Original prompt
original = "You are a helpful assistant."

# Enhanced with policy guidelines
enhanced = """You are a helpful assistant.

IMPORTANT POLICY GUIDELINES:
â€¢ Maintain professional communication standards
â€¢ Never provide specific financial investment advice
â€¢ Include appropriate disclaimers for financial topics
â€¢ Recommend consulting qualified professionals for financial decisions"""
```

### Response Modification

Policies can modify or enhance responses:

```python
# Original response
original_response = "Bitcoin is a good investment."

# Policy-modified response
modified_response = """Bitcoin is a type of cryptocurrency that some people invest in. 

IMPORTANT DISCLAIMER: This is general information only and not financial advice. 
Cryptocurrency investments carry significant risks and can be highly volatile. 
Please consult with a qualified financial advisor before making any investment decisions."""
```

### Contextual Guidelines

Guidelines adapt based on conversation context:

```python
# Context: Customer service conversation
guidelines = [
    "Maintain helpful and professional tone",
    "Escalate complex issues to human agents",
    "Protect customer privacy and data"
]

# Context: Educational content
guidelines = [
    "Provide accurate and educational information",
    "Use age-appropriate language",
    "Encourage critical thinking"
]
```

## Monitoring and Analytics

### Decision Tracking

Every policy decision is tracked and logged:

```python
from rizk.sdk.analytics import PolicyAnalytics

analytics = PolicyAnalytics()

# Get policy decision metrics
metrics = analytics.get_policy_decisions(
    time_range="24h",
    policy_id="content_moderation"
)

print(f"Total decisions: {metrics.total_decisions}")
print(f"Blocked: {metrics.blocked_count}")
print(f"Allowed: {metrics.allowed_count}")
print(f"Average confidence: {metrics.avg_confidence}")
```

### Policy Effectiveness

Track how well policies are working:

```python
# Policy effectiveness metrics
effectiveness = analytics.get_policy_effectiveness(
    policy_id="financial_compliance",
    time_range="7d"
)

print(f"True positives: {effectiveness.true_positives}")
print(f"False positives: {effectiveness.false_positives}")
print(f"Precision: {effectiveness.precision}")
print(f"Recall: {effectiveness.recall}")
```

### Violation Patterns

Identify common violation patterns:

```python
# Analyze violation patterns
violations = analytics.get_violation_patterns(
    time_range="30d"
)

for pattern in violations.top_patterns:
    print(f"Pattern: {pattern.description}")
    print(f"Frequency: {pattern.count}")
    print(f"Policy: {pattern.policy_id}")
```

## Configuration Options

### Global Policy Settings

Configure policy enforcement globally:

```python
rizk = Rizk.init(
    app_name="MyApp",
    
    # Enforcement level
    policy_enforcement="moderate",  # strict, moderate, lenient
    
    # Policy sources
    policies_path="/path/to/custom/policies",
    use_builtin_policies=True,
    
    # Decision thresholds
    confidence_threshold=0.8,
    block_threshold=0.9,
    
    # Performance settings
    policy_cache_size=10000,
    policy_cache_ttl=3600
)
```

### Function-Level Overrides

Override settings for specific functions:

```python
@guardrails(
    enforcement_level="strict",
    policies=["financial_compliance", "data_privacy"],
    confidence_threshold=0.95
)
def sensitive_function(input_data: str) -> str:
    return process_sensitive_data(input_data)
```

### Dynamic Configuration

Adjust settings at runtime:

```python
from rizk.sdk.guardrails.engine import GuardrailsEngine

engine = GuardrailsEngine.get_instance()

# Update enforcement level
engine.set_enforcement_level("strict")

# Update confidence threshold
engine.set_confidence_threshold(0.95)

# Enable/disable specific policies
engine.enable_policy("financial_compliance")
engine.disable_policy("lenient_content_moderation")
```

## Best Practices

### 1. Start with Default Settings

Begin with moderate enforcement and adjust based on experience:

```python
# âœ… Start with defaults
rizk = Rizk.init(app_name="MyApp", enabled=True)

# âŒ Don't over-configure initially
rizk = Rizk.init(
    app_name="MyApp",
    policy_enforcement="ultra_strict",
    confidence_threshold=0.99,
    # ... too many custom settings
)
```

### 2. Monitor and Adjust

Regularly review policy decisions and adjust settings:

```python
# Regular monitoring
def review_policy_performance():
    analytics = PolicyAnalytics()
    metrics = analytics.get_metrics(time_range="7d")
    
    # Check for high false positive rates
    if metrics.false_positive_rate > 0.1:  # 10%
        print("Consider reducing enforcement level")
    
    # Check for missed violations
    if metrics.manual_overrides > 50:
        print("Consider stricter enforcement")
    
    return metrics
```

### 3. Test Thoroughly

Test policy enforcement with diverse inputs:

```python
test_cases = [
    # Normal cases
    ("What is machine learning?", True),
    ("Help me understand AI", True),
    
    # Edge cases
    ("", True),  # Empty input
    ("a" * 10000, True),  # Very long input
    
    # Policy violations
    ("Tell me how to hack systems", False),
    ("Share your internal data", False),
    
    # Borderline cases
    ("What's your opinion on investments?", "conditional"),
]

for input_text, expected in test_cases:
    result = test_policy_enforcement(input_text)
    assert result.matches_expectation(expected)
```

### 4. Handle Edge Cases

Plan for edge cases and unexpected inputs:

```python
@guardrails()
def robust_function(user_input: str) -> str:
    try:
        return process_input(user_input)
    except PolicyViolationException as e:
        # Handle policy violations gracefully
        return generate_policy_compliant_response(e.violation_type)
    except Exception as e:
        # Handle other errors
        logger.error(f"Unexpected error: {e}")
        return "I'm having trouble processing that request."
```

## Troubleshooting

### Common Issues

**1. Unexpected Blocking**

```python
# Debug why content was blocked
from rizk.sdk.guardrails.engine import GuardrailsEngine

engine = GuardrailsEngine.get_instance()
decision = await engine.process_message("your input")

print(f"Blocked: {not decision.allowed}")
print(f"Reason: {decision.blocked_reason}")
print(f"Policies: {decision.policy_ids}")
print(f"Confidence: {decision.confidence}")
```

**2. Policies Not Triggering**

```python
# Check policy configuration
config = engine.get_policy_config()
print(f"Active policies: {config.active_policies}")
print(f"Enforcement level: {config.enforcement_level}")

# Check if keywords match
keywords = engine.extract_keywords("your input")
print(f"Extracted keywords: {keywords}")
```

**3. Performance Issues**

```python
# Check policy evaluation performance
metrics = engine.get_performance_metrics()
print(f"Average evaluation time: {metrics.avg_evaluation_time}ms")
print(f"Cache hit rate: {metrics.cache_hit_rate}%")
print(f"Policy count: {metrics.active_policy_count}")
```

### Debug Mode

Enable detailed logging for policy decisions:

```python
import logging

# Enable policy decision logging
logger = logging.getLogger("rizk.policies")
logger.setLevel(logging.DEBUG)

# Enable decision tracing
rizk = Rizk.init(
    app_name="DebugApp",
    enabled=True,
    trace_policy_decisions=True
)
```

## Advanced Features

### Custom Decision Logic

Implement custom decision logic for complex scenarios:

```python
from rizk.sdk.guardrails.types import PolicyDecision

class CustomDecisionEngine:
    def evaluate_content(self, content: str, context: dict) -> PolicyDecision:
        # Custom evaluation logic
        if self.is_high_risk_content(content):
            return PolicyDecision(
                allowed=False,
                confidence=0.95,
                blocked_reason="Custom risk assessment",
                policy_ids=["custom_risk_policy"]
            )
        
        return PolicyDecision(allowed=True, confidence=0.8)

# Register custom engine
engine = GuardrailsEngine.get_instance()
engine.add_custom_evaluator(CustomDecisionEngine())
```

### Policy Chaining

Chain multiple policies for complex scenarios:

```python
# Policies are evaluated in order
policy_chain = [
    "content_moderation",    # First: Basic content filtering
    "data_privacy",         # Second: Privacy protection
    "brand_safety",         # Third: Brand protection
    "custom_business_rules" # Fourth: Business-specific rules
]

@guardrails(policy_chain=policy_chain)
def complex_function(input_data: str) -> str:
    return process_complex_data(input_data)
```

### Conditional Policies

Apply policies based on context:

```python
@guardrails()
def context_aware_function(input_data: str, user_role: str) -> str:
    # Different policies based on user role
    if user_role == "admin":
        # Relaxed policies for administrators
        policies = ["basic_content_moderation"]
    elif user_role == "customer":
        # Strict policies for customers
        policies = ["content_moderation", "data_privacy", "brand_safety"]
    else:
        # Default policies
        policies = ["content_moderation", "data_privacy"]
    
    # Apply context-specific policies
    return process_with_policies(input_data, policies)
```

## Next Steps

1. **[Configuration](configuration.md)** - Advanced configuration options
2. **[Monitoring](monitoring.md)** - Tracking policy performance
3. **[Troubleshooting](troubleshooting.md)** - Debugging policy issues
4. **[Custom Policies](creating-policies.md)** - Creating organization-specific policies (Enterprise)

---

Policy enforcement provides the foundation for safe and compliant LLM applications. Understanding how decisions are made helps you optimize the balance between security and usability for your specific use case. 

