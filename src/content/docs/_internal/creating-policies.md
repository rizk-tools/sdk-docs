---
title: "Creating Custom Guardrail Policies"
description: "Complete guide for creating custom guardrail policies for the Rizk SDK, including policy structure, examples, and testing."
---

# Creating Custom Guardrail Policies

This guide walks you through creating custom guardrail policies for the Rizk SDK. Policies define the rules and guidelines that govern your LLM applications, ensuring compliance, safety, and alignment with your organization's requirements.

## Overview

Rizk's guardrail system uses a multi-layer approach:
1. **Fast Rules** - Regex-based pattern matching for immediate blocking
2. **Policy Augmentation** - Context-aware prompt enhancement with guidelines  
3. **LLM Fallback** - LLM-based evaluation for complex cases

Custom policies can leverage all three layers to provide comprehensive governance.

## Policy Structure

Policies are defined in YAML format with the following structure:

```yaml
version: "1.0.0"
updated_at: "2025-01-01"
policies:
  - id: "unique_policy_id"
    name: "Human-Readable Policy Name"
    domains: ["keyword1", "keyword2", "keyword3"]
    description: "Detailed description of what this policy does"
    action: "block" | "allow"
    guidelines: 
      - "Guideline 1 for LLM behavior"
      - "Guideline 2 for LLM behavior"
    patterns:
      - "regex_pattern_1"
      - "regex_pattern_2"
```

### Required Fields

- **`id`**: Unique identifier for the policy (string)
- **`name`**: Human-readable name for the policy
- **`description`**: Detailed explanation of the policy's purpose
- **`action`**: Either `"block"` (prevent execution) or `"allow"` (permit with guidelines)

### Optional Fields

- **`domains`**: Keywords that trigger this policy (used for fast matching)
- **`guidelines`**: Instructions for the LLM on how to behave
- **`patterns`**: Regex patterns for pattern-based detection

## Step-by-Step Policy Creation

### Step 1: Define Your Policy Requirements

Before writing the policy, clearly define:

1. **What behavior you want to control**
2. **When the policy should trigger**
3. **What action to take (block vs. guide)**
4. **What guidance to provide to the LLM**

**Example**: Create a policy to prevent financial advice while allowing general financial education.

### Step 2: Create Your Policy Directory

Create a custom policies directory and file:

```bash
# Create directory for custom policies
mkdir -p policies/custom

# Create your policy file
touch policies/custom/financial_guidance.yaml
```

### Step 3: Write Your Policy Definition

```yaml
version: "1.0.0"
updated_at: "2025-01-01"
policies:
  - id: "financial_advice_prevention"
    name: "Financial Advice Prevention Policy"
    domains: ["investment", "stock", "crypto", "trading", "financial", "advice", "money", "portfolio"]
    description: "Prevents providing specific financial advice while allowing general financial education"
    action: "allow"
    guidelines:
      - "Provide general financial education only, never specific investment advice"
      - "Always recommend consulting with qualified financial advisors for investment decisions"
      - "Do not suggest specific stocks, cryptocurrencies, or financial products to buy or sell"
      - "Emphasize that all investments carry risk and past performance doesn't guarantee future results"
      - "Clearly state that your response is for educational purposes only and not financial advice"
    patterns:
      - "(?i)you should (buy|sell|invest in|purchase) (.*(?:stock|crypto|bitcoin|shares|bonds))"
      - "(?i)guaranteed (profit|returns|gains) of \\d+%"
      - "(?i)this (?:investment|stock|crypto) (?:will|guaranteed to) (?:double|triple|make you rich)"
      - "(?i)(?:definitely|certainly|sure to) (?:go up|increase|moon|crash)"
      - "(?i)(?:get rich quick|easy money|passive income) (?:scheme|strategy|method)"
```

### Step 4: Configure Rizk to Use Your Policies

Set the environment variable to point to your custom policies:

```bash
# Point to your custom policies directory
export RIZK_POLICIES_PATH="/path/to/your/policies/custom"
```

Or configure it programmatically:

```python
from rizk.sdk import Rizk

rizk = Rizk.init(
    app_name="MyApp",
    api_key="your-api-key",
    policies_path="/path/to/your/policies/custom"
)
```

### Step 5: Test Your Policy

Create a test script to validate your policy:

```python
import asyncio
from rizk.sdk.guardrails.engine import GuardrailsEngine

async def test_financial_policy():
    # Initialize the guardrails engine
    engine = GuardrailsEngine.get_instance()
    
    # Test cases that should trigger the policy
    test_cases = [
        "Should I buy Tesla stock?",
        "What's the best cryptocurrency to invest in?",
        "Tell me about investment portfolio diversification",  # Educational - should pass
        "You should definitely buy Bitcoin now, it will double!"  # Should be blocked/guided
    ]
    
    for test_case in test_cases:
        result = await engine.process_message(test_case)
        print(f"Input: {test_case}")
        print(f"Allowed: {result.allowed}")
        print(f"Guidelines: {result.guidelines if hasattr(result, 'guidelines') else 'None'}")
        print(f"Reason: {result.blocked_reason if not result.allowed else 'N/A'}")
        print("-" * 50)

# Run the test
asyncio.run(test_financial_policy())
```

## Policy Examples

### Example 1: Content Moderation Policy

```yaml
- id: "content_moderation_001"
  name: "Professional Communication Policy"
  domains: ["profanity", "inappropriate", "offensive", "harassment"]
  description: "Ensures all communications maintain professional standards"
  action: "block"
  guidelines:
    - "Maintain professional and respectful communication at all times"
    - "Avoid any language that could be considered offensive or inappropriate"
    - "If encountering inappropriate input, politely redirect to constructive topics"
  patterns:
    - "(?i)\\b(damn|hell|crap|stupid|idiot|moron)\\b"
    - "(?i)\\b(shut up|go away|you suck)\\b"
    - "(?i)(offensive|inappropriate).*language"
```

### Example 2: Data Privacy Policy

```yaml
- id: "data_privacy_001"
  name: "Personal Data Protection Policy"
  domains: ["personal", "private", "confidential", "pii", "data", "information"]
  description: "Prevents collection or processing of personal identifiable information"
  action: "block"
  guidelines:
    - "Never request, store, or process personal identifiable information"
    - "If users share personal information, remind them not to do so and don't reference the specific information"
    - "Suggest using placeholder or anonymized data for examples"
    - "Recommend proper data protection practices when discussing data handling"
  patterns:
    - "\\b\\d{3}-?\\d{2}-?\\d{4}\\b"  # SSN pattern
    - "\\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Z|a-z]{2,}\\b"  # Email pattern
    - "\\b\\d{3}[.-]?\\d{3}[.-]?\\d{4}\\b"  # Phone number pattern
    - "(?i)what.*(is|s) (your|my) (name|address|phone|email|ssn)"
```

### Example 3: Brand Safety Policy

```yaml
- id: "brand_safety_001"
  name: "Brand Safety and Reputation Policy"
  domains: ["competitor", "brand", "company", "competitor", "negative", "criticism"]
  description: "Protects brand reputation and handles competitor discussions appropriately"
  action: "allow"
  guidelines:
    - "Always represent our brand professionally and positively"
    - "When discussing competitors, be factual and objective, never disparaging"
    - "Focus on our unique value propositions rather than criticizing alternatives"
    - "If asked about controversies, provide balanced, factual information"
    - "Redirect negative conversations toward constructive solutions"
  patterns:
    - "(?i)(our|this) company.*(sucks|terrible|awful|worst)"
    - "(?i)(competitor.*better|switch to.*competitor)"
    - "(?i)why.*(not choose|avoid).*us"
```

### Example 4: Technical Security Policy

```yaml
- id: "security_001"
  name: "Security Information Protection Policy"
  domains: ["password", "security", "vulnerability", "exploit", "hack", "breach"]
  description: "Prevents sharing of security-sensitive information and practices"
  action: "block"
  guidelines:
    - "Never provide information that could compromise security"
    - "Do not share specific vulnerability details or exploit methods"
    - "For security education, focus on defensive strategies and best practices"
    - "Direct users to official security resources and responsible disclosure practices"
    - "Encourage following established security protocols"
  patterns:
    - "(?i)what.*(is|s) (your|the|our) (password|api key|secret|token)"
    - "(?i)how to (hack|crack|exploit|bypass) (.*(?:system|security|password))"
    - "(?i)(sql injection|xss|buffer overflow).*exploit"
    - "(?i)default.*(password|credentials|login)"
```

## Best Practices

### 1. Policy Design Principles

**Start Broad, Then Refine**
```yaml
# âŒ Too specific initially
patterns:
  - "(?i)buy AAPL stock now"

# âœ… Broader pattern, refined over time
patterns:
  - "(?i)buy.*stock.*now"
```

**Use Meaningful Guidelines**
```yaml
# âŒ Vague guidelines
guidelines:
  - "Be careful about financial advice"

# âœ… Specific, actionable guidelines
guidelines:
  - "Provide general financial education only, never specific investment advice"
  - "Always recommend consulting with qualified financial advisors"
```

### 2. Pattern Writing Best Practices

**Use Case-Insensitive Matching**
```yaml
patterns:
  - "(?i)sensitive.*pattern"  # âœ… Case insensitive
  - "sensitive.*pattern"      # âŒ Case sensitive
```

**Escape Special Characters**
```yaml
patterns:
  - "\\$\\d+\\.\\d{2}"       # âœ… Properly escaped currency
  - "$\d+.\d{2}"             # âŒ Unescaped special chars
```

**Use Word Boundaries for Exact Matches**
```yaml
patterns:
  - "\\bpassword\\b"         # âœ… Matches "password" but not "passwording"
  - "password"               # âŒ Matches "passwording"
```

### 3. Testing and Validation

**Create Comprehensive Test Cases**
```python
test_cases = [
    # Cases that should trigger the policy
    ("Buy Tesla stock now!", True),
    ("Guaranteed 50% returns", True),
    
    # Cases that should NOT trigger
    ("Learn about portfolio diversification", False),
    ("Understanding risk management", False),
    
    # Edge cases
    ("Buy groceries for dinner", False),
    ("BUY TESLA STOCK NOW!", True),  # Test case sensitivity
]
```

**Monitor Policy Performance**
```python
# Add logging to track policy effectiveness
import logging

logger = logging.getLogger("custom_policies")

async def test_policy_effectiveness():
    true_positives = 0
    false_positives = 0
    
    for test_input, expected_trigger in test_cases:
        result = await engine.process_message(test_input)
        triggered = not result.allowed or len(result.guidelines or []) > 0
        
        if triggered == expected_trigger:
            true_positives += 1
        else:
            false_positives += 1
            logger.warning(f"Policy mismatch: '{test_input}' - Expected: {expected_trigger}, Got: {triggered}")
    
    accuracy = true_positives / len(test_cases)
    logger.info(f"Policy accuracy: {accuracy:.2%}")
```

## Advanced Policy Features

### Dynamic Policy Loading

For policies that need to be updated frequently:

```python
from rizk.sdk.guardrails.engine import GuardrailsEngine
import asyncio

async def reload_policies():
    """Reload policies without restarting the application"""
    engine = GuardrailsEngine.get_instance()
    
    # Clear current policies
    engine.fast_rules.policies = []
    engine.policy_augmentation.policies = []
    
    # Reload from disk
    engine.fast_rules._load_and_process_policies()
    engine.policy_augmentation._load_and_process_policies()
    
    print("Policies reloaded successfully")

# Use in production with a scheduler
asyncio.run(reload_policies())
```

### Environment-Specific Policies

```yaml
# Development environment policies
- id: "dev_debugging_001"
  name: "Development Debugging Policy"
  domains: ["debug", "test", "development"]
  description: "Allow verbose debugging in development"
  action: "allow"
  guidelines:
    - "Provide detailed debugging information"
    - "Include internal state information when helpful"
  patterns: []

# Production environment policies  
- id: "prod_security_001"
  name: "Production Security Policy"
  domains: ["debug", "internal", "system"]
  description: "Restrict internal information in production"
  action: "block"
  guidelines:
    - "Never expose internal system information"
    - "Provide user-friendly error messages only"
  patterns:
    - "(?i)(debug|internal|system).*(info|details|state)"
```

### Policy Inheritance and Composition

```yaml
# Base policy for all financial content
- id: "financial_base"
  name: "Base Financial Policy"
  domains: ["financial", "money", "investment"]
  description: "Base guidelines for all financial discussions"
  action: "allow"
  guidelines:
    - "Always include risk disclaimers"
    - "Recommend professional consultation"

# Specific policy that extends the base
- id: "crypto_specific"
  name: "Cryptocurrency Policy"
  domains: ["crypto", "bitcoin", "ethereum", "blockchain"]
  description: "Additional guidelines for cryptocurrency discussions"
  action: "allow"
  guidelines:
    - "Emphasize the high volatility of cryptocurrencies"
    - "Mention regulatory uncertainty"
    - "Include base financial guidelines"
  patterns:
    - "(?i)guaranteed.*crypto.*profit"
    - "(?i)bitcoin.*will.*moon"
```

## Troubleshooting Common Issues

### Issue 1: Policy Not Triggering

**Symptoms**: Your policy patterns aren't matching expected inputs

**Solutions**:
1. **Test patterns separately**:
```python
import re

pattern = re.compile("(?i)buy.*stock", re.IGNORECASE)
test_input = "You should buy Tesla stock"
match = pattern.search(test_input)
print(f"Pattern matched: {match is not None}")
```

2. **Check domain keywords**:
```python
# Ensure your domains include relevant keywords
domains = ["investment", "stock", "buy", "sell", "trading"]
input_words = test_input.lower().split()
domain_match = any(domain in input_words for domain in domains)
print(f"Domain matched: {domain_match}")
```

### Issue 2: Too Many False Positives

**Symptoms**: Policy triggers on benign inputs

**Solutions**:
1. **Make patterns more specific**:
```yaml
# âŒ Too broad
patterns:
  - "(?i)buy.*"

# âœ… More specific
patterns:
  - "(?i)buy.*(?:stock|crypto|investment|securities)"
```

2. **Use negative lookaheads**:
```yaml
patterns:
  - "(?i)buy(?!.*(?:groceries|food|clothing)).*(?:stock|investment)"
```

### Issue 3: Performance Issues

**Symptoms**: Slow policy evaluation

**Solutions**:
1. **Optimize regex patterns**:
```yaml
# âŒ Inefficient backtracking
patterns:
  - "(?i)(a+)+b"

# âœ… More efficient
patterns:
  - "(?i)a+b"
```

2. **Order patterns by frequency**:
```yaml
# Place most common patterns first
patterns:
  - "(?i)common.*pattern"      # Frequent
  - "(?i)less.*common.*pattern" # Less frequent
  - "(?i)rare.*edge.*case"     # Rare
```

## Policy Governance and Lifecycle

### Version Control

Keep your policies in version control:

```bash
# Initialize git repository for policies
cd policies/
git init
git add *.yaml
git commit -m "Initial policy definitions"

# Track changes over time
git log --oneline custom/financial_guidance.yaml
```

### Policy Review Process

1. **Create policy in development environment**
2. **Test thoroughly with diverse inputs**
3. **Review with stakeholders (legal, compliance, business)**
4. **Deploy to staging for integration testing**
5. **Monitor metrics and adjust as needed**
6. **Deploy to production with monitoring**

### Monitoring and Analytics

```python
# Track policy effectiveness
from rizk.sdk.analytics import PolicyAnalytics

analytics = PolicyAnalytics()

# Monitor policy triggers
policy_metrics = analytics.get_policy_metrics(
    policy_id="financial_advice_prevention",
    time_range="24h"
)

print(f"Triggers: {policy_metrics.trigger_count}")
print(f"Blocks: {policy_metrics.block_count}")
print(f"False positive rate: {policy_metrics.false_positive_rate}")
```

## Next Steps

1. **Start Simple**: Begin with basic policies and gradually add complexity
2. **Test Thoroughly**: Use diverse test cases and monitor performance
3. **Iterate**: Continuously improve based on real-world usage
4. **Document**: Keep clear documentation of policy intentions and changes
5. **Monitor**: Track policy effectiveness and adjust as needed

For more advanced policy features, see:
- **[Policy YAML Reference](policy-yaml-reference.md)** - Complete format specification
- **[Policy Examples](policy-examples.md)** - Real-world implementations
- **[Troubleshooting Policies](troubleshooting-policies.md)** - Advanced debugging techniques

---

**Remember**: Effective policies balance security and usability. Start restrictive and gradually relax constraints based on real-world feedback and usage patterns. 
