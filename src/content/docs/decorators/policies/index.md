---
title: "@policies Decorator"
description: "@policies Decorator"
---

# @policies Decorator

The `@policies` decorator provides fine-grained control over custom policy application and enforcement. It allows you to apply specific organizational policies, compliance rules, and governance frameworks to individual functions, workflows, or agents.

## Overview

The `@policies` decorator enables targeted application of custom policies without the full guardrails system. It's ideal when you need specific policy enforcement for particular functions or when you want to apply different policy sets to different parts of your application.

## Basic Usage

```python
from rizk.sdk import Rizk
from rizk.sdk.decorators import policies, workflow

# Initialize Rizk
rizk = Rizk.init(app_name="PoliciesApp", enabled=True)

@policies(["data_privacy", "content_moderation"])
@workflow(name="user_content_processing")
def process_user_content(content: str, user_id: str) -> dict:
    """Process user content with specific policy enforcement."""
    
    # Process the content
    processed_content = {
        "original_content": content,
        "user_id": user_id,
        "processed_at": datetime.now().isoformat(),
        "content_length": len(content),
        "word_count": len(content.split())
    }
    
    # Add content analysis
    if "question" in content.lower():
        processed_content["content_type"] = "question"
    elif "complaint" in content.lower():
        processed_content["content_type"] = "complaint"
        processed_content["priority"] = "high"
    else:
        processed_content["content_type"] = "general"
    
    return processed_content

# Usage
result = process_user_content("I have a question about my account", "user_123")
print(f"Processed content: {result}")
```

## Parameters Reference

### Core Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `policy_names` | `list[str]` | Yes | List of policy names to apply |
| `enforcement_mode` | `str` | No | `"strict"`, `"warn"`, `"log"` (default: `"strict"`) |
| `policy_path` | `str` | No | Custom path to policy files |
| `context` | `dict` | No | Additional context for policy evaluation |

### Advanced Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `priority` | `int` | Policy priority (1-10, higher = more important) |
| `override_global` | `bool` | Override global policy settings |
| `custom_evaluator` | `callable` | Custom policy evaluation function |
| `cache_duration` | `int` | Policy evaluation cache duration in seconds |

## Policy Types

### Data Privacy Policies

```python
@policies(["gdpr_compliance", "data_minimization", "consent_tracking"])
@workflow(name="personal_data_processor")
def process_personal_data(data: dict, purpose: str, consent_id: str) -> dict:
    """Process personal data with GDPR compliance policies."""
    
    # Validate consent
    if not consent_id:
        raise ValueError("Consent ID required for personal data processing")
    
    # Apply data minimization
    allowed_fields = {
        "marketing": ["name", "email", "preferences"],
        "analytics": ["user_id", "session_data", "timestamps"],
        "support": ["name", "email", "issue_description"]
    }
    
    minimized_data = {
        field: data[field] 
        for field in allowed_fields.get(purpose, []) 
        if field in data
    }
    
    processing_result = {
        "processed_data": minimized_data,
        "purpose": purpose,
        "consent_id": consent_id,
        "processing_timestamp": datetime.now().isoformat(),
        "data_fields_processed": list(minimized_data.keys()),
        "compliance_status": "gdpr_compliant"
    }
    
    return processing_result

# Example usage
user_data = {
    "name": "John Doe",
    "email": "john@example.com",
    "phone": "+1234567890",
    "preferences": {"newsletter": True},
    "session_data": {"last_login": "2024-01-01"}
}

result = process_personal_data(user_data, "marketing", "consent_123")
print(f"GDPR compliant processing: {result}")
```

### Content Moderation Policies

```python
@policies(["content_safety", "community_guidelines", "brand_standards"])
@workflow(name="content_moderation")
def moderate_user_content(content: str, content_type: str, user_tier: str = "standard") -> dict:
    """Moderate user content with comprehensive policy enforcement."""
    
    moderation_result = {
        "original_content": content,
        "content_type": content_type,
        "user_tier": user_tier,
        "moderation_timestamp": datetime.now().isoformat(),
        "status": "pending",
        "flags": [],
        "score": 0
    }
    
    # Content safety checks
    safety_score = 0
    if any(word in content.lower() for word in ["spam", "scam", "fake"]):
        moderation_result["flags"].append("potential_spam")
        safety_score -= 20
    
    # Community guidelines
    if len(content) > 1000 and content_type == "comment":
        moderation_result["flags"].append("excessive_length")
        safety_score -= 10
    
    if content.count("!") > 5:
        moderation_result["flags"].append("excessive_punctuation")
        safety_score -= 5
    
    # Brand standards
    if any(word in content.lower() for word in ["professional", "quality", "excellent"]):
        safety_score += 10
        moderation_result["flags"].append("positive_brand_language")
    
    # User tier adjustments
    if user_tier == "premium":
        safety_score += 5
    elif user_tier == "verified":
        safety_score += 10
    
    moderation_result["score"] = max(0, min(100, 50 + safety_score))
    
    # Final status determination
    if moderation_result["score"] >= 70:
        moderation_result["status"] = "approved"
    elif moderation_result["score"] >= 40:
        moderation_result["status"] = "review_required"
    else:
        moderation_result["status"] = "rejected"
    
    return moderation_result

# Example usage
content_result = moderate_user_content(
    "This is a professional and excellent review of the product!", 
    "review", 
    "premium"
)
print(f"Moderation result: {content_result}")
```

### Financial Compliance Policies

```python
@policies(["sox_compliance", "financial_reporting", "audit_trail"], 
          enforcement_mode="strict",
          priority=9)
@workflow(name="financial_transaction_processor")
def process_financial_transaction(transaction: dict, approver_id: str) -> dict:
    """Process financial transactions with SOX compliance."""
    
    # Validate required fields
    required_fields = ["amount", "account_from", "account_to", "purpose", "transaction_id"]
    missing_fields = [field for field in required_fields if field not in transaction]
    
    if missing_fields:
        raise ValueError(f"Missing required fields: {missing_fields}")
    
    # SOX compliance checks
    amount = float(transaction["amount"])
    
    # Materiality threshold check
    if amount > 10000:
        if not approver_id:
            raise ValueError("Approver ID required for transactions over $10,000")
    
    # Segregation of duties
    if transaction.get("created_by") == approver_id:
        raise ValueError("Creator and approver cannot be the same person")
    
    processing_result = {
        "transaction_id": transaction["transaction_id"],
        "amount": amount,
        "status": "processed",
        "approver_id": approver_id,
        "processing_timestamp": datetime.now().isoformat(),
        "compliance_checks": {
            "sox_compliant": True,
            "materiality_check": "passed",
            "segregation_of_duties": "verified",
            "audit_trail": "complete"
        },
        "risk_level": "low" if amount < 10000 else "medium" if amount < 100000 else "high"
    }
    
    return processing_result

# Example usage
transaction_data = {
    "transaction_id": "TXN_001",
    "amount": "15000.00",
    "account_from": "ACC_123",
    "account_to": "ACC_456",
    "purpose": "Equipment purchase",
    "created_by": "user_001"
}

result = process_financial_transaction(transaction_data, "manager_002")
print(f"Financial processing result: {result}")
```

## Policy Enforcement Modes

### Strict Mode

```python
@policies(["security_policies"], enforcement_mode="strict")
@workflow(name="secure_data_access")
def access_secure_data(user_id: str, data_category: str, access_reason: str) -> dict:
    """Access secure data with strict policy enforcement."""
    
    # Strict mode: Any policy violation blocks execution
    access_result = {
        "user_id": user_id,
        "data_category": data_category,
        "access_reason": access_reason,
        "access_timestamp": datetime.now().isoformat(),
        "access_granted": False,
        "security_level": "high"
    }
    
    # Security policy checks
    if data_category in ["pii", "financial", "medical"]:
        if not access_reason or len(access_reason) < 10:
            raise ValueError("Detailed access reason required for sensitive data")
    
    # Simulate access control
    access_result["access_granted"] = True
    access_result["data_sample"] = f"Secure data for category: {data_category}"
    
    return access_result
```

### Warning Mode

```python
@policies(["content_guidelines"], enforcement_mode="warn")
@workflow(name="content_publisher")
def publish_content(content: str, category: str) -> dict:
    """Publish content with warning-level policy enforcement."""
    
    publish_result = {
        "content": content,
        "category": category,
        "published_at": datetime.now().isoformat(),
        "status": "published",
        "warnings": []
    }
    
    # Warning mode: Policy violations generate warnings but don't block
    if len(content) < 100:
        publish_result["warnings"].append("Content is quite short")
    
    if not any(char.isupper() for char in content):
        publish_result["warnings"].append("Consider adding proper capitalization")
    
    return publish_result
```

### Logging Mode

```python
@policies(["usage_tracking"], enforcement_mode="log")
@workflow(name="api_endpoint")
def api_endpoint_handler(request_data: dict, endpoint: str) -> dict:
    """Handle API requests with logging-only policy enforcement."""
    
    # Logging mode: Policies are evaluated and logged but don't affect execution
    response = {
        "endpoint": endpoint,
        "request_data": request_data,
        "response_timestamp": datetime.now().isoformat(),
        "status": "success"
    }
    
    # Process the request normally
    if endpoint == "user_profile":
        response["data"] = {"profile": "user profile data"}
    elif endpoint == "settings":
        response["data"] = {"settings": "user settings data"}
    else:
        response["data"] = {"message": "endpoint not found"}
        response["status"] = "not_found"
    
    return response
```

## Custom Policy Evaluators

```python
def custom_healthcare_evaluator(content: str, context: dict) -> dict:
    """Custom evaluator for healthcare content policies."""
    
    evaluation_result = {
        "compliant": True,
        "violations": [],
        "recommendations": [],
        "confidence": 0.95
    }
    
    # HIPAA compliance checks
    if any(pattern in content.lower() for pattern in ["ssn", "social security", "patient id"]):
        evaluation_result["violations"].append("Potential PHI disclosure")
        evaluation_result["compliant"] = False
    
    # Medical accuracy checks
    if "diagnosis" in content.lower() and context.get("user_role") != "physician":
        evaluation_result["violations"].append("Medical diagnosis by non-physician")
        evaluation_result["compliant"] = False
    
    # Recommendations
    if "medication" in content.lower():
        evaluation_result["recommendations"].append("Consider adding medication disclaimer")
    
    return evaluation_result

@policies(["healthcare_compliance"], 
          custom_evaluator=custom_healthcare_evaluator,
          context={"user_role": "nurse"})
@workflow(name="healthcare_content_processor")
def process_healthcare_content(content: str, patient_context: dict) -> dict:
    """Process healthcare content with custom policy evaluation."""
    
    processing_result = {
        "content": content,
        "patient_context": patient_context,
        "processed_at": datetime.now().isoformat(),
        "compliance_status": "evaluated"
    }
    
    # Add healthcare-specific processing
    if "symptoms" in content.lower():
        processing_result["content_type"] = "symptom_report"
    elif "treatment" in content.lower():
        processing_result["content_type"] = "treatment_plan"
    else:
        processing_result["content_type"] = "general_healthcare"
    
    return processing_result
```

## Policy Composition

```python
@policies(["base_security"], priority=1)
@policies(["industry_specific"], priority=5)
@policies(["company_custom"], priority=10)
@workflow(name="multi_policy_processor")
def process_with_multiple_policies(data: dict) -> dict:
    """Process data with multiple policy layers."""
    
    # Policies are applied in priority order (higher priority first)
    processing_result = {
        "data": data,
        "processed_at": datetime.now().isoformat(),
        "policy_layers": ["company_custom", "industry_specific", "base_security"],
        "compliance_status": "multi_layer_compliant"
    }
    
    return processing_result
```

## Dynamic Policy Application

```python
@policies([], enforcement_mode="strict")  # Empty list - policies added dynamically
@workflow(name="dynamic_policy_processor")
def process_with_dynamic_policies(content: str, content_type: str, user_role: str) -> dict:
    """Process content with dynamically determined policies."""
    
    # Determine policies based on context
    dynamic_policies = ["base_content_policy"]
    
    if content_type == "financial":
        dynamic_policies.extend(["financial_compliance", "sox_requirements"])
    elif content_type == "healthcare":
        dynamic_policies.extend(["hipaa_compliance", "medical_accuracy"])
    elif content_type == "legal":
        dynamic_policies.extend(["legal_compliance", "confidentiality"])
    
    if user_role == "admin":
        dynamic_policies.append("admin_privileges")
    elif user_role == "guest":
        dynamic_policies.append("guest_restrictions")
    
    # Apply policies dynamically (this would be handled by the decorator in practice)
    processing_result = {
        "content": content,
        "content_type": content_type,
        "user_role": user_role,
        "applied_policies": dynamic_policies,
        "processed_at": datetime.now().isoformat(),
        "status": "processed"
    }
    
    return processing_result
```

## Policy Testing

```python
import pytest
from unittest.mock import patch

def test_policy_application():
    """Test basic policy application."""
    
    @policies(["test_policy"])
    def test_function(data: str) -> str:
        return f"Processed: {data}"
    
    result = test_function("test data")
    assert "Processed:" in result

def test_policy_enforcement_modes():
    """Test different enforcement modes."""
    
    @policies(["strict_policy"], enforcement_mode="strict")
    def strict_function(data: str) -> str:
        return f"Strict: {data}"
    
    @policies(["warn_policy"], enforcement_mode="warn")
    def warn_function(data: str) -> str:
        return f"Warn: {data}"
    
    strict_result = strict_function("test")
    warn_result = warn_function("test")
    
    assert "Strict:" in strict_result
    assert "Warn:" in warn_result

@patch('rizk.sdk.policies.PolicyEngine')
def test_custom_evaluator(mock_engine):
    """Test custom policy evaluator."""
    
    def custom_eval(content, context):
        return {"compliant": True, "violations": []}
    
    mock_engine.return_value.evaluate.return_value = {"compliant": True}
    
    @policies(["custom_policy"], custom_evaluator=custom_eval)
    def custom_function(data: str) -> str:
        return f"Custom: {data}"
    
    result = custom_function("test")
    assert "Custom:" in result

def test_policy_priority():
    """Test policy priority handling."""
    
    @policies(["low_priority"], priority=1)
    @policies(["high_priority"], priority=10)
    def priority_function(data: str) -> str:
        return f"Priority: {data}"
    
    result = priority_function("test")
    assert "Priority:" in result
```

## Best Practices

### 1. **Policy Organization**
```python
# Good: Organize policies by domain and scope
@policies(["gdpr_base", "gdpr_marketing", "company_privacy"])
def marketing_processor(data): pass

# Avoid: Mixing unrelated policies
@policies(["gdpr", "financial_sox", "content_moderation"])  # Too broad
def generic_processor(data): pass
```

### 2. **Enforcement Mode Selection**
```python
# Critical systems: Use strict enforcement
@policies(["security"], enforcement_mode="strict")
def security_function(data): pass

# User-facing features: Use warn mode
@policies(["content_guidelines"], enforcement_mode="warn")
def user_content_function(data): pass

# Analytics: Use log mode
@policies(["usage_tracking"], enforcement_mode="log")
def analytics_function(data): pass
```

### 3. **Policy Priority**
```python
# Higher priority for more specific policies
@policies(["general_security"], priority=1)
@policies(["financial_security"], priority=5)
@policies(["company_financial"], priority=10)
def financial_processor(data): pass
```

### 4. **Context Provision**
```python
# Provide relevant context for policy evaluation
@policies(["content_policy"], 
          context={"user_role": "editor", "content_type": "article"})
def content_processor(content): pass
```

## Related Documentation

- **[@guardrails Decorator](guardrails.md)** - Comprehensive policy enforcement
- **[Policy Configuration](../05-configuration/policies.md)** - Setting up custom policies
- **[Compliance](../09-compliance/overview.md)** - Enterprise compliance features
- **[Decorator Composition](decorator-composition.md)** - Combining decorators effectively

---

The `@policies` decorator provides fine-grained control over custom policy application, enabling targeted enforcement of organizational policies, compliance rules, and governance frameworks for specific functions and workflows. 

