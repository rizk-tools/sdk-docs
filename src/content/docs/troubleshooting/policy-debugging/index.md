---
title: "Policy Debugging"
description: "Policy Debugging"
---

# Policy Debugging

This guide helps you debug guardrails policies, understand policy evaluation decisions, and troubleshoot policy-related issues in Rizk SDK.

## ðŸ›¡ï¸ Policy System Overview

Rizk SDK uses a multi-layer guardrails system:

1. **Fast Rules Engine**: Regex-based pattern matching for immediate decisions
2. **Policy Augmentation**: Context-aware prompt enhancement with guidelines  
3. **LLM Fallback**: LLM-based evaluation for complex edge cases
4. **State Management**: Conversation context and history tracking

## ðŸ” Policy Debugging Tools

### 1. Policy Evaluation Debugger

Debug policy evaluation step-by-step:

```python
from rizk.sdk.guardrails.engine import GuardrailsEngine
from rizk.sdk.guardrails.fast_rules import FastRulesEngine
from rizk.sdk.guardrails.policy_augmentation import PolicyAugmentation
from rizk.sdk.config import get_policies_path

def debug_policy_evaluation(test_message: str, direction: str = "inbound"):
    """Debug policy evaluation step by step."""
    
    print(f"ðŸ›¡ï¸ Policy Evaluation Debug for: '{test_message[:50]}...'")
    print("=" * 60)
    
    try:
        # 1. Check policy loading
        print("\nðŸ“‹ Step 1: Policy Loading")
        policies_path = get_policies_path()
        print(f"Policies path: {policies_path}")
        
        fast_rules = FastRulesEngine(policies_path)
        print(f"âœ… Loaded {len(fast_rules.policies)} policies")
        
        # Show policy summaries
        print("\nðŸ“ Loaded Policies:")
        for policy in fast_rules.policies[:5]:  # Show first 5
            print(f"  â€¢ {policy.id}: {policy.name} ({policy.action})")
        if len(fast_rules.policies) > 5:
            print(f"  ... and {len(fast_rules.policies) - 5} more")
        
        # 2. Fast Rules Evaluation
        print(f"\nâš¡ Step 2: Fast Rules Evaluation")
        fast_result = fast_rules.evaluate(test_message, direction=direction)
        print(f"  Result: {'ðŸš« BLOCKED' if fast_result.blocked else 'âœ… ALLOWED'}")
        print(f"  Confidence: {fast_result.confidence:.2f}")
        print(f"  Reason: {fast_result.reason}")
        if fast_result.matched_patterns:
            print(f"  Matched patterns: {fast_result.matched_patterns}")
        
        # 3. Policy Augmentation
        print(f"\nðŸ“ Step 3: Policy Augmentation")
        policy_aug = PolicyAugmentation(policies_path)
        guidelines = policy_aug.get_guidelines(test_message)
        
        if guidelines:
            print(f"  âœ… Found {len(guidelines)} guidelines")
            print("  Guidelines preview:")
            for i, guideline in enumerate(guidelines[:3]):
                print(f"    {i+1}. {guideline}")
            if len(guidelines) > 3:
                print(f"    ... and {len(guidelines) - 3} more")
        else:
            print(f"  â„¹ï¸ No specific guidelines found")
        
        # 4. Full Guardrails Engine Evaluation
        print(f"\nðŸ”’ Step 4: Full Guardrails Evaluation")
        engine = GuardrailsEngine.get_instance()
        
        context = {
            "conversation_id": "debug_conversation",
            "user_id": "debug_user"
        }
        
        full_result = engine.evaluate(test_message, direction=direction, context=context)
        
        print(f"  Final Result: {'âœ… ALLOWED' if full_result.allowed else 'ðŸš« BLOCKED'}")
        print(f"  Confidence: {full_result.confidence:.2f}")
        print(f"  Decision Layer: {full_result.decision_layer}")
        
        if not full_result.allowed:
            print(f"  Block Reason: {full_result.blocked_reason}")
            print(f"  Violated Policies: {full_result.violated_policies}")
        
        return full_result
        
    except Exception as e:
        print(f"âŒ Debug process failed: {e}")
        import traceback
        traceback.print_exc()
        return None

# Example usage:
test_messages = [
    "Hello, how can I help you?",
    "My SSN is 123-45-6789",
    "Let me think step by step about this problem...",
    "The secret API key is sk-abc123def456"
]

for message in test_messages:
    print("\n" + "="*80)
    debug_policy_evaluation(message, direction="inbound")
    print("\n" + "-"*40 + " OUTBOUND TEST " + "-"*40)
    debug_policy_evaluation(message, direction="outbound")
```

### 2. Policy File Validator

Validate policy file syntax and structure:

```python
import yaml
import re
from typing import List, Dict, Any

def validate_policy_file(policies_path: str) -> Dict[str, Any]:
    """Validate policy file structure and content."""
    
    validation_report = {
        "file_exists": False,
        "valid_yaml": False,
        "valid_structure": False,
        "policies_count": 0,
        "errors": [],
        "warnings": [],
        "policy_details": []
    }
    
    try:
        # Check file exists
        import os
        if not os.path.exists(policies_path):
            validation_report["errors"].append(f"Policy file not found: {policies_path}")
            return validation_report
        
        validation_report["file_exists"] = True
        
        # Load and parse YAML
        with open(policies_path, 'r', encoding='utf-8') as f:
            content = yaml.safe_load(f)
        
        validation_report["valid_yaml"] = True
        
        # Validate structure
        if not isinstance(content, dict):
            validation_report["errors"].append("Root level must be a dictionary")
            return validation_report
        
        if "policies" not in content:
            validation_report["errors"].append("Missing 'policies' key in root")
            return validation_report
        
        if not isinstance(content["policies"], list):
            validation_report["errors"].append("'policies' must be a list")
            return validation_report
        
        validation_report["valid_structure"] = True
        validation_report["policies_count"] = len(content["policies"])
        
        # Validate individual policies
        required_fields = ["id", "name", "action"]
        optional_fields = ["domains", "description", "guidelines", "patterns", "confidence"]
        
        for i, policy in enumerate(content["policies"]):
            policy_errors = []
            policy_warnings = []
            
            # Check required fields
            for field in required_fields:
                if field not in policy:
                    policy_errors.append(f"Missing required field: {field}")
            
            # Validate field types
            if "id" in policy and not isinstance(policy["id"], str):
                policy_errors.append("'id' must be a string")
            
            if "name" in policy and not isinstance(policy["name"], str):
                policy_errors.append("'name' must be a string")
            
            if "action" in policy and policy["action"] not in ["allow", "block", "augment"]:
                policy_errors.append("'action' must be 'allow', 'block', or 'augment'")
            
            if "patterns" in policy:
                if not isinstance(policy["patterns"], list):
                    policy_errors.append("'patterns' must be a list")
                else:
                    # Validate regex patterns
                    for j, pattern in enumerate(policy["patterns"]):
                        try:
                            re.compile(pattern)
                        except re.error as e:
                            policy_errors.append(f"Invalid regex pattern {j}: {e}")
            
            if "guidelines" in policy and not isinstance(policy["guidelines"], list):
                policy_errors.append("'guidelines' must be a list")
            
            if "confidence" in policy:
                try:
                    conf = float(policy["confidence"])
                    if not (0.0 <= conf <= 1.0):
                        policy_warnings.append("'confidence' should be between 0.0 and 1.0")
                except (ValueError, TypeError):
                    policy_errors.append("'confidence' must be a number")
            
            # Policy completeness warnings
            if "patterns" not in policy and "guidelines" not in policy:
                policy_warnings.append("Policy has no patterns or guidelines - may not be effective")
            
            if policy.get("action") == "block" and "patterns" not in policy:
                policy_warnings.append("Block policy without patterns may not trigger correctly")
            
            validation_report["policy_details"].append({
                "index": i,
                "id": policy.get("id", f"policy_{i}"),
                "name": policy.get("name", "Unnamed"),
                "action": policy.get("action", "unknown"),
                "has_patterns": "patterns" in policy,
                "has_guidelines": "guidelines" in policy,
                "errors": policy_errors,
                "warnings": policy_warnings
            })
            
            validation_report["errors"].extend([f"Policy {i} ({policy.get('id', 'unknown')}): {err}" for err in policy_errors])
            validation_report["warnings"].extend([f"Policy {i} ({policy.get('id', 'unknown')}): {warn}" for warn in policy_warnings])
        
    except yaml.YAMLError as e:
        validation_report["errors"].append(f"YAML parsing error: {e}")
    except Exception as e:
        validation_report["errors"].append(f"Validation error: {e}")
    
    return validation_report

def print_validation_report(policies_path: str):
    """Print a formatted validation report."""
    
    print(f"ðŸ” Policy File Validation: {policies_path}")
    print("=" * 60)
    
    report = validate_policy_file(policies_path)
    
    # Overall status
    if report["errors"]:
        print("âŒ VALIDATION FAILED")
    elif report["warnings"]:
        print("âš ï¸ VALIDATION PASSED WITH WARNINGS")
    else:
        print("âœ… VALIDATION PASSED")
    
    print(f"File exists: {'âœ…' if report['file_exists'] else 'âŒ'}")
    print(f"Valid YAML: {'âœ…' if report['valid_yaml'] else 'âŒ'}")
    print(f"Valid structure: {'âœ…' if report['valid_structure'] else 'âŒ'}")
    print(f"Policies found: {report['policies_count']}")
    
    # Errors
    if report["errors"]:
        print(f"\nâŒ Errors ({len(report['errors'])}):")
        for error in report["errors"]:
            print(f"  â€¢ {error}")
    
    # Warnings
    if report["warnings"]:
        print(f"\nâš ï¸ Warnings ({len(report['warnings'])}):")
        for warning in report["warnings"]:
            print(f"  â€¢ {warning}")
    
    # Policy summary
    if report["policy_details"]:
        print(f"\nðŸ“‹ Policy Summary:")
        for policy in report["policy_details"]:
            status = "âŒ" if policy["errors"] else "âš ï¸" if policy["warnings"] else "âœ…"
            print(f"  {status} {policy['id']}: {policy['name']} ({policy['action']})")
            if policy["errors"]:
                for error in policy["errors"]:
                    print(f"    âŒ {error}")
            if policy["warnings"]:
                for warning in policy["warnings"]:
                    print(f"    âš ï¸ {warning}")

# Example usage:
from rizk.sdk.config import get_policies_path
policies_path = get_policies_path()
print_validation_report(policies_path)
```

### 3. Policy Pattern Tester

Test regex patterns against sample inputs:

```python
import re
from typing import List, Tuple

def test_policy_patterns(patterns: List[str], test_inputs: List[str]) -> Dict[str, Any]:
    """Test regex patterns against sample inputs."""
    
    results = {
        "pattern_results": [],
        "input_results": [],
        "summary": {
            "total_patterns": len(patterns),
            "total_inputs": len(test_inputs),
            "working_patterns": 0,
            "failing_patterns": 0,
            "matches_found": 0
        }
    }
    
    # Test each pattern
    for i, pattern in enumerate(patterns):
        pattern_result = {
            "index": i,
            "pattern": pattern,
            "valid_regex": False,
            "matches": [],
            "error": None
        }
        
        try:
            compiled_pattern = re.compile(pattern, re.IGNORECASE)
            pattern_result["valid_regex"] = True
            results["summary"]["working_patterns"] += 1
            
            # Test against all inputs
            for j, test_input in enumerate(test_inputs):
                match = compiled_pattern.search(test_input)
                if match:
                    pattern_result["matches"].append({
                        "input_index": j,
                        "input": test_input,
                        "match": match.group(),
                        "start": match.start(),
                        "end": match.end()
                    })
                    results["summary"]["matches_found"] += 1
        
        except re.error as e:
            pattern_result["error"] = str(e)
            results["summary"]["failing_patterns"] += 1
        
        results["pattern_results"].append(pattern_result)
    
    # Test each input
    for j, test_input in enumerate(test_inputs):
        input_result = {
            "index": j,
            "input": test_input,
            "matching_patterns": []
        }
        
        for i, pattern_result in enumerate(results["pattern_results"]):
            if pattern_result["valid_regex"]:
                for match in pattern_result["matches"]:
                    if match["input_index"] == j:
                        input_result["matching_patterns"].append({
                            "pattern_index": i,
                            "pattern": pattern_result["pattern"],
                            "match": match["match"]
                        })
        
        results["input_results"].append(input_result)
    
    return results

def print_pattern_test_results(patterns: List[str], test_inputs: List[str]):
    """Print formatted pattern test results."""
    
    print("ðŸ§ª Policy Pattern Testing")
    print("=" * 50)
    
    results = test_policy_patterns(patterns, test_inputs)
    
    # Summary
    summary = results["summary"]
    print(f"Patterns tested: {summary['total_patterns']}")
    print(f"Working patterns: {summary['working_patterns']}")
    print(f"Failing patterns: {summary['failing_patterns']}")
    print(f"Total matches: {summary['matches_found']}")
    
    # Pattern results
    print(f"\nðŸ“‹ Pattern Results:")
    for pattern_result in results["pattern_results"]:
        status = "âœ…" if pattern_result["valid_regex"] else "âŒ"
        matches_count = len(pattern_result["matches"])
        
        print(f"  {status} Pattern {pattern_result['index']}: {matches_count} matches")
        print(f"    Pattern: {pattern_result['pattern']}")
        
        if pattern_result["error"]:
            print(f"    âŒ Error: {pattern_result['error']}")
        elif pattern_result["matches"]:
            print(f"    Matches:")
            for match in pattern_result["matches"][:3]:  # Show first 3 matches
                print(f"      â€¢ '{match['match']}' in input {match['input_index']}")
            if len(pattern_result["matches"]) > 3:
                print(f"      ... and {len(pattern_result['matches']) - 3} more")
    
    # Input results
    print(f"\nðŸ“ Input Results:")
    for input_result in results["input_results"]:
        pattern_count = len(input_result["matching_patterns"])
        status = "ðŸš«" if pattern_count > 0 else "âœ…"
        
        print(f"  {status} Input {input_result['index']}: {pattern_count} pattern matches")
        print(f"    Input: '{input_result['input'][:60]}{'...' if len(input_result['input']) > 60 else ''}'")
        
        if input_result["matching_patterns"]:
            for match in input_result["matching_patterns"]:
                print(f"      â€¢ Pattern {match['pattern_index']}: '{match['match']}'")

# Example usage with common sensitive patterns:
sensitive_patterns = [
    r'\b\d{3}-\d{2}-\d{4}\b',  # SSN
    r'\b\d{4}[-\s]\d{4}[-\s]\d{4}[-\s]\d{4}\b',  # Credit card
    r'\bsk-[a-zA-Z0-9]{48}\b',  # OpenAI API key
    r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Email
    r'\blet me think step by step\b',  # Chain of thought
    r'\bI need to think about this\b'  # Reasoning leak
]

test_inputs = [
    "Hello, how can I help you?",
    "My SSN is 123-45-6789 and my email is user@example.com",
    "Here's my credit card: 1234 5678 9012 3456",
    "The API key is sk-abcd1234efgh5678ijkl9012mnop3456qrst7890uvwx1234",
    "Let me think step by step about this problem...",
    "I need to think about this carefully before responding",
    "This is a normal conversation message"
]

print_pattern_test_results(sensitive_patterns, test_inputs)
```

## ðŸ”§ Common Policy Issues

### 1. Policies Not Loading

**Symptoms**:
- No policies found warnings
- All messages pass through without evaluation
- Policy file appears correct

**Diagnosis**:
```python
import os
from rizk.sdk.config import get_policies_path
from rizk.sdk.guardrails.fast_rules import FastRulesEngine

# Check policies path
policies_path = get_policies_path()
print(f"Policies path: {policies_path}")
print(f"Path exists: {os.path.exists(policies_path)}")

if os.path.exists(policies_path):
    print(f"Files in directory: {os.listdir(os.path.dirname(policies_path))}")
    
    # Try loading manually
    try:
        engine = FastRulesEngine(policies_path)
        print(f"Policies loaded: {len(engine.policies)}")
        for policy in engine.policies[:3]:
            print(f"  â€¢ {policy.id}: {policy.name}")
    except Exception as e:
        print(f"Loading error: {e}")
```

**Solutions**:
```python
import os

# 1. Set explicit policies path
os.environ["RIZK_POLICIES_PATH"] = "/absolute/path/to/policies.yaml"

# 2. Check file permissions
policies_path = get_policies_path()
if os.path.exists(policies_path):
    print(f"File readable: {os.access(policies_path, os.R_OK)}")

# 3. Validate file content
print_validation_report(policies_path)

# 4. Use default policies if custom ones fail
from rizk.sdk.guardrails.fast_rules import FastRulesEngine
try:
    # Try custom policies first
    engine = FastRulesEngine(policies_path)
except:
    # Fallback to default policies
    engine = FastRulesEngine()
```

### 2. Regex Patterns Not Matching

**Symptoms**:
- Expected content not being blocked
- Patterns work in regex testers but not in Rizk
- Inconsistent matching behavior

**Diagnosis**:
```python
# Test pattern matching directly
import re

pattern = r'\b\d{3}-\d{2}-\d{4}\b'  # SSN pattern
test_text = "My SSN is 123-45-6789"

# Test basic matching
match = re.search(pattern, test_text)
print(f"Basic match: {match.group() if match else 'No match'}")

# Test with flags (Rizk uses IGNORECASE)
match_ci = re.search(pattern, test_text, re.IGNORECASE)
print(f"Case-insensitive match: {match_ci.group() if match_ci else 'No match'}")

# Test with Rizk's evaluation
from rizk.sdk.guardrails.fast_rules import FastRulesEngine
engine = FastRulesEngine()
result = engine.evaluate(test_text)
print(f"Rizk evaluation: {'BLOCKED' if result.blocked else 'ALLOWED'}")
print(f"Matched patterns: {result.matched_patterns}")
```

**Solutions**:
```python
# 1. Use proper regex escaping
patterns = [
    r'\bssn\s*:?\s*\d{3}[-\s]?\d{2}[-\s]?\d{4}\b',  # More flexible SSN
    r'\bcredit\s+card\s*:?\s*\d{4}[-\s]*\d{4}[-\s]*\d{4}[-\s]*\d{4}\b',  # Credit card
    r'\bapi\s*key\s*:?\s*sk-[a-zA-Z0-9]{48}\b'  # API key
]

# 2. Test patterns thoroughly
for pattern in patterns:
    print(f"Testing pattern: {pattern}")
    test_cases = [
        "SSN: 123-45-6789",
        "ssn 123456789", 
        "Credit card: 1234 5678 9012 3456",
        "API key: sk-abcd1234..."
    ]
    
    for test_case in test_cases:
        match = re.search(pattern, test_case, re.IGNORECASE)
        print(f"  '{test_case}': {'âœ…' if match else 'âŒ'}")

# 3. Use word boundaries carefully
# âŒ Won't match: 'email@domain.com' (no word boundary before @)
bad_pattern = r'\b\w+@\w+\.\w+\b'

# âœ… Will match: 'email@domain.com'
good_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
```

### 3. Policy Conflicts

**Symptoms**:
- Unexpected blocking or allowing
- Multiple policies triggering
- Inconsistent confidence scores

**Diagnosis**:
```python
def analyze_policy_conflicts(test_message: str):
    """Analyze which policies trigger for a message."""
    
    from rizk.sdk.guardrails.fast_rules import FastRulesEngine
    engine = FastRulesEngine()
    
    print(f"ðŸ” Policy Conflict Analysis: '{test_message}'")
    print("=" * 50)
    
    triggered_policies = []
    
    # Check each policy individually
    for policy in engine.policies:
        if hasattr(policy, 'patterns'):
            for pattern in policy.patterns:
                try:
                    import re
                    if re.search(pattern, test_message, re.IGNORECASE):
                        triggered_policies.append({
                            "policy_id": policy.id,
                            "policy_name": policy.name,
                            "action": policy.action,
                            "pattern": pattern,
                            "confidence": getattr(policy, 'confidence', 0.5)
                        })
                except re.error:
                    pass
    
    if triggered_policies:
        print(f"Triggered policies: {len(triggered_policies)}")
        for tp in triggered_policies:
            print(f"  â€¢ {tp['policy_id']}: {tp['action']} (confidence: {tp['confidence']})")
            print(f"    Pattern: {tp['pattern']}")
        
        # Determine final action
        block_policies = [p for p in triggered_policies if p['action'] == 'block']
        if block_policies:
            highest_conf = max(block_policies, key=lambda x: x['confidence'])
            print(f"  â†’ Final action: BLOCK (policy: {highest_conf['policy_id']})")
        else:
            print(f"  â†’ Final action: ALLOW")
    else:
        print("No policies triggered")

# Test with potentially conflicting content
analyze_policy_conflicts("My email is user@company.com and my SSN is 123-45-6789")
```

### 4. Performance Issues with Policies

**Symptoms**:
- Slow policy evaluation
- High CPU usage during guardrails processing
- Timeouts on large messages

**Diagnosis**:
```python
import time
from rizk.sdk.guardrails.fast_rules import FastRulesEngine

def benchmark_policy_performance():
    """Benchmark policy evaluation performance."""
    
    engine = FastRulesEngine()
    
    test_messages = [
        "Short message",
        "Medium length message with some more content to evaluate for potential policy violations",
        "Very long message " + "with lots of content " * 50,
        "Message with SSN 123-45-6789 and email user@domain.com"
    ]
    
    print("âš¡ Policy Performance Benchmark")
    print("=" * 40)
    
    for i, message in enumerate(test_messages):
        # Warm up
        for _ in range(5):
            engine.evaluate(message)
        
        # Benchmark
        times = []
        for _ in range(100):
            start = time.time()
            result = engine.evaluate(message)
            times.append((time.time() - start) * 1000)
        
        avg_time = sum(times) / len(times)
        print(f"Message {i+1} ({len(message)} chars): {avg_time:.2f}ms avg")
    
    # Test individual patterns
    print("\nðŸ“‹ Pattern Performance:")
    for policy in engine.policies[:5]:  # Test first 5 policies
        if hasattr(policy, 'patterns'):
            start = time.time()
            for _ in range(1000):
                for pattern in policy.patterns:
                    try:
                        import re
                        re.search(pattern, "test message", re.IGNORECASE)
                    except:
                        pass
            
            elapsed = (time.time() - start) * 1000
            print(f"  {policy.id}: {elapsed:.2f}ms for 1000 evaluations")

benchmark_policy_performance()
```

**Solutions**:
```python
# 1. Optimize regex patterns
# âŒ Slow: Complex nested groups with backtracking
slow_pattern = r'(\w+\s*)*@(\w+\s*)*\.(\w+\s*)*'

# âœ… Fast: Specific character classes without backtracking
fast_pattern = r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}'

# 2. Limit message length for evaluation
def evaluate_with_length_limit(message: str, max_length: int = 10000):
    if len(message) > max_length:
        # Evaluate first and last parts
        start_part = message[:max_length//2]
        end_part = message[-max_length//2:]
        truncated_message = start_part + "..." + end_part
        return engine.evaluate(truncated_message)
    return engine.evaluate(message)

# 3. Use policy caching for repeated evaluations
from functools import lru_cache

@lru_cache(maxsize=1000)
def cached_policy_evaluation(message_hash: str, message: str):
    return engine.evaluate(message)

# Usage: cached_policy_evaluation(hash(message), message)

# 4. Disable slow policies in production
import os
os.environ["RIZK_FAST_POLICIES_ONLY"] = "true"  # Skip complex patterns
```

## ðŸ“Š Policy Analytics

### Policy Usage Statistics

Track which policies are triggered most frequently:

```python
from collections import defaultdict, Counter
import json

class PolicyAnalytics:
    """Track policy usage and performance."""
    
    def __init__(self):
        self.policy_triggers = Counter()
        self.pattern_triggers = Counter()
        self.evaluation_times = defaultdict(list)
        self.blocked_messages = []
        self.allowed_messages = []
    
    def record_evaluation(self, message: str, result, evaluation_time: float):
        """Record a policy evaluation."""
        
        self.evaluation_times['total'].append(evaluation_time)
        
        if result.blocked:
            self.blocked_messages.append({
                'message': message[:100],  # First 100 chars
                'reason': result.reason,
                'confidence': result.confidence,
                'policy_id': result.policy_id
            })
            
            if result.policy_id:
                self.policy_triggers[result.policy_id] += 1
            
            if result.matched_patterns:
                for pattern in result.matched_patterns:
                    self.pattern_triggers[pattern] += 1
        else:
            self.allowed_messages.append(message[:100])
    
    def get_report(self) -> dict:
        """Generate analytics report."""
        
        total_evaluations = len(self.blocked_messages) + len(self.allowed_messages)
        avg_eval_time = sum(self.evaluation_times['total']) / len(self.evaluation_times['total']) if self.evaluation_times['total'] else 0
        
        return {
            'total_evaluations': total_evaluations,
            'blocked_count': len(self.blocked_messages),
            'allowed_count': len(self.allowed_messages),
            'block_rate': len(self.blocked_messages) / total_evaluations if total_evaluations > 0 else 0,
            'avg_evaluation_time_ms': avg_eval_time * 1000,
            'top_triggered_policies': dict(self.policy_triggers.most_common(10)),
            'top_triggered_patterns': dict(self.pattern_triggers.most_common(10)),
            'recent_blocked_messages': self.blocked_messages[-10:],  # Last 10
        }
    
    def print_report(self):
        """Print formatted analytics report."""
        
        report = self.get_report()
        
        print("ðŸ“Š Policy Analytics Report")
        print("=" * 40)
        print(f"Total evaluations: {report['total_evaluations']}")
        print(f"Blocked: {report['blocked_count']} ({report['block_rate']:.1%})")
        print(f"Allowed: {report['allowed_count']}")
        print(f"Avg evaluation time: {report['avg_evaluation_time_ms']:.2f}ms")
        
        if report['top_triggered_policies']:
            print(f"\nðŸ”¥ Top Triggered Policies:")
            for policy_id, count in report['top_triggered_policies'].items():
                print(f"  â€¢ {policy_id}: {count} times")
        
        if report['top_triggered_patterns']:
            print(f"\nðŸŽ¯ Top Triggered Patterns:")
            for pattern, count in report['top_triggered_patterns'].items():
                print(f"  â€¢ {pattern[:50]}...: {count} times")
        
        if report['recent_blocked_messages']:
            print(f"\nðŸš« Recent Blocked Messages:")
            for msg in report['recent_blocked_messages'][-5:]:
                print(f"  â€¢ {msg['message']} (reason: {msg['reason']})")

# Example usage:
analytics = PolicyAnalytics()

# Record some evaluations
test_messages = [
    "Hello world",
    "My SSN is 123-45-6789", 
    "Secret API key: sk-abc123",
    "Normal conversation",
    "Credit card: 1234-5678-9012-3456"
]

for message in test_messages:
    start_time = time.time()
    result = debug_policy_evaluation(message)
    eval_time = time.time() - start_time
    
    if result:
        analytics.record_evaluation(message, result, eval_time)

analytics.print_report()
```

This comprehensive policy debugging guide provides the tools and techniques needed to understand, troubleshoot, and optimize your Rizk SDK guardrails policies. Use these debugging methods to ensure your policies work correctly and efficiently protect your LLM applications.


