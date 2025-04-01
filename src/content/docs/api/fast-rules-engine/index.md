---
title: "FastRulesEngine Class"
description: "Documentation for FastRulesEngine Class"
---

The `FastRulesEngine` class provides fast, rule-based policy evaluation for messages and AI responses. It uses predefined rules to quickly check for policy violations.

## Class Definition

```python
class FastRulesEngine:
    """Fast rule-based policy evaluation engine."""
```

## Instance Methods

### evaluate

```python
def evaluate(
    message: str,
    context: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]
```

Evaluate a message against fast rules.

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `message` | `str` | - | The message to evaluate |
| `context` | `Optional[Dict[str, Any]]` | `None` | Additional context |

#### Returns

- `Dict[str, Any]`: Evaluation result with confidence and violation information

#### Example

```python
result = fast_rules.evaluate(
    message="User message here",
    context={"conversation_id": "unique_id"}
)

if not result["allowed"]:
    print(f"Violation detected: {result['violation_reason']}")
```

### add_rule

```python
def add_rule(
    rule_id: str,
    pattern: str,
    action: str,
    priority: int = 0
) -> None
```

Add a new fast rule.

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `rule_id` | `str` | - | Unique identifier for the rule |
| `pattern` | `str` | - | Regex pattern to match |
| `action` | `str` | - | Action to take on match ("block", "flag", "augment") |
| `priority` | `int` | `0` | Rule priority (higher numbers evaluated first) |

#### Example

```python
fast_rules.add_rule(
    rule_id="sensitive_data",
    pattern=r"(?i)(ssn|credit card|password)",
    action="block",
    priority=100
)
```

### remove_rule

```python
def remove_rule(rule_id: str) -> None
```

Remove a fast rule.

#### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `rule_id` | `str` | ID of the rule to remove |

#### Example

```python
fast_rules.remove_rule("sensitive_data")
```

### get_rules

```python
def get_rules() -> List[Dict[str, Any]]
```

Get all configured fast rules.

#### Returns

- `List[Dict[str, Any]]`: List of rule configurations

#### Example

```python
rules = fast_rules.get_rules()
for rule in rules:
    print(f"Rule {rule['id']}: {rule['pattern']}")
```

## Properties

| Property | Type | Description |
|----------|------|-------------|
| `rules` | `List[Dict[str, Any]]` | List of configured rules |
| `tracer` | `Tracer` | The OpenTelemetry tracer |

## Rule Configuration

Each rule is a dictionary with the following structure:

```python
{
    "id": str,          # Unique rule identifier
    "pattern": str,     # Regex pattern to match
    "action": str,      # Action to take ("block", "flag", "augment")
    "priority": int,    # Rule priority
    "metadata": dict    # Additional rule metadata
}
```

## Related Documentation

- [Guardrails Guide](../core-concepts/guardrails)
- [Policy Enforcement Guide](../core-concepts/policy-enforcement)
<!-- - [Examples](../examples/advanced-guardrails) -->
- [API Reference](../api/rizk)