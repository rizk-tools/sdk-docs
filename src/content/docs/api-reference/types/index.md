---
title: "Types API Reference"
description: "Types API Reference"
---

# Types API Reference

This document provides a comprehensive reference for all type definitions, protocols, and data structures used in the Rizk SDK.

## Core Types

### `Decision`

**Basic decision object returned by simple policy evaluations.**

```python
@dataclass
class Decision:
    """Represents a guardrail decision result."""
    
    allowed: bool
    reason: Optional[str] = None
    policy_id: Optional[str] = None
    confidence: Optional[float] = None
```

#### Fields

| Field | Type | Description |
|-------|------|-------------|
| `allowed` | `bool` | Whether the content is allowed |
| `reason` | `Optional[str]` | Reason for blocking if not allowed |
| `policy_id` | `Optional[str]` | ID of the triggered policy |
| `confidence` | `Optional[float]` | Confidence score (0.0-1.0) |

#### Example

```python
# Create a decision
decision = Decision(
    allowed=False,
    reason="Contains sensitive information",
    policy_id="pii_detection",
    confidence=0.95
)

# Check result
if not decision.allowed:
    print(f"Blocked: {decision.reason}")
```

---

### `GuardrailProcessingResult`

**Comprehensive result from guardrail message processing.**

```python
@dataclass
class GuardrailProcessingResult:
    """Result from processing a message through guardrails."""
    
    allowed: bool
    confidence: float
    decision_layer: str
    violated_policies: Optional[List[str]] = None
    applied_policies: Optional[List[str]] = None
    response: Optional[str] = None
    blocked_reason: Optional[str] = None
    error: Optional[str] = None
    processing_time_ms: Optional[float] = None
```

#### Fields

| Field | Type | Description |
|-------|------|-------------|
| `allowed` | `bool` | Whether the message is allowed |
| `confidence` | `float` | Confidence score (0.0-1.0) |
| `decision_layer` | `str` | Layer that made decision: "fast_rules", "policy_augmentation", "llm_fallback", "error" |
| `violated_policies` | `Optional[List[str]]` | IDs of policies that were violated |
| `applied_policies` | `Optional[List[str]]` | IDs of policies that were applied |
| `response` | `Optional[str]` | Alternative response if blocked |
| `blocked_reason` | `Optional[str]` | Human-readable reason for blocking |
| `error` | `Optional[str]` | Error description if processing failed |
| `processing_time_ms` | `Optional[float]` | Processing time in milliseconds |

#### Example

```python
result = GuardrailProcessingResult(
    allowed=False,
    confidence=0.98,
    decision_layer="fast_rules",
    violated_policies=["profanity_filter", "hate_speech"],
    blocked_reason="Content violates community guidelines",
    response="I cannot process messages with inappropriate content.",
    processing_time_ms=15.3
)
```

---

### `GuardrailOutputCheckResult`

**Result from checking AI-generated output.**

```python
@dataclass
class GuardrailOutputCheckResult:
    """Result from checking AI output against policies."""
    
    allowed: bool
    confidence: float
    decision_layer: str
    violated_policies: Optional[List[str]] = None
    blocked_reason: Optional[str] = None
    transformed_response: Optional[str] = None
    error: Optional[str] = None
    processing_time_ms: Optional[float] = None
```

#### Fields

| Field | Type | Description |
|-------|------|-------------|
| `allowed` | `bool` | Whether the output is allowed |
| `confidence` | `float` | Confidence score (0.0-1.0) |
| `decision_layer` | `str` | Layer that made the decision |
| `violated_policies` | `Optional[List[str]]` | IDs of violated policies |
| `blocked_reason` | `Optional[str]` | Reason for blocking |
| `transformed_response` | `Optional[str]` | Redacted or modified response |
| `error` | `Optional[str]` | Error description if check failed |
| `processing_time_ms` | `Optional[float]` | Processing time in milliseconds |

#### Example

```python
result = GuardrailOutputCheckResult(
    allowed=False,
    confidence=0.92,
    decision_layer="llm_fallback",
    violated_policies=["pii_detection"],
    blocked_reason="Output contains personally identifiable information",
    transformed_response="User information has been redacted for privacy.",
    processing_time_ms=234.7
)
```

## Protocol Types

### `LLMServiceProtocol`

**Protocol for LLM service implementations used in guardrails.**

```python
class LLMServiceProtocol(Protocol):
    """Protocol for LLM services used in guardrail evaluation."""
    
    async def evaluate_policy(
        self, 
        message: str, 
        policy: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Evaluate a message against a specific policy.
        
        Args:
            message: The message to evaluate
            policy: Policy definition dictionary
            
        Returns:
            Dictionary with evaluation results:
            {
                "allowed": bool,
                "confidence": float,
                "reasoning": str
            }
        """
        ...
    
    async def generate_guidelines(
        self, 
        message: str, 
        policies: List[Dict[str, Any]]
    ) -> List[str]:
        """Generate policy guidelines for a message.
        
        Args:
            message: The user message
            policies: List of applicable policies
            
        Returns:
            List of guideline strings
        """
        ...
```

#### Implementation Example

```python
class CustomLLMService:
    """Custom LLM service implementation."""
    
    def __init__(self, model_name: str, api_key: str):
        self.model_name = model_name
        self.api_key = api_key
    
    async def evaluate_policy(
        self, 
        message: str, 
        policy: Dict[str, Any]
    ) -> Dict[str, Any]:
        # Custom evaluation logic
        evaluation_prompt = f"""
        Evaluate if this message violates the policy:
        
        Message: {message}
        Policy: {policy['description']}
        
        Respond with JSON: {{"allowed": true/false, "confidence": 0.0-1.0, "reasoning": "..."}}
        """
        
        # Call your LLM service
        response = await self.call_llm(evaluation_prompt)
        return parse_json_response(response)
    
    async def generate_guidelines(
        self, 
        message: str, 
        policies: List[Dict[str, Any]]
    ) -> List[str]:
        # Generate guidelines based on policies
        guidelines = []
        for policy in policies:
            guideline = f"Remember to {policy['guideline']}"
            guidelines.append(guideline)
        return guidelines
```

---

### `FrameworkAdapterProtocol`

**Protocol for framework-specific adapters.**

```python
class FrameworkAdapterProtocol(Protocol):
    """Protocol for framework adapters."""
    
    def adapt_workflow(
        self, 
        func: Callable, 
        name: Optional[str] = None,
        **kwargs: Any
    ) -> Callable:
        """Adapt a workflow function for the specific framework."""
        ...
    
    def adapt_task(
        self, 
        func: Callable, 
        name: Optional[str] = None,
        **kwargs: Any
    ) -> Callable:
        """Adapt a task function for the specific framework."""
        ...
    
    def adapt_agent(
        self, 
        func: Callable, 
        name: Optional[str] = None,
        **kwargs: Any
    ) -> Callable:
        """Adapt an agent function for the specific framework."""
        ...
    
    def adapt_tool(
        self, 
        func_or_class: Union[Callable, Type], 
        name: Optional[str] = None,
        **kwargs: Any
    ) -> Union[Callable, Type]:
        """Adapt a tool function or class for the specific framework."""
        ...
    
    async def apply_input_guardrails(
        self, 
        args: Tuple, 
        kwargs: Dict[str, Any],
        func_name: str,
        strategy: str
    ) -> Tuple[Tuple, Dict[str, Any]]:
        """Apply input guardrails to function arguments."""
        ...
    
    async def apply_output_guardrails(
        self, 
        result: Any,
        func_name: str
    ) -> Any:
        """Apply output guardrails to function result."""
        ...
    
    def apply_augmentation(
        self, 
        args: Tuple, 
        kwargs: Dict[str, Any],
        guidelines: List[str],
        func_name: str
    ) -> Tuple[Tuple, Dict[str, Any]]:
        """Apply policy augmentation to function arguments."""
        ...
```

## Enumeration Types

### `ViolationMode`

**Enumeration for guardrail violation handling modes.**

```python
class ViolationMode(str, Enum):
    """Violation handling modes for guardrails."""
    
    BLOCK = "block"         # Block the request entirely
    AUGMENT = "augment"     # Augment with guidelines and continue
    WARN = "warn"          # Log warning but allow
    ALTERNATIVE = "alternative"  # Return alternative response
    REDACT = "redact"      # Redact sensitive content
```

#### Usage

```python
from rizk.sdk.types import ViolationMode

# Configure guardrails with specific violation mode
@guardrails(
    on_input_violation=ViolationMode.BLOCK,
    on_output_violation=ViolationMode.REDACT
)
def sensitive_function(input_data: str) -> str:
    return process_sensitive_data(input_data)
```

---

### `DecisionLayer`

**Enumeration for guardrail decision layers.**

```python
class DecisionLayer(str, Enum):
    """Layers in the guardrail decision process."""
    
    FAST_RULES = "fast_rules"
    POLICY_AUGMENTATION = "policy_augmentation"
    LLM_FALLBACK = "llm_fallback"
    ERROR = "error"
```

#### Usage

```python
result = await guardrails.process_message("Hello world")
print(f"Decision made by: {result.decision_layer}")

if result.decision_layer == DecisionLayer.FAST_RULES:
    print("Quickly processed by pattern matching")
elif result.decision_layer == DecisionLayer.LLM_FALLBACK:
    print("Required LLM evaluation")
```

## Context Types

### `RizkContext`

**Context information passed through the SDK.**

```python
@dataclass
class RizkContext:
    """Context information for Rizk SDK operations."""
    
    conversation_id: Optional[str] = None
    organization_id: Optional[str] = None
    project_id: Optional[str] = None
    agent_id: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    request_id: Optional[str] = None
    
    # Framework-specific context
    framework: Optional[str] = None
    adapter_name: Optional[str] = None
    
    # Policy context
    policy_ids: Optional[List[str]] = None
    enforcement_level: Optional[str] = None
    
    # Additional metadata
    metadata: Optional[Dict[str, Any]] = None
```

#### Example

```python
context = RizkContext(
    conversation_id="conv_123",
    organization_id="acme_corp",
    project_id="customer_service",
    user_id="user_456",
    framework="langchain",
    policy_ids=["pii_policy", "content_safety"],
    metadata={"user_role": "customer", "priority": "high"}
)
```

---

### `PolicyContext`

**Context for policy evaluation.**

```python
@dataclass
class PolicyContext:
    """Context for policy evaluation and enforcement."""
    
    message: str
    direction: str  # "inbound" or "outbound"
    conversation_history: Optional[List[str]] = None
    user_context: Optional[Dict[str, Any]] = None
    system_context: Optional[Dict[str, Any]] = None
    
    # Timing context
    timestamp: Optional[datetime] = None
    timezone: Optional[str] = None
    
    # Conversation state
    turn_number: Optional[int] = None
    previous_violations: Optional[List[str]] = None
```

## Policy Types

### `Policy`

**Policy definition structure.**

```python
@dataclass
class Policy:
    """Policy definition for guardrails."""
    
    id: str
    name: str
    description: str
    category: str
    
    # Rule definition
    rules: List[Dict[str, Any]]
    patterns: Optional[List[str]] = None
    
    # Behavior configuration
    action: str  # "block", "warn", "augment", etc.
    confidence_threshold: float = 0.8
    
    # Metadata
    version: str = "1.0"
    enabled: bool = True
    tags: Optional[List[str]] = None
    
    # Guidelines
    guidelines: Optional[List[str]] = None
    violation_message: Optional[str] = None
```

#### Example

```python
policy = Policy(
    id="pii_detection",
    name="PII Detection Policy",
    description="Detects and blocks personally identifiable information",
    category="privacy",
    rules=[
        {
            "type": "regex",
            "pattern": r"\b\d{3}-\d{2}-\d{4}\b",
            "description": "Social Security Number"
        }
    ],
    action="block",
    confidence_threshold=0.9,
    guidelines=[
        "Never share social security numbers",
        "Protect user privacy at all times"
    ],
    violation_message="Cannot share personal identification numbers"
)
```

---

### `PolicySet`

**Collection of policies.**

```python
@dataclass
class PolicySet:
    """Collection of policies for evaluation."""
    
    id: str
    name: str
    description: str
    
    policies: List[Policy]
    
    # Configuration
    evaluation_order: Optional[List[str]] = None
    default_action: str = "allow"
    
    # Metadata
    version: str = "1.0"
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
```

## Result Types

### `FastRulesResult`

**Result from fast rules evaluation.**

```python
@dataclass
class FastRulesResult:
    """Result from fast rules pattern matching."""
    
    blocked: bool
    matched_rules: List[str]
    confidence: float
    reason: Optional[str] = None
    category: Optional[str] = None
    processing_time_ms: Optional[float] = None
```

---

### `PolicyAugmentationResult`

**Result from policy augmentation.**

```python
@dataclass
class PolicyAugmentationResult:
    """Result from policy augmentation process."""
    
    applicable_policies: List[str]
    guidelines: List[str]
    augmented_prompt: Optional[str] = None
    processing_time_ms: Optional[float] = None
```

---

### `LLMFallbackResult`

**Result from LLM fallback evaluation.**

```python
@dataclass
class LLMFallbackResult:
    """Result from LLM-based policy evaluation."""
    
    allowed: bool
    confidence: float
    reasoning: str
    evaluated_policies: List[str]
    violated_policies: List[str]
    processing_time_ms: Optional[float] = None
```

## Streaming Types

### `StreamChunk`

**Individual chunk in a streaming response.**

```python
@dataclass
class StreamChunk:
    """Individual chunk in a streaming response."""
    
    content: str
    chunk_id: int
    timestamp: datetime
    
    # Metadata
    token_count: Optional[int] = None
    is_final: bool = False
    
    # Guardrail evaluation
    guardrail_result: Optional[GuardrailOutputCheckResult] = None
```

---

### `StreamResult`

**Complete result from streaming processing.**

```python
@dataclass
class StreamResult:
    """Complete result from streaming processing."""
    
    chunks: List[StreamChunk]
    total_content: str
    
    # Aggregated metrics
    total_tokens: Optional[int] = None
    processing_time_ms: float
    
    # Guardrail summary
    violations_detected: List[str]
    total_chunks_blocked: int
```

## Error Types

### `RizkSDKError`

**Base exception for Rizk SDK errors.**

```python
class RizkSDKError(Exception):
    """Base exception for Rizk SDK errors."""
    
    def __init__(
        self, 
        message: str, 
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.details = details or {}
```

---

### `PolicyViolationError`

**Exception raised when a policy is violated.**

```python
class PolicyViolationError(RizkSDKError):
    """Exception raised when a policy violation is detected."""
    
    def __init__(
        self, 
        message: str,
        policy_id: str,
        violation_type: str,
        confidence: Optional[float] = None
    ):
        super().__init__(message, error_code="POLICY_VIOLATION")
        self.policy_id = policy_id
        self.violation_type = violation_type
        self.confidence = confidence
```

---

### `ConfigurationError`

**Exception raised for configuration issues.**

```python
class ConfigurationError(RizkSDKError):
    """Exception raised for configuration issues."""
    
    def __init__(
        self, 
        message: str,
        config_field: Optional[str] = None,
        validation_errors: Optional[List[str]] = None
    ):
        super().__init__(message, error_code="CONFIGURATION_ERROR")
        self.config_field = config_field
        self.validation_errors = validation_errors or []
```

## Type Aliases

### Common Type Aliases

```python
# Function types
F = TypeVar('F', bound=Callable[..., Any])
C = TypeVar('C', bound=Type)

# Configuration types
ConfigDict = Dict[str, Any]
EnvironmentVariables = Dict[str, str]

# Context types
ContextDict = Dict[str, Any]
MetadataDict = Dict[str, Any]

# Policy types
PolicyDict = Dict[str, Any]
RuleDict = Dict[str, Any]

# Result types
GuardrailResult = Union[GuardrailProcessingResult, GuardrailOutputCheckResult]
EvaluationResult = Union[Decision, GuardrailProcessingResult]
```

## Usage Examples

### Type Checking with mypy

```python
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rizk.sdk.types import GuardrailProcessingResult, RizkContext

async def process_with_types(
    message: str,
    context: "RizkContext"
) -> "GuardrailProcessingResult":
    """Process message with proper type annotations."""
    guardrails = GuardrailsEngine.get_instance()
    
    result: GuardrailProcessingResult = await guardrails.process_message(
        message, 
        context.to_dict() if context else None
    )
    
    return result
```

### Custom Protocol Implementation

```python
from rizk.sdk.types import LLMServiceProtocol

class MyCustomLLMService:
    """Custom LLM service implementation."""
    
    async def evaluate_policy(
        self, 
        message: str, 
        policy: Dict[str, Any]
    ) -> Dict[str, Any]:
        # Implementation
        return {
            "allowed": True,
            "confidence": 0.95,
            "reasoning": "Content appears safe"
        }
    
    async def generate_guidelines(
        self, 
        message: str, 
        policies: List[Dict[str, Any]]
    ) -> List[str]:
        # Implementation
        return ["Be helpful and safe"]

# Type checking ensures protocol compliance
service: LLMServiceProtocol = MyCustomLLMService()
```

### Result Processing Patterns

```python
from rizk.sdk.types import GuardrailProcessingResult, DecisionLayer

def handle_guardrail_result(result: GuardrailProcessingResult) -> str:
    """Handle guardrail result with proper typing."""
    
    if result.error:
        # Handle system error
        logger.error(f"Guardrail error: {result.error}")
        return "System temporarily unavailable"
    
    if not result.allowed:
        # Handle policy violation
        if result.response:
            return result.response
        else:
            return f"Request blocked: {result.blocked_reason}"
    
    # Request approved
    if result.decision_layer == DecisionLayer.FAST_RULES:
        logger.info("Quick approval via pattern matching")
    elif result.decision_layer == DecisionLayer.LLM_FALLBACK:
        logger.info("Approved after LLM evaluation")
    
    return "proceed"
```

## Related APIs

- **[Rizk Class API](./rizk-class.md)** - Main SDK interface
- **[GuardrailsEngine API](./guardrails-api.md)** - Policy enforcement engine
- **[Decorators API](./decorators-api.md)** - Function decorators
- **[Configuration API](./configuration-api.md)** - Configuration management 

