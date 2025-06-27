---
title: "OpenAI Agents Integration"
description: "OpenAI Agents Integration"
---

# OpenAI Agents Integration

The Rizk SDK provides seamless integration with OpenAI Agents SDK, offering comprehensive observability, tracing, and governance for agent-based applications. This guide covers everything from basic setup to advanced enterprise patterns.

## Overview

OpenAI Agents SDK enables building autonomous agents that can use tools, maintain conversation context, and execute complex workflows. Rizk SDK enhances these capabilities with:

- **Automatic Agent Instrumentation**: Zero-configuration observability for all agents
- **Tool-Level Tracing**: Detailed monitoring of tool usage and performance
- **Conversation Context Management**: Hierarchical tracing across agent interactions
- **Policy Enforcement**: Real-time governance for agent behaviors
- **Performance Analytics**: Comprehensive metrics and insights

## Prerequisites

```bash
pip install rizk openai
```

## Quick Start

### Basic Agent with Monitoring

```python
import os
from rizk.sdk import Rizk
from rizk.sdk.decorators import agent, tool, workflow

# Initialize Rizk SDK
rizk = Rizk.init(
    app_name="OpenAI-Agents-Demo",
    enabled=True
)

@tool(
    name="calculator",
    organization_id="demo_org",
    project_id="agents_project"
)
def calculate(expression: str) -> str:
    """
    Safely evaluate mathematical expressions.
    
    Args:
        expression: Mathematical expression to evaluate (e.g., "2 + 3 * 4")
    
    Returns:
        Result of the calculation as a string
    """
    try:
        # Simple expression evaluation (production would use safer parsing)
        allowed_chars = set('0123456789+-*/.() ')
        if not all(c in allowed_chars for c in expression):
            return "Error: Invalid characters in expression"
        
        result = eval(expression)
        return f"Result: {result}"
    except Exception as e:
        return f"Error: {str(e)}"

@workflow(
    name="agent_conversation",
    organization_id="demo_org", 
    project_id="agents_project"
)
def run_agent_conversation(user_message: str) -> str:
    """Run a conversation with a math agent."""
    
    # For demo purposes, simulate agent behavior
    # In production, this would integrate with actual OpenAI Agents SDK
    
    if "calculate" in user_message.lower() or any(op in user_message for op in ['+', '-', '*', '/']):
        # Extract mathematical expression (simplified)
        import re
        math_expr = re.search(r'[\d\+\-\*/\.\(\)\s]+', user_message)
        if math_expr:
            return calculate(math_expr.group().strip())
    
    return f"I'm a math assistant. I can help you with calculations. You asked: '{user_message}'"

# Test the agent
if __name__ == "__main__":
    test_queries = [
        "What is 15 * 24 + 100?",
        "Calculate 25 + 15",
        "Help me solve: (100 - 25) / 3"
    ]
    
    print("ðŸ¤– Testing OpenAI Agent with Rizk SDK...")
    print("=" * 50)
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{i}. Query: {query}")
        try:
            result = run_agent_conversation(query)
            print(f"   Response: {result}")
        except Exception as e:
            print(f"   Error: {str(e)}")
```

## Advanced Agent Patterns

### Multi-Agent Collaboration

```python
import asyncio
from typing import List, Dict, Any
from rizk.sdk.decorators import workflow, agent, tool

@tool(name="web_search", organization_id="enterprise", project_id="research")
def web_search(query: str) -> str:
    """Simulate web search functionality."""
    # In production, integrate with real search API
    return f"Search results for '{query}': [Found 5 relevant articles about {query}]"

@tool(name="data_analysis", organization_id="enterprise", project_id="research") 
def analyze_data(data: str) -> str:
    """Analyze provided data and extract insights."""
    # In production, integrate with analytics tools
    word_count = len(data.split())
    return f"Analysis of {word_count} words: Key trends identified in the data suggest growing interest in the topic."

@workflow(name="research_workflow", organization_id="enterprise", project_id="research")
async def run_research_workflow(topic: str) -> Dict[str, Any]:
    """
    Coordinate multiple agents for comprehensive research.
    
    Args:
        topic: Research topic to investigate
        
    Returns:
        Dictionary containing research results and analysis
    """
    
    # Phase 1: Research (simulated)
    research_data = web_search(topic)
    
    # Phase 2: Analysis (simulated)
    analysis_result = analyze_data(research_data)
    
    return {
        "topic": topic,
        "research": research_data,
        "analysis": analysis_result,
        "timestamp": "2024-01-01T00:00:00Z"
    }

# Example usage
async def demo_research():
    topic = "AI trends in enterprise software 2024"
    results = await run_research_workflow(topic)
    
    print(f"Research Topic: {results['topic']}")
    print(f"Research: {results['research']}")
    print(f"Analysis: {results['analysis']}")

# asyncio.run(demo_research())
```

### Agent with Context Management

```python
from dataclasses import dataclass
from typing import Optional, List
import json

@dataclass
class ConversationContext:
    """Manage conversation context across agent interactions."""
    user_id: str
    session_id: str
    conversation_history: List[Dict[str, str]]
    user_preferences: Dict[str, Any]
    
    def add_message(self, role: str, content: str):
        """Add a message to conversation history."""
        self.conversation_history.append({
            "role": role,
            "content": content,
            "timestamp": "2024-01-01T00:00:00Z"
        })
    
    def get_context_summary(self) -> str:
        """Get a summary of the conversation context."""
        recent_messages = self.conversation_history[-5:]  # Last 5 messages
        return json.dumps({
            "recent_messages": recent_messages,
            "preferences": self.user_preferences
        })

@tool(name="context_aware_search", organization_id="enterprise", project_id="support")
def context_aware_search(query: str, context: str) -> str:
    """Search with conversation context awareness."""
    return f"Context-aware search for '{query}' found relevant help articles based on user context."

@workflow(name="contextual_support", organization_id="enterprise", project_id="support")
def handle_support_request(
    user_id: str,
    session_id: str, 
    message: str,
    context: Optional[ConversationContext] = None
) -> Dict[str, Any]:
    """Handle support request with context management."""
    
    # Initialize or load context
    if context is None:
        context = ConversationContext(
            user_id=user_id,
            session_id=session_id,
            conversation_history=[],
            user_preferences={"language": "en", "support_level": "standard"}
        )
    
    # Add user message to context
    context.add_message("user", message)
    
    # Process request (simulated agent response)
    if "login" in message.lower():
        response = "I can help you with login issues. Let me search for relevant solutions..."
        search_result = context_aware_search("login problems", context.get_context_summary())
        response += f" {search_result}"
    else:
        response = f"I understand you need help with: {message}. Let me assist you."
    
    # Add agent response to context
    context.add_message("assistant", response)
    
    return {
        "response": response,
        "context": context,
        "session_id": session_id
    }

# Example usage
def demo_contextual_support():
    """Demonstrate contextual support workflow."""
    
    context = None
    user_id = "user_123"
    session_id = "session_456"
    
    messages = [
        "I'm having trouble with my account login",
        "I tried resetting my password but didn't receive the email",
        "My email is john@example.com"
    ]
    
    print("ðŸŽ§ Contextual Support Demo")
    print("=" * 30)
    
    for i, message in enumerate(messages, 1):
        print(f"\n{i}. User: {message}")
        
        result = handle_support_request(user_id, session_id, message, context)
        context = result["context"]
        
        print(f"   Agent: {result['response'][:100]}...")

# demo_contextual_support()
```

## Production Patterns

### Enterprise Agent Configuration

```python
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

@dataclass
class AgentConfig:
    """Enterprise agent configuration."""
    model: str = "gpt-4"
    temperature: float = 0.1
    max_tokens: int = 2000
    timeout_seconds: int = 30
    retry_attempts: int = 3
    enable_function_calling: bool = True
    safety_checks: bool = True

class EnterpriseAgentManager:
    """Manage enterprise agents with advanced configuration."""
    
    def __init__(self, config: AgentConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    @workflow(name="safe_agent_execution", organization_id="production", project_id="main")
    def execute_safely(self, agent_instructions: str, user_message: str) -> Dict[str, Any]:
        """Execute agent with enterprise safety measures."""
        
        try:
            # Pre-execution validation
            if self.config.safety_checks:
                if not self._validate_input(user_message):
                    return {
                        "success": False,
                        "error": "Input validation failed",
                        "response": None
                    }
            
            # Execute with retry logic
            response = self._execute_with_retry(agent_instructions, user_message)
            
            # Post-execution validation
            if self.config.safety_checks:
                response = self._sanitize_response(response)
            
            return {
                "success": True,
                "error": None,
                "response": response
            }
            
        except Exception as e:
            self.logger.error(f"Agent execution failed: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "response": None
            }
    
    def _validate_input(self, message: str) -> bool:
        """Validate input message for safety."""
        if len(message) > 10000:  # Too long
            return False
        
        # Check for potentially harmful content
        harmful_patterns = ["<script>", "DROP TABLE", "rm -rf"]
        if any(pattern in message.lower() for pattern in harmful_patterns):
            return False
        
        return True
    
    def _execute_with_retry(self, instructions: str, message: str) -> str:
        """Execute agent with retry logic."""
        for attempt in range(self.config.retry_attempts):
            try:
                # Simulate agent execution
                response = f"Agent response following '{instructions}' to message: {message}"
                return response
            except Exception as e:
                if attempt == self.config.retry_attempts - 1:
                    raise e
                self.logger.warning(f"Attempt {attempt + 1} failed: {str(e)}")
        
        raise Exception("All retry attempts failed")
    
    def _sanitize_response(self, response: str) -> str:
        """Sanitize agent response for safety."""
        import re
        
        # Remove email addresses
        response = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL]', response)
        
        # Remove phone numbers
        response = re.sub(r'\b\d{3}-\d{3}-\d{4}\b', '[PHONE]', response)
        
        return response

# Example usage
def demo_enterprise_agent():
    """Demonstrate enterprise agent management."""
    
    config = AgentConfig(
        model="gpt-4",
        temperature=0.1,
        safety_checks=True,
        retry_attempts=2
    )
    
    manager = EnterpriseAgentManager(config)
    
    # Test safe execution
    instructions = "You are a customer service representative. Help users professionally."
    test_message = "How can I update my account information?"
    result = manager.execute_safely(instructions, test_message)
    
    print("ðŸ¢ Enterprise Agent Demo")
    print("=" * 25)
    print(f"Success: {result['success']}")
    if result['success']:
        print(f"Response: {result['response'][:100]}...")
    else:
        print(f"Error: {result['error']}")

# demo_enterprise_agent()
```

### Performance Monitoring

```python
import time
from typing import Dict, List
from dataclasses import dataclass, field
from datetime import datetime

@dataclass
class AgentMetrics:
    """Track agent performance metrics."""
    agent_name: str
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    average_response_time: float = 0.0
    response_times: List[float] = field(default_factory=list)
    error_types: Dict[str, int] = field(default_factory=dict)
    
    def add_request(self, response_time: float, success: bool, error_type: str = None):
        """Add a request to metrics."""
        self.total_requests += 1
        self.response_times.append(response_time)
        
        if success:
            self.successful_requests += 1
        else:
            self.failed_requests += 1
            if error_type:
                self.error_types[error_type] = self.error_types.get(error_type, 0) + 1
        
        # Update average response time
        self.average_response_time = sum(self.response_times) / len(self.response_times)
    
    def get_success_rate(self) -> float:
        """Calculate success rate percentage."""
        if self.total_requests == 0:
            return 0.0
        return (self.successful_requests / self.total_requests) * 100
    
    def get_p95_response_time(self) -> float:
        """Calculate 95th percentile response time."""
        if not self.response_times:
            return 0.0
        sorted_times = sorted(self.response_times)
        index = int(0.95 * len(sorted_times))
        return sorted_times[index]

class AgentMonitor:
    """Monitor and analyze agent performance."""
    
    def __init__(self):
        self.metrics: Dict[str, AgentMetrics] = {}
    
    def track_agent_call(self, agent_name: str):
        """Decorator to track agent performance."""
        def decorator(func):
            def wrapper(*args, **kwargs):
                start_time = time.time()
                success = False
                error_type = None
                
                try:
                    result = func(*args, **kwargs)
                    success = True
                    return result
                except Exception as e:
                    error_type = type(e).__name__
                    raise
                finally:
                    response_time = time.time() - start_time
                    
                    # Initialize metrics if not exists
                    if agent_name not in self.metrics:
                        self.metrics[agent_name] = AgentMetrics(agent_name)
                    
                    # Record metrics
                    self.metrics[agent_name].add_request(response_time, success, error_type)
            
            return wrapper
        return decorator
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        report = {
            "timestamp": datetime.now().isoformat(),
            "agents": {}
        }
        
        for agent_name, metrics in self.metrics.items():
            report["agents"][agent_name] = {
                "total_requests": metrics.total_requests,
                "success_rate": round(metrics.get_success_rate(), 2),
                "average_response_time": round(metrics.average_response_time, 3),
                "p95_response_time": round(metrics.get_p95_response_time(), 3),
                "error_breakdown": metrics.error_types
            }
        
        return report

# Global monitor instance
monitor = AgentMonitor()

@workflow(name="monitored_agent_workflow", organization_id="production", project_id="analytics")
@monitor.track_agent_call("customer_service_agent")
def run_monitored_agent(message: str) -> str:
    """Run agent with performance monitoring."""
    
    # Simulate processing time
    time.sleep(0.1)  # Remove in production
    
    # Simulate agent response
    return f"Monitored agent response to: {message}"

# Example usage
def demo_performance_monitoring():
    """Demonstrate agent performance monitoring."""
    
    print("ðŸ“Š Agent Performance Monitoring Demo")
    print("=" * 40)
    
    # Simulate multiple agent calls
    test_messages = [
        "How do I reset my password?",
        "What are your business hours?", 
        "I need help with billing",
        "Can you help me track my order?",
        "I want to cancel my subscription"
    ]
    
    for message in test_messages:
        try:
            response = run_monitored_agent(message)
            print(f"âœ“ Processed: {message[:30]}...")
        except Exception as e:
            print(f"âœ— Failed: {message[:30]}... - {str(e)}")
    
    # Generate performance report
    report = monitor.get_performance_report()
    
    print("\nðŸ“ˆ Performance Report:")
    for agent_name, stats in report["agents"].items():
        print(f"\nAgent: {agent_name}")
        print(f"  Total Requests: {stats['total_requests']}")
        print(f"  Success Rate: {stats['success_rate']}%")
        print(f"  Avg Response Time: {stats['average_response_time']}s")
        print(f"  P95 Response Time: {stats['p95_response_time']}s")

# demo_performance_monitoring()
```

## Configuration and Best Practices

### Environment Configuration

```python
import os
from typing import Optional, List
from dataclasses import dataclass

@dataclass  
class OpenAIAgentsConfig:
    """Configuration for OpenAI Agents integration."""
    
    # API Configuration
    api_key: str
    base_url: Optional[str] = None
    organization: Optional[str] = None
    
    # Agent Defaults
    default_model: str = "gpt-4"
    default_temperature: float = 0.1
    default_max_tokens: int = 2000
    
    # Performance Settings
    request_timeout: int = 30
    max_retries: int = 3
    retry_delay: float = 1.0
    
    # Monitoring Settings
    enable_tracing: bool = True
    enable_metrics: bool = True
    log_level: str = "INFO"
    
    # Safety Settings
    enable_content_filtering: bool = True
    max_input_length: int = 10000
    max_output_length: int = 5000
    
    @classmethod
    def from_environment(cls) -> 'OpenAIAgentsConfig':
        """Load configuration from environment variables."""
        return cls(
            api_key=os.getenv("OPENAI_API_KEY", ""),
            base_url=os.getenv("OPENAI_BASE_URL"),
            organization=os.getenv("OPENAI_ORGANIZATION"),
            default_model=os.getenv("OPENAI_DEFAULT_MODEL", "gpt-4"),
            default_temperature=float(os.getenv("OPENAI_DEFAULT_TEMPERATURE", "0.1")),
            request_timeout=int(os.getenv("OPENAI_REQUEST_TIMEOUT", "30")),
            enable_tracing=os.getenv("RIZK_TRACING_ENABLED", "true").lower() == "true",
            enable_metrics=os.getenv("RIZK_METRICS_ENABLED", "true").lower() == "true",
            log_level=os.getenv("LOG_LEVEL", "INFO")
        )
    
    def validate(self) -> List[str]:
        """Validate configuration and return any errors."""
        errors = []
        
        if not self.api_key:
            errors.append("OpenAI API key is required")
        
        if self.default_temperature < 0 or self.default_temperature > 2:
            errors.append("Temperature must be between 0 and 2")
        
        if self.request_timeout <= 0:
            errors.append("Request timeout must be positive")
        
        return errors

# Example configuration usage
def setup_production_config():
    """Setup production configuration with validation."""
    
    config = OpenAIAgentsConfig.from_environment()
    
    # Validate configuration
    errors = config.validate()
    if errors:
        raise ValueError(f"Configuration errors: {', '.join(errors)}")
    
    # Initialize Rizk with production settings
    rizk = Rizk.init(
        app_name="OpenAI-Agents-Production",
        enabled=config.enable_tracing
    )
    
    return config, rizk

# Example .env file content:
"""
OPENAI_API_KEY=sk-your-api-key-here
OPENAI_DEFAULT_MODEL=gpt-4
OPENAI_DEFAULT_TEMPERATURE=0.1
OPENAI_REQUEST_TIMEOUT=30
RIZK_TRACING_ENABLED=true
RIZK_METRICS_ENABLED=true
LOG_LEVEL=INFO
"""
```

## Testing and Validation

### Test Framework

```python
import unittest
from unittest.mock import Mock, patch
import asyncio

class TestOpenAIAgentsIntegration(unittest.TestCase):
    """Test suite for OpenAI Agents integration."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = OpenAIAgentsConfig(
            api_key="test-key",
            default_model="gpt-3.5-turbo",
            request_timeout=10
        )
    
    def test_agent_creation(self):
        """Test agent creation with proper configuration."""
        manager = EnterpriseAgentManager(AgentConfig())
        
        # Test safe execution
        result = manager.execute_safely(
            "You are a test agent",
            "Hello, world!"
        )
        
        # Verify result structure
        self.assertIn("success", result)
        self.assertIn("response", result)
        self.assertIn("error", result)
    
    def test_metrics_tracking(self):
        """Test performance metrics tracking."""
        metrics = AgentMetrics("test_agent")
        
        # Add test requests
        metrics.add_request(0.5, True)
        metrics.add_request(1.0, True)
        metrics.add_request(0.8, False, "TimeoutError")
        
        # Verify metrics
        self.assertEqual(metrics.total_requests, 3)
        self.assertEqual(metrics.successful_requests, 2)
        self.assertEqual(metrics.failed_requests, 1)
        self.assertAlmostEqual(metrics.get_success_rate(), 66.67, places=1)
        self.assertEqual(metrics.error_types["TimeoutError"], 1)
    
    def test_configuration_validation(self):
        """Test configuration validation."""
        # Valid configuration
        valid_config = OpenAIAgentsConfig(api_key="test-key")
        self.assertEqual(len(valid_config.validate()), 0)
        
        # Invalid configuration
        invalid_config = OpenAIAgentsConfig(
            api_key="",  # Missing API key
            default_temperature=3.0  # Invalid temperature
        )
        errors = invalid_config.validate()
        self.assertGreater(len(errors), 0)
        self.assertTrue(any("API key" in error for error in errors))
    
    def test_context_management(self):
        """Test conversation context management."""
        context = ConversationContext(
            user_id="test_user",
            session_id="test_session", 
            conversation_history=[],
            user_preferences={}
        )
        
        # Add messages
        context.add_message("user", "Hello")
        context.add_message("assistant", "Hi there")
        
        # Verify context
        self.assertEqual(len(context.conversation_history), 2)
        self.assertEqual(context.conversation_history[0]["role"], "user")
        self.assertEqual(context.conversation_history[1]["role"], "assistant")

if __name__ == "__main__":
    unittest.main()
```

## Troubleshooting

### Common Issues and Solutions

| Issue | Symptoms | Solution |
|-------|----------|----------|
| **API Key Authentication** | `401 Unauthorized` errors | Verify `OPENAI_API_KEY` is set correctly |
| **Rate Limiting** | `429 Too Many Requests` | Implement exponential backoff and request queuing |
| **Timeout Errors** | Requests hanging or timing out | Adjust `request_timeout` and implement async patterns |
| **Tool Registration** | Tools not available to agents | Ensure tools are decorated with `@tool` and passed to agent |
| **Memory Issues** | High memory usage | Implement conversation context cleanup and limits |
| **Tracing Not Working** | No traces in dashboard | Verify Rizk SDK initialization and API key |

### Debug Mode

```python
import logging

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)

# Initialize Rizk with verbose logging
rizk = Rizk.init(
    app_name="Debug-OpenAI-Agents",
    enabled=True,
    verbose=True
)

@workflow(name="debug_workflow", organization_id="debug", project_id="test")
def debug_agent_workflow(message: str) -> Dict[str, Any]:
    """Debug workflow with comprehensive logging."""
    
    logger = logging.getLogger(__name__)
    logger.info(f"Processing message: {message}")
    
    try:
        result = f"Debug response to: {message}"
        logger.info(f"Generated response: {result}")
        
        return {
            "success": True,
            "response": result,
            "debug_info": {
                "message_length": len(message),
                "response_length": len(result),
                "timestamp": datetime.now().isoformat()
            }
        }
        
    except Exception as e:
        logger.error(f"Error in debug workflow: {str(e)}", exc_info=True)
        raise

# Test debug workflow
if __name__ == "__main__":
    result = debug_agent_workflow("Debug test message")
    print(f"Debug result: {result}")
```

## Next Steps

1. **Explore Advanced Features**: Check out [Multi-Agent Workflows](../10-examples/multi-agent-workflow.md)
2. **Production Deployment**: See [Production Setup](../advanced-config/production-setup.md)
3. **Monitoring Setup**: Configure [Observability](../observability/dashboards.md)
4. **Custom Tools**: Learn about [Tool Development](../decorators/tool.md)
5. **Security**: Review [Security Best Practices](../advanced-config/security.md)

---

**Enterprise Support**: For enterprise-specific requirements, advanced configurations, or custom integrations, contact our enterprise team at [enterprise@rizk.tools](mailto:enterprise@rizk.tools). 

