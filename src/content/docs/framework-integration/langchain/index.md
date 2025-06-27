---
title: "LangChain Integration"
description: "LangChain Integration"
---

# LangChain Integration

The Rizk SDK provides seamless integration with LangChain, offering comprehensive observability, tracing, and governance for LLM applications built with the LangChain framework. This guide covers everything from basic setup to advanced enterprise patterns.

## Overview

LangChain is a powerful framework for developing applications with large language models. Rizk SDK enhances LangChain applications with:

- **Automatic Chain Instrumentation**: Zero-configuration observability for all chains
- **Agent Monitoring**: Detailed tracing of agent decisions and tool usage
- **Callback Integration**: Native LangChain callback handlers for seamless integration
- **Memory Management**: Enhanced conversation memory with persistence and analytics
- **Policy Enforcement**: Real-time governance for LangChain workflows
- **Performance Analytics**: Comprehensive metrics for chains, agents, and tools

## Prerequisites

```bash
pip install rizk langchain langchain-openai
```

## Quick Start

### Basic Chain with Monitoring

```python
import os
from rizk.sdk import Rizk
from rizk.sdk.decorators import workflow, tool
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage

# Initialize Rizk SDK
rizk = Rizk.init(
    app_name="LangChain-Demo",
    enabled=True
)

@workflow(
    name="simple_chat",
    organization_id="demo_org",
    project_id="langchain_project"
)
def simple_chat_workflow(user_message: str) -> str:
    """Simple chat workflow with LangChain."""
    
    # For demo purposes, simulate LLM response
    # In production, this would use actual OpenAI API
    if "hello" in user_message.lower():
        return "Hello! How can I assist you today?"
    elif "weather" in user_message.lower():
        return "I'd be happy to help with weather information, but I need your location first."
    else:
        return f"I understand you said: '{user_message}'. How can I help you with that?"

@tool(
    name="search_tool",
    organization_id="demo_org", 
    project_id="langchain_project"
)
def search_tool(query: str) -> str:
    """Simulate a search tool for LangChain agents."""
    return f"Search results for '{query}': [Found relevant information about {query}]"

# Test the workflow
if __name__ == "__main__":
    test_messages = [
        "Hello there!",
        "What's the weather like?",
        "Tell me about artificial intelligence"
    ]
    
    print("ðŸ¦œ Testing LangChain with Rizk SDK...")
    print("=" * 50)
    
    for i, message in enumerate(test_messages, 1):
        print(f"\n{i}. User: {message}")
        response = simple_chat_workflow(message)
        print(f"   Assistant: {response}")
```

## Advanced LangChain Patterns

### Agent with Tools Integration

```python
from typing import List, Dict, Any
from rizk.sdk.decorators import workflow, agent, tool

@tool(name="calculator", organization_id="enterprise", project_id="langchain_agents")
def calculator_tool(expression: str) -> str:
    """Calculate mathematical expressions safely."""
    try:
        # Simple expression evaluation (production would use safer parsing)
        allowed_chars = set('0123456789+-*/.() ')
        if not all(c in allowed_chars for c in expression):
            return "Error: Invalid characters in expression"
        
        result = eval(expression)
        return f"Calculation result: {result}"
    except Exception as e:
        return f"Error calculating: {str(e)}"

@tool(name="weather_lookup", organization_id="enterprise", project_id="langchain_agents")
def weather_lookup_tool(location: str) -> str:
    """Look up weather information for a location."""
    # Simulate weather API call
    return f"Weather in {location}: Sunny, 72Â°F (22Â°C), light breeze"

@tool(name="web_search", organization_id="enterprise", project_id="langchain_agents")
def web_search_tool(query: str) -> str:
    """Search the web for information."""
    # Simulate web search
    return f"Web search for '{query}' found: Latest articles and information about {query}"

@workflow(name="langchain_agent_workflow", organization_id="enterprise", project_id="langchain_agents")
def run_langchain_agent(user_query: str) -> Dict[str, Any]:
    """
    Run a LangChain agent with multiple tools.
    
    Args:
        user_query: User's query to process
        
    Returns:
        Dictionary containing agent response and metadata
    """
    
    # Simulate agent reasoning and tool selection
    query_lower = user_query.lower()
    
    if any(word in query_lower for word in ['calculate', 'math', '+', '-', '*', '/']):
        # Use calculator tool
        import re
        math_expr = re.search(r'[\d\+\-\*/\.\(\)\s]+', user_query)
        if math_expr:
            tool_result = calculator_tool(math_expr.group().strip())
            response = f"I'll calculate that for you. {tool_result}"
        else:
            response = "I can help with calculations, but I need a valid mathematical expression."
        
        tools_used = ["calculator"]
    
    elif any(word in query_lower for word in ['weather', 'temperature', 'forecast']):
        # Use weather tool
        # Extract location (simplified)
        location = "your location"  # In production, use NER to extract location
        tool_result = weather_lookup_tool(location)
        response = f"Let me check the weather for you. {tool_result}"
        tools_used = ["weather_lookup"]
    
    elif any(word in query_lower for word in ['search', 'find', 'look up', 'information']):
        # Use web search tool
        tool_result = web_search_tool(user_query)
        response = f"I'll search for that information. {tool_result}"
        tools_used = ["web_search"]
    
    else:
        # General response without tools
        response = f"I understand you're asking about: {user_query}. How can I help you further?"
        tools_used = []
    
    return {
        "response": response,
        "tools_used": tools_used,
        "query": user_query,
        "timestamp": "2024-01-01T00:00:00Z"
    }

# Example usage
def demo_langchain_agent():
    """Demonstrate LangChain agent with tools."""
    
    test_queries = [
        "What is 25 * 4 + 100?",
        "What's the weather like today?",
        "Search for information about Python programming",
        "Tell me a joke"
    ]
    
    print("ðŸ¤– LangChain Agent with Tools Demo")
    print("=" * 35)
    
    for query in test_queries:
        print(f"\nðŸ“ Query: {query}")
        result = run_langchain_agent(query)
        print(f"ðŸ¤– Response: {result['response']}")
        print(f"ðŸ”§ Tools used: {result['tools_used']}")

# demo_langchain_agent()
```

### Chain Composition with Memory

```python
from dataclasses import dataclass
from typing import Optional, List
import json

@dataclass
class ConversationMemory:
    """Manage conversation memory for LangChain applications."""
    session_id: str
    messages: List[Dict[str, str]]
    context: Dict[str, Any]
    max_messages: int = 10
    
    def add_message(self, role: str, content: str):
        """Add a message to memory."""
        self.messages.append({
            "role": role,
            "content": content,
            "timestamp": "2024-01-01T00:00:00Z"
        })
        
        # Keep only recent messages
        if len(self.messages) > self.max_messages:
            self.messages = self.messages[-self.max_messages:]
    
    def get_context(self) -> str:
        """Get conversation context as string."""
        recent_messages = self.messages[-5:]  # Last 5 messages
        return "\n".join([f"{msg['role']}: {msg['content']}" for msg in recent_messages])
    
    def update_context(self, key: str, value: Any):
        """Update conversation context."""
        self.context[key] = value

@tool(name="memory_search", organization_id="enterprise", project_id="langchain_memory")
def memory_search_tool(query: str, context: str) -> str:
    """Search with conversation context."""
    return f"Context-aware search for '{query}' considering conversation history: {context[:100]}..."

@workflow(name="conversational_chain", organization_id="enterprise", project_id="langchain_memory")
def conversational_chain_workflow(
    user_message: str,
    session_id: str,
    memory: Optional[ConversationMemory] = None
) -> Dict[str, Any]:
    """
    Conversational chain with memory management.
    
    Args:
        user_message: User's current message
        session_id: Conversation session identifier
        memory: Optional existing conversation memory
        
    Returns:
        Dictionary containing response and updated memory
    """
    
    # Initialize or use existing memory
    if memory is None:
        memory = ConversationMemory(
            session_id=session_id,
            messages=[],
            context={"user_preferences": {}, "topics_discussed": []}
        )
    
    # Add user message to memory
    memory.add_message("user", user_message)
    
    # Generate response based on message and context
    context = memory.get_context()
    
    if "remember" in user_message.lower():
        # Extract information to remember
        info = user_message.replace("remember", "").strip()
        memory.update_context("user_info", info)
        response = f"I'll remember that: {info}"
    
    elif "what did" in user_message.lower() and "say" in user_message.lower():
        # Recall previous conversation
        if len(memory.messages) > 1:
            prev_messages = [msg for msg in memory.messages[-5:] if msg['role'] == 'user']
            if prev_messages:
                response = f"You recently mentioned: {prev_messages[-1]['content']}"
            else:
                response = "I don't have any previous messages from you in this conversation."
        else:
            response = "This is the start of our conversation."
    
    elif "search" in user_message.lower():
        # Use context-aware search
        search_result = memory_search_tool(user_message, context)
        response = f"Let me search for that with our conversation context. {search_result}"
    
    else:
        # General response with context awareness
        if len(memory.messages) > 2:
            response = f"Continuing our conversation about previous topics, regarding '{user_message}': I can help you with that."
        else:
            response = f"I understand you're asking about: {user_message}. How can I assist you?"
    
    # Add assistant response to memory
    memory.add_message("assistant", response)
    
    return {
        "response": response,
        "memory": memory,
        "session_id": session_id,
        "context_summary": memory.get_context()
    }

# Example usage
def demo_conversational_chain():
    """Demonstrate conversational chain with memory."""
    
    session_id = "demo_session_123"
    memory = None
    
    conversation = [
        "Hello, I'm working on a Python project",
        "Remember that I prefer using pandas for data analysis",
        "What did I say about my project?",
        "Search for Python best practices"
    ]
    
    print("ðŸ’­ Conversational Chain with Memory Demo")
    print("=" * 40)
    
    for i, message in enumerate(conversation, 1):
        print(f"\n{i}. User: {message}")
        
        result = conversational_chain_workflow(message, session_id, memory)
        memory = result["memory"]
        
        print(f"   Assistant: {result['response']}")
        print(f"   Memory size: {len(memory.messages)} messages")

# demo_conversational_chain()
```

## Production Patterns

### Enterprise LangChain Configuration

```python
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from rizk.sdk.decorators import workflow

@dataclass
class LangChainConfig:
    """Enterprise LangChain configuration."""
    
    # LLM Configuration
    model_name: str = "gpt-4"
    temperature: float = 0.1
    max_tokens: int = 2000
    
    # Chain Configuration
    max_chain_length: int = 10
    enable_memory: bool = True
    memory_ttl_seconds: int = 3600
    
    # Safety Configuration
    enable_content_filtering: bool = True
    max_input_length: int = 10000
    allowed_tools: List[str] = None
    
    # Performance Configuration
    request_timeout: int = 30
    max_retries: int = 3
    enable_caching: bool = True
    
    def __post_init__(self):
        if self.allowed_tools is None:
            self.allowed_tools = ["search", "calculator", "weather"]

class EnterpriseLangChainManager:
    """Manage LangChain applications with enterprise features."""
    
    def __init__(self, config: LangChainConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self._chain_cache = {}
    
    @workflow(name="enterprise_chain_execution", organization_id="production", project_id="langchain")
    def execute_chain_safely(
        self,
        chain_type: str,
        user_input: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Execute LangChain workflows with enterprise safety measures."""
        
        try:
            # Pre-execution validation
            if not self._validate_input(user_input):
                return {
                    "success": False,
                    "error": "Input validation failed",
                    "response": None
                }
            
            # Execute chain with monitoring
            response = self._execute_chain_with_retry(chain_type, user_input, context)
            
            # Post-execution validation
            if self.config.enable_content_filtering:
                response = self._filter_response(response)
            
            return {
                "success": True,
                "response": response,
                "chain_type": chain_type,
                "metadata": {
                    "input_length": len(user_input),
                    "output_length": len(response),
                    "model_used": self.config.model_name
                }
            }
            
        except Exception as e:
            self.logger.error(f"Chain execution failed: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "response": "I apologize, but I'm experiencing technical difficulties. Please try again later."
            }
    
    def _validate_input(self, user_input: str) -> bool:
        """Validate user input for safety."""
        
        # Check input length
        if len(user_input) > self.config.max_input_length:
            self.logger.warning(f"Input too long: {len(user_input)} characters")
            return False
        
        # Check for harmful content
        harmful_patterns = [
            "<script>", "javascript:", "data:", "vbscript:",
            "DROP TABLE", "DELETE FROM", "INSERT INTO",
            "rm -rf", "sudo", "chmod 777"
        ]
        
        user_input_lower = user_input.lower()
        for pattern in harmful_patterns:
            if pattern.lower() in user_input_lower:
                self.logger.warning(f"Harmful pattern detected: {pattern}")
                return False
        
        return True
    
    def _execute_chain_with_retry(
        self,
        chain_type: str,
        user_input: str,
        context: Optional[Dict[str, Any]]
    ) -> str:
        """Execute chain with retry logic."""
        
        for attempt in range(self.config.max_retries):
            try:
                # Check cache first
                if self.config.enable_caching:
                    cache_key = f"{chain_type}:{hash(user_input)}"
                    if cache_key in self._chain_cache:
                        self.logger.info("Returning cached response")
                        return self._chain_cache[cache_key]
                
                # Execute chain based on type
                if chain_type == "qa":
                    response = self._execute_qa_chain(user_input, context)
                elif chain_type == "agent":
                    response = self._execute_agent_chain(user_input, context)
                elif chain_type == "summarization":
                    response = self._execute_summarization_chain(user_input, context)
                else:
                    response = self._execute_general_chain(user_input, context)
                
                # Cache successful response
                if self.config.enable_caching:
                    self._chain_cache[cache_key] = response
                
                return response
                
            except Exception as e:
                if attempt == self.config.max_retries - 1:
                    raise e
                self.logger.warning(f"Attempt {attempt + 1} failed: {str(e)}")
        
        raise Exception("All retry attempts failed")
    
    def _execute_qa_chain(self, user_input: str, context: Optional[Dict[str, Any]]) -> str:
        """Execute Q&A chain."""
        # Simulate Q&A processing
        return f"Q&A response to: {user_input}"
    
    def _execute_agent_chain(self, user_input: str, context: Optional[Dict[str, Any]]) -> str:
        """Execute agent chain."""
        # Simulate agent processing
        return f"Agent response to: {user_input}"
    
    def _execute_summarization_chain(self, user_input: str, context: Optional[Dict[str, Any]]) -> str:
        """Execute summarization chain."""
        # Simulate summarization
        word_count = len(user_input.split())
        return f"Summary of {word_count} words: Key points extracted from the provided text."
    
    def _execute_general_chain(self, user_input: str, context: Optional[Dict[str, Any]]) -> str:
        """Execute general purpose chain."""
        # Simulate general processing
        return f"General response to: {user_input}"
    
    def _filter_response(self, response: str) -> str:
        """Filter response for harmful content."""
        # Implement content filtering
        import re
        
        # Remove potential sensitive information
        response = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL]', response)
        response = re.sub(r'\b\d{3}-\d{3}-\d{4}\b', '[PHONE]', response)
        response = re.sub(r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b', '[CARD]', response)
        
        return response

# Example usage
def demo_enterprise_langchain():
    """Demonstrate enterprise LangChain management."""
    
    config = LangChainConfig(
        model_name="gpt-4",
        temperature=0.1,
        enable_content_filtering=True,
        enable_caching=True
    )
    
    manager = EnterpriseLangChainManager(config)
    
    test_cases = [
        ("qa", "What is the capital of France?"),
        ("agent", "Find information about renewable energy"),
        ("summarization", "Please summarize this long text about machine learning and artificial intelligence applications in modern business environments."),
        ("general", "Hello, how are you?")
    ]
    
    print("ðŸ¢ Enterprise LangChain Management Demo")
    print("=" * 40)
    
    for chain_type, user_input in test_cases:
        print(f"\nðŸ”— Chain Type: {chain_type}")
        print(f"ðŸ“ Input: {user_input}")
        
        result = manager.execute_chain_safely(chain_type, user_input)
        
        if result["success"]:
            print(f"âœ… Response: {result['response']}")
            print(f"ðŸ“Š Metadata: {result['metadata']}")
        else:
            print(f"âŒ Error: {result['error']}")

# demo_enterprise_langchain()
```

### Performance Monitoring and Analytics

```python
import time
from typing import Dict, List, Any
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict

@dataclass
class LangChainMetrics:
    """Track LangChain application metrics."""
    
    chain_type: str
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    
    # Performance metrics
    response_times: List[float] = field(default_factory=list)
    token_usage: List[int] = field(default_factory=list)
    
    # Error tracking
    error_types: Dict[str, int] = field(default_factory=dict)
    
    # Chain-specific metrics
    tool_usage: Dict[str, int] = field(default_factory=dict)
    memory_usage: List[int] = field(default_factory=list)
    
    def add_request(
        self,
        response_time: float,
        success: bool,
        tokens_used: int = 0,
        tools_used: List[str] = None,
        memory_size: int = 0,
        error_type: str = None
    ):
        """Add a request to metrics."""
        self.total_requests += 1
        self.response_times.append(response_time)
        self.token_usage.append(tokens_used)
        self.memory_usage.append(memory_size)
        
        if success:
            self.successful_requests += 1
        else:
            self.failed_requests += 1
            if error_type:
                self.error_types[error_type] = self.error_types.get(error_type, 0) + 1
        
        # Track tool usage
        if tools_used:
            for tool in tools_used:
                self.tool_usage[tool] = self.tool_usage.get(tool, 0) + 1
    
    def get_success_rate(self) -> float:
        """Calculate success rate percentage."""
        if self.total_requests == 0:
            return 0.0
        return (self.successful_requests / self.total_requests) * 100
    
    def get_average_response_time(self) -> float:
        """Calculate average response time."""
        if not self.response_times:
            return 0.0
        return sum(self.response_times) / len(self.response_times)
    
    def get_average_tokens(self) -> float:
        """Calculate average token usage."""
        if not self.token_usage:
            return 0.0
        return sum(self.token_usage) / len(self.token_usage)

class LangChainMonitor:
    """Monitor LangChain application performance."""
    
    def __init__(self):
        self.metrics: Dict[str, LangChainMetrics] = defaultdict(lambda: LangChainMetrics("unknown"))
        self.session_data: Dict[str, Dict] = {}
    
    def track_chain_execution(self, chain_type: str):
        """Decorator to track chain execution."""
        def decorator(func):
            def wrapper(*args, **kwargs):
                start_time = time.time()
                success = False
                error_type = None
                tools_used = []
                tokens_used = 0
                memory_size = 0
                
                try:
                    result = func(*args, **kwargs)
                    success = True
                    
                    # Extract metrics from result if available
                    if isinstance(result, dict):
                        tools_used = result.get("tools_used", [])
                        tokens_used = result.get("tokens_used", 0)
                        memory_size = result.get("memory_size", 0)
                    
                    return result
                    
                except Exception as e:
                    error_type = type(e).__name__
                    raise
                    
                finally:
                    response_time = time.time() - start_time
                    
                    # Initialize metrics if not exists
                    if chain_type not in self.metrics:
                        self.metrics[chain_type] = LangChainMetrics(chain_type)
                    
                    # Record metrics
                    self.metrics[chain_type].add_request(
                        response_time=response_time,
                        success=success,
                        tokens_used=tokens_used,
                        tools_used=tools_used,
                        memory_size=memory_size,
                        error_type=error_type
                    )
            
            return wrapper
        return decorator
    
    def get_comprehensive_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        report = {
            "timestamp": datetime.now().isoformat(),
            "chains": {},
            "summary": {
                "total_chains": len(self.metrics),
                "total_requests": sum(m.total_requests for m in self.metrics.values()),
                "overall_success_rate": 0.0
            }
        }
        
        total_successful = sum(m.successful_requests for m in self.metrics.values())
        total_requests = sum(m.total_requests for m in self.metrics.values())
        
        if total_requests > 0:
            report["summary"]["overall_success_rate"] = (total_successful / total_requests) * 100
        
        for chain_type, metrics in self.metrics.items():
            report["chains"][chain_type] = {
                "total_requests": metrics.total_requests,
                "success_rate": round(metrics.get_success_rate(), 2),
                "average_response_time": round(metrics.get_average_response_time(), 3),
                "average_tokens": round(metrics.get_average_tokens(), 1),
                "tool_usage": dict(metrics.tool_usage),
                "error_breakdown": dict(metrics.error_types),
                "performance_trends": {
                    "p50_response_time": self._calculate_percentile(metrics.response_times, 0.5),
                    "p95_response_time": self._calculate_percentile(metrics.response_times, 0.95),
                    "p99_response_time": self._calculate_percentile(metrics.response_times, 0.99)
                }
            }
        
        return report
    
    def _calculate_percentile(self, values: List[float], percentile: float) -> float:
        """Calculate percentile of values."""
        if not values:
            return 0.0
        sorted_values = sorted(values)
        index = int(percentile * len(sorted_values))
        return round(sorted_values[min(index, len(sorted_values) - 1)], 3)

# Global monitor instance
langchain_monitor = LangChainMonitor()

@workflow(name="monitored_qa_chain", organization_id="production", project_id="langchain_monitoring")
@langchain_monitor.track_chain_execution("qa")
def monitored_qa_chain(question: str) -> Dict[str, Any]:
    """Q&A chain with monitoring."""
    
    # Simulate processing time
    time.sleep(0.05)  # Small delay for testing
    
    # Simulate token usage and tools
    tokens_used = len(question.split()) * 10  # Rough estimate
    tools_used = ["search"] if "search" in question.lower() else []
    
    return {
        "response": f"Answer to: {question}",
        "tokens_used": tokens_used,
        "tools_used": tools_used,
        "memory_size": 5
    }

@workflow(name="monitored_agent_chain", organization_id="production", project_id="langchain_monitoring")
@langchain_monitor.track_chain_execution("agent")
def monitored_agent_chain(task: str) -> Dict[str, Any]:
    """Agent chain with monitoring."""
    
    # Simulate processing time
    time.sleep(0.08)  # Small delay for testing
    
    # Simulate more complex processing
    tokens_used = len(task.split()) * 15
    tools_used = ["search", "calculator"] if "calculate" in task.lower() else ["search"]
    
    return {
        "response": f"Agent completed task: {task}",
        "tokens_used": tokens_used,
        "tools_used": tools_used,
        "memory_size": 10
    }

# Example usage
def demo_langchain_monitoring():
    """Demonstrate LangChain monitoring capabilities."""
    
    print("ðŸ“Š LangChain Performance Monitoring Demo")
    print("=" * 45)
    
    # Test Q&A chains
    qa_questions = [
        "What is machine learning?",
        "How does neural network training work?",
        "Explain natural language processing"
    ]
    
    print("\nðŸ”— Testing Q&A Chains:")
    for question in qa_questions:
        result = monitored_qa_chain(question)
        print(f"  âœ“ Q: {question[:40]}...")
        print(f"    Tokens: {result['tokens_used']}, Tools: {result['tools_used']}")
    
    # Test Agent chains
    agent_tasks = [
        "Research the latest AI developments",
        "Calculate the ROI of our ML project",
        "Summarize quarterly performance reports"
    ]
    
    print("\nðŸ¤– Testing Agent Chains:")
    for task in agent_tasks:
        result = monitored_agent_chain(task)
        print(f"  âœ“ Task: {task[:40]}...")
        print(f"    Tokens: {result['tokens_used']}, Tools: {result['tools_used']}")
    
    # Generate comprehensive report
    report = langchain_monitor.get_comprehensive_report()
    
    print("\nðŸ“ˆ Performance Report:")
    print(f"  Total Chains: {report['summary']['total_chains']}")
    print(f"  Total Requests: {report['summary']['total_requests']}")
    print(f"  Overall Success Rate: {report['summary']['overall_success_rate']:.1f}%")
    
    for chain_type, stats in report["chains"].items():
        print(f"\n  ðŸ“Š {chain_type.upper()} Chain:")
        print(f"    Requests: {stats['total_requests']}")
        print(f"    Success Rate: {stats['success_rate']}%")
        print(f"    Avg Response Time: {stats['average_response_time']}s")
        print(f"    Avg Tokens: {stats['average_tokens']}")
        print(f"    Tools Used: {stats['tool_usage']}")

# demo_langchain_monitoring()
```

## Configuration and Best Practices

### Environment Configuration

```python
import os
from typing import Optional, List
from dataclasses import dataclass

@dataclass
class LangChainProductionConfig:
    """Production configuration for LangChain integration."""
    
    # OpenAI Configuration
    openai_api_key: str = ""
    openai_model: str = "gpt-4"
    openai_temperature: float = 0.1
    openai_max_tokens: int = 2000
    
    # LangChain Configuration
    langchain_cache_enabled: bool = True
    langchain_verbose: bool = False
    langchain_debug: bool = False
    
    # Rizk SDK Configuration
    rizk_api_key: str = ""
    rizk_app_name: str = "LangChain-Production"
    rizk_enabled: bool = True
    rizk_trace_content: bool = False
    
    # Performance Configuration
    max_concurrent_requests: int = 10
    request_timeout: int = 30
    retry_attempts: int = 3
    
    # Safety Configuration
    enable_content_moderation: bool = True
    max_input_tokens: int = 8000
    max_output_tokens: int = 2000
    
    @classmethod
    def from_environment(cls) -> 'LangChainProductionConfig':
        """Load configuration from environment variables."""
        return cls(
            # OpenAI
            openai_api_key=os.getenv("OPENAI_API_KEY", ""),
            openai_model=os.getenv("OPENAI_MODEL", "gpt-4"),
            openai_temperature=float(os.getenv("OPENAI_TEMPERATURE", "0.1")),
            openai_max_tokens=int(os.getenv("OPENAI_MAX_TOKENS", "2000")),
            
            # LangChain
            langchain_cache_enabled=os.getenv("LANGCHAIN_CACHE_ENABLED", "true").lower() == "true",
            langchain_verbose=os.getenv("LANGCHAIN_VERBOSE", "false").lower() == "true",
            langchain_debug=os.getenv("LANGCHAIN_DEBUG", "false").lower() == "true",
            
            # Rizk SDK
            rizk_api_key=os.getenv("RIZK_API_KEY", ""),
            rizk_app_name=os.getenv("RIZK_APP_NAME", "LangChain-Production"),
            rizk_enabled=os.getenv("RIZK_ENABLED", "true").lower() == "true",
            rizk_trace_content=os.getenv("RIZK_TRACE_CONTENT", "false").lower() == "true",
            
            # Performance
            max_concurrent_requests=int(os.getenv("MAX_CONCURRENT_REQUESTS", "10")),
            request_timeout=int(os.getenv("REQUEST_TIMEOUT", "30")),
            retry_attempts=int(os.getenv("RETRY_ATTEMPTS", "3")),
            
            # Safety
            enable_content_moderation=os.getenv("ENABLE_CONTENT_MODERATION", "true").lower() == "true",
            max_input_tokens=int(os.getenv("MAX_INPUT_TOKENS", "8000")),
            max_output_tokens=int(os.getenv("MAX_OUTPUT_TOKENS", "2000"))
        )
    
    def validate(self) -> List[str]:
        """Validate configuration and return any errors."""
        errors = []
        
        if not self.openai_api_key:
            errors.append("OpenAI API key is required")
        
        if not self.rizk_api_key:
            errors.append("Rizk API key is required")
        
        if self.openai_temperature < 0 or self.openai_temperature > 2:
            errors.append("OpenAI temperature must be between 0 and 2")
        
        if self.max_concurrent_requests <= 0:
            errors.append("Max concurrent requests must be positive")
        
        if self.request_timeout <= 0:
            errors.append("Request timeout must be positive")
        
        return errors

def setup_production_environment():
    """Setup production environment for LangChain with Rizk."""
    
    # Load configuration
    config = LangChainProductionConfig.from_environment()
    
    # Validate configuration
    errors = config.validate()
    if errors:
        raise ValueError(f"Configuration errors: {', '.join(errors)}")
    
    # Initialize Rizk SDK
    rizk = Rizk.init(
        app_name=config.rizk_app_name,
        api_key=config.rizk_api_key,
        enabled=config.rizk_enabled,
        trace_content=config.rizk_trace_content
    )
    
    # Set OpenAI configuration
    os.environ["OPENAI_API_KEY"] = config.openai_api_key
    
    print(f"âœ… Production environment configured:")
    print(f"   - App: {config.rizk_app_name}")
    print(f"   - Model: {config.openai_model}")
    print(f"   - Tracing: {config.rizk_enabled}")
    print(f"   - Content Moderation: {config.enable_content_moderation}")
    
    return config, rizk

# Example .env file for production:
"""
# OpenAI Configuration
OPENAI_API_KEY=sk-your-openai-key-here
OPENAI_MODEL=gpt-4
OPENAI_TEMPERATURE=0.1
OPENAI_MAX_TOKENS=2000

# LangChain Configuration
LANGCHAIN_CACHE_ENABLED=true
LANGCHAIN_VERBOSE=false
LANGCHAIN_DEBUG=false

# Rizk SDK Configuration
RIZK_API_KEY=your-rizk-api-key-here
RIZK_APP_NAME=LangChain-Production
RIZK_ENABLED=true
RIZK_TRACE_CONTENT=false

# Performance Configuration
MAX_CONCURRENT_REQUESTS=10
REQUEST_TIMEOUT=30
RETRY_ATTEMPTS=3

# Safety Configuration
ENABLE_CONTENT_MODERATION=true
MAX_INPUT_TOKENS=8000
MAX_OUTPUT_TOKENS=2000
"""
```

## Testing and Validation

### Test Framework

```python
import unittest
from unittest.mock import Mock, patch
import asyncio

class TestLangChainIntegration(unittest.TestCase):
    """Test suite for LangChain integration."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = LangChainProductionConfig(
            openai_api_key="test-key",
            rizk_api_key="test-rizk-key",
            rizk_app_name="TestApp"
        )
    
    def test_simple_chat_workflow(self):
        """Test basic chat workflow."""
        from rizk.sdk import Rizk
        
        # Initialize Rizk for testing
        rizk = Rizk.init(app_name="TestLangChain", enabled=True)
        
        # Test the workflow
        result = simple_chat_workflow("Hello there!")
        
        self.assertIsInstance(result, str)
        self.assertIn("Hello", result)
    
    def test_agent_workflow(self):
        """Test agent workflow with tools."""
        test_queries = [
            "What is 25 * 4?",
            "What's the weather like?",
            "Search for Python tutorials"
        ]
        
        for query in test_queries:
            result = run_langchain_agent(query)
            
            self.assertIsInstance(result, dict)
            self.assertIn("response", result)
            self.assertIn("tools_used", result)
            self.assertIn("query", result)
    
    def test_conversational_memory(self):
        """Test conversational chain with memory."""
        session_id = "test_session"
        memory = None
        
        # First message
        result1 = conversational_chain_workflow(
            "Hello, I'm working on a Python project",
            session_id,
            memory
        )
        memory = result1["memory"]
        
        # Second message with memory
        result2 = conversational_chain_workflow(
            "Remember that I prefer pandas",
            session_id,
            memory
        )
        memory = result2["memory"]
        
        # Verify memory persistence
        self.assertEqual(len(memory.messages), 4)  # 2 user + 2 assistant
        self.assertEqual(memory.session_id, session_id)
        self.assertIn("pandas", memory.context.get("user_info", ""))
    
    def test_enterprise_manager(self):
        """Test enterprise LangChain manager."""
        config = LangChainConfig()
        manager = EnterpriseLangChainManager(config)
        
        # Test valid input
        result = manager.execute_chain_safely("qa", "What is AI?")
        self.assertTrue(result["success"])
        self.assertIn("response", result)
        
        # Test invalid input (too long)
        long_input = "x" * 20000
        result = manager.execute_chain_safely("qa", long_input)
        self.assertFalse(result["success"])
        self.assertIn("validation failed", result["error"])
    
    def test_performance_monitoring(self):
        """Test performance monitoring."""
        monitor = LangChainMonitor()
        
        @monitor.track_chain_execution("test")
        def test_chain(input_text: str) -> Dict[str, Any]:
            return {
                "response": f"Test response to {input_text}",
                "tokens_used": 100,
                "tools_used": ["test_tool"]
            }
        
        # Execute test chain
        result = test_chain("test input")
        
        # Verify monitoring
        self.assertIn("test", monitor.metrics)
        metrics = monitor.metrics["test"]
        self.assertEqual(metrics.total_requests, 1)
        self.assertEqual(metrics.successful_requests, 1)
        self.assertEqual(len(metrics.response_times), 1)
    
    def test_configuration_validation(self):
        """Test configuration validation."""
        # Valid configuration
        valid_config = LangChainProductionConfig(
            openai_api_key="test-key",
            rizk_api_key="test-rizk-key"
        )
        errors = valid_config.validate()
        self.assertEqual(len(errors), 0)
        
        # Invalid configuration
        invalid_config = LangChainProductionConfig(
            openai_api_key="",  # Missing
            rizk_api_key="test-rizk-key",
            openai_temperature=3.0  # Invalid
        )
        errors = invalid_config.validate()
        self.assertGreater(len(errors), 0)

if __name__ == "__main__":
    unittest.main()
```

## Troubleshooting

### Common Issues and Solutions

| Issue | Symptoms | Solution |
|-------|----------|----------|
| **API Key Issues** | `401 Unauthorized` errors | Verify `OPENAI_API_KEY` and `RIZK_API_KEY` are set |
| **Import Errors** | `ModuleNotFoundError` for LangChain | Install with `pip install langchain langchain-openai` |
| **Memory Leaks** | Increasing memory usage | Implement memory cleanup and limits |
| **Rate Limiting** | `429 Too Many Requests` | Implement exponential backoff and request queuing |
| **Chain Timeouts** | Slow or hanging chains | Adjust timeout settings and implement async patterns |
| **Tool Errors** | Tools not working in agents | Ensure tools are properly decorated and registered |

### Debug Mode

```python
import logging

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)

# Initialize with debug settings
rizk = Rizk.init(
    app_name="Debug-LangChain",
    enabled=True,
    verbose=True,
    trace_content=True  # Include content in traces (be careful with sensitive data)
)

@workflow(name="debug_chain", organization_id="debug", project_id="test")
def debug_langchain_workflow(user_input: str) -> Dict[str, Any]:
    """Debug workflow with comprehensive logging."""
    
    logger = logging.getLogger(__name__)
    logger.info(f"Processing LangChain request: {user_input}")
    
    try:
        # Your LangChain logic here
        result = f"Debug LangChain response to: {user_input}"
        logger.info(f"Generated LangChain response: {result}")
        
        return {
            "success": True,
            "response": result,
            "debug_info": {
                "input_length": len(user_input),
                "output_length": len(result),
                "chain_type": "debug",
                "timestamp": datetime.now().isoformat()
            }
        }
        
    except Exception as e:
        logger.error(f"LangChain workflow error: {str(e)}", exc_info=True)
        raise

# Test debug workflow
if __name__ == "__main__":
    result = debug_langchain_workflow("Debug test message")
    print(f"Debug result: {result}")
```

## Next Steps

1. **Explore Advanced Patterns**: Check out [Multi-Agent Workflows](../10-examples/multi-agent-workflow.md)
2. **Production Deployment**: See [Production Setup](../advanced-config/production-setup.md)
3. **Custom Tools**: Learn about [Tool Development](../decorators/tool.md)
4. **Memory Management**: Explore [Advanced Memory Patterns](../10-examples/rag-application.md)
5. **Security**: Review [Security Best Practices](../advanced-config/security.md)

---

**Enterprise Support**: For enterprise-specific LangChain integrations, custom chain development, or advanced configurations, contact our enterprise team at [enterprise@rizk.tools](mailto:enterprise@rizk.tools). 

