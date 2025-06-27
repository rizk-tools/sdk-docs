---
title: "Adapters"
description: "Adapters"
---

# Adapters

Rizk SDK's adapter system enables universal LLM framework integration through a sophisticated adapter pattern. This document explains how adapters work, the framework integration approach, and how to extend the system for new frameworks.

## Overview

The adapter system provides a unified interface for integrating with any LLM framework while preserving framework-specific functionality and patterns. Each adapter translates between Rizk SDK's universal API and the framework's native operations.

```python
# Universal API works across all frameworks
from rizk.sdk.decorators import workflow, agent, tool

# OpenAI Agents
@agent(name="openai_assistant")
def create_openai_agent():
    return {"agent": "assistant"}

# LangChain  
@workflow(name="langchain_process")
def run_langchain_chain():
    return "chain result"

# CrewAI
@workflow(name="crew_process")
def run_crew():
    return "crew result"
```

## Adapter Architecture

### Base Adapter Interface

All adapters implement a common interface defined by `BaseAdapter`:

```python
from abc import ABC, abstractmethod
from typing import Any, Callable, Optional

class BaseAdapter(ABC):
    """Base class for all framework adapters."""
    
    @abstractmethod
    def adapt_workflow(self, func: Callable, name: str = None, **kwargs) -> Callable:
        """Adapt workflow-level functions."""
        pass
    
    @abstractmethod
    def adapt_task(self, func: Callable, name: str = None, **kwargs) -> Callable:
        """Adapt task-level functions."""
        pass
    
    @abstractmethod
    def adapt_agent(self, func: Callable, name: str = None, **kwargs) -> Callable:
        """Adapt agent-level functions."""
        pass
    
    @abstractmethod
    def adapt_tool(self, func_or_class: Any, name: str = None, **kwargs) -> Any:
        """Adapt tool functions or classes."""
        pass
```

### Adapter Registration

Adapters are automatically registered and selected based on framework detection:

```python
# Framework adapters registry
FRAMEWORK_ADAPTERS = {
    "openai_agents": "OpenAIAgentsAdapter",
    "langchain": "LangChainAdapter", 
    "crewai": "CrewAIAdapter",
    "llama_index": "LlamaIndexAdapter",
    "langgraph": "LangGraphAdapter",
    "standard": "StandardAdapter"  # Default fallback
}

# LLM client adapters registry  
LLM_CLIENT_ADAPTERS = {
    "openai": "OpenAICompletionAdapter",
    "anthropic": "AnthropicAdapter",
    "gemini": "GeminiAdapter",
    "ollama": "OllamaAdapter"
}
```

## Framework Adapters

### 1. OpenAI Agents Adapter

Integrates with the OpenAI Agents SDK for native agent operations:

```python
# OpenAI Agents integration example
from rizk.sdk.decorators import agent, tool

@tool(name="calculator")
def calculate(expression: str) -> str:
    """Calculator tool with automatic tracing."""
    try:
        result = eval(expression)  # Use safely in production
        return str(result)
    except Exception as e:
        return f"Error: {str(e)}"

@agent(name="math_assistant")
def create_math_agent():
    """Create math agent with OpenAI Agents SDK."""
    # Would integrate with actual OpenAI Agents SDK if available
    return {
        "name": "MathBot",
        "instructions": "You are a helpful math assistant",
        "functions": [calculate]
    }

# Usage
agent = create_math_agent()
print(f"Created agent: {agent['name']}")
```

### 2. LangChain Adapter

Integrates with LangChain through callback handlers and chain wrapping:

```python
# LangChain integration example
from rizk.sdk.decorators import workflow, agent

@agent(name="langchain_agent")
def create_langchain_agent():
    """Create LangChain agent with automatic callback integration."""
    # Simulates LangChain AgentExecutor
    return {
        "type": "langchain_agent",
        "tools": ["search", "calculator"],
        "callbacks": []  # Rizk callbacks automatically added
    }

@workflow(name="langchain_workflow")
def run_langchain_process(query: str):
    """Run LangChain process with full tracing."""
    agent = create_langchain_agent()
    
    # Simulate LangChain processing
    result = f"LangChain processed: {query}"
    
    return {
        "agent": agent,
        "query": query,
        "result": result
    }

# Usage
result = run_langchain_process("What is the weather today?")
print(f"LangChain result: {result['result']}")
```

### 3. CrewAI Adapter

Integrates with CrewAI for multi-agent workflow orchestration:

```python
# CrewAI integration example
from rizk.sdk.decorators import crew, agent, task

@agent(name="researcher")
def create_researcher():
    """Create research agent."""
    return {
        "role": "Researcher",
        "goal": "Research topics thoroughly",
        "backstory": "Expert researcher with analytical skills"
    }

@task(name="research_task")
def create_research_task():
    """Create research task."""
    return {
        "description": "Research the given topic",
        "expected_output": "Comprehensive research report"
    }

@crew(name="research_crew")
def create_research_crew():
    """Create and run research crew."""
    researcher = create_researcher()
    task = create_research_task()
    
    # Simulate CrewAI crew execution
    crew_result = {
        "agents": [researcher],
        "tasks": [task],
        "process": "sequential",
        "status": "completed"
    }
    
    return crew_result

# Usage
crew_result = create_research_crew()
print(f"Crew execution: {crew_result['status']}")
```

### 4. LlamaIndex Adapter

Integrates with LlamaIndex for document processing and retrieval:

```python
# LlamaIndex integration example
from rizk.sdk.decorators import workflow, tool

@tool(name="document_loader")
def load_documents(directory: str):
    """Load documents with tracing."""
    # Simulate document loading
    return {
        "documents": [f"doc1.txt", f"doc2.txt"],
        "directory": directory,
        "count": 2
    }

@workflow(name="document_query")
def create_query_engine(directory: str):
    """Create query engine with automatic tracing."""
    documents = load_documents(directory)
    
    # Simulate LlamaIndex query engine
    query_engine = {
        "type": "vector_store_index",
        "documents": documents,
        "query_method": "similarity_search"
    }
    
    return query_engine

def query_documents(query_engine: dict, query: str):
    """Query documents using the engine."""
    return {
        "query": query,
        "engine": query_engine["type"],
        "result": f"Answer based on {query_engine['documents']['count']} documents"
    }

# Usage
engine = create_query_engine("./docs")
response = query_documents(engine, "What is this about?")
print(f"Query response: {response['result']}")
```

## Standard Adapter

The fallback adapter for frameworks without specific integration:

```python
# Standard adapter example
from rizk.sdk.decorators import workflow, task, agent

@workflow(name="standard_workflow")
def standard_processing(data: dict) -> dict:
    """Standard workflow processing."""
    # Works with any framework or no framework
    processed_data = {
        "original": data,
        "processed": True,
        "framework": "standard"
    }
    
    return processed_data

@task(name="data_validation")
def validate_data(data: dict) -> bool:
    """Validate input data."""
    required_fields = ["id", "content"]
    return all(field in data for field in required_fields)

@agent(name="processing_agent")
def create_processing_agent():
    """Create a generic processing agent."""
    return {
        "name": "ProcessingAgent",
        "capabilities": ["data_processing", "validation"],
        "framework": "standard"
    }

# Usage
agent = create_processing_agent()
test_data = {"id": "123", "content": "test content"}

if validate_data(test_data):
    result = standard_processing(test_data)
    print(f"Processing result: {result}")
```

## Adapter Selection and Registration

### Automatic Adapter Selection

Adapters are selected automatically based on framework detection:

```python
# Example of how adapters are selected
def demonstrate_adapter_selection():
    """Demonstrate automatic adapter selection."""
    
    # Framework detection happens automatically
    from rizk.sdk.utils.framework_detection import detect_framework
    
    framework = detect_framework()
    print(f"Detected framework: {framework}")
    
    # Appropriate adapter is selected automatically
    @workflow(name="auto_adapted_workflow")
    def auto_workflow():
        return f"Adapted for {framework} framework"
    
    result = auto_workflow()
    print(f"Result: {result}")

# Usage
demonstrate_adapter_selection()
```

### Manual Adapter Override

Override automatic detection when needed:

```python
# Manual adapter override example
from rizk.sdk.decorators import workflow

@workflow(name="manual_override", framework="standard")
def manually_adapted_workflow():
    """Workflow with manual framework specification."""
    return "Using manually specified adapter"

# Force specific framework globally
import os
os.environ["RIZK_FORCE_FRAMEWORK"] = "langchain"

@workflow(name="forced_framework")
def forced_framework_workflow():
    """Workflow using globally forced framework."""
    return "Using globally forced framework"

# Usage
manual_result = manually_adapted_workflow()
forced_result = forced_framework_workflow()

print(f"Manual override: {manual_result}")
print(f"Forced framework: {forced_result}")
```

## Adapter Development Guide

### Creating a Custom Adapter

```python
# Custom adapter implementation
from rizk.sdk.adapters.base import BaseAdapter
from rizk.sdk.utils.framework_registry import register_framework_adapter

class CustomFrameworkAdapter(BaseAdapter):
    """Custom adapter for a proprietary framework."""
    
    def __init__(self):
        super().__init__()
        self.framework_name = "custom_framework"
    
    def adapt_workflow(self, func, name=None, **kwargs):
        """Adapt workflow functions for custom framework."""
        
        def wrapper(*args, **kwargs):
            print(f"Custom adapter: Executing workflow {name}")
            
            # Custom framework-specific logic here
            result = func(*args, **kwargs)
            
            # Add custom framework metadata
            if isinstance(result, dict):
                result["framework"] = "custom_framework"
                result["adapter"] = "CustomFrameworkAdapter"
            
            return result
        
        return wrapper
    
    def adapt_task(self, func, name=None, **kwargs):
        """Adapt task functions."""
        def wrapper(*args, **kwargs):
            print(f"Custom adapter: Executing task {name}")
            return func(*args, **kwargs)
        return wrapper
    
    def adapt_agent(self, func, name=None, **kwargs):
        """Adapt agent functions."""
        def wrapper(*args, **kwargs):
            print(f"Custom adapter: Creating agent {name}")
            agent = func(*args, **kwargs)
            
            # Add custom agent capabilities
            if isinstance(agent, dict):
                agent["custom_capabilities"] = ["monitoring", "logging"]
            
            return agent
        return wrapper
    
    def adapt_tool(self, func_or_class, name=None, **kwargs):
        """Adapt tool functions."""
        if callable(func_or_class):
            def wrapper(*args, **kwargs):
                print(f"Custom adapter: Using tool {name}")
                return func_or_class(*args, **kwargs)
            return wrapper
        return func_or_class

# Register the custom adapter
register_framework_adapter("custom_framework", CustomFrameworkAdapter)

# Usage with custom adapter
@workflow(name="custom_workflow", framework="custom_framework")
def test_custom_adapter():
    """Test function using custom adapter."""
    return {"message": "Custom framework workflow executed"}

result = test_custom_adapter()
print(f"Custom adapter result: {result}")
```

## Performance Optimization

### Adapter Caching

```python
# Adapter caching example
class CachedAdapter:
    """Demonstrate adapter caching for performance."""
    
    def __init__(self):
        self._cache = {}
    
    def get_cached_adapter(self, framework: str):
        """Get cached adapter instance."""
        if framework not in self._cache:
            print(f"Creating new adapter for {framework}")
            # Simulate adapter creation
            self._cache[framework] = f"{framework}_adapter_instance"
        else:
            print(f"Using cached adapter for {framework}")
        
        return self._cache[framework]

# Demonstrate caching
cache_demo = CachedAdapter()

# First call creates adapter
adapter1 = cache_demo.get_cached_adapter("langchain")

# Second call uses cached adapter
adapter2 = cache_demo.get_cached_adapter("langchain")

print(f"Same adapter instance: {adapter1 == adapter2}")
```

### Lazy Loading

```python
# Lazy loading example
class LazyAdapter:
    """Demonstrate lazy loading of framework dependencies."""
    
    def __init__(self):
        self._framework_module = None
    
    @property
    def framework_module(self):
        """Lazy load framework module."""
        if self._framework_module is None:
            try:
                # Simulate framework import
                print("Lazy loading framework module...")
                self._framework_module = "simulated_framework_module"
            except Exception as e:
                print(f"Framework not available: {e}")
                self._framework_module = None
        
        return self._framework_module
    
    def use_framework(self):
        """Use framework module when needed."""
        module = self.framework_module
        if module:
            return f"Using {module}"
        else:
            return "Framework not available"

# Demonstrate lazy loading
lazy_adapter = LazyAdapter()
result = lazy_adapter.use_framework()
print(f"Lazy loading result: {result}")
```

## Integration Examples

### Multi-Framework Application

```python
# Multi-framework application example
from rizk.sdk.decorators import workflow, agent

@workflow(name="langchain_part")
def langchain_processing(data: str):
    """LangChain-specific processing."""
    return f"LangChain processed: {data}"

@workflow(name="crewai_part") 
def crewai_processing(data: str):
    """CrewAI-specific processing."""
    return f"CrewAI processed: {data}"

@workflow(name="standard_part")
def standard_processing(data: str):
    """Standard processing."""
    return f"Standard processed: {data}"

def multi_framework_workflow(data: str, framework_preference: str = "auto"):
    """Workflow that can use different frameworks."""
    
    if framework_preference == "langchain":
        return langchain_processing(data)
    elif framework_preference == "crewai":
        return crewai_processing(data)
    else:
        return standard_processing(data)

# Usage
test_data = "Hello, world!"

langchain_result = multi_framework_workflow(test_data, "langchain")
crewai_result = multi_framework_workflow(test_data, "crewai")
standard_result = multi_framework_workflow(test_data, "standard")

print(f"LangChain: {langchain_result}")
print(f"CrewAI: {crewai_result}")
print(f"Standard: {standard_result}")
```

### Framework Migration

```python
# Framework migration example
def demonstrate_framework_migration():
    """Demonstrate migrating between frameworks."""
    
    # Original implementation (e.g., LangChain)
    @workflow(name="original_workflow", framework="langchain")
    def original_implementation(query: str):
        return f"LangChain implementation: {query}"
    
    # New implementation (e.g., CrewAI)
    @workflow(name="new_workflow", framework="crewai")
    def new_implementation(query: str):
        return f"CrewAI implementation: {query}"
    
    # Migration wrapper
    def migrated_workflow(query: str, use_new_implementation: bool = False):
        if use_new_implementation:
            return new_implementation(query)
        else:
            return original_implementation(query)
    
    # Test both implementations
    test_query = "What is AI?"
    
    old_result = migrated_workflow(test_query, use_new_implementation=False)
    new_result = migrated_workflow(test_query, use_new_implementation=True)
    
    print(f"Old implementation: {old_result}")
    print(f"New implementation: {new_result}")

demonstrate_framework_migration()
```

## Best Practices

### 1. Adapter Design Principles

```python
# Good adapter design example
class WellDesignedAdapter(BaseAdapter):
    """Example of well-designed adapter following best practices."""
    
    def __init__(self):
        super().__init__()
        self.framework_name = "well_designed"
    
    def adapt_workflow(self, func, name=None, **kwargs):
        """Well-designed workflow adaptation."""
        
        def wrapper(*args, **kwargs):
            # 1. Clear logging
            print(f"Executing workflow: {name}")
            
            # 2. Error handling
            try:
                result = func(*args, **kwargs)
                
                # 3. Consistent result format
                return self._normalize_result(result, "workflow")
                
            except Exception as e:
                print(f"Workflow error: {e}")
                return {"error": str(e), "type": "workflow"}
        
        return wrapper
    
    def _normalize_result(self, result, operation_type):
        """Normalize results to consistent format."""
        if isinstance(result, dict):
            result["operation_type"] = operation_type
            result["adapter"] = self.framework_name
        
        return result
```

### 2. Error Handling

```python
# Error handling best practices
def robust_adapter_example():
    """Demonstrate robust error handling in adapters."""
    
    @workflow(name="robust_workflow")
    def potentially_failing_workflow(data: dict):
        """Workflow that might fail."""
        
        if not data:
            raise ValueError("No data provided")
        
        if "error" in data:
            raise RuntimeError("Simulated error")
        
        return {"processed": data, "status": "success"}
    
    # Test error handling
    test_cases = [
        {"valid": "data"},
        {},  # Empty data
        {"error": "trigger"},  # Error case
    ]
    
    for i, test_data in enumerate(test_cases):
        try:
            result = potentially_failing_workflow(test_data)
            print(f"Test {i+1}: Success - {result}")
        except Exception as e:
            print(f"Test {i+1}: Error handled - {type(e).__name__}: {e}")

robust_adapter_example()
```

### 3. Performance Monitoring

```python
# Performance monitoring example
import time

def performance_monitoring_example():
    """Demonstrate performance monitoring in adapters."""
    
    @workflow(name="monitored_workflow")
    def monitored_workflow(data: str):
        """Workflow with performance monitoring."""
        start_time = time.time()
        
        # Simulate processing
        time.sleep(0.1)
        result = f"Processed: {data}"
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        return {
            "result": result,
            "performance": {
                "processing_time_ms": processing_time * 1000,
                "start_time": start_time,
                "end_time": end_time
            }
        }
    
    # Test performance monitoring
    result = monitored_workflow("test data")
    print(f"Result: {result['result']}")
    print(f"Processing time: {result['performance']['processing_time_ms']:.2f}ms")

performance_monitoring_example()
```

## Summary

Rizk SDK's adapter system provides:

âœ… **Universal Integration** - Single API for all LLM frameworks  
âœ… **Framework Preservation** - Maintains native framework patterns  
âœ… **Automatic Detection** - Adapters selected automatically  
âœ… **Extensible Design** - Easy to add new framework support  
âœ… **Performance Optimized** - Caching and lazy loading  
âœ… **Comprehensive Coverage** - Workflow, agent, task, and tool adaptation  
âœ… **Error Handling** - Robust error management and recovery  
âœ… **Production Ready** - Performance monitoring and best practices  

The adapter system enables Rizk SDK to provide universal LLM framework integration while preserving the unique capabilities and patterns of each framework, making it truly framework-agnostic for enterprise LLM applications. 

