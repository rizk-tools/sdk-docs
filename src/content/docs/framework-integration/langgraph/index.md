---
title: "LangGraph Integration Guide"
description: "LangGraph Integration Guide"
---

# LangGraph Integration Guide

LangGraph is a powerful library for building stateful, multi-actor applications with LLMs. This guide shows you how to integrate Rizk SDK with LangGraph for comprehensive observability, tracing, and governance of your graph-based workflows.

## Overview

LangGraph excels at creating complex, stateful workflows where multiple agents or tools collaborate. Rizk SDK provides:

- **Automatic Graph Tracing**: Monitor node execution, state transitions, and graph performance
- **State-Aware Guardrails**: Apply policies based on graph state and node outputs
- **Node-Level Observability**: Track individual node performance and dependencies
- **Graph Workflow Analytics**: Understand execution patterns and optimization opportunities

## Quick Start

### Installation

```bash
pip install rizk langgraph
```

### Basic Setup

```python
from rizk.sdk import Rizk
from rizk.sdk.decorators import workflow, task, agent, tool
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, Sequence
import operator

# Initialize Rizk
rizk = Rizk.init(
    app_name="LangGraph-Demo",
    enabled=True
)

# Define state
class AgentState(TypedDict):
    messages: Annotated[Sequence[str], operator.add]
    next: str
```

## Core Integration Patterns

### 1. Basic Graph Workflow

```python
@workflow(
    name="research_graph", 
    organization_id="demo", 
    project_id="langgraph"
)
def create_research_graph():
    """Create a research workflow graph with monitoring."""
    
    @task(name="researcher_node", organization_id="demo", project_id="langgraph")
    def researcher(state: AgentState) -> AgentState:
        """Research node with automatic tracing."""
        query = state["messages"][-1] if state["messages"] else "default query"
        
        # Simulate research
        research_result = f"Research findings for: {query}"
        
        return {
            "messages": [research_result],
            "next": "analyzer"
        }
    
    @task(name="analyzer_node", organization_id="demo", project_id="langgraph")
    def analyzer(state: AgentState) -> AgentState:
        """Analysis node with automatic tracing."""
        research_data = state["messages"][-1] if state["messages"] else ""
        
        # Simulate analysis
        analysis_result = f"Analysis: {research_data} shows promising trends"
        
        return {
            "messages": [analysis_result],
            "next": END
        }
    
    # Build graph
    workflow = StateGraph(AgentState)
    workflow.add_node("researcher", researcher)
    workflow.add_node("analyzer", analyzer)
    
    # Define edges
    workflow.set_entry_point("researcher")
    workflow.add_edge("researcher", "analyzer")
    workflow.add_edge("analyzer", END)
    
    return workflow.compile()

# Usage
if __name__ == "__main__":
    graph = create_research_graph()
    
    result = graph.invoke({
        "messages": ["What are the latest trends in AI?"],
        "next": ""
    })
    
    print("Final result:", result["messages"][-1])
```

### 2. Multi-Agent Collaboration Graph

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict, Literal

class CollaborationState(TypedDict):
    task: str
    research_data: str
    analysis: str
    final_report: str
    next: Literal["researcher", "analyst", "writer", "__end__"]

@workflow(
    name="collaboration_graph",
    organization_id="enterprise", 
    project_id="multi_agent"
)
def create_collaboration_graph():
    """Multi-agent collaboration with comprehensive monitoring."""
    
    @agent(name="research_agent", organization_id="enterprise", project_id="multi_agent")
    def research_agent(state: CollaborationState) -> CollaborationState:
        """Research agent with guardrails."""
        task = state.get("task", "")
        
        # Simulate research with potential sensitive content check
        research_data = f"Comprehensive research on {task}. Key findings include market analysis, competitor review, and trend identification."
        
        return {
            **state,
            "research_data": research_data,
            "next": "analyst"
        }
    
    @agent(name="analysis_agent", organization_id="enterprise", project_id="multi_agent")
    def analysis_agent(state: CollaborationState) -> CollaborationState:
        """Analysis agent with performance tracking."""
        research_data = state.get("research_data", "")
        
        # Simulate analysis
        analysis = f"Based on the research: {research_data[:100]}..., we recommend strategic focus on emerging opportunities."
        
        return {
            **state,
            "analysis": analysis,
            "next": "writer"
        }
    
    @agent(name="writing_agent", organization_id="enterprise", project_id="multi_agent")
    def writing_agent(state: CollaborationState) -> CollaborationState:
        """Writing agent with content governance."""
        research = state.get("research_data", "")
        analysis = state.get("analysis", "")
        
        # Create final report
        final_report = f"""
        Executive Summary Report
        
        Research Findings:
        {research}
        
        Strategic Analysis:
        {analysis}
        
        Recommendations:
        1. Implement data-driven decision making
        2. Focus on customer-centric solutions
        3. Invest in emerging technologies
        """
        
        return {
            **state,
            "final_report": final_report,
            "next": "__end__"
        }
    
    # Build the collaboration graph
    workflow = StateGraph(CollaborationState)
    
    # Add nodes
    workflow.add_node("researcher", research_agent)
    workflow.add_node("analyst", analysis_agent)
    workflow.add_node("writer", writing_agent)
    
    # Define flow
    workflow.set_entry_point("researcher")
    workflow.add_edge("researcher", "analyst")
    workflow.add_edge("analyst", "writer")
    workflow.add_edge("writer", END)
    
    return workflow.compile()

# Test collaboration
collaboration_graph = create_collaboration_graph()
result = collaboration_graph.invoke({
    "task": "Market analysis for sustainable technology solutions",
    "research_data": "",
    "analysis": "",
    "final_report": "",
    "next": "researcher"
})

print("Collaboration Result:")
print(result["final_report"])
```

### 3. Conditional Graph with Decision Logic

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict, Literal

class DecisionState(TypedDict):
    query: str
    intent: str
    response: str
    confidence: float
    next: Literal["classifier", "simple_response", "complex_analysis", "__end__"]

@workflow(
    name="decision_graph",
    organization_id="ai_systems", 
    project_id="intelligent_routing"
)
def create_decision_graph():
    """Conditional graph with intelligent routing."""
    
    @task(name="intent_classifier", organization_id="ai_systems", project_id="intelligent_routing")
    def classify_intent(state: DecisionState) -> DecisionState:
        """Classify user intent with confidence scoring."""
        query = state.get("query", "").lower()
        
        # Simple intent classification
        if any(word in query for word in ["analyze", "complex", "detailed", "research"]):
            intent = "complex"
            confidence = 0.85
            next_node = "complex_analysis"
        else:
            intent = "simple"
            confidence = 0.90
            next_node = "simple_response"
        
        return {
            **state,
            "intent": intent,
            "confidence": confidence,
            "next": next_node
        }
    
    @task(name="simple_responder", organization_id="ai_systems", project_id="intelligent_routing")
    def simple_response(state: DecisionState) -> DecisionState:
        """Handle simple queries efficiently."""
        query = state.get("query", "")
        response = f"Quick response to: {query}"
        
        return {
            **state,
            "response": response,
            "next": "__end__"
        }
    
    @task(name="complex_analyzer", organization_id="ai_systems", project_id="intelligent_routing")
    def complex_analysis(state: DecisionState) -> DecisionState:
        """Handle complex queries with detailed analysis."""
        query = state.get("query", "")
        
        # Simulate complex analysis
        response = f"""
        Detailed Analysis of: {query}
        
        1. Context Understanding: Query requires comprehensive analysis
        2. Data Processing: Multiple data sources considered  
        3. Synthesis: Cross-referencing relevant information
        4. Conclusion: Providing structured, detailed response
        
        Final Answer: Based on the analysis, here's a comprehensive response to your query about {query}.
        """
        
        return {
            **state,
            "response": response,
            "next": "__end__"
        }
    
    # Define routing logic
    def route_decision(state: DecisionState) -> str:
        """Route based on classification results."""
        return state.get("next", "__end__")
    
    # Build decision graph
    workflow = StateGraph(DecisionState)
    
    # Add nodes
    workflow.add_node("classifier", classify_intent)
    workflow.add_node("simple_response", simple_response)
    workflow.add_node("complex_analysis", complex_analysis)
    
    # Define conditional routing
    workflow.set_entry_point("classifier")
    workflow.add_conditional_edges(
        "classifier",
        route_decision,
        {
            "simple_response": "simple_response",
            "complex_analysis": "complex_analysis",
            "__end__": END
        }
    )
    workflow.add_edge("simple_response", END)
    workflow.add_edge("complex_analysis", END)
    
    return workflow.compile()

# Test decision routing
decision_graph = create_decision_graph()

# Test simple query
simple_result = decision_graph.invoke({
    "query": "Hello, how are you?",
    "intent": "",
    "response": "",
    "confidence": 0.0,
    "next": "classifier"
})

print("Simple Query Result:", simple_result["response"])

# Test complex query
complex_result = decision_graph.invoke({
    "query": "Please analyze the market trends for renewable energy",
    "intent": "",
    "response": "",
    "confidence": 0.0,
    "next": "classifier"
})

print("Complex Query Result:", complex_result["response"])
```

## Configuration and Testing

### Environment Configuration

```python
import os
from rizk.sdk import Rizk

# Production configuration
rizk = Rizk.init(
    app_name="LangGraph-Production",
    api_key=os.getenv("RIZK_API_KEY"),
    opentelemetry_endpoint=os.getenv("RIZK_OTLP_ENDPOINT"),
    enabled=True,
    # LangGraph-specific settings
    policies_path="./policies",
    trace_content=False,  # Disable content tracing for privacy
    disable_batch=True,   # Better for graph workflows
    verbose=False
)
```

### Testing Framework

```python
import unittest
from unittest.mock import patch, MagicMock

class TestLangGraphIntegration(unittest.TestCase):
    """Test LangGraph integration with Rizk SDK."""
    
    def setUp(self):
        """Set up test environment."""
        self.rizk = Rizk.init(
            app_name="LangGraph-Test",
            enabled=True,
            verbose=True
        )
    
    def test_basic_graph_tracing(self):
        """Test basic graph workflow tracing."""
        
        @workflow(name="test_graph", organization_id="test", project_id="langgraph")
        def create_test_graph():
            @task(name="test_node", organization_id="test", project_id="langgraph")
            def test_node(state):
                return {"result": "success", "next": "__end__"}
            
            workflow = StateGraph(dict)
            workflow.add_node("test_node", test_node)
            workflow.set_entry_point("test_node")
            workflow.add_edge("test_node", END)
            return workflow.compile()
        
        graph = create_test_graph()
        result = graph.invoke({"input": "test"})
        
        self.assertIn("result", result)
        self.assertEqual(result["result"], "success")

if __name__ == "__main__":
    unittest.main()
```

## Troubleshooting

### Common Issues and Solutions

#### 1. Graph State Validation Errors

```python
# Problem: State validation fails
# Solution: Ensure TypedDict definitions match your state structure

from typing import TypedDict, Optional

class ValidatedState(TypedDict):
    # Required fields
    query: str
    result: str
    
    # Optional fields  
    metadata: Optional[dict]
    next: Optional[str]

# Validate state before graph execution
def validate_state(state: dict) -> bool:
    required_fields = ["query", "result"]
    return all(field in state for field in required_fields)
```

#### 2. Node Execution Timeouts

```python
# Problem: Nodes taking too long to execute
# Solution: Add timeout handling

import asyncio
from functools import wraps

def with_timeout(timeout_seconds: int):
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                return await asyncio.wait_for(
                    func(*args, **kwargs), 
                    timeout=timeout_seconds
                )
            except asyncio.TimeoutError:
                logger.error(f"Node {func.__name__} timed out after {timeout_seconds}s")
                raise
        return wrapper
    return decorator

@task(name="timeout_protected_node", organization_id="reliability", project_id="timeouts")
@with_timeout(30)  # 30 second timeout
async def protected_node(state):
    # Your node logic here
    await asyncio.sleep(0.1)  # Simulate work
    return state
```

This comprehensive guide covers LangGraph integration with Rizk SDK, providing enterprise-grade observability, governance, and performance monitoring for your graph-based workflows.


