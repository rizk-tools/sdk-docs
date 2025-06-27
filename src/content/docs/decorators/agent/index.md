---
title: "@agent Decorator"
description: "The @agent decorator is designed for instrumenting autonomous AI components that can make decisions, use tools, and interact with users or other systems."
---

# @agent Decorator

The `@agent` decorator is designed for instrumenting autonomous AI components that can make decisions, use tools, and interact with users or other systems. It provides comprehensive observability for agent-based architectures while automatically adapting to different AI frameworks.

## Overview

An **agent** represents an autonomous component that can reason, make decisions, use tools, and take actions to achieve specific goals. The `@agent` decorator provides framework-agnostic instrumentation for agent creation, execution, and interaction patterns across OpenAI Agents, LangChain, CrewAI, and custom implementations.

## Basic Usage

```python
from rizk.sdk import Rizk
from rizk.sdk.decorators import agent, tool

# Initialize Rizk
rizk = Rizk.init(app_name="AgentApp", enabled=True)

@tool(
    name="weather_tool",
    organization_id="assistant_team",
    project_id="weather_service"
)
def get_weather(location: str) -> str:
    """Get current weather for a location."""
    # Mock weather service
    weather_data = {
        "New York": "Sunny, 72Â°F",
        "London": "Cloudy, 18Â°C", 
        "Tokyo": "Rainy, 22Â°C"
    }
    return weather_data.get(location, f"Weather data not available for {location}")

@agent(
    name="weather_assistant",
    organization_id="assistant_team",
    project_id="weather_service",
    agent_id="weather_agent_001"
)
def create_weather_agent() -> dict:
    """Create a weather assistant agent."""
    
    agent_config = {
        "name": "WeatherAssistant",
        "role": "Weather Information Specialist",
        "instructions": "You are a helpful weather assistant. Use the get_weather tool to provide current weather information for any location.",
        "tools": [get_weather],
        "capabilities": ["weather_lookup", "location_parsing"],
        "created_at": datetime.now().isoformat()
    }
    
    return agent_config

# Usage
def run_weather_assistant(user_query: str) -> str:
    """Run the weather assistant with a user query."""
    agent_config = create_weather_agent()
    
    # Simulate agent processing
    if "weather" in user_query.lower():
        # Extract location (simplified)
        words = user_query.split()
        location = "New York"  # Default
        for word in words:
            if word.title() in ["New York", "London", "Tokyo"]:
                location = word.title()
                break
        
        weather_result = get_weather(location)
        return f"The weather in {location} is: {weather_result}"
    else:
        return "I can help you with weather information. Please ask about the weather in a specific location."

# Test the agent
result = run_weather_assistant("What's the weather like in London?")
print(f"Agent response: {result}")
```

## Parameters Reference

### Core Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `name` | `str` | No | Agent name (defaults to function name) |
| `version` | `int` | No | Agent version for tracking changes |
| `organization_id` | `str` | No | Organization identifier for hierarchical context |
| `project_id` | `str` | No | Project identifier for grouping agents |
| `agent_id` | `str` | No | Specific agent identifier for unique identification |

### Advanced Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `**kwargs` | `Any` | Framework-specific parameters passed to underlying adapters |

## Framework-Specific Behavior

### OpenAI Agents Integration

When used with OpenAI Agents, `@agent` integrates with the native Agent and Runner system:

```python
from agents import Agent, Runner
from rizk.sdk.decorators import agent, tool

@tool(
    name="calculator_tool",
    organization_id="math_team",
    project_id="calculator_service"
)
def calculate(expression: str) -> str:
    """Safely calculate mathematical expressions."""
    try:
        # Safe evaluation for basic math
        allowed_chars = set('0123456789+-*/().')
        if not all(c in allowed_chars or c.isspace() for c in expression):
            return f"Error: Invalid characters in expression"
        
        result = eval(expression)
        return f"The result of {expression} is {result}"
    except Exception as e:
        return f"Error calculating {expression}: {str(e)}"

@agent(
    name="math_assistant",
    organization_id="math_team",
    project_id="calculator_service",
    agent_id="math_agent_001"
)
def create_math_agent() -> Agent:
    """Create a math assistant agent using OpenAI Agents SDK."""
    
    agent = Agent(
        name="MathAssistant",
        instructions="""You are a helpful math assistant. You can help users with mathematical calculations.
        
        When a user asks for a calculation:
        1. Use the calculate tool to perform the calculation
        2. Provide a clear explanation of the result
        3. If the calculation fails, explain what went wrong
        
        Always be helpful and educational in your responses.""",
        model="gpt-4",
        tools=[calculate]
    )
    
    return agent

def run_math_session(query: str) -> str:
    """Run a math session with the agent."""
    agent = create_math_agent()
    runner = Runner()
    
    result = runner.run(
        agent=agent,
        messages=[{"role": "user", "content": query}]
    )
    
    return result.messages[-1]["content"]

# Example usage
math_result = run_math_session("Can you calculate 15 * 23 + 7?")
print(f"Math agent response: {math_result}")
```

### LangChain Integration

With LangChain, `@agent` works with agent executors and callback handlers:

```python
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.tools import Tool
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from rizk.sdk.decorators import agent, tool

@tool(
    name="search_tool",
    organization_id="research_team",
    project_id="information_retrieval"
)
def search_information(query: str) -> str:
    """Search for information on a given topic."""
    # Mock search results
    search_results = {
        "python": "Python is a high-level programming language known for its simplicity and readability.",
        "ai": "Artificial Intelligence (AI) refers to the simulation of human intelligence in machines.",
        "machine learning": "Machine Learning is a subset of AI that enables computers to learn without explicit programming."
    }
    
    # Simple keyword matching
    for keyword, result in search_results.items():
        if keyword.lower() in query.lower():
            return result
    
    return f"No specific information found for '{query}'. Try searching for 'python', 'ai', or 'machine learning'."

@agent(
    name="research_assistant",
    organization_id="research_team",
    project_id="information_retrieval",
    agent_id="research_agent_001"
)
def create_research_agent() -> AgentExecutor:
    """Create a research assistant using LangChain."""
    
    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")
    
    # Convert our decorated tool to LangChain tool
    langchain_search_tool = Tool(
        name="search_information",
        description="Search for information on any topic",
        func=search_information
    )
    
    tools = [langchain_search_tool]
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful research assistant. Use the search_information tool to find information about topics users ask about."),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])
    
    agent = create_openai_tools_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    
    return agent_executor

def run_research_query(query: str) -> str:
    """Run a research query with the agent."""
    agent_executor = create_research_agent()
    
    result = agent_executor.invoke({"input": query})
    return result["output"]

# Example usage
research_result = run_research_query("Tell me about Python programming language")
print(f"Research agent response: {research_result}")
```

### CrewAI Integration

For CrewAI, `@agent` can be used to instrument agent creation and crew participation:

```python
from crewai import Agent, Task, Crew, Process
from rizk.sdk.decorators import agent, task

@agent(
    name="content_researcher",
    organization_id="content_team",
    project_id="content_creation",
    agent_id="researcher_001"
)
def create_content_researcher() -> Agent:
    """Create a content researcher agent for CrewAI."""
    
    return Agent(
        role="Content Researcher",
        goal="Research comprehensive and accurate information about given topics",
        backstory="""You are an expert content researcher with 10+ years of experience 
        in gathering, analyzing, and synthesizing information from various sources. 
        You have a keen eye for detail and always verify facts before presenting them.""",
        verbose=True,
        allow_delegation=False
    )

@agent(
    name="content_writer",
    organization_id="content_team",
    project_id="content_creation",
    agent_id="writer_001"
)
def create_content_writer() -> Agent:
    """Create a content writer agent for CrewAI."""
    
    return Agent(
        role="Content Writer",
        goal="Create engaging, well-structured content based on research findings",
        backstory="""You are a skilled content writer with expertise in creating 
        compelling articles, blog posts, and marketing content. You know how to 
        translate complex research into accessible, engaging prose that resonates 
        with target audiences.""",
        verbose=True,
        allow_delegation=False
    )

@task(
    name="research_task",
    organization_id="content_team",
    project_id="content_creation"
)
def create_research_task(topic: str, researcher: Agent) -> Task:
    """Create a research task for the content researcher."""
    
    return Task(
        description=f"""Research the topic '{topic}' thoroughly. 
        Gather key facts, statistics, trends, and insights.
        Focus on recent developments and authoritative sources.
        Provide a comprehensive research summary.""",
        agent=researcher,
        expected_output="A detailed research report with key findings, statistics, and insights"
    )

@task(
    name="writing_task",
    organization_id="content_team",
    project_id="content_creation"
)
def create_writing_task(topic: str, writer: Agent) -> Task:
    """Create a writing task for the content writer."""
    
    return Task(
        description=f"""Based on the research findings, write a comprehensive 
        article about '{topic}'. The article should be:
        - Well-structured with clear headings
        - Engaging and accessible to a general audience
        - Factually accurate based on the research
        - Between 800-1200 words""",
        agent=writer,
        expected_output="A well-written, engaging article ready for publication"
    )

def run_content_creation_crew(topic: str) -> str:
    """Run a content creation crew with researched agents."""
    
    # Create agents
    researcher = create_content_researcher()
    writer = create_content_writer()
    
    # Create tasks
    research_task = create_research_task(topic, researcher)
    writing_task = create_writing_task(topic, writer)
    
    # Create and run crew
    crew = Crew(
        agents=[researcher, writer],
        tasks=[research_task, writing_task],
        process=Process.sequential,
        verbose=True
    )
    
    result = crew.kickoff()
    return str(result)

# Example usage
content_result = run_content_creation_crew("The Future of Artificial Intelligence")
print(f"Content creation result: {content_result}")
```

## Multi-Agent Coordination

The `@agent` decorator supports complex multi-agent workflows:

```python
from rizk.sdk.decorators import agent, workflow, task
import asyncio

@agent(
    name="data_analyst",
    organization_id="analytics_team",
    project_id="business_intelligence"
)
def create_data_analyst() -> dict:
    """Create a data analyst agent."""
    return {
        "role": "Data Analyst",
        "specialization": "Statistical Analysis",
        "tools": ["data_processing", "statistical_analysis"],
        "capabilities": ["trend_analysis", "pattern_recognition"]
    }

@agent(
    name="report_generator",
    organization_id="analytics_team", 
    project_id="business_intelligence"
)
def create_report_generator() -> dict:
    """Create a report generator agent."""
    return {
        "role": "Report Generator",
        "specialization": "Business Reporting",
        "tools": ["report_creation", "data_visualization"],
        "capabilities": ["executive_summaries", "chart_generation"]
    }

@agent(
    name="quality_reviewer",
    organization_id="analytics_team",
    project_id="business_intelligence"
)
def create_quality_reviewer() -> dict:
    """Create a quality reviewer agent."""
    return {
        "role": "Quality Reviewer",
        "specialization": "Quality Assurance",
        "tools": ["fact_checking", "consistency_validation"],
        "capabilities": ["accuracy_verification", "completeness_check"]
    }

@task(name="analyze_data")
def analyze_data(data: dict, analyst_config: dict) -> dict:
    """Analyze data using the data analyst agent."""
    # Simulate data analysis
    analysis_results = {
        "trends": ["Upward trend in Q3", "Seasonal patterns detected"],
        "insights": ["Revenue increased 15%", "Customer satisfaction improved"],
        "metrics": {"accuracy": 0.95, "confidence": 0.88},
        "analyst": analyst_config["role"]
    }
    return analysis_results

@task(name="generate_report")
def generate_report(analysis: dict, generator_config: dict) -> dict:
    """Generate report using the report generator agent."""
    # Simulate report generation
    report = {
        "title": "Business Intelligence Report",
        "summary": "Quarterly analysis shows positive trends across key metrics",
        "sections": ["Executive Summary", "Key Findings", "Recommendations"],
        "charts": ["revenue_trend", "satisfaction_score"],
        "generator": generator_config["role"]
    }
    return report

@task(name="review_quality")
def review_quality(report: dict, reviewer_config: dict) -> dict:
    """Review report quality using the quality reviewer agent."""
    # Simulate quality review
    review_results = {
        "quality_score": 0.92,
        "issues_found": 0,
        "recommendations": ["Add more context to chart titles"],
        "approved": True,
        "reviewer": reviewer_config["role"]
    }
    return review_results

@workflow(
    name="multi_agent_analysis_workflow",
    organization_id="analytics_team",
    project_id="business_intelligence"
)
def run_multi_agent_analysis(business_data: dict) -> dict:
    """Run multi-agent business analysis workflow."""
    
    # Create agent configurations
    analyst = create_data_analyst()
    generator = create_report_generator()
    reviewer = create_quality_reviewer()
    
    # Execute workflow steps
    analysis = analyze_data(business_data, analyst)
    report = generate_report(analysis, generator)
    quality_review = review_quality(report, reviewer)
    
    return {
        "analysis": analysis,
        "report": report,
        "quality_review": quality_review,
        "workflow_status": "completed",
        "agents_used": [analyst["role"], generator["role"], reviewer["role"]]
    }

# Example usage
sample_data = {
    "revenue": [100000, 110000, 125000, 130000],
    "customers": [1000, 1050, 1100, 1150],
    "satisfaction": [4.2, 4.3, 4.5, 4.6]
}

workflow_result = run_multi_agent_analysis(sample_data)
print(f"Multi-agent workflow result: {workflow_result}")
```

## Async Agent Support

The `@agent` decorator supports asynchronous agent operations:

```python
import asyncio
from rizk.sdk.decorators import agent, task

@agent(
    name="async_data_processor",
    organization_id="data_team",
    project_id="real_time_processing"
)
async def create_async_data_processor() -> dict:
    """Create an async data processing agent."""
    
    # Simulate async agent initialization
    await asyncio.sleep(0.1)
    
    return {
        "name": "AsyncDataProcessor",
        "type": "streaming_processor",
        "capabilities": ["real_time_processing", "batch_processing"],
        "max_concurrent_tasks": 10,
        "initialized_at": datetime.now().isoformat()
    }

@task(name="process_data_stream")
async def process_data_stream(data_stream: list, processor_config: dict) -> dict:
    """Process data stream using async agent."""
    
    processed_items = []
    
    # Process items concurrently
    async def process_item(item):
        await asyncio.sleep(0.01)  # Simulate processing time
        return f"processed_{item}"
    
    # Process in batches
    batch_size = processor_config.get("max_concurrent_tasks", 5)
    for i in range(0, len(data_stream), batch_size):
        batch = data_stream[i:i + batch_size]
        batch_results = await asyncio.gather(*[process_item(item) for item in batch])
        processed_items.extend(batch_results)
    
    return {
        "total_items": len(data_stream),
        "processed_items": processed_items,
        "processing_time": "real_time",
        "processor": processor_config["name"]
    }

@agent(
    name="async_coordinator",
    organization_id="coordination_team",
    project_id="distributed_processing"
)
async def create_async_coordinator() -> dict:
    """Create an async coordination agent."""
    
    await asyncio.sleep(0.05)
    
    return {
        "name": "AsyncCoordinator",
        "type": "coordination_agent",
        "capabilities": ["task_distribution", "result_aggregation"],
        "managed_agents": [],
        "initialized_at": datetime.now().isoformat()
    }

async def run_async_agent_workflow(data_batches: list) -> dict:
    """Run async workflow with multiple agents."""
    
    # Create agents concurrently
    processor_task = create_async_data_processor()
    coordinator_task = create_async_coordinator()
    
    processor, coordinator = await asyncio.gather(processor_task, coordinator_task)
    
    # Process batches concurrently
    processing_tasks = [
        process_data_stream(batch, processor)
        for batch in data_batches
    ]
    
    batch_results = await asyncio.gather(*processing_tasks)
    
    # Aggregate results
    total_processed = sum(result["total_items"] for result in batch_results)
    all_processed_items = []
    for result in batch_results:
        all_processed_items.extend(result["processed_items"])
    
    return {
        "coordinator": coordinator["name"],
        "processor": processor["name"],
        "total_batches": len(data_batches),
        "total_items_processed": total_processed,
        "sample_results": all_processed_items[:5],  # Show first 5 results
        "workflow_status": "completed"
    }

# Example usage
async def test_async_agents():
    sample_batches = [
        ["item1", "item2", "item3"],
        ["item4", "item5", "item6"],
        ["item7", "item8", "item9"]
    ]
    
    result = await run_async_agent_workflow(sample_batches)
    print(f"Async agent workflow result: {result}")

# asyncio.run(test_async_agents())
```

## Agent State Management

Agents can maintain state across interactions:

```python
from rizk.sdk.decorators import agent, task
from typing import Dict, Any

class AgentState:
    """Simple agent state management."""
    
    def __init__(self):
        self.memory: Dict[str, Any] = {}
        self.interaction_count = 0
        self.session_id = None
    
    def update_memory(self, key: str, value: Any):
        self.memory[key] = value
    
    def get_memory(self, key: str, default=None):
        return self.memory.get(key, default)
    
    def increment_interactions(self):
        self.interaction_count += 1

# Global state storage (in production, use proper state management)
agent_states: Dict[str, AgentState] = {}

@agent(
    name="stateful_assistant",
    organization_id="assistant_team",
    project_id="conversational_ai"
)
def create_stateful_assistant(session_id: str) -> dict:
    """Create a stateful assistant agent."""
    
    # Initialize or retrieve agent state
    if session_id not in agent_states:
        agent_states[session_id] = AgentState()
        agent_states[session_id].session_id = session_id
    
    state = agent_states[session_id]
    
    return {
        "name": "StatefulAssistant",
        "session_id": session_id,
        "interaction_count": state.interaction_count,
        "memory_keys": list(state.memory.keys()),
        "capabilities": ["memory_retention", "context_awareness"]
    }

@task(name="process_user_input")
def process_user_input(user_input: str, session_id: str, agent_config: dict) -> dict:
    """Process user input with state management."""
    
    state = agent_states.get(session_id)
    if not state:
        return {"error": "Session not found"}
    
    state.increment_interactions()
    
    # Parse user input for memory updates
    if "remember" in user_input.lower():
        # Extract key-value pairs (simplified)
        if "my name is" in user_input.lower():
            name = user_input.lower().split("my name is")[1].strip()
            state.update_memory("user_name", name)
            response = f"I'll remember that your name is {name}."
        else:
            response = "I'll try to remember that information."
    
    elif "what do you know about me" in user_input.lower():
        user_name = state.get_memory("user_name", "unknown")
        preferences = state.get_memory("preferences", [])
        response = f"I know your name is {user_name}. "
        if preferences:
            response += f"Your preferences include: {', '.join(preferences)}."
        else:
            response += "I don't have any recorded preferences yet."
    
    elif "i like" in user_input.lower():
        preference = user_input.lower().split("i like")[1].strip()
        current_prefs = state.get_memory("preferences", [])
        current_prefs.append(preference)
        state.update_memory("preferences", current_prefs)
        response = f"Noted! I'll remember that you like {preference}."
    
    else:
        user_name = state.get_memory("user_name", "there")
        response = f"Hello {user_name}! How can I help you today?"
    
    return {
        "response": response,
        "interaction_count": state.interaction_count,
        "session_id": session_id,
        "memory_updated": True
    }

def run_stateful_conversation(session_id: str, messages: list) -> list:
    """Run a stateful conversation with the agent."""
    
    agent_config = create_stateful_assistant(session_id)
    conversation_log = []
    
    for message in messages:
        result = process_user_input(message, session_id, agent_config)
        conversation_log.append({
            "user_input": message,
            "agent_response": result["response"],
            "interaction_count": result["interaction_count"]
        })
    
    return conversation_log

# Example usage
conversation_messages = [
    "Hello there!",
    "Remember that my name is Alice",
    "I like chocolate ice cream",
    "What do you know about me?",
    "I also like reading books",
    "What do you know about me now?"
]

conversation_log = run_stateful_conversation("session_123", conversation_messages)
for entry in conversation_log:
    print(f"User: {entry['user_input']}")
    print(f"Agent: {entry['agent_response']}")
    print(f"Interaction #{entry['interaction_count']}\n")
```

## Error Handling and Resilience

Agents include robust error handling patterns:

```python
from rizk.sdk.decorators import agent, task
from rizk.sdk.utils.error_handler import handle_errors

@agent(
    name="resilient_agent",
    organization_id="production_team",
    project_id="critical_systems"
)
@handle_errors(fail_closed=False, max_retries=3)
def create_resilient_agent() -> dict:
    """Create a resilient agent with error handling."""
    
    return {
        "name": "ResilientAgent",
        "error_handling": "enabled",
        "retry_policy": "exponential_backoff",
        "fallback_strategies": ["graceful_degradation", "alternative_response"],
        "monitoring": "comprehensive"
    }

@task(name="resilient_task_execution")
def execute_resilient_task(task_input: dict, agent_config: dict) -> dict:
    """Execute task with comprehensive error handling."""
    
    try:
        # Validate input
        if not isinstance(task_input, dict):
            raise ValueError("Task input must be a dictionary")
        
        if "action" not in task_input:
            raise ValueError("Task input must contain 'action' field")
        
        action = task_input["action"]
        
        # Simulate different actions with potential failures
        if action == "process_data":
            if task_input.get("data") is None:
                raise ValueError("No data provided for processing")
            
            # Simulate processing
            result = {
                "action": action,
                "status": "completed",
                "processed_items": len(task_input["data"]),
                "agent": agent_config["name"]
            }
            
        elif action == "external_api_call":
            # Simulate external API call that might fail
            if task_input.get("simulate_failure", False):
                raise ConnectionError("External API is temporarily unavailable")
            
            result = {
                "action": action,
                "status": "completed",
                "api_response": "Success",
                "agent": agent_config["name"]
            }
            
        elif action == "complex_calculation":
            # Simulate complex calculation that might timeout
            if task_input.get("complexity", 1) > 10:
                raise TimeoutError("Calculation too complex, operation timed out")
            
            result = {
                "action": action,
                "status": "completed",
                "calculation_result": task_input.get("complexity", 1) * 42,
                "agent": agent_config["name"]
            }
            
        else:
            # Fallback for unknown actions
            result = {
                "action": action,
                "status": "unknown_action",
                "message": f"Agent doesn't know how to handle action: {action}",
                "agent": agent_config["name"]
            }
        
        return result
        
    except ValueError as e:
        # Input validation errors - don't retry
        return {
            "status": "validation_error",
            "error": str(e),
            "error_type": "validation",
            "agent": agent_config["name"]
        }
    
    except (ConnectionError, TimeoutError) as e:
        # Recoverable errors - will be retried by error handler
        logger.warning(f"Recoverable error in agent task: {e}")
        raise  # Re-raise for retry mechanism
    
    except Exception as e:
        # Unexpected errors
        logger.error(f"Unexpected error in agent task: {e}")
        return {
            "status": "unexpected_error",
            "error": str(e),
            "error_type": "unexpected",
            "agent": agent_config["name"]
        }

def run_resilient_agent_tests():
    """Test resilient agent with various scenarios."""
    
    agent_config = create_resilient_agent()
    
    test_cases = [
        # Success case
        {
            "action": "process_data",
            "data": ["item1", "item2", "item3"]
        },
        # Validation error
        {
            "invalid_input": "no action field"
        },
        # Recoverable error (will be retried)
        {
            "action": "external_api_call",
            "simulate_failure": True
        },
        # Success after retry
        {
            "action": "external_api_call",
            "simulate_failure": False
        },
        # Timeout error
        {
            "action": "complex_calculation",
            "complexity": 15  # Too complex
        },
        # Unknown action (graceful handling)
        {
            "action": "unknown_action",
            "data": "some data"
        }
    ]
    
    results = []
    for i, test_case in enumerate(test_cases):
        print(f"\nTest case {i+1}: {test_case}")
        result = execute_resilient_task(test_case, agent_config)
        print(f"Result: {result}")
        results.append(result)
    
    return results

# Example usage
test_results = run_resilient_agent_tests()
print(f"\nCompleted {len(test_results)} test cases")
```

## Testing Agents

Here's how to test agent-decorated functions:

```python
import pytest
from unittest.mock import Mock, patch
from rizk.sdk import Rizk
from rizk.sdk.decorators import agent, tool

@pytest.fixture
def rizk_setup():
    """Setup Rizk for testing."""
    return Rizk.init(app_name="AgentTest", enabled=True)

def test_basic_agent_creation(rizk_setup):
    """Test basic agent creation."""
    
    @agent(
        name="test_agent",
        organization_id="test_org",
        project_id="test_project"
    )
    def create_test_agent() -> dict:
        return {
            "name": "TestAgent",
            "role": "Testing Assistant",
            "capabilities": ["testing", "validation"]
        }
    
    # Test agent creation
    agent_config = create_test_agent()
    
    assert agent_config["name"] == "TestAgent"
    assert agent_config["role"] == "Testing Assistant"
    assert "testing" in agent_config["capabilities"]

def test_agent_with_tools(rizk_setup):
    """Test agent with tools integration."""
    
    @tool(name="test_tool")
    def test_tool(input_data: str) -> str:
        return f"Processed: {input_data}"
    
    @agent(
        name="tool_using_agent",
        organization_id="test_org",
        project_id="test_project"
    )
    def create_tool_agent() -> dict:
        return {
            "name": "ToolAgent",
            "tools": [test_tool],
            "capabilities": ["tool_usage"]
        }
    
    # Test agent with tools
    agent_config = create_tool_agent()
    
    assert agent_config["name"] == "ToolAgent"
    assert len(agent_config["tools"]) == 1
    
    # Test tool functionality
    tool_result = agent_config["tools"][0]("test input")
    assert tool_result == "Processed: test input"

def test_async_agent(rizk_setup):
    """Test async agent functionality."""
    
    @agent(
        name="async_test_agent",
        organization_id="test_org",
        project_id="test_project"
    )
    async def create_async_agent() -> dict:
        await asyncio.sleep(0.01)
        return {
            "name": "AsyncAgent",
            "type": "asynchronous",
            "status": "ready"
        }
    
    # Test async agent creation
    async def run_test():
        agent_config = await create_async_agent()
        assert agent_config["name"] == "AsyncAgent"
        assert agent_config["type"] == "asynchronous"
        assert agent_config["status"] == "ready"
    
    asyncio.run(run_test())

def test_agent_error_handling(rizk_setup):
    """Test agent error handling."""
    
    @agent(
        name="error_test_agent",
        organization_id="test_org",
        project_id="test_project"
    )
    def create_error_agent(should_fail: bool = False) -> dict:
        if should_fail:
            raise ValueError("Test error")
        return {"name": "ErrorTestAgent", "status": "success"}
    
    # Test successful creation
    success_config = create_error_agent(False)
    assert success_config["status"] == "success"
    
    # Test error handling
    with pytest.raises(ValueError):
        create_error_agent(True)

@patch('time.sleep')
def test_agent_performance(mock_sleep, rizk_setup):
    """Test agent performance monitoring."""
    mock_sleep.return_value = None
    
    @agent(
        name="performance_test_agent",
        organization_id="test_org",
        project_id="test_project"
    )
    def create_performance_agent() -> dict:
        start_time = time.time()
        
        # Simulate agent initialization
        time.sleep(0.1)  # This will be mocked
        
        end_time = time.time()
        
        return {
            "name": "PerformanceAgent",
            "initialization_time": end_time - start_time,
            "status": "ready"
        }
    
    agent_config = create_performance_agent()
    
    assert agent_config["name"] == "PerformanceAgent"
    assert "initialization_time" in agent_config
    assert agent_config["status"] == "ready"
    mock_sleep.assert_called_once_with(0.1)
```

## Best Practices

### 1. **Clear Agent Roles**
```python
# Good: Clear, specific role definition
@agent(name="customer_support_specialist")
def create_support_agent():
    return {
        "role": "Customer Support Specialist",
        "specialization": "Technical Issues",
        "capabilities": ["troubleshooting", "escalation", "documentation"]
    }

# Avoid: Vague or overly broad roles
@agent(name="general_agent")
def create_general_agent():
    return {"role": "General Assistant"}  # Too vague
```

### 2. **Tool Integration**
```python
# Good: Well-defined tools with clear purposes
@tool(name="database_query")
def query_database(query: str) -> dict:
    # Implementation
    pass

@agent(name="data_analyst")
def create_analyst():
    return {
        "tools": [query_database],
        "capabilities": ["data_analysis"]
    }
```

### 3. **Error Recovery**
```python
# Good: Comprehensive error handling
@agent(name="robust_agent")
def create_robust_agent():
    return {
        "error_handling": {
            "retry_policy": "exponential_backoff",
            "max_retries": 3,
            "fallback_strategies": ["graceful_degradation"]
        }
    }
```

### 4. **State Management**
```python
# Good: Explicit state management
@agent(name="stateful_agent")
def create_stateful_agent(session_id: str):
    return {
        "session_id": session_id,
        "state_management": "enabled",
        "memory_retention": True
    }
```

## Related Documentation

- **[@workflow Decorator](workflow.md)** - For orchestrating agent workflows
- **[@task Decorator](task.md)** - For individual agent tasks
- **[@tool Decorator](tool.md)** - For agent tools and capabilities
- **[OpenAI Agents Integration](../framework-integration/openai-agents.md)** - OpenAI-specific patterns
- **[LangChain Integration](../framework-integration/langchain.md)** - LangChain agent patterns
- **[CrewAI Integration](../framework-integration/crewai.md)** - Multi-agent crew patterns

---

The `@agent` decorator provides comprehensive observability for autonomous AI components while maintaining framework compatibility. It enables sophisticated agent architectures with proper monitoring, error handling, and state management across any AI framework.

