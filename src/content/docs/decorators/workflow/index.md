---
title: "@workflow Decorator"
description: "@workflow Decorator"
---

# @workflow Decorator

The `@workflow` decorator is Rizk SDK's primary decorator for instrumenting high-level processes and business workflows. It provides automatic framework detection, distributed tracing, and hierarchical context management for complex multi-step operations.

## Overview

A **workflow** represents a high-level business process that orchestrates multiple tasks, agents, and tools to achieve a specific outcome. The `@workflow` decorator automatically adapts to your framework (OpenAI Agents, LangChain, CrewAI, etc.) while providing consistent observability and governance.

## Basic Usage

```python
from rizk.sdk import Rizk
from rizk.sdk.decorators import workflow

# Initialize Rizk
rizk = Rizk.init(app_name="WorkflowApp", enabled=True)

@workflow(
    name="data_processing_workflow",
    organization_id="acme_corp",
    project_id="analytics_platform"
)
def process_customer_data(customer_id: str, data_source: str) -> dict:
    """Process customer data through multiple stages."""
    # Extract data
    raw_data = extract_data(customer_id, data_source)
    
    # Transform data  
    cleaned_data = transform_data(raw_data)
    
    # Load into analytics platform
    result = load_data(cleaned_data)
    
    return {
        "customer_id": customer_id,
        "records_processed": len(cleaned_data),
        "status": "completed"
    }
```

## Parameters Reference

### Core Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `name` | `str` | No | Workflow name (defaults to function name) |
| `version` | `int` | No | Workflow version for tracking changes |
| `organization_id` | `str` | No | Organization identifier for hierarchical context |
| `project_id` | `str` | No | Project identifier for grouping workflows |

### Advanced Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `**kwargs` | `Any` | Framework-specific parameters passed to underlying adapters |

## Framework-Specific Behavior

### OpenAI Agents Integration

When used with OpenAI Agents, `@workflow` integrates with the native workflow system:

```python
from agents import Agent, Runner
from rizk.sdk.decorators import workflow

@workflow(
    name="customer_support_workflow",
    organization_id="support_team",
    project_id="helpdesk_v2"
)
def create_support_workflow(customer_query: str) -> str:
    """Create a customer support workflow with multiple agents."""
    
    # Create specialized agents
    triage_agent = Agent(
        name="TriageAgent",
        instructions="Classify and route customer queries",
        model="gpt-4"
    )
    
    resolution_agent = Agent(
        name="ResolutionAgent", 
        instructions="Provide detailed solutions to customer problems",
        model="gpt-4"
    )
    
    # Run workflow
    runner = Runner()
    
    # Step 1: Triage the query
    triage_result = runner.run(
        agent=triage_agent,
        messages=[{"role": "user", "content": f"Classify this query: {customer_query}"}]
    )
    
    # Step 2: Generate resolution
    resolution = runner.run(
        agent=resolution_agent,
        messages=[
            {"role": "user", "content": customer_query},
            {"role": "assistant", "content": triage_result.messages[-1]["content"]}
        ]
    )
    
    return resolution.messages[-1]["content"]
```

### LangChain Integration

With LangChain, `@workflow` works with chains and agent executors:

```python
from langchain.chains import LLMChain
from langchain.agents import AgentExecutor
from langchain_openai import ChatOpenAI
from rizk.sdk.decorators import workflow

@workflow(
    name="research_workflow", 
    organization_id="research_team",
    project_id="market_analysis"
)
def research_market_trends(topic: str, depth: str = "comprehensive") -> dict:
    """Research market trends using LangChain workflow."""
    
    llm = ChatOpenAI(temperature=0)
    
    # Research chain
    research_chain = LLMChain(
        llm=llm,
        prompt=PromptTemplate(
            input_variables=["topic", "depth"],
            template="Research {topic} with {depth} analysis. Provide structured insights."
        )
    )
    
    # Analysis chain
    analysis_chain = LLMChain(
        llm=llm,
        prompt=PromptTemplate(
            input_variables=["research_data"],
            template="Analyze this research data and provide actionable insights: {research_data}"
        )
    )
    
    # Execute workflow
    research_data = research_chain.run(topic=topic, depth=depth)
    analysis = analysis_chain.run(research_data=research_data)
    
    return {
        "topic": topic,
        "research_data": research_data,
        "analysis": analysis,
        "timestamp": datetime.now().isoformat()
    }
```

### CrewAI Integration

For CrewAI, `@workflow` orchestrates crew creation and execution:

```python
from crewai import Agent, Task, Crew, Process
from rizk.sdk.decorators import workflow

@workflow(
    name="content_creation_workflow",
    organization_id="marketing_team", 
    project_id="content_strategy"
)
def create_marketing_content(topic: str, target_audience: str) -> str:
    """Create marketing content using CrewAI workflow."""
    
    # Define agents
    researcher = Agent(
        role="Content Researcher",
        goal="Research comprehensive information about the topic",
        backstory="Expert researcher with deep knowledge of market trends",
        verbose=True
    )
    
    writer = Agent(
        role="Content Writer",
        goal="Create engaging content based on research",
        backstory="Skilled writer who creates compelling marketing content",
        verbose=True
    )
    
    # Define tasks
    research_task = Task(
        description=f"Research {topic} for {target_audience} audience",
        agent=researcher,
        expected_output="Comprehensive research report with key insights"
    )
    
    writing_task = Task(
        description=f"Write marketing content about {topic} for {target_audience}",
        agent=writer,
        expected_output="Engaging marketing content ready for publication"
    )
    
    # Create and run crew
    crew = Crew(
        agents=[researcher, writer],
        tasks=[research_task, writing_task],
        process=Process.sequential,
        verbose=True
    )
    
    result = crew.kickoff()
    return str(result)
```

## Async Workflow Support

The `@workflow` decorator supports both synchronous and asynchronous workflows:

```python
import asyncio
from rizk.sdk.decorators import workflow

@workflow(
    name="async_data_pipeline",
    organization_id="data_team",
    project_id="real_time_analytics"
)
async def process_streaming_data(stream_id: str, batch_size: int = 100) -> dict:
    """Process streaming data asynchronously."""
    
    processed_count = 0
    error_count = 0
    
    async def process_batch(batch_data):
        nonlocal processed_count, error_count
        try:
            # Simulate async processing
            await asyncio.sleep(0.1)
            processed_count += len(batch_data)
            return True
        except Exception:
            error_count += 1
            return False
    
    # Simulate streaming data processing
    for i in range(0, 1000, batch_size):
        batch_data = [f"record_{j}" for j in range(i, min(i + batch_size, 1000))]
        await process_batch(batch_data)
    
    return {
        "stream_id": stream_id,
        "processed_count": processed_count,
        "error_count": error_count,
        "status": "completed"
    }

# Usage
async def main():
    result = await process_streaming_data("stream_001", batch_size=50)
    print(f"Processed {result['processed_count']} records")

# asyncio.run(main())
```

## Error Handling and Resilience

The `@workflow` decorator includes built-in error handling and recovery mechanisms:

```python
from rizk.sdk.decorators import workflow
from rizk.sdk.utils.error_handler import handle_errors

@workflow(
    name="resilient_workflow",
    organization_id="production_team",
    project_id="critical_systems"
)
@handle_errors(fail_closed=False, max_retries=3)
def resilient_data_processing(data_source: str) -> dict:
    """Workflow with built-in error handling and retries."""
    
    try:
        # Step 1: Validate input
        if not data_source or len(data_source) < 3:
            raise ValueError("Invalid data source provided")
        
        # Step 2: Process data with potential failures
        results = []
        for i in range(5):
            try:
                # Simulate processing that might fail
                if i == 2:  # Simulate a failure
                    raise ConnectionError("Temporary connection issue")
                results.append(f"processed_item_{i}")
            except ConnectionError as e:
                logger.warning(f"Retrying after connection error: {e}")
                # Built-in retry mechanism will handle this
                raise
        
        return {
            "status": "success",
            "processed_items": results,
            "data_source": data_source
        }
        
    except Exception as e:
        # Workflow-level error handling
        logger.error(f"Workflow failed: {e}")
        return {
            "status": "failed",
            "error": str(e),
            "data_source": data_source
        }
```

## Hierarchical Context Management

Workflows automatically establish hierarchical context for nested operations:

```python
from rizk.sdk.decorators import workflow, task, agent

@workflow(
    name="hierarchical_workflow",
    organization_id="enterprise_corp",
    project_id="customer_onboarding"
)
def onboard_customer(customer_data: dict) -> dict:
    """Multi-step customer onboarding workflow."""
    
    # Context is automatically propagated to nested decorators
    validation_result = validate_customer_data(customer_data)
    
    if validation_result["valid"]:
        account_result = create_customer_account(customer_data)
        notification_result = send_welcome_notification(customer_data["email"])
        
        return {
            "status": "completed",
            "customer_id": account_result["customer_id"],
            "validation": validation_result,
            "account": account_result,
            "notification": notification_result
        }
    else:
        return {
            "status": "failed",
            "validation": validation_result
        }

@task(name="validate_customer_data")  # Inherits workflow context
def validate_customer_data(data: dict) -> dict:
    """Validate customer data."""
    required_fields = ["name", "email", "phone"]
    missing_fields = [field for field in required_fields if field not in data]
    
    return {
        "valid": len(missing_fields) == 0,
        "missing_fields": missing_fields
    }

@agent(name="account_creation_agent")  # Inherits workflow context
def create_customer_account(data: dict) -> dict:
    """Create customer account."""
    import uuid
    return {
        "customer_id": str(uuid.uuid4()),
        "account_status": "active",
        "created_at": datetime.now().isoformat()
    }

@task(name="notification_task")  # Inherits workflow context
def send_welcome_notification(email: str) -> dict:
    """Send welcome notification."""
    # Simulate email sending
    return {
        "email": email,
        "notification_sent": True,
        "sent_at": datetime.now().isoformat()
    }
```

## Performance Monitoring

Workflows include comprehensive performance monitoring:

```python
import time
from rizk.sdk.decorators import workflow
from rizk.sdk.tracing import create_span, set_span_attribute

@workflow(
    name="performance_monitored_workflow",
    organization_id="performance_team",
    project_id="optimization_analysis"
)
def performance_critical_workflow(dataset_size: int) -> dict:
    """Workflow with detailed performance monitoring."""
    
    start_time = time.time()
    
    # Custom performance tracking
    with create_span("data_loading") as span:
        set_span_attribute(span, "dataset.size", dataset_size)
        
        # Simulate data loading
        load_time = min(dataset_size / 1000, 5.0)  # Cap at 5 seconds
        time.sleep(load_time)
        
        set_span_attribute(span, "data_loading.duration_seconds", load_time)
    
    # Processing phase
    with create_span("data_processing") as span:
        set_span_attribute(span, "processing.algorithm", "optimized_batch")
        
        # Simulate processing
        process_time = min(dataset_size / 2000, 3.0)  # Cap at 3 seconds
        time.sleep(process_time)
        
        set_span_attribute(span, "data_processing.duration_seconds", process_time)
        set_span_attribute(span, "processing.throughput_records_per_second", dataset_size / process_time)
    
    total_time = time.time() - start_time
    
    return {
        "dataset_size": dataset_size,
        "total_duration_seconds": total_time,
        "load_duration_seconds": load_time,
        "process_duration_seconds": process_time,
        "throughput_records_per_second": dataset_size / total_time,
        "status": "completed"
    }
```

## Multi-Framework Workflows

Workflows can orchestrate multiple frameworks within a single process:

```python
from rizk.sdk.decorators import workflow
from rizk.sdk.utils.framework_detection import detect_framework

@workflow(
    name="multi_framework_workflow",
    organization_id="integration_team",
    project_id="hybrid_ai_system"
)
def hybrid_ai_workflow(user_query: str) -> dict:
    """Workflow that uses multiple AI frameworks."""
    
    results = {}
    
    # Use LangChain for initial processing
    try:
        from langchain_openai import ChatOpenAI
        from langchain.chains import LLMChain
        
        langchain_llm = ChatOpenAI(temperature=0)
        langchain_result = langchain_llm.invoke([{"role": "user", "content": f"Analyze: {user_query}"}])
        results["langchain_analysis"] = langchain_result.content
        
    except ImportError:
        results["langchain_analysis"] = "LangChain not available"
    
    # Use OpenAI Agents for specialized processing
    try:
        from agents import Agent, Runner
        
        specialist_agent = Agent(
            name="SpecialistAgent",
            instructions="Provide specialized insights based on the analysis",
            model="gpt-4"
        )
        
        runner = Runner()
        specialist_result = runner.run(
            agent=specialist_agent,
            messages=[{"role": "user", "content": f"Based on this analysis: {results.get('langchain_analysis', '')}, provide specialized insights for: {user_query}"}]
        )
        
        results["specialist_insights"] = specialist_result.messages[-1]["content"]
        
    except ImportError:
        results["specialist_insights"] = "OpenAI Agents not available"
    
    # Use CrewAI for collaborative processing
    try:
        from crewai import Agent, Task, Crew, Process
        
        reviewer = Agent(
            role="Quality Reviewer",
            goal="Review and synthesize all analyses",
            backstory="Expert reviewer who synthesizes multiple AI perspectives"
        )
        
        review_task = Task(
            description=f"Review and synthesize these analyses: {results}",
            agent=reviewer,
            expected_output="Comprehensive synthesis of all analyses"
        )
        
        crew = Crew(
            agents=[reviewer],
            tasks=[review_task],
            process=Process.sequential
        )
        
        synthesis = crew.kickoff()
        results["final_synthesis"] = str(synthesis)
        
    except ImportError:
        results["final_synthesis"] = "CrewAI not available"
    
    # Detect which frameworks were actually used
    detected_framework = detect_framework()
    results["detected_framework"] = detected_framework
    results["query"] = user_query
    
    return results
```

## Testing Workflows

Here's how to test workflow-decorated functions:

```python
import pytest
from unittest.mock import Mock, patch
from rizk.sdk import Rizk
from rizk.sdk.decorators import workflow

# Test setup
@pytest.fixture
def rizk_setup():
    """Setup Rizk for testing."""
    return Rizk.init(app_name="WorkflowTest", enabled=True)

def test_basic_workflow(rizk_setup):
    """Test basic workflow functionality."""
    
    @workflow(
        name="test_workflow",
        organization_id="test_org",
        project_id="test_project"
    )
    def simple_workflow(input_data: str) -> dict:
        return {
            "processed": input_data.upper(),
            "status": "success"
        }
    
    # Test execution
    result = simple_workflow("hello world")
    
    assert result["processed"] == "HELLO WORLD"
    assert result["status"] == "success"
    assert hasattr(simple_workflow, "__name__")

def test_async_workflow(rizk_setup):
    """Test async workflow functionality."""
    
    @workflow(
        name="async_test_workflow",
        organization_id="test_org", 
        project_id="test_project"
    )
    async def async_workflow(delay: float) -> dict:
        await asyncio.sleep(delay)
        return {
            "delay": delay,
            "status": "completed"
        }
    
    # Test async execution
    async def run_test():
        result = await async_workflow(0.1)
        assert result["delay"] == 0.1
        assert result["status"] == "completed"
    
    asyncio.run(run_test())

def test_workflow_error_handling(rizk_setup):
    """Test workflow error handling."""
    
    @workflow(
        name="error_test_workflow",
        organization_id="test_org",
        project_id="test_project"
    )
    def error_workflow(should_fail: bool) -> dict:
        if should_fail:
            raise ValueError("Intentional test error")
        return {"status": "success"}
    
    # Test successful execution
    result = error_workflow(False)
    assert result["status"] == "success"
    
    # Test error handling
    with pytest.raises(ValueError):
        error_workflow(True)

@patch('rizk.sdk.utils.framework_detection.detect_framework')
def test_workflow_framework_detection(mock_detect, rizk_setup):
    """Test workflow framework detection."""
    mock_detect.return_value = "langchain"
    
    @workflow(
        name="framework_test_workflow",
        organization_id="test_org",
        project_id="test_project"
    )
    def framework_workflow() -> str:
        from rizk.sdk.utils.framework_detection import detect_framework
        return detect_framework()
    
    result = framework_workflow()
    assert result == "langchain"
    mock_detect.assert_called()
```

## Best Practices

### 1. **Naming Conventions**
```python
# Good: Descriptive workflow names
@workflow(name="customer_onboarding_workflow")
@workflow(name="data_processing_pipeline") 
@workflow(name="ai_content_generation_workflow")

# Avoid: Generic or unclear names
@workflow(name="workflow1")
@workflow(name="process")
@workflow(name="main")
```

### 2. **Hierarchical Organization**
```python
# Use consistent organization and project IDs
@workflow(
    name="user_management_workflow",
    organization_id="acme_corp",           # Company/org level
    project_id="customer_portal_v2"        # Project level
)
```

### 3. **Error Handling**
```python
# Always include comprehensive error handling
@workflow(name="robust_workflow")
def robust_workflow(data: dict) -> dict:
    try:
        # Validate inputs
        if not data:
            raise ValueError("No data provided")
        
        # Process with error recovery
        result = process_data(data)
        
        return {"status": "success", "result": result}
        
    except Exception as e:
        logger.error(f"Workflow failed: {e}")
        return {"status": "error", "error": str(e)}
```

### 4. **Performance Considerations**
```python
# Use spans for performance tracking
@workflow(name="performance_aware_workflow")
def performance_workflow(large_dataset: list) -> dict:
    with create_span("data_validation") as span:
        set_span_attribute(span, "dataset.size", len(large_dataset))
        # Validation logic here
    
    with create_span("data_processing") as span:
        # Processing logic here
        pass
    
    return {"processed": len(large_dataset)}
```

### 5. **Version Management**
```python
# Use versions for tracking workflow changes
@workflow(
    name="evolving_workflow",
    version=2,  # Increment when making breaking changes
    organization_id="acme_corp",
    project_id="production_system"
)
def evolving_workflow_v2(data: dict) -> dict:
    # New implementation
    pass
```

## Common Patterns

### 1. **Data Pipeline Workflow**
```python
@workflow(name="etl_pipeline")
def etl_pipeline(source: str, destination: str) -> dict:
    """Extract, Transform, Load pipeline."""
    
    # Extract
    with create_span("extract") as span:
        data = extract_data(source)
        set_span_attribute(span, "records.extracted", len(data))
    
    # Transform
    with create_span("transform") as span:
        transformed_data = transform_data(data)
        set_span_attribute(span, "records.transformed", len(transformed_data))
    
    # Load
    with create_span("load") as span:
        load_result = load_data(transformed_data, destination)
        set_span_attribute(span, "records.loaded", load_result["count"])
    
    return {
        "source": source,
        "destination": destination,
        "records_processed": len(transformed_data),
        "status": "completed"
    }
```

### 2. **AI Agent Orchestration**
```python
@workflow(name="multi_agent_workflow")
def multi_agent_workflow(task_description: str) -> dict:
    """Orchestrate multiple AI agents."""
    
    results = {}
    
    # Research phase
    research_result = research_agent(task_description)
    results["research"] = research_result
    
    # Analysis phase
    analysis_result = analysis_agent(research_result)
    results["analysis"] = analysis_result
    
    # Synthesis phase
    synthesis_result = synthesis_agent(results)
    results["synthesis"] = synthesis_result
    
    return results
```

### 3. **Event-Driven Workflow**
```python
@workflow(name="event_driven_workflow")
async def event_driven_workflow(event_data: dict) -> dict:
    """Process events asynchronously."""
    
    event_type = event_data.get("type")
    
    if event_type == "user_signup":
        return await handle_user_signup(event_data)
    elif event_type == "order_placed":
        return await handle_order_placed(event_data)
    elif event_type == "payment_received":
        return await handle_payment_received(event_data)
    else:
        return {"status": "unknown_event", "event_type": event_type}
```

## Troubleshooting

### Common Issues

1. **Framework Not Detected**
```python
# Issue: Framework detection fails
# Solution: Ensure framework is properly imported before decoration

# Before decorator application
import langchain  # or your framework
from rizk.sdk.decorators import workflow

@workflow(name="my_workflow")
def my_workflow():
    pass
```

2. **Context Not Propagated**
```python
# Issue: Nested functions don't inherit context
# Solution: Ensure nested functions are also decorated

@workflow(name="parent_workflow")
def parent_workflow():
    return nested_operation()  # This should also be decorated

@task(name="nested_operation")  # Add appropriate decorator
def nested_operation():
    pass
```

3. **Performance Issues**
```python
# Issue: Workflow is slow
# Solution: Use spans to identify bottlenecks

@workflow(name="optimized_workflow")
def optimized_workflow():
    with create_span("expensive_operation") as span:
        # Monitor this operation specifically
        result = expensive_operation()
        set_span_attribute(span, "operation.duration", time.time() - start)
    return result
```

## Related Documentation

- **[@task Decorator](task.md)** - For individual operations within workflows
- **[@agent Decorator](agent.md)** - For autonomous components in workflows  
- **[@tool Decorator](tool.md)** - For utility functions used by workflows
- **[Observability Guide](../observability/tracing.md)** - Understanding tracing and spans
- **[Framework Integration](../framework-integration/)** - Framework-specific workflow patterns
- **[Performance Monitoring](../observability/performance-monitoring.md)** - Optimizing workflow performance

---

The `@workflow` decorator is the foundation of Rizk SDK's observability system. It provides automatic framework adaptation, comprehensive tracing, and hierarchical context management that scales from simple functions to complex multi-framework AI systems.

