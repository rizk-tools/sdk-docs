---
title: "@task Decorator"
description: "@task Decorator"
---

# @task Decorator

The `@task` decorator is designed for instrumenting individual operations and discrete units of work within larger workflows. It provides fine-grained observability for specific functions while maintaining hierarchical context with parent workflows.

## Overview

A **task** represents a distinct, focused operation that performs a specific function within a larger workflow. Tasks are the building blocks of complex processes, providing detailed tracing and monitoring for individual operations while automatically inheriting context from parent workflows.

## Basic Usage

```python
from rizk.sdk import Rizk
from rizk.sdk.decorators import task, workflow

# Initialize Rizk
rizk = Rizk.init(app_name="TaskApp", enabled=True)

@task(
    name="data_validation_task",
    organization_id="data_team",
    project_id="quality_control"
)
def validate_data(data: dict, schema: dict) -> dict:
    """Validate data against a schema."""
    errors = []
    
    # Check required fields
    for field in schema.get("required", []):
        if field not in data:
            errors.append(f"Missing required field: {field}")
    
    # Check data types
    for field, expected_type in schema.get("types", {}).items():
        if field in data and not isinstance(data[field], expected_type):
            errors.append(f"Invalid type for {field}: expected {expected_type.__name__}")
    
    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "data": data
    }

# Usage
schema = {
    "required": ["name", "email"],
    "types": {"name": str, "email": str, "age": int}
}

result = validate_data(
    {"name": "John", "email": "john@example.com", "age": 30},
    schema
)
print(f"Validation result: {result}")
```

## Parameters Reference

### Core Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `name` | `str` | No | Task name (defaults to function name) |
| `version` | `int` | No | Task version for tracking changes |
| `organization_id` | `str` | No | Organization identifier for hierarchical context |
| `project_id` | `str` | No | Project identifier for grouping tasks |
| `task_id` | `str` | No | Specific task identifier for unique identification |

### Advanced Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `**kwargs` | `Any` | Framework-specific parameters passed to underlying adapters |

## Hierarchical Context Integration

Tasks automatically integrate with workflow context when used within decorated workflows:

```python
from rizk.sdk.decorators import workflow, task

@workflow(
    name="data_processing_workflow",
    organization_id="analytics_team",
    project_id="customer_insights"
)
def process_customer_data(customer_records: list) -> dict:
    """Process customer data through multiple tasks."""
    
    # Tasks inherit workflow context automatically
    validated_records = validate_records(customer_records)
    enriched_records = enrich_records(validated_records)
    analyzed_data = analyze_records(enriched_records)
    
    return {
        "total_records": len(customer_records),
        "valid_records": len(validated_records),
        "enriched_records": len(enriched_records),
        "analysis": analyzed_data
    }

@task(name="validate_records")  # Inherits workflow context
def validate_records(records: list) -> list:
    """Validate individual customer records."""
    valid_records = []
    
    for record in records:
        if record.get("email") and "@" in record["email"]:
            if record.get("name") and len(record["name"]) > 0:
                valid_records.append(record)
    
    return valid_records

@task(name="enrich_records")  # Inherits workflow context
def enrich_records(records: list) -> list:
    """Enrich records with additional data."""
    enriched = []
    
    for record in records:
        # Simulate data enrichment
        enriched_record = record.copy()
        enriched_record["enriched_at"] = datetime.now().isoformat()
        enriched_record["source"] = "customer_database"
        enriched.append(enriched_record)
    
    return enriched

@task(name="analyze_records")  # Inherits workflow context
def analyze_records(records: list) -> dict:
    """Analyze enriched records."""
    if not records:
        return {"summary": "No records to analyze"}
    
    domains = [record["email"].split("@")[1] for record in records]
    domain_counts = {}
    for domain in domains:
        domain_counts[domain] = domain_counts.get(domain, 0) + 1
    
    return {
        "total_records": len(records),
        "unique_domains": len(domain_counts),
        "top_domains": sorted(domain_counts.items(), key=lambda x: x[1], reverse=True)[:5]
    }
```

## Framework-Specific Behavior

### OpenAI Agents Integration

When used with OpenAI Agents, `@task` integrates with agent tool execution:

```python
from agents import Agent, Runner
from rizk.sdk.decorators import task, agent

@task(
    name="calculation_task",
    organization_id="math_team",
    project_id="calculator_agent"
)
def calculate_expression(expression: str) -> str:
    """Safely evaluate mathematical expressions."""
    try:
        # Safe evaluation for basic math
        allowed_operators = {'+', '-', '*', '/', '(', ')', ' ', '.'}
        allowed_chars = set('0123456789') | allowed_operators
        
        if not all(c in allowed_chars for c in expression):
            return f"Error: Invalid characters in expression: {expression}"
        
        result = eval(expression)
        return f"Result: {result}"
    except Exception as e:
        return f"Error: {str(e)}"

@agent(
    name="math_assistant",
    organization_id="math_team",
    project_id="calculator_agent"
)
def create_math_agent() -> Agent:
    """Create a math assistant agent with calculation tools."""
    
    agent = Agent(
        name="MathAssistant",
        instructions="You are a helpful math assistant. Use the calculate_expression tool for calculations.",
        model="gpt-4",
        tools=[calculate_expression]  # Task is automatically wrapped as a tool
    )
    
    return agent

# Usage
def run_math_assistant(query: str) -> str:
    """Run the math assistant with a query."""
    agent = create_math_agent()
    runner = Runner()
    
    result = runner.run(
        agent=agent,
        messages=[{"role": "user", "content": query}]
    )
    
    return result.messages[-1]["content"]
```

### LangChain Integration

With LangChain, `@task` works with individual chain components:

```python
from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from rizk.sdk.decorators import task, workflow

@task(
    name="text_summarization_task",
    organization_id="content_team",
    project_id="document_processing"
)
def summarize_text(text: str, max_length: int = 100) -> str:
    """Summarize text using LangChain."""
    
    llm = ChatOpenAI(temperature=0)
    
    prompt = PromptTemplate(
        input_variables=["text", "max_length"],
        template="Summarize the following text in no more than {max_length} words:\n\n{text}\n\nSummary:"
    )
    
    chain = LLMChain(llm=llm, prompt=prompt)
    
    summary = chain.run(text=text, max_length=max_length)
    return summary.strip()

@task(
    name="keyword_extraction_task",
    organization_id="content_team",
    project_id="document_processing"
)
def extract_keywords(text: str, num_keywords: int = 5) -> list:
    """Extract keywords from text."""
    
    llm = ChatOpenAI(temperature=0)
    
    prompt = PromptTemplate(
        input_variables=["text", "num_keywords"],
        template="Extract the top {num_keywords} most important keywords from this text:\n\n{text}\n\nKeywords (comma-separated):"
    )
    
    chain = LLMChain(llm=llm, prompt=prompt)
    
    keywords_str = chain.run(text=text, num_keywords=num_keywords)
    keywords = [kw.strip() for kw in keywords_str.split(",")]
    
    return keywords[:num_keywords]

@workflow(
    name="document_analysis_workflow",
    organization_id="content_team",
    project_id="document_processing"
)
def analyze_document(document_text: str) -> dict:
    """Analyze a document using multiple tasks."""
    
    # Tasks execute with inherited context
    summary = summarize_text(document_text, max_length=150)
    keywords = extract_keywords(document_text, num_keywords=8)
    
    return {
        "original_length": len(document_text),
        "summary": summary,
        "keywords": keywords,
        "analysis_complete": True
    }
```

### CrewAI Integration

For CrewAI, `@task` can be used to instrument individual task functions:

```python
from crewai import Agent, Task, Crew, Process
from rizk.sdk.decorators import task, workflow

@task(
    name="research_task_function",
    organization_id="research_team",
    project_id="market_analysis"
)
def execute_research(topic: str, depth: str = "comprehensive") -> str:
    """Execute research task with monitoring."""
    
    # Simulate research process
    research_points = [
        f"Market size analysis for {topic}",
        f"Competitive landscape in {topic}",
        f"Growth trends and projections for {topic}",
        f"Key challenges and opportunities in {topic}"
    ]
    
    if depth == "comprehensive":
        research_points.extend([
            f"Regulatory environment for {topic}",
            f"Technology disruptions in {topic}",
            f"Consumer behavior patterns related to {topic}"
        ])
    
    return "\n".join(f"â€¢ {point}" for point in research_points)

@task(
    name="analysis_task_function",
    organization_id="research_team",
    project_id="market_analysis"
)
def execute_analysis(research_data: str) -> str:
    """Execute analysis task with monitoring."""
    
    # Simulate analysis process
    analysis_sections = [
        "## Key Findings",
        "Based on the research data, several key patterns emerge:",
        "",
        "## Strategic Recommendations", 
        "The following recommendations are proposed:",
        "",
        "## Risk Assessment",
        "Potential risks and mitigation strategies:"
    ]
    
    return "\n".join(analysis_sections)

@workflow(
    name="crewai_research_workflow",
    organization_id="research_team",
    project_id="market_analysis"
)
def run_research_crew(topic: str) -> str:
    """Run CrewAI research crew with monitored tasks."""
    
    # Create agents
    researcher = Agent(
        role="Market Researcher",
        goal=f"Research comprehensive information about {topic}",
        backstory="Expert market researcher with 10+ years experience"
    )
    
    analyst = Agent(
        role="Market Analyst", 
        goal="Analyze research data and provide insights",
        backstory="Senior analyst specializing in market trend analysis"
    )
    
    # Create tasks using monitored functions
    research_task = Task(
        description=f"Research {topic} market comprehensively",
        agent=researcher,
        expected_output="Detailed research findings"
    )
    
    analysis_task = Task(
        description="Analyze research findings and provide recommendations",
        agent=analyst,
        expected_output="Strategic analysis and recommendations"
    )
    
    # Create and run crew
    crew = Crew(
        agents=[researcher, analyst],
        tasks=[research_task, analysis_task],
        process=Process.sequential
    )
    
    result = crew.kickoff()
    return str(result)
```

## Async Task Support

The `@task` decorator supports both synchronous and asynchronous operations:

```python
import asyncio
import aiohttp
from rizk.sdk.decorators import task, workflow

@task(
    name="async_api_call_task",
    organization_id="integration_team",
    project_id="external_apis"
)
async def fetch_user_data(user_id: str, api_endpoint: str) -> dict:
    """Fetch user data from external API asynchronously."""
    
    async with aiohttp.ClientSession() as session:
        url = f"{api_endpoint}/users/{user_id}"
        
        try:
            async with session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    return {
                        "user_id": user_id,
                        "data": data,
                        "status": "success"
                    }
                else:
                    return {
                        "user_id": user_id,
                        "error": f"HTTP {response.status}",
                        "status": "error"
                    }
        except Exception as e:
            return {
                "user_id": user_id,
                "error": str(e),
                "status": "error"
            }

@task(
    name="async_data_processing_task",
    organization_id="integration_team",
    project_id="external_apis"
)
async def process_user_data(user_data: dict) -> dict:
    """Process user data asynchronously."""
    
    if user_data["status"] != "success":
        return user_data  # Return error as-is
    
    # Simulate async processing
    await asyncio.sleep(0.1)
    
    processed_data = {
        "user_id": user_data["user_id"],
        "name": user_data["data"].get("name", "Unknown"),
        "email": user_data["data"].get("email", ""),
        "processed_at": datetime.now().isoformat(),
        "status": "processed"
    }
    
    return processed_data

@workflow(
    name="async_user_processing_workflow",
    organization_id="integration_team", 
    project_id="external_apis"
)
async def process_multiple_users(user_ids: list, api_endpoint: str) -> dict:
    """Process multiple users asynchronously."""
    
    # Fetch all user data concurrently
    fetch_tasks = [
        fetch_user_data(user_id, api_endpoint) 
        for user_id in user_ids
    ]
    
    user_data_list = await asyncio.gather(*fetch_tasks)
    
    # Process all user data concurrently
    process_tasks = [
        process_user_data(user_data)
        for user_data in user_data_list
    ]
    
    processed_users = await asyncio.gather(*process_tasks)
    
    # Aggregate results
    successful = [u for u in processed_users if u["status"] == "processed"]
    failed = [u for u in processed_users if u["status"] == "error"]
    
    return {
        "total_users": len(user_ids),
        "successful": len(successful),
        "failed": len(failed),
        "processed_users": successful,
        "failed_users": failed
    }
```

## Error Handling and Resilience

Tasks include comprehensive error handling patterns:

```python
from rizk.sdk.decorators import task
from rizk.sdk.utils.error_handler import handle_errors

@task(
    name="resilient_database_task",
    organization_id="data_team",
    project_id="database_operations"
)
@handle_errors(fail_closed=False, max_retries=3, retry_delay=1.0)
def query_database(query: str, connection_string: str) -> dict:
    """Execute database query with retry logic."""
    
    try:
        # Simulate database connection and query
        if "invalid" in query.lower():
            raise ValueError("Invalid SQL query")
        
        if "timeout" in query.lower():
            raise TimeoutError("Database query timeout")
        
        # Simulate successful query
        mock_results = [
            {"id": 1, "name": "John Doe", "email": "john@example.com"},
            {"id": 2, "name": "Jane Smith", "email": "jane@example.com"}
        ]
        
        return {
            "query": query,
            "results": mock_results,
            "row_count": len(mock_results),
            "status": "success"
        }
        
    except ValueError as e:
        # Don't retry on validation errors
        return {
            "query": query,
            "error": str(e),
            "error_type": "validation",
            "status": "failed"
        }
    
    except (TimeoutError, ConnectionError) as e:
        # These will be retried by the error handler
        logger.warning(f"Database operation failed, will retry: {e}")
        raise

@task(
    name="data_validation_with_fallback",
    organization_id="data_team",
    project_id="data_quality"
)
def validate_with_fallback(data: dict, strict_mode: bool = False) -> dict:
    """Validate data with fallback strategies."""
    
    validation_result = {
        "original_data": data,
        "validated_data": {},
        "errors": [],
        "warnings": [],
        "status": "unknown"
    }
    
    try:
        # Primary validation
        validated_data = {}
        
        # Validate required fields
        required_fields = ["id", "name", "email"]
        for field in required_fields:
            if field not in data:
                if strict_mode:
                    raise ValueError(f"Missing required field: {field}")
                else:
                    validation_result["warnings"].append(f"Missing optional field: {field}")
                    validated_data[field] = None
            else:
                validated_data[field] = data[field]
        
        # Validate email format
        if validated_data.get("email") and "@" not in validated_data["email"]:
            if strict_mode:
                raise ValueError("Invalid email format")
            else:
                validation_result["warnings"].append("Email format appears invalid")
        
        validation_result["validated_data"] = validated_data
        validation_result["status"] = "success"
        
    except Exception as e:
        validation_result["errors"].append(str(e))
        validation_result["status"] = "failed"
        
        # Fallback: return partial data
        if not strict_mode:
            validation_result["validated_data"] = {
                k: v for k, v in data.items() if k in ["id", "name"]
            }
            validation_result["status"] = "partial_success"
    
    return validation_result
```

## Performance Monitoring

Tasks provide detailed performance monitoring capabilities:

```python
import time
from rizk.sdk.decorators import task
from rizk.sdk.tracing import create_span, set_span_attribute

@task(
    name="performance_critical_task",
    organization_id="performance_team",
    project_id="optimization"
)
def process_large_dataset(dataset: list, algorithm: str = "standard") -> dict:
    """Process large dataset with performance monitoring."""
    
    start_time = time.time()
    
    # Phase 1: Data preparation
    with create_span("data_preparation") as span:
        set_span_attribute(span, "dataset.size", len(dataset))
        set_span_attribute(span, "algorithm.type", algorithm)
        
        prep_start = time.time()
        
        # Simulate data preparation
        prepared_data = [item for item in dataset if item is not None]
        
        prep_time = time.time() - prep_start
        set_span_attribute(span, "preparation.duration_seconds", prep_time)
        set_span_attribute(span, "preparation.items_processed", len(prepared_data))
    
    # Phase 2: Core processing
    with create_span("core_processing") as span:
        set_span_attribute(span, "processing.algorithm", algorithm)
        
        process_start = time.time()
        
        # Simulate different algorithms
        if algorithm == "fast":
            time.sleep(min(len(prepared_data) / 10000, 0.1))
            processed_items = len(prepared_data)
        elif algorithm == "accurate":
            time.sleep(min(len(prepared_data) / 5000, 0.2))
            processed_items = len(prepared_data)
        else:  # standard
            time.sleep(min(len(prepared_data) / 7500, 0.15))
            processed_items = len(prepared_data)
        
        process_time = time.time() - process_start
        set_span_attribute(span, "processing.duration_seconds", process_time)
        set_span_attribute(span, "processing.items_processed", processed_items)
        set_span_attribute(span, "processing.throughput_items_per_second", processed_items / process_time)
    
    # Phase 3: Result aggregation
    with create_span("result_aggregation") as span:
        agg_start = time.time()
        
        # Aggregate results
        result_summary = {
            "total_items": len(dataset),
            "processed_items": processed_items,
            "algorithm_used": algorithm,
            "success_rate": processed_items / len(dataset) if dataset else 0
        }
        
        agg_time = time.time() - agg_start
        set_span_attribute(span, "aggregation.duration_seconds", agg_time)
    
    total_time = time.time() - start_time
    
    return {
        **result_summary,
        "performance_metrics": {
            "total_duration_seconds": total_time,
            "preparation_duration_seconds": prep_time,
            "processing_duration_seconds": process_time,
            "aggregation_duration_seconds": agg_time,
            "overall_throughput_items_per_second": processed_items / total_time
        }
    }

@task(
    name="memory_efficient_task",
    organization_id="performance_team",
    project_id="resource_optimization"
)
def process_streaming_chunks(data_generator, chunk_size: int = 1000) -> dict:
    """Process data in chunks to manage memory usage."""
    
    processed_chunks = 0
    total_items = 0
    errors = 0
    
    with create_span("streaming_processing") as span:
        set_span_attribute(span, "processing.chunk_size", chunk_size)
        
        for chunk in data_generator:
            chunk_start = time.time()
            
            try:
                # Process chunk
                chunk_items = len(chunk) if hasattr(chunk, '__len__') else sum(1 for _ in chunk)
                
                # Simulate processing
                time.sleep(chunk_items / 100000)  # Very fast processing
                
                processed_chunks += 1
                total_items += chunk_items
                
                # Update span attributes periodically
                if processed_chunks % 10 == 0:
                    set_span_attribute(span, "processing.chunks_completed", processed_chunks)
                    set_span_attribute(span, "processing.items_processed", total_items)
                
            except Exception as e:
                errors += 1
                logger.warning(f"Error processing chunk {processed_chunks}: {e}")
        
        # Final span attributes
        set_span_attribute(span, "processing.total_chunks", processed_chunks)
        set_span_attribute(span, "processing.total_items", total_items)
        set_span_attribute(span, "processing.error_count", errors)
    
    return {
        "processed_chunks": processed_chunks,
        "total_items": total_items,
        "errors": errors,
        "average_chunk_size": total_items / processed_chunks if processed_chunks > 0 else 0
    }
```

## Testing Tasks

Here's how to test task-decorated functions:

```python
import pytest
from unittest.mock import Mock, patch
from rizk.sdk import Rizk
from rizk.sdk.decorators import task, workflow

@pytest.fixture
def rizk_setup():
    """Setup Rizk for testing."""
    return Rizk.init(app_name="TaskTest", enabled=True)

def test_basic_task(rizk_setup):
    """Test basic task functionality."""
    
    @task(
        name="test_task",
        organization_id="test_org",
        project_id="test_project"
    )
    def simple_task(input_value: str) -> dict:
        return {
            "input": input_value,
            "output": input_value.upper(),
            "status": "completed"
        }
    
    # Test execution
    result = simple_task("hello world")
    
    assert result["input"] == "hello world"
    assert result["output"] == "HELLO WORLD"
    assert result["status"] == "completed"

def test_task_error_handling(rizk_setup):
    """Test task error handling."""
    
    @task(
        name="error_test_task",
        organization_id="test_org",
        project_id="test_project"
    )
    def error_task(should_fail: bool) -> dict:
        if should_fail:
            raise ValueError("Test error")
        return {"status": "success"}
    
    # Test successful execution
    result = error_task(False)
    assert result["status"] == "success"
    
    # Test error handling
    with pytest.raises(ValueError):
        error_task(True)

def test_async_task(rizk_setup):
    """Test async task functionality."""
    
    @task(
        name="async_test_task",
        organization_id="test_org",
        project_id="test_project"
    )
    async def async_task(delay: float) -> dict:
        await asyncio.sleep(delay)
        return {
            "delay": delay,
            "status": "completed"
        }
    
    # Test async execution
    async def run_test():
        result = await async_task(0.01)
        assert result["delay"] == 0.01
        assert result["status"] == "completed"
    
    asyncio.run(run_test())

def test_task_in_workflow_context(rizk_setup):
    """Test task within workflow context."""
    
    @workflow(
        name="test_workflow",
        organization_id="test_org",
        project_id="test_project"
    )
    def parent_workflow(data: list) -> dict:
        processed_data = process_data_task(data)
        validated_data = validate_data_task(processed_data)
        
        return {
            "original_count": len(data),
            "processed_count": len(processed_data),
            "valid_count": len(validated_data)
        }
    
    @task(name="process_data_task")
    def process_data_task(data: list) -> list:
        return [item.upper() if isinstance(item, str) else item for item in data]
    
    @task(name="validate_data_task")
    def validate_data_task(data: list) -> list:
        return [item for item in data if item and len(str(item)) > 0]
    
    # Test workflow with tasks
    test_data = ["hello", "world", "", "test"]
    result = parent_workflow(test_data)
    
    assert result["original_count"] == 4
    assert result["processed_count"] == 4
    assert result["valid_count"] == 3  # Empty string filtered out

@patch('time.sleep')
def test_performance_monitoring_task(mock_sleep, rizk_setup):
    """Test task performance monitoring."""
    mock_sleep.return_value = None  # Skip actual sleep
    
    @task(
        name="performance_test_task",
        organization_id="test_org",
        project_id="test_project"
    )
    def performance_task(dataset_size: int) -> dict:
        start_time = time.time()
        
        # Simulate processing
        time.sleep(0.1)  # This will be mocked
        
        end_time = time.time()
        
        return {
            "dataset_size": dataset_size,
            "duration": end_time - start_time,
            "throughput": dataset_size / (end_time - start_time)
        }
    
    result = performance_task(1000)
    
    assert result["dataset_size"] == 1000
    assert "duration" in result
    assert "throughput" in result
    mock_sleep.assert_called_once_with(0.1)
```

## Best Practices

### 1. **Single Responsibility**
```python
# Good: Task has a single, clear purpose
@task(name="validate_email")
def validate_email(email: str) -> bool:
    return "@" in email and "." in email.split("@")[1]

# Avoid: Task doing multiple unrelated things
@task(name="validate_and_send_email")  # Too broad
def validate_and_send_email(email: str, message: str):
    # Validation and sending are separate concerns
    pass
```

### 2. **Clear Input/Output Contracts**
```python
# Good: Clear input validation and structured output
@task(name="process_user_data")
def process_user_data(user_data: dict) -> dict:
    if not isinstance(user_data, dict):
        raise TypeError("user_data must be a dictionary")
    
    if "id" not in user_data:
        raise ValueError("user_data must contain 'id' field")
    
    return {
        "user_id": user_data["id"],
        "processed": True,
        "timestamp": datetime.now().isoformat()
    }
```

### 3. **Error Handling**
```python
# Good: Comprehensive error handling with context
@task(name="safe_api_call")
def safe_api_call(endpoint: str) -> dict:
    try:
        # API call logic
        result = make_api_call(endpoint)
        return {"status": "success", "data": result}
    
    except requests.ConnectionError as e:
        logger.error(f"Connection error for {endpoint}: {e}")
        return {"status": "connection_error", "endpoint": endpoint}
    
    except requests.Timeout as e:
        logger.error(f"Timeout error for {endpoint}: {e}")
        return {"status": "timeout", "endpoint": endpoint}
    
    except Exception as e:
        logger.error(f"Unexpected error for {endpoint}: {e}")
        return {"status": "error", "error": str(e), "endpoint": endpoint}
```

### 4. **Performance Awareness**
```python
# Good: Monitor performance-critical operations
@task(name="large_data_processing")
def process_large_data(data: list) -> dict:
    start_time = time.time()
    
    with create_span("data_processing") as span:
        set_span_attribute(span, "data_size", len(data))
        
        # Processing logic
        result = expensive_operation(data)
        
        duration = time.time() - start_time
        set_span_attribute(span, "processing_duration", duration)
        
    return {"result": result, "processing_time": duration}
```

### 5. **Context Inheritance**
```python
# Good: Let tasks inherit context from workflows
@workflow(name="data_pipeline")
def data_pipeline(source: str) -> dict:
    raw_data = extract_data(source)      # Task inherits context
    clean_data = clean_data_task(raw_data)  # Task inherits context
    return {"processed": len(clean_data)}

@task(name="extract_data")  # No need to repeat context
def extract_data(source: str) -> list:
    # Implementation
    pass

@task(name="clean_data_task")  # Context inherited automatically
def clean_data_task(data: list) -> list:
    # Implementation
    pass
```

## Common Patterns

### 1. **Data Processing Pipeline**
```python
@task(name="extract_task")
def extract_data(source: str) -> list:
    """Extract data from source."""
    # Extraction logic
    return extracted_data

@task(name="transform_task")
def transform_data(data: list) -> list:
    """Transform extracted data."""
    # Transformation logic
    return transformed_data

@task(name="load_task")
def load_data(data: list, destination: str) -> dict:
    """Load data to destination."""
    # Loading logic
    return {"loaded_count": len(data)}
```

### 2. **Validation Chain**
```python
@task(name="schema_validation")
def validate_schema(data: dict, schema: dict) -> dict:
    """Validate data against schema."""
    # Schema validation
    return validation_result

@task(name="business_validation")
def validate_business_rules(data: dict) -> dict:
    """Validate business rules."""
    # Business rule validation
    return validation_result

@task(name="security_validation")
def validate_security(data: dict) -> dict:
    """Validate security constraints."""
    # Security validation
    return validation_result
```

### 3. **Async Processing Pattern**
```python
@task(name="async_fetch")
async def fetch_data(url: str) -> dict:
    """Fetch data asynchronously."""
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return await response.json()

@task(name="async_process")
async def process_data(data: dict) -> dict:
    """Process data asynchronously."""
    # Async processing
    return processed_data

@task(name="async_save")
async def save_data(data: dict, destination: str) -> dict:
    """Save data asynchronously."""
    # Async save operation
    return {"saved": True}
```

## Troubleshooting

### Common Issues

1. **Task Not Inheriting Context**
```python
# Issue: Task not getting workflow context
# Solution: Ensure task is called within workflow scope

@workflow(name="parent")
def parent_workflow():
    return child_task()  # This will inherit context

@task(name="child")
def child_task():
    # Will inherit parent workflow context
    pass
```

2. **Performance Issues**
```python
# Issue: Task is slow
# Solution: Add performance monitoring

@task(name="monitored_task")
def slow_task(data):
    with create_span("processing") as span:
        start = time.time()
        result = process(data)
        set_span_attribute(span, "duration", time.time() - start)
    return result
```

3. **Error Handling**
```python
# Issue: Tasks failing silently
# Solution: Add comprehensive error handling

@task(name="robust_task")
def robust_task(data):
    try:
        return process(data)
    except Exception as e:
        logger.error(f"Task failed: {e}")
        return {"error": str(e), "status": "failed"}
```

## Related Documentation

- **[@workflow Decorator](workflow.md)** - For high-level processes that orchestrate tasks
- **[@agent Decorator](agent.md)** - For autonomous components that may use tasks
- **[@tool Decorator](tool.md)** - For utility functions similar to tasks
- **[Performance Monitoring](../observability/performance-monitoring.md)** - Optimizing task performance
- **[Error Handling](../troubleshooting/debugging.md)** - Advanced error handling patterns

---

The `@task` decorator provides fine-grained observability for individual operations while seamlessly integrating with larger workflows. It's the building block for creating comprehensive, monitored AI applications with detailed operational insights.

