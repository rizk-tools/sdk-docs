---
title: "@tool Decorator"
description: "@tool Decorator"
---

# @tool Decorator

The `@tool` decorator is designed for instrumenting utility functions and capabilities that can be used by agents, workflows, and other components. It provides automatic framework adaptation and observability for tool execution across different AI frameworks.

## Overview

A **tool** represents a specific capability or function that can be invoked by agents, workflows, or other components to perform discrete operations. The `@tool` decorator automatically adapts tools for use with different frameworks (OpenAI Agents, LangChain, CrewAI) while providing comprehensive monitoring and error handling.

## Basic Usage

```python
from rizk.sdk import Rizk
from rizk.sdk.decorators import tool, agent

# Initialize Rizk
rizk = Rizk.init(app_name="ToolApp", enabled=True)

@tool(
    name="text_analyzer",
    organization_id="nlp_team",
    project_id="text_processing"
)
def analyze_text(text: str, analysis_type: str = "sentiment") -> dict:
    """Analyze text for various properties."""
    
    # Mock text analysis
    results = {
        "text": text,
        "analysis_type": analysis_type,
        "length": len(text),
        "word_count": len(text.split()),
        "timestamp": datetime.now().isoformat()
    }
    
    if analysis_type == "sentiment":
        # Simple sentiment analysis
        positive_words = ["good", "great", "excellent", "amazing", "wonderful"]
        negative_words = ["bad", "terrible", "awful", "horrible", "disappointing"]
        
        text_lower = text.lower()
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        if positive_count > negative_count:
            sentiment = "positive"
            confidence = min(0.9, 0.5 + (positive_count - negative_count) * 0.1)
        elif negative_count > positive_count:
            sentiment = "negative"
            confidence = min(0.9, 0.5 + (negative_count - positive_count) * 0.1)
        else:
            sentiment = "neutral"
            confidence = 0.5
        
        results.update({
            "sentiment": sentiment,
            "confidence": confidence,
            "positive_indicators": positive_count,
            "negative_indicators": negative_count
        })
    
    elif analysis_type == "keywords":
        # Simple keyword extraction
        words = text.lower().split()
        word_freq = {}
        for word in words:
            if len(word) > 3:  # Only words longer than 3 characters
                word_freq[word] = word_freq.get(word, 0) + 1
        
        keywords = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:5]
        results.update({
            "keywords": [word for word, freq in keywords],
            "keyword_frequencies": dict(keywords)
        })
    
    return results

# Usage
text_sample = "This is a great example of amazing text analysis capabilities!"
result = analyze_text(text_sample, "sentiment")
print(f"Analysis result: {result}")
```

## Parameters Reference

### Core Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `name` | `str` | No | Tool name (defaults to function name) |
| `version` | `int` | No | Tool version for tracking changes |
| `organization_id` | `str` | No | Organization identifier for hierarchical context |
| `project_id` | `str` | No | Project identifier for grouping tools |
| `tool_id` | `str` | No | Specific tool identifier for unique identification |

### Advanced Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `**kwargs` | `Any` | Framework-specific parameters passed to underlying adapters |

## Framework-Specific Behavior

### OpenAI Agents Integration

```python
from agents import Agent, Runner
from rizk.sdk.decorators import tool, agent

@tool(
    name="calculator",
    organization_id="math_team",
    project_id="computation_tools"
)
def calculate(expression: str) -> str:
    """Safely evaluate mathematical expressions."""
    try:
        # Safe evaluation for basic math
        allowed_chars = set('0123456789+-*/().')
        if not all(c in allowed_chars or c.isspace() for c in expression):
            return f"Error: Invalid characters in expression"
        
        result = eval(expression)
        return f"The result is {result}"
    except Exception as e:
        return f"Error: {str(e)}"

@tool(
    name="unit_converter",
    organization_id="math_team",
    project_id="computation_tools"
)
def convert_units(value: float, from_unit: str, to_unit: str) -> str:
    """Convert between different units."""
    
    # Length conversions (to meters)
    length_to_meters = {
        "mm": 0.001, "cm": 0.01, "m": 1.0, "km": 1000.0,
        "in": 0.0254, "ft": 0.3048, "yd": 0.9144, "mi": 1609.34
    }
    
    # Temperature conversions
    if from_unit == "celsius" and to_unit == "fahrenheit":
        result = (value * 9/5) + 32
        return f"{value}Â°C = {result}Â°F"
    elif from_unit == "fahrenheit" and to_unit == "celsius":
        result = (value - 32) * 5/9
        return f"{value}Â°F = {result}Â°C"
    
    # Length conversions
    elif from_unit in length_to_meters and to_unit in length_to_meters:
        meters = value * length_to_meters[from_unit]
        result = meters / length_to_meters[to_unit]
        return f"{value} {from_unit} = {result} {to_unit}"
    
    else:
        return f"Conversion from {from_unit} to {to_unit} not supported"

@agent(
    name="math_assistant",
    organization_id="math_team",
    project_id="computation_tools"
)
def create_math_agent() -> Agent:
    """Create a math assistant with calculation tools."""
    
    agent = Agent(
        name="MathAssistant",
        instructions="You are a helpful math assistant. Use the available tools for calculations and unit conversions.",
        model="gpt-4",
        tools=[calculate, convert_units]  # Tools are automatically integrated
    )
    
    return agent
```

### LangChain Integration

```python
from langchain.tools import Tool
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_openai import ChatOpenAI
from rizk.sdk.decorators import tool, agent

@tool(
    name="web_search",
    organization_id="research_team",
    project_id="information_tools"
)
def search_web(query: str, max_results: int = 5) -> str:
    """Search the web for information."""
    # Mock web search results
    mock_results = [
        f"Result 1 for '{query}': Comprehensive information about {query}",
        f"Result 2 for '{query}': Latest developments in {query}",
        f"Result 3 for '{query}': Expert analysis of {query}",
        f"Result 4 for '{query}': Historical context of {query}",
        f"Result 5 for '{query}': Future trends in {query}"
    ]
    
    results = mock_results[:max_results]
    return "\n".join(results)

@tool(
    name="summarize_text",
    organization_id="research_team",
    project_id="information_tools"
)
def summarize_text(text: str, max_sentences: int = 3) -> str:
    """Summarize text to key points."""
    sentences = text.split('.')
    sentences = [s.strip() for s in sentences if s.strip()]
    
    # Simple summarization - take first few sentences
    summary_sentences = sentences[:max_sentences]
    return '. '.join(summary_sentences) + '.'

@agent(
    name="research_assistant",
    organization_id="research_team",
    project_id="information_tools"
)
def create_research_agent() -> AgentExecutor:
    """Create a research assistant with information tools."""
    
    llm = ChatOpenAI(temperature=0)
    
    # Convert Rizk tools to LangChain tools
    langchain_tools = [
        Tool(
            name="web_search",
            description="Search the web for information on any topic",
            func=search_web
        ),
        Tool(
            name="summarize_text", 
            description="Summarize long text into key points",
            func=summarize_text
        )
    ]
    
    # Create agent with tools
    agent = create_openai_tools_agent(llm, langchain_tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=langchain_tools)
    
    return agent_executor
```

## Tool Composition and Chaining

Tools can be composed and chained together:

```python
from rizk.sdk.decorators import tool, workflow

@tool(
    name="data_fetcher",
    organization_id="data_team",
    project_id="data_pipeline"
)
def fetch_data(source: str, filters: dict = None) -> dict:
    """Fetch data from various sources."""
    
    # Mock data fetching
    mock_data = {
        "database": [
            {"id": 1, "name": "Alice", "score": 95},
            {"id": 2, "name": "Bob", "score": 87},
            {"id": 3, "name": "Charlie", "score": 92}
        ],
        "api": [
            {"user_id": "u1", "activity": "login", "timestamp": "2024-01-01T10:00:00"},
            {"user_id": "u2", "activity": "purchase", "timestamp": "2024-01-01T11:00:00"}
        ],
        "file": [
            {"product": "Widget A", "sales": 150, "region": "North"},
            {"product": "Widget B", "sales": 200, "region": "South"}
        ]
    }
    
    data = mock_data.get(source, [])
    
    # Apply filters if provided
    if filters:
        filtered_data = []
        for item in data:
            match = True
            for key, value in filters.items():
                if key in item and item[key] != value:
                    match = False
                    break
            if match:
                filtered_data.append(item)
        data = filtered_data
    
    return {
        "source": source,
        "data": data,
        "count": len(data),
        "filters_applied": filters or {}
    }

@tool(
    name="data_transformer",
    organization_id="data_team",
    project_id="data_pipeline"
)
def transform_data(data_result: dict, transformations: list) -> dict:
    """Transform data using specified transformations."""
    
    data = data_result["data"]
    transformed_data = []
    
    for item in data:
        transformed_item = item.copy()
        
        for transformation in transformations:
            if transformation["type"] == "add_field":
                transformed_item[transformation["field"]] = transformation["value"]
            elif transformation["type"] == "multiply_field":
                field = transformation["field"]
                if field in transformed_item and isinstance(transformed_item[field], (int, float)):
                    transformed_item[field] = transformed_item[field] * transformation["factor"]
            elif transformation["type"] == "uppercase_field":
                field = transformation["field"]
                if field in transformed_item and isinstance(transformed_item[field], str):
                    transformed_item[field] = transformed_item[field].upper()
        
        transformed_data.append(transformed_item)
    
    return {
        "source": data_result["source"],
        "data": transformed_data,
        "count": len(transformed_data),
        "transformations_applied": transformations
    }

@tool(
    name="data_aggregator",
    organization_id="data_team",
    project_id="data_pipeline"
)
def aggregate_data(data_result: dict, group_by: str, aggregate_field: str, operation: str = "sum") -> dict:
    """Aggregate data by grouping and applying operations."""
    
    data = data_result["data"]
    groups = {}
    
    # Group data
    for item in data:
        if group_by in item:
            group_key = item[group_by]
            if group_key not in groups:
                groups[group_key] = []
            groups[group_key].append(item)
    
    # Aggregate within groups
    aggregated_results = []
    for group_key, group_items in groups.items():
        if aggregate_field in group_items[0]:
            values = [item[aggregate_field] for item in group_items if aggregate_field in item]
            
            if operation == "sum":
                aggregated_value = sum(values)
            elif operation == "avg":
                aggregated_value = sum(values) / len(values)
            elif operation == "count":
                aggregated_value = len(values)
            elif operation == "max":
                aggregated_value = max(values)
            elif operation == "min":
                aggregated_value = min(values)
            else:
                aggregated_value = sum(values)  # Default to sum
            
            aggregated_results.append({
                group_by: group_key,
                f"{operation}_{aggregate_field}": aggregated_value,
                "item_count": len(group_items)
            })
    
    return {
        "source": data_result["source"],
        "aggregated_data": aggregated_results,
        "group_by": group_by,
        "aggregate_field": aggregate_field,
        "operation": operation
    }

@workflow(
    name="data_processing_pipeline",
    organization_id="data_team",
    project_id="data_pipeline"
)
def process_data_pipeline(source: str, filters: dict = None) -> dict:
    """Process data through a pipeline of tools."""
    
    # Step 1: Fetch data
    raw_data = fetch_data(source, filters)
    
    # Step 2: Transform data
    transformations = [
        {"type": "add_field", "field": "processed_at", "value": datetime.now().isoformat()},
        {"type": "multiply_field", "field": "score", "factor": 1.1}  # Boost scores by 10%
    ]
    transformed_data = transform_data(raw_data, transformations)
    
    # Step 3: Aggregate data (if applicable)
    if source == "database":
        aggregated_data = aggregate_data(transformed_data, "name", "score", "avg")
    else:
        aggregated_data = {"message": "Aggregation not applicable for this data type"}
    
    return {
        "pipeline_steps": ["fetch", "transform", "aggregate"],
        "raw_data_count": raw_data["count"],
        "transformed_data_count": transformed_data["count"],
        "final_result": aggregated_data,
        "processing_complete": True
    }

# Usage
result = process_data_pipeline("database", {"score": 95})
print(f"Pipeline result: {result}")
```

## Async Tool Support

```python
import asyncio
import aiohttp
from rizk.sdk.decorators import tool, agent

@tool(
    name="async_api_client",
    organization_id="integration_team",
    project_id="external_apis"
)
async def fetch_api_data(url: str, headers: dict = None) -> dict:
    """Fetch data from external API asynchronously."""
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    return {
                        "status": "success",
                        "data": data,
                        "url": url,
                        "response_code": response.status
                    }
                else:
                    return {
                        "status": "error",
                        "error": f"HTTP {response.status}",
                        "url": url,
                        "response_code": response.status
                    }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "url": url,
            "response_code": None
        }

@tool(
    name="async_data_processor",
    organization_id="integration_team",
    project_id="external_apis"
)
async def process_api_data(api_result: dict) -> dict:
    """Process API data asynchronously."""
    
    if api_result["status"] != "success":
        return api_result  # Return error as-is
    
    # Simulate async processing
    await asyncio.sleep(0.1)
    
    data = api_result["data"]
    processed_data = {
        "original_count": len(data) if isinstance(data, list) else 1,
        "processed_at": datetime.now().isoformat(),
        "processing_time": "100ms",
        "summary": f"Processed data from {api_result['url']}"
    }
    
    return {
        "status": "processed",
        "original_result": api_result,
        "processed_data": processed_data
    }

@agent(
    name="async_integration_agent",
    organization_id="integration_team",
    project_id="external_apis"
)
async def create_async_agent() -> dict:
    """Create an async integration agent."""
    
    return {
        "name": "AsyncIntegrationAgent",
        "capabilities": ["api_integration", "async_processing"],
        "tools": [fetch_api_data, process_api_data],
        "async_support": True
    }

async def run_async_tool_workflow(urls: list) -> dict:
    """Run async workflow with multiple API calls."""
    
    agent = await create_async_agent()
    
    # Fetch data from multiple URLs concurrently
    fetch_tasks = [fetch_api_data(url) for url in urls]
    api_results = await asyncio.gather(*fetch_tasks)
    
    # Process results concurrently
    process_tasks = [process_api_data(result) for result in api_results]
    processed_results = await asyncio.gather(*process_tasks)
    
    return {
        "agent": agent["name"],
        "urls_processed": len(urls),
        "successful_fetches": sum(1 for r in api_results if r["status"] == "success"),
        "processed_results": processed_results,
        "workflow_complete": True
    }
```

## Tool Error Handling and Validation

```python
from rizk.sdk.decorators import tool
from rizk.sdk.utils.error_handler import handle_errors

@tool(
    name="robust_file_processor",
    organization_id="file_team",
    project_id="file_operations"
)
@handle_errors(fail_closed=False, max_retries=3)
def process_file(file_path: str, operation: str, options: dict = None) -> dict:
    """Process files with comprehensive error handling."""
    
    # Input validation
    if not isinstance(file_path, str) or not file_path.strip():
        raise ValueError("file_path must be a non-empty string")
    
    if operation not in ["read", "analyze", "transform"]:
        raise ValueError("operation must be one of: read, analyze, transform")
    
    options = options or {}
    
    try:
        # Simulate file operations
        if operation == "read":
            if "simulate_not_found" in options:
                raise FileNotFoundError(f"File not found: {file_path}")
            
            # Mock file reading
            content = f"Mock content from {file_path}"
            return {
                "operation": operation,
                "file_path": file_path,
                "content": content,
                "size": len(content),
                "status": "success"
            }
        
        elif operation == "analyze":
            if "simulate_corruption" in options:
                raise IOError(f"File corrupted: {file_path}")
            
            # Mock file analysis
            analysis = {
                "file_type": "text",
                "encoding": "utf-8",
                "line_count": 42,
                "word_count": 150
            }
            return {
                "operation": operation,
                "file_path": file_path,
                "analysis": analysis,
                "status": "success"
            }
        
        elif operation == "transform":
            if "simulate_permission_error" in options:
                raise PermissionError(f"Permission denied: {file_path}")
            
            # Mock file transformation
            return {
                "operation": operation,
                "file_path": file_path,
                "output_path": f"{file_path}.transformed",
                "transformation": options.get("transform_type", "default"),
                "status": "success"
            }
    
    except FileNotFoundError as e:
        return {
            "operation": operation,
            "file_path": file_path,
            "status": "file_not_found",
            "error": str(e)
        }
    
    except (IOError, PermissionError) as e:
        # These errors will be retried by the error handler
        logger.warning(f"Recoverable file operation error: {e}")
        raise
    
    except Exception as e:
        return {
            "operation": operation,
            "file_path": file_path,
            "status": "unexpected_error",
            "error": str(e)
        }

@tool(
    name="validated_calculator",
    organization_id="math_team",
    project_id="secure_computation"
)
def secure_calculate(expression: str, precision: int = 2) -> dict:
    """Calculate with input validation and security checks."""
    
    # Input validation
    if not isinstance(expression, str):
        raise TypeError("expression must be a string")
    
    if not isinstance(precision, int) or precision < 0 or precision > 10:
        raise ValueError("precision must be an integer between 0 and 10")
    
    # Security validation
    expression = expression.strip()
    if not expression:
        raise ValueError("expression cannot be empty")
    
    # Check for dangerous patterns
    dangerous_patterns = ["import", "__", "exec", "eval", "open", "file"]
    for pattern in dangerous_patterns:
        if pattern in expression.lower():
            raise SecurityError(f"Dangerous pattern detected: {pattern}")
    
    # Allowed characters for mathematical expressions
    allowed_chars = set('0123456789+-*/().')
    if not all(c in allowed_chars or c.isspace() for c in expression):
        invalid_chars = set(expression) - allowed_chars - set(' ')
        raise ValueError(f"Invalid characters in expression: {invalid_chars}")
    
    try:
        # Safe evaluation
        result = eval(expression)
        
        # Format result with specified precision
        if isinstance(result, float):
            formatted_result = round(result, precision)
        else:
            formatted_result = result
        
        return {
            "expression": expression,
            "result": formatted_result,
            "precision": precision,
            "status": "success"
        }
    
    except ZeroDivisionError:
        return {
            "expression": expression,
            "status": "division_by_zero",
            "error": "Cannot divide by zero"
        }
    
    except Exception as e:
        return {
            "expression": expression,
            "status": "calculation_error",
            "error": str(e)
        }

class SecurityError(Exception):
    """Custom security exception."""
    pass
```

## Tool Performance Monitoring

```python
import time
from rizk.sdk.decorators import tool
from rizk.sdk.tracing import create_span, set_span_attribute

@tool(
    name="performance_monitored_tool",
    organization_id="performance_team",
    project_id="monitoring"
)
def process_with_monitoring(data: list, algorithm: str = "standard") -> dict:
    """Process data with detailed performance monitoring."""
    
    start_time = time.time()
    
    with create_span("data_processing") as span:
        set_span_attribute(span, "data_size", len(data))
        set_span_attribute(span, "algorithm", algorithm)
        
        # Phase 1: Validation
        with create_span("validation") as validation_span:
            validation_start = time.time()
            
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            
            valid_items = [item for item in data if item is not None]
            validation_time = time.time() - validation_start
            
            set_span_attribute(validation_span, "validation_time", validation_time)
            set_span_attribute(validation_span, "valid_items", len(valid_items))
        
        # Phase 2: Processing
        with create_span("core_processing") as processing_span:
            processing_start = time.time()
            
            if algorithm == "fast":
                # Fast algorithm
                processed_items = [f"fast_{item}" for item in valid_items]
                time.sleep(0.01)  # Simulate fast processing
            elif algorithm == "accurate":
                # Accurate algorithm
                processed_items = [f"accurate_{item}_verified" for item in valid_items]
                time.sleep(0.05)  # Simulate slower but more accurate processing
            else:
                # Standard algorithm
                processed_items = [f"standard_{item}" for item in valid_items]
                time.sleep(0.03)  # Simulate standard processing
            
            processing_time = time.time() - processing_start
            
            set_span_attribute(processing_span, "processing_time", processing_time)
            set_span_attribute(processing_span, "processed_items", len(processed_items))
            set_span_attribute(processing_span, "throughput", len(processed_items) / processing_time)
        
        # Phase 3: Output formatting
        with create_span("output_formatting") as output_span:
            output_start = time.time()
            
            result = {
                "algorithm": algorithm,
                "input_size": len(data),
                "valid_items": len(valid_items),
                "processed_items": len(processed_items),
                "sample_output": processed_items[:3],  # Show first 3 items
                "processing_complete": True
            }
            
            output_time = time.time() - output_start
            set_span_attribute(output_span, "output_formatting_time", output_time)
    
    total_time = time.time() - start_time
    
    # Add performance metrics to result
    result["performance_metrics"] = {
        "total_time": total_time,
        "validation_time": validation_time,
        "processing_time": processing_time,
        "output_time": output_time,
        "items_per_second": len(processed_items) / total_time if total_time > 0 else 0
    }
    
    return result
```

## Best Practices

### 1. **Clear Tool Purpose**
```python
# Good: Specific, focused tool
@tool(name="email_validator")
def validate_email(email: str) -> bool:
    return "@" in email and "." in email.split("@")[1]

# Avoid: Multi-purpose tool
@tool(name="general_validator")  # Too broad
def validate_everything(data, type):
    pass
```

### 2. **Input Validation**
```python
# Good: Comprehensive input validation
@tool(name="safe_processor")
def process_data(data: list, options: dict = None) -> dict:
    if not isinstance(data, list):
        raise TypeError("data must be a list")
    
    if options and not isinstance(options, dict):
        raise TypeError("options must be a dictionary")
    
    # Processing logic
    return {"processed": True}
```

### 3. **Error Handling**
```python
# Good: Structured error responses
@tool(name="robust_tool")
def robust_operation(input_data):
    try:
        result = process(input_data)
        return {"status": "success", "result": result}
    except ValueError as e:
        return {"status": "validation_error", "error": str(e)}
    except Exception as e:
        return {"status": "error", "error": str(e)}
```

### 4. **Documentation**
```python
# Good: Clear documentation with types and examples
@tool(name="documented_tool")
def well_documented_tool(param1: str, param2: int = 10) -> dict:
    """
    Process data with specified parameters.
    
    Args:
        param1: Input string to process
        param2: Processing intensity (1-100, default: 10)
    
    Returns:
        Dictionary with processing results
    
    Example:
        result = well_documented_tool("hello", 5)
    """
    pass
```

## Related Documentation

- **[@agent Decorator](agent.md)** - For agents that use tools
- **[@workflow Decorator](workflow.md)** - For workflows that orchestrate tools
- **[@task Decorator](task.md)** - For tasks that may use tools
- **[Framework Integration](../framework-integration/)** - Framework-specific tool patterns
- **[Error Handling](../troubleshooting/debugging.md)** - Advanced error handling patterns

---

The `@tool` decorator provides comprehensive observability and framework adaptation for utility functions, enabling them to be seamlessly used across different AI frameworks while maintaining proper monitoring and error handling.

