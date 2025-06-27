---
title: "Decorator Composition"
description: "Decorator Composition"
---

# Decorator Composition

Combining multiple Rizk decorators enables you to create sophisticated, enterprise-grade workflows with comprehensive observability, policy enforcement, and framework integration. This guide covers best practices for decorator composition, execution order, and advanced patterns.

## Overview

Decorator composition in Rizk allows you to layer multiple capabilities onto a single function:

- **Observability**: Workflow, task, and agent tracing
- **Policy Enforcement**: Guardrails and custom policies
- **Framework Integration**: Automatic adaptation to your LLM framework
- **Context Management**: Hierarchical organization and project tracking

## Basic Composition Patterns

### Workflow with Guardrails

```python
from rizk.sdk import Rizk
from rizk.sdk.decorators import workflow, guardrails
from datetime import datetime

# Initialize Rizk
rizk = Rizk.init(app_name="CompositionApp", enabled=True)

@guardrails(enforcement_level="strict")
@workflow(name="content_generation", 
          organization_id="acme_corp", 
          project_id="marketing_automation")
def generate_marketing_content(topic: str, audience: str, tone: str = "professional") -> dict:
    """Generate marketing content with policy enforcement and workflow tracking."""
    
    # Content generation logic
    content_templates = {
        "professional": f"We are pleased to present insights on {topic} for {audience}.",
        "casual": f"Hey {audience}! Let's talk about {topic}.",
        "technical": f"Technical analysis of {topic} for {audience} stakeholders."
    }
    
    generated_content = {
        "topic": topic,
        "audience": audience,
        "tone": tone,
        "content": content_templates.get(tone, content_templates["professional"]),
        "generated_at": datetime.now().isoformat(),
        "word_count": len(content_templates.get(tone, "").split()),
        "compliance_status": "approved"
    }
    
    return generated_content

# Usage
result = generate_marketing_content("AI Innovation", "enterprise_clients", "professional")
print(f"Generated content: {result}")
```

### Multi-Agent Coordination

```python
from rizk.sdk.decorators import agent, task, workflow, guardrails

@workflow(name="document_analysis_pipeline",
          organization_id="legal_firm",
          project_id="contract_review")
def document_analysis_pipeline(document_path: str, analysis_type: str) -> dict:
    """Orchestrate multi-agent document analysis."""
    
    # Initialize pipeline results
    pipeline_result = {
        "document_path": document_path,
        "analysis_type": analysis_type,
        "started_at": datetime.now().isoformat(),
        "stages": []
    }
    
    # Stage 1: Document extraction
    extraction_result = extract_document_content(document_path)
    pipeline_result["stages"].append({
        "stage": "extraction",
        "result": extraction_result,
        "completed_at": datetime.now().isoformat()
    })
    
    # Stage 2: Legal analysis
    analysis_result = analyze_legal_content(
        extraction_result["content"], 
        analysis_type
    )
    pipeline_result["stages"].append({
        "stage": "analysis", 
        "result": analysis_result,
        "completed_at": datetime.now().isoformat()
    })
    
    # Stage 3: Risk assessment
    risk_result = assess_document_risk(
        analysis_result["findings"],
        analysis_type
    )
    pipeline_result["stages"].append({
        "stage": "risk_assessment",
        "result": risk_result,
        "completed_at": datetime.now().isoformat()
    })
    
    # Final pipeline summary
    pipeline_result["completed_at"] = datetime.now().isoformat()
    pipeline_result["status"] = "completed"
    pipeline_result["overall_risk"] = risk_result.get("risk_level", "unknown")
    
    return pipeline_result

@guardrails(enforcement_level="moderate")
@agent(name="document_extractor",
       organization_id="legal_firm", 
       project_id="contract_review")
def extract_document_content(document_path: str) -> dict:
    """Extract content from legal documents."""
    
    # Simulate document extraction
    extraction_result = {
        "document_path": document_path,
        "content": f"Extracted content from {document_path}",
        "metadata": {
            "pages": 15,
            "sections": ["Introduction", "Terms", "Conditions", "Signatures"],
            "document_type": "contract",
            "extraction_confidence": 0.95
        },
        "extracted_at": datetime.now().isoformat()
    }
    
    return extraction_result

@guardrails(enforcement_level="strict")
@agent(name="legal_analyzer",
       organization_id="legal_firm",
       project_id="contract_review")
def analyze_legal_content(content: str, analysis_type: str) -> dict:
    """Analyze legal content for compliance and risk factors."""
    
    analysis_result = {
        "content": content,
        "analysis_type": analysis_type,
        "findings": [],
        "analyzed_at": datetime.now().isoformat()
    }
    
    # Simulate legal analysis
    if analysis_type == "contract_review":
        analysis_result["findings"] = [
            {"type": "clause", "content": "Termination clause found", "risk": "low"},
            {"type": "liability", "content": "Limited liability clause", "risk": "medium"},
            {"type": "compliance", "content": "GDPR compliance section", "risk": "low"}
        ]
    elif analysis_type == "risk_assessment":
        analysis_result["findings"] = [
            {"type": "financial", "content": "Payment terms analysis", "risk": "low"},
            {"type": "operational", "content": "Service level agreements", "risk": "medium"}
        ]
    
    analysis_result["total_findings"] = len(analysis_result["findings"])
    analysis_result["high_risk_findings"] = len([f for f in analysis_result["findings"] if f["risk"] == "high"])
    
    return analysis_result

@task(name="risk_assessor",
      organization_id="legal_firm",
      project_id="contract_review")
def assess_document_risk(findings: list, analysis_type: str) -> dict:
    """Assess overall risk based on legal analysis findings."""
    
    risk_assessment = {
        "findings": findings,
        "analysis_type": analysis_type,
        "assessed_at": datetime.now().isoformat(),
        "risk_factors": []
    }
    
    # Calculate risk score
    risk_score = 0
    for finding in findings:
        if finding["risk"] == "high":
            risk_score += 30
            risk_assessment["risk_factors"].append(f"High risk: {finding['content']}")
        elif finding["risk"] == "medium":
            risk_score += 15
            risk_assessment["risk_factors"].append(f"Medium risk: {finding['content']}")
        elif finding["risk"] == "low":
            risk_score += 5
    
    # Determine overall risk level
    if risk_score >= 50:
        risk_level = "high"
    elif risk_score >= 25:
        risk_level = "medium"
    else:
        risk_level = "low"
    
    risk_assessment["risk_score"] = risk_score
    risk_assessment["risk_level"] = risk_level
    risk_assessment["recommendation"] = {
        "high": "Requires legal review before proceeding",
        "medium": "Consider additional review",
        "low": "Standard processing acceptable"
    }[risk_level]
    
    return risk_assessment

# Usage
pipeline_result = document_analysis_pipeline("/documents/contract_001.pdf", "contract_review")
print(f"Pipeline completed with risk level: {pipeline_result['overall_risk']}")
```

## Advanced Composition Patterns

### Policy Layering with Custom Evaluators

```python
from rizk.sdk.decorators import policies, guardrails, workflow

def financial_compliance_evaluator(content: str, context: dict) -> dict:
    """Custom evaluator for financial compliance."""
    
    evaluation = {
        "compliant": True,
        "violations": [],
        "recommendations": [],
        "confidence": 0.9
    }
    
    # Check for financial regulations
    if "investment advice" in content.lower() and context.get("user_role") != "advisor":
        evaluation["violations"].append("Unlicensed investment advice")
        evaluation["compliant"] = False
    
    if any(term in content.lower() for term in ["guaranteed returns", "risk-free"]):
        evaluation["violations"].append("Misleading financial claims")
        evaluation["compliant"] = False
    
    return evaluation

@policies(["sox_compliance", "finra_rules"], 
          custom_evaluator=financial_compliance_evaluator,
          enforcement_mode="strict",
          priority=10)
@guardrails(enforcement_level="strict")
@workflow(name="financial_advisory_system",
          organization_id="wealth_management",
          project_id="client_advisory")
def provide_financial_advice(client_profile: dict, question: str, advisor_id: str) -> dict:
    """Provide financial advice with comprehensive compliance checking."""
    
    advice_response = {
        "client_profile": client_profile,
        "question": question,
        "advisor_id": advisor_id,
        "response_timestamp": datetime.now().isoformat(),
        "compliance_status": "evaluated"
    }
    
    # Generate advice based on client profile
    risk_tolerance = client_profile.get("risk_tolerance", "moderate")
    investment_timeline = client_profile.get("timeline", "medium_term")
    
    if risk_tolerance == "conservative":
        advice_response["recommendation"] = {
            "strategy": "Conservative portfolio with focus on bonds and stable investments",
            "allocation": {"bonds": 60, "stocks": 30, "cash": 10},
            "expected_return": "4-6% annually",
            "risk_level": "low"
        }
    elif risk_tolerance == "aggressive":
        advice_response["recommendation"] = {
            "strategy": "Growth-focused portfolio with higher equity allocation",
            "allocation": {"stocks": 70, "bonds": 20, "alternatives": 10},
            "expected_return": "8-12% annually",
            "risk_level": "high"
        }
    else:  # moderate
        advice_response["recommendation"] = {
            "strategy": "Balanced portfolio suitable for moderate risk tolerance",
            "allocation": {"stocks": 50, "bonds": 40, "cash": 10},
            "expected_return": "6-8% annually",
            "risk_level": "medium"
        }
    
    # Add compliance disclaimers
    advice_response["disclaimers"] = [
        "Past performance does not guarantee future results",
        "All investments carry risk of loss",
        "Consider your financial situation before investing",
        "Consult with a qualified financial advisor"
    ]
    
    advice_response["advisor_certification"] = f"Advice provided by certified advisor {advisor_id}"
    
    return advice_response

# Usage
client_data = {
    "client_id": "CLIENT_001",
    "risk_tolerance": "moderate",
    "timeline": "long_term",
    "investment_amount": 50000
}

advice = provide_financial_advice(
    client_data, 
    "What's the best investment strategy for retirement?", 
    "ADVISOR_123"
)
print(f"Financial advice provided: {advice['recommendation']['strategy']}")
```

## Decorator Execution Order

Understanding decorator execution order is crucial for effective composition:

```python
# Decorators are applied bottom-to-top (closest to function first)
@policies(["policy_a"])           # Applied 4th
@guardrails(enforcement_level="strict")  # Applied 3rd  
@workflow(name="example")         # Applied 2nd
@task(name="subtask")            # Applied 1st (closest to function)
def example_function():
    pass

# Execution order:
# 1. @task wrapper executes
# 2. @workflow wrapper executes  
# 3. @guardrails wrapper executes
# 4. @policies wrapper executes
# 5. Original function executes
# 6. Results bubble back up through wrappers
```

### Controlling Execution Order

```python
from functools import wraps

def execution_order_example():
    """Demonstrate decorator execution order."""
    
    def order_tracker(name):
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                print(f"Entering {name}")
                result = func(*args, **kwargs)
                print(f"Exiting {name}")
                return result
            return wrapper
        return decorator
    
    @order_tracker("Fourth")
    @order_tracker("Third") 
    @order_tracker("Second")
    @order_tracker("First")
    def test_function():
        print("Executing original function")
        return "result"
    
    return test_function()

# Output:
# Entering Fourth
# Entering Third
# Entering Second  
# Entering First
# Executing original function
# Exiting First
# Exiting Second
# Exiting Third
# Exiting Fourth
```

## Performance Considerations

### Decorator Overhead

```python
import time
from functools import wraps

def performance_monitoring_decorator(func):
    """Monitor decorator performance impact."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        
        # Pre-execution overhead
        overhead_start = time.time()
        # Simulate decorator processing
        time.sleep(0.001)  # 1ms overhead
        overhead_end = time.time()
        
        # Execute function
        result = func(*args, **kwargs)
        
        # Post-execution overhead
        post_overhead_start = time.time()
        # Simulate result processing
        time.sleep(0.001)  # 1ms overhead
        post_overhead_end = time.time()
        
        total_time = time.time() - start_time
        decorator_overhead = (overhead_end - overhead_start) + (post_overhead_end - post_overhead_start)
        
        print(f"Total execution time: {total_time:.4f}s")
        print(f"Decorator overhead: {decorator_overhead:.4f}s")
        print(f"Overhead percentage: {(decorator_overhead/total_time)*100:.2f}%")
        
        return result
    return wrapper

@performance_monitoring_decorator
@workflow(name="performance_test")
def performance_test_function():
    """Test function for performance monitoring."""
    time.sleep(0.01)  # Simulate work
    return "completed"

# Usage
result = performance_test_function()
```

## Best Practices

### 1. **Logical Decorator Ordering**

```python
# Recommended order (bottom to top):
@policies(["custom_policies"])      # Highest level: Custom business policies
@guardrails(enforcement_level="strict")  # Policy enforcement
@workflow(name="business_process")  # Business process tracking
@task(name="data_processing")       # Technical task tracking
def well_ordered_function():
    pass
```

### 2. **Avoid Over-Decoration**

```python
# Good: Focused decorator usage
@guardrails(enforcement_level="moderate")
@workflow(name="user_registration")
def register_user(user_data): pass

# Avoid: Excessive decoration
@policies(["policy1", "policy2", "policy3"])
@guardrails(enforcement_level="strict") 
@workflow(name="workflow")
@task(name="task")
@agent(name="agent")
def over_decorated_function(): pass
```

### 3. **Context Consistency**

```python
# Good: Consistent context across decorators
ORG_ID = "acme_corp"
PROJECT_ID = "user_management"

@guardrails(enforcement_level="strict")
@workflow(name="user_workflow", organization_id=ORG_ID, project_id=PROJECT_ID)
@task(name="validation_task", organization_id=ORG_ID, project_id=PROJECT_ID)
def consistent_context_function(): pass
```

### 4. **Performance Monitoring**

```python
# Monitor decorator performance impact
@workflow(name="performance_monitored", enable_metrics=True)
def monitored_function():
    # Function implementation
    pass
```

## Framework-Specific Patterns

### OpenAI Agents

```python
from agents import Agent

@workflow(name="openai_agent_workflow")
def create_openai_agent():
    """Create OpenAI agent with Rizk integration."""
    
    @tool(name="calculator")
    def calculate(expression: str) -> str:
        return str(eval(expression))
    
    @agent(name="math_assistant")
    def create_math_agent():
        return Agent(
            name="MathBot",
            instructions="You are a helpful math assistant",
            tools=[calculate]
        )
    
    return create_math_agent()
```

### LangChain

```python
from langchain.agents import AgentExecutor
from langchain.tools import Tool

@workflow(name="langchain_workflow")
def create_langchain_agent():
    """Create LangChain agent with Rizk integration."""
    
    @tool(name="search_tool")
    def search_function(query: str) -> str:
        return f"Search results for: {query}"
    
    tools = [Tool(
        name="Search",
        func=search_function,
        description="Search for information"
    )]
    
    return AgentExecutor.from_agent_and_tools(
        agent=agent,
        tools=tools,
        verbose=True
    )
```

### CrewAI

```python
from crewai import Crew, Agent, Task

@crew(name="research_crew")
@workflow(name="crewai_workflow")
def create_research_crew():
    """Create CrewAI crew with Rizk integration."""
    
    @agent(name="researcher")
    def create_researcher():
        return Agent(
            role="Research Analyst",
            goal="Conduct thorough research",
            backstory="Expert researcher with attention to detail"
        )
    
    @task(name="research_task")
    def create_research_task(agent):
        return Task(
            description="Research the given topic thoroughly",
            agent=agent
        )
    
    researcher = create_researcher()
    research_task = create_research_task(researcher)
    
    return Crew(
        agents=[researcher],
        tasks=[research_task],
        verbose=True
    )
```

## Related Documentation

- **[Workflow Decorator](workflow.md)** - High-level process orchestration
- **[Guardrails Decorator](guardrails.md)** - Policy enforcement and safety
- **[Agent Decorator](agent.md)** - Autonomous AI component management
- **[Configuration](../05-configuration/overview.md)** - SDK configuration options
- **[Best Practices](../10-best-practices/decorator-usage.md)** - Advanced decorator patterns

---

Effective decorator composition enables you to build sophisticated, enterprise-grade LLM applications with comprehensive observability, policy enforcement, and framework integration. Follow the patterns and best practices outlined in this guide to create robust, maintainable, and scalable AI workflows. 

