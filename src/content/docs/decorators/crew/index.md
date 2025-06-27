---
title: "@crew Decorator"
description: "@crew Decorator"
---

# @crew Decorator

The `@crew` decorator is specifically designed for instrumenting CrewAI multi-agent workflows. It provides specialized observability and management for crew-based operations, including agent coordination, task distribution, and process management.

## Overview

A **crew** represents a coordinated group of agents working together to accomplish complex objectives through structured processes. The `@crew` decorator provides comprehensive monitoring for crew creation, agent coordination, task execution, and result aggregation in CrewAI workflows.

## Basic Usage

```python
from crewai import Agent, Task, Crew, Process
from rizk.sdk import Rizk
from rizk.sdk.decorators import crew, agent, task

# Initialize Rizk
rizk = Rizk.init(app_name="CrewApp", enabled=True)

@crew(
    name="content_creation_crew",
    organization_id="marketing_team",
    project_id="content_strategy"
)
def create_content_crew() -> Crew:
    """Create a content creation crew with researcher and writer agents."""
    
    # Create agents
    researcher = Agent(
        role="Content Researcher",
        goal="Research comprehensive information about given topics",
        backstory="Expert researcher with 10+ years experience in content strategy",
        verbose=True,
        allow_delegation=False
    )
    
    writer = Agent(
        role="Content Writer", 
        goal="Create engaging content based on research findings",
        backstory="Skilled writer specializing in marketing and educational content",
        verbose=True,
        allow_delegation=False
    )
    
    # Create tasks
    research_task = Task(
        description="Research the latest trends in AI and machine learning",
        agent=researcher,
        expected_output="Comprehensive research report with key insights"
    )
    
    writing_task = Task(
        description="Write an engaging blog post based on the research findings",
        agent=writer,
        expected_output="Well-structured blog post ready for publication"
    )
    
    # Create and return crew
    crew = Crew(
        agents=[researcher, writer],
        tasks=[research_task, writing_task],
        process=Process.sequential,
        verbose=True
    )
    
    return crew

# Usage
content_crew = create_content_crew()
result = content_crew.kickoff()
print(f"Crew result: {result}")
```

## Parameters Reference

### Core Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `name` | `str` | No | Crew name (defaults to function name) |
| `version` | `int` | No | Crew version for tracking changes |
| `organization_id` | `str` | No | Organization identifier for hierarchical context |
| `project_id` | `str` | No | Project identifier for grouping crews |
| `crew_id` | `str` | No | Specific crew identifier for unique identification |

### Advanced Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `**kwargs` | `Any` | CrewAI-specific parameters passed to underlying adapters |

## Crew Process Types

### Sequential Process

```python
@crew(
    name="sequential_analysis_crew",
    organization_id="analytics_team",
    project_id="business_intelligence"
)
def create_sequential_crew() -> Crew:
    """Create a crew with sequential task processing."""
    
    data_analyst = Agent(
        role="Data Analyst",
        goal="Analyze business data and identify trends",
        backstory="Statistical expert with business intelligence experience"
    )
    
    report_writer = Agent(
        role="Report Writer",
        goal="Create executive reports from analysis results",
        backstory="Business communication specialist"
    )
    
    quality_reviewer = Agent(
        role="Quality Reviewer",
        goal="Review and validate report quality",
        backstory="Quality assurance expert with attention to detail"
    )
    
    # Sequential tasks - each depends on the previous
    analysis_task = Task(
        description="Analyze Q4 business metrics and identify key trends",
        agent=data_analyst,
        expected_output="Statistical analysis with trend identification"
    )
    
    report_task = Task(
        description="Create executive summary based on analysis results",
        agent=report_writer,
        expected_output="Executive report with recommendations"
    )
    
    review_task = Task(
        description="Review report for accuracy and completeness",
        agent=quality_reviewer,
        expected_output="Quality-assured final report"
    )
    
    return Crew(
        agents=[data_analyst, report_writer, quality_reviewer],
        tasks=[analysis_task, report_task, review_task],
        process=Process.sequential,
        verbose=True
    )
```

### Hierarchical Process

```python
@crew(
    name="hierarchical_project_crew",
    organization_id="project_management",
    project_id="software_development"
)
def create_hierarchical_crew() -> Crew:
    """Create a crew with hierarchical management structure."""
    
    project_manager = Agent(
        role="Project Manager",
        goal="Coordinate team activities and ensure project success",
        backstory="Experienced PM with software development background",
        allow_delegation=True  # Can delegate tasks
    )
    
    senior_developer = Agent(
        role="Senior Developer", 
        goal="Lead technical implementation and mentor junior developers",
        backstory="10+ years software development experience",
        allow_delegation=True
    )
    
    junior_developer = Agent(
        role="Junior Developer",
        goal="Implement features under senior guidance",
        backstory="2 years development experience, eager to learn"
    )
    
    qa_engineer = Agent(
        role="QA Engineer",
        goal="Ensure software quality through comprehensive testing",
        backstory="Quality assurance specialist with automation expertise"
    )
    
    # Hierarchical tasks with delegation
    planning_task = Task(
        description="Create project plan and assign development tasks",
        agent=project_manager,
        expected_output="Detailed project plan with task assignments"
    )
    
    development_task = Task(
        description="Implement core application features",
        agent=senior_developer,
        expected_output="Working application with core features"
    )
    
    testing_task = Task(
        description="Perform comprehensive testing of implemented features",
        agent=qa_engineer,
        expected_output="Test results and quality report"
    )
    
    return Crew(
        agents=[project_manager, senior_developer, junior_developer, qa_engineer],
        tasks=[planning_task, development_task, testing_task],
        process=Process.hierarchical,
        manager_llm="gpt-4",  # LLM for management decisions
        verbose=True
    )
```

## Advanced Crew Patterns

### Multi-Stage Crew Workflow

```python
@crew(
    name="multi_stage_research_crew",
    organization_id="research_division",
    project_id="market_analysis"
)
def create_multi_stage_crew() -> Crew:
    """Create a crew with multiple research stages."""
    
    # Primary research agents
    market_researcher = Agent(
        role="Market Researcher",
        goal="Conduct primary market research and data collection",
        backstory="Market research specialist with quantitative analysis skills"
    )
    
    competitive_analyst = Agent(
        role="Competitive Analyst", 
        goal="Analyze competitive landscape and positioning",
        backstory="Business strategy expert with competitive intelligence experience"
    )
    
    trend_analyst = Agent(
        role="Trend Analyst",
        goal="Identify emerging trends and future opportunities",
        backstory="Futurist with expertise in trend analysis and forecasting"
    )
    
    # Synthesis agent
    strategy_synthesizer = Agent(
        role="Strategy Synthesizer",
        goal="Synthesize research findings into actionable strategy",
        backstory="Strategic consultant with synthesis and recommendation expertise"
    )
    
    # Stage 1: Parallel research tasks
    market_research_task = Task(
        description="Conduct comprehensive market size and opportunity analysis",
        agent=market_researcher,
        expected_output="Market research report with size, growth, and opportunity data"
    )
    
    competitive_analysis_task = Task(
        description="Analyze top 5 competitors and their positioning strategies",
        agent=competitive_analyst,
        expected_output="Competitive analysis with SWOT and positioning insights"
    )
    
    trend_analysis_task = Task(
        description="Identify key trends affecting the market over next 3 years",
        agent=trend_analyst,
        expected_output="Trend analysis with impact assessment and timeline"
    )
    
    # Stage 2: Synthesis task (depends on all research tasks)
    strategy_synthesis_task = Task(
        description="Synthesize all research findings into strategic recommendations",
        agent=strategy_synthesizer,
        expected_output="Strategic recommendations with prioritized action items"
    )
    
    return Crew(
        agents=[market_researcher, competitive_analyst, trend_analyst, strategy_synthesizer],
        tasks=[market_research_task, competitive_analysis_task, trend_analysis_task, strategy_synthesis_task],
        process=Process.sequential,  # Will handle dependencies automatically
        verbose=True
    )
```

### Specialized Domain Crew

```python
@crew(
    name="legal_document_crew",
    organization_id="legal_department", 
    project_id="contract_analysis"
)
def create_legal_crew() -> Crew:
    """Create a specialized crew for legal document processing."""
    
    contract_analyzer = Agent(
        role="Contract Analyzer",
        goal="Analyze contract terms and identify key clauses",
        backstory="Legal expert specializing in contract law and risk assessment",
        tools=["contract_parser", "clause_extractor", "risk_assessor"]
    )
    
    compliance_checker = Agent(
        role="Compliance Checker",
        goal="Verify compliance with regulations and company policies",
        backstory="Compliance specialist with regulatory expertise",
        tools=["regulation_database", "policy_checker", "compliance_validator"]
    )
    
    risk_assessor = Agent(
        role="Risk Assessor", 
        goal="Assess legal and business risks in contracts",
        backstory="Risk management expert with legal background",
        tools=["risk_calculator", "precedent_analyzer", "impact_assessor"]
    )
    
    legal_reviewer = Agent(
        role="Legal Reviewer",
        goal="Provide final legal review and recommendations",
        backstory="Senior attorney with contract negotiation experience",
        tools=["legal_database", "precedent_search", "recommendation_generator"]
    )
    
    # Specialized legal tasks
    contract_analysis_task = Task(
        description="Analyze contract structure, terms, and key provisions",
        agent=contract_analyzer,
        expected_output="Detailed contract analysis with clause breakdown"
    )
    
    compliance_check_task = Task(
        description="Verify contract compliance with applicable regulations",
        agent=compliance_checker,
        expected_output="Compliance report with any violations or concerns"
    )
    
    risk_assessment_task = Task(
        description="Assess legal and business risks associated with contract",
        agent=risk_assessor,
        expected_output="Risk assessment with mitigation recommendations"
    )
    
    legal_review_task = Task(
        description="Provide comprehensive legal review and final recommendations",
        agent=legal_reviewer,
        expected_output="Legal opinion with approval/rejection recommendation"
    )
    
    return Crew(
        agents=[contract_analyzer, compliance_checker, risk_assessor, legal_reviewer],
        tasks=[contract_analysis_task, compliance_check_task, risk_assessment_task, legal_review_task],
        process=Process.sequential,
        verbose=True,
        memory=True,  # Enable crew memory for context retention
        embedder={"provider": "openai"}  # For semantic memory
    )
```

## Crew Monitoring and Analytics

```python
@crew(
    name="monitored_development_crew",
    organization_id="engineering_team",
    project_id="feature_development"
)
def create_monitored_crew() -> Crew:
    """Create a crew with comprehensive monitoring."""
    
    # Import monitoring tools
    from rizk.sdk.monitoring import CrewMonitor, TaskMetrics, AgentPerformance
    
    # Create agents with performance tracking
    architect = Agent(
        role="Software Architect",
        goal="Design system architecture and technical specifications",
        backstory="Senior architect with 15+ years experience in scalable systems"
    )
    
    developer = Agent(
        role="Full Stack Developer",
        goal="Implement features according to architectural specifications", 
        backstory="Experienced developer proficient in multiple technologies"
    )
    
    tester = Agent(
        role="Test Engineer",
        goal="Create comprehensive tests and ensure quality",
        backstory="QA engineer specializing in automated testing and quality assurance"
    )
    
    # Create tasks with metrics tracking
    design_task = Task(
        description="Design system architecture for new feature set",
        agent=architect,
        expected_output="Technical architecture document with implementation plan",
        metrics=TaskMetrics(
            complexity="high",
            estimated_duration="4 hours",
            priority="critical"
        )
    )
    
    implementation_task = Task(
        description="Implement features according to architectural design",
        agent=developer,
        expected_output="Working implementation with unit tests",
        metrics=TaskMetrics(
            complexity="medium",
            estimated_duration="8 hours", 
            priority="high"
        )
    )
    
    testing_task = Task(
        description="Create integration tests and perform quality validation",
        agent=tester,
        expected_output="Test suite with quality validation report",
        metrics=TaskMetrics(
            complexity="medium",
            estimated_duration="3 hours",
            priority="high"
        )
    )
    
    # Create crew with monitoring
    crew = Crew(
        agents=[architect, developer, tester],
        tasks=[design_task, implementation_task, testing_task],
        process=Process.sequential,
        verbose=True,
        memory=True,
        planning=True,  # Enable planning for better coordination
        planning_llm="gpt-4"
    )
    
    # Add monitoring capabilities
    monitor = CrewMonitor(crew)
    monitor.track_performance = True
    monitor.log_interactions = True
    monitor.measure_efficiency = True
    
    return crew

def run_monitored_crew_with_analytics(crew: Crew, input_data: dict) -> dict:
    """Run crew with comprehensive analytics."""
    
    start_time = time.time()
    
    # Execute crew
    result = crew.kickoff(inputs=input_data)
    
    end_time = time.time()
    
    # Collect analytics
    analytics = {
        "execution_time": end_time - start_time,
        "agents_used": len(crew.agents),
        "tasks_completed": len(crew.tasks),
        "success_rate": 1.0,  # Simplified for example
        "agent_performance": {
            agent.role: {
                "tasks_assigned": len([t for t in crew.tasks if t.agent == agent]),
                "efficiency_score": 0.85  # Mock efficiency score
            }
            for agent in crew.agents
        },
        "crew_result": result
    }
    
    return analytics
```

## Error Handling and Resilience

```python
@crew(
    name="resilient_data_crew",
    organization_id="data_engineering",
    project_id="data_pipeline"
)
def create_resilient_crew() -> Crew:
    """Create a crew with robust error handling."""
    
    from rizk.sdk.utils.error_handler import handle_errors
    
    @handle_errors(fail_closed=False, max_retries=3)
    def create_data_extractor() -> Agent:
        return Agent(
            role="Data Extractor",
            goal="Extract data from various sources with error recovery",
            backstory="Data engineer with expertise in resilient data extraction",
            tools=["database_connector", "api_client", "file_reader"]
        )
    
    @handle_errors(fail_closed=False, max_retries=2)
    def create_data_transformer() -> Agent:
        return Agent(
            role="Data Transformer",
            goal="Transform and clean extracted data",
            backstory="Data scientist specializing in data preprocessing",
            tools=["data_cleaner", "transformer", "validator"]
        )
    
    @handle_errors(fail_closed=True, max_retries=1)
    def create_data_loader() -> Agent:
        return Agent(
            role="Data Loader",
            goal="Load processed data into target systems",
            backstory="ETL specialist with database optimization expertise",
            tools=["database_loader", "data_validator", "performance_monitor"]
        )
    
    # Create agents with error handling
    extractor = create_data_extractor()
    transformer = create_data_transformer()
    loader = create_data_loader()
    
    # Create tasks with fallback strategies
    extraction_task = Task(
        description="Extract data from source systems with retry logic",
        agent=extractor,
        expected_output="Raw data with extraction metadata",
        fallback_strategy="partial_extraction"
    )
    
    transformation_task = Task(
        description="Transform and validate extracted data",
        agent=transformer,
        expected_output="Cleaned and transformed data",
        fallback_strategy="basic_cleaning"
    )
    
    loading_task = Task(
        description="Load data into target warehouse",
        agent=loader,
        expected_output="Data loading confirmation with metrics",
        fallback_strategy="staging_load"
    )
    
    return Crew(
        agents=[extractor, transformer, loader],
        tasks=[extraction_task, transformation_task, loading_task],
        process=Process.sequential,
        verbose=True,
        max_execution_time=3600,  # 1 hour timeout
        step_callback=lambda step: logger.info(f"Completed step: {step}")
    )
```

## Testing Crews

```python
import pytest
from unittest.mock import Mock, patch
from crewai import Agent, Task, Crew, Process

def test_crew_creation():
    """Test basic crew creation and configuration."""
    
    @crew(
        name="test_crew",
        organization_id="test_org",
        project_id="test_project"
    )
    def create_test_crew() -> Crew:
        agent = Agent(
            role="Test Agent",
            goal="Perform testing tasks",
            backstory="Testing specialist"
        )
        
        task = Task(
            description="Execute test scenario",
            agent=agent,
            expected_output="Test results"
        )
        
        return Crew(
            agents=[agent],
            tasks=[task],
            process=Process.sequential
        )
    
    crew = create_test_crew()
    
    assert len(crew.agents) == 1
    assert len(crew.tasks) == 1
    assert crew.process == Process.sequential

@patch('crewai.Crew.kickoff')
def test_crew_execution(mock_kickoff):
    """Test crew execution with mocked results."""
    
    mock_kickoff.return_value = "Test execution completed successfully"
    
    @crew(name="execution_test_crew")
    def create_execution_crew() -> Crew:
        agent = Agent(role="Executor", goal="Execute tasks", backstory="Test executor")
        task = Task(description="Test task", agent=agent, expected_output="Results")
        return Crew(agents=[agent], tasks=[task], process=Process.sequential)
    
    test_crew = create_execution_crew()
    result = test_crew.kickoff()
    
    assert result == "Test execution completed successfully"
    mock_kickoff.assert_called_once()

def test_crew_error_handling():
    """Test crew error handling scenarios."""
    
    @crew(name="error_test_crew")
    def create_error_crew() -> Crew:
        # This should handle errors gracefully
        try:
            agent = Agent(role="Error Tester", goal="Test errors", backstory="Error specialist")
            task = Task(description="Error task", agent=agent, expected_output="Error results")
            return Crew(agents=[agent], tasks=[task], process=Process.sequential)
        except Exception as e:
            # Return a minimal crew for error testing
            fallback_agent = Agent(role="Fallback", goal="Handle errors", backstory="Fallback agent")
            fallback_task = Task(description="Fallback task", agent=fallback_agent, expected_output="Fallback results")
            return Crew(agents=[fallback_agent], tasks=[fallback_task], process=Process.sequential)
    
    crew = create_error_crew()
    assert crew is not None
    assert len(crew.agents) >= 1
```

## Best Practices

### 1. **Clear Agent Roles**
```python
# Good: Specific, complementary roles
@crew(name="content_crew")
def create_content_crew():
    researcher = Agent(role="Content Researcher", goal="Research topics")
    writer = Agent(role="Content Writer", goal="Write articles")
    editor = Agent(role="Content Editor", goal="Edit and polish")
    return Crew(agents=[researcher, writer, editor], ...)

# Avoid: Overlapping or vague roles
@crew(name="general_crew")
def create_general_crew():
    agent1 = Agent(role="General Assistant", ...)  # Too vague
    agent2 = Agent(role="Helper", ...)  # Unclear purpose
```

### 2. **Task Dependencies**
```python
# Good: Clear task flow with dependencies
research_task = Task(description="Research topic X", ...)
writing_task = Task(description="Write based on research", ...)
editing_task = Task(description="Edit the written content", ...)
```

### 3. **Process Selection**
```python
# Sequential: When tasks depend on each other
process=Process.sequential

# Hierarchical: When you need management and delegation
process=Process.hierarchical, manager_llm="gpt-4"
```

### 4. **Memory and Context**
```python
# Good: Enable memory for complex crews
Crew(
    agents=agents,
    tasks=tasks,
    memory=True,
    embedder={"provider": "openai"}
)
```

## Related Documentation

- **[CrewAI Integration](../framework-integration/crewai.md)** - Detailed CrewAI patterns
- **[@agent Decorator](agent.md)** - For individual agent instrumentation
- **[@task Decorator](task.md)** - For task-level monitoring
- **[@workflow Decorator](workflow.md)** - For broader workflow orchestration

---

The `@crew` decorator provides specialized observability for CrewAI multi-agent workflows, enabling comprehensive monitoring of agent coordination, task execution, and crew performance while maintaining the full power of CrewAI's collaborative agent framework.

