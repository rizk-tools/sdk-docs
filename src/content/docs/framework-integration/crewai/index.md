---
title: "CrewAI Integration"
description: "CrewAI Integration"
---

# CrewAI Integration

The Rizk SDK provides seamless integration with CrewAI, offering comprehensive observability, tracing, and governance for multi-agent collaborative workflows. This guide covers everything from basic setup to advanced enterprise patterns.

## Overview

CrewAI enables building teams of AI agents that work together to accomplish complex tasks. Rizk SDK enhances CrewAI applications with:

- **Crew-Level Monitoring**: Complete observability for crew workflows and processes
- **Agent Performance Tracking**: Individual agent metrics and decision tracing
- **Task Execution Analytics**: Detailed insights into task completion and handoffs
- **Inter-Agent Communication**: Tracing of agent collaboration and information sharing
- **Policy Enforcement**: Real-time governance for crew behaviors and outputs
- **Resource Management**: Monitoring of tool usage and computational resources

## Prerequisites

```bash
pip install rizk crewai langchain-openai
```

## Quick Start

### Basic Crew with Monitoring

```python
import os
from rizk.sdk import Rizk
from rizk.sdk.decorators import workflow, agent, tool, crew

# Initialize Rizk SDK
rizk = Rizk.init(
    app_name="CrewAI-Demo",
    enabled=True
)

@tool(
    name="research_tool",
    organization_id="demo_org",
    project_id="crewai_project"
)
def research_tool(topic: str) -> str:
    """Research information about a given topic."""
    # Simulate research API call
    return f"Research findings on {topic}: Key insights and latest developments discovered."

@tool(
    name="writing_tool", 
    organization_id="demo_org",
    project_id="crewai_project"
)
def writing_tool(content: str) -> str:
    """Format and enhance written content."""
    # Simulate content enhancement
    word_count = len(content.split())
    return f"Enhanced content ({word_count} words): {content[:100]}... [Content optimized for readability and impact]"

@crew(
    name="research_crew",
    organization_id="demo_org",
    project_id="crewai_project"
)
def create_research_crew():
    """Create a research crew with specialized agents."""
    
    # For demo purposes, simulate CrewAI crew creation
    # In production, this would use actual CrewAI classes
    
    crew_config = {
        "name": "Research and Writing Crew",
        "agents": [
            {
                "role": "researcher",
                "goal": "Research comprehensive information on given topics",
                "backstory": "Expert researcher with access to multiple information sources",
                "tools": ["research_tool"]
            },
            {
                "role": "writer", 
                "goal": "Create compelling written content from research",
                "backstory": "Professional writer skilled in various content formats",
                "tools": ["writing_tool"]
            }
        ],
        "tasks": [
            {
                "description": "Research the given topic thoroughly",
                "agent": "researcher"
            },
            {
                "description": "Write engaging content based on research",
                "agent": "writer"
            }
        ]
    }
    
    return crew_config

@workflow(
    name="crew_execution",
    organization_id="demo_org",
    project_id="crewai_project" 
)
def execute_crew_workflow(topic: str) -> dict:
    """Execute a crew workflow for a given topic."""
    
    # Create the crew
    crew = create_research_crew()
    
    # Simulate crew execution
    print(f"ðŸš€ Starting crew workflow for topic: {topic}")
    
    # Phase 1: Research
    print("ðŸ“š Researcher agent starting...")
    research_result = research_tool(topic)
    print(f"âœ… Research completed: {research_result[:50]}...")
    
    # Phase 2: Writing
    print("âœï¸ Writer agent starting...")
    writing_result = writing_tool(research_result)
    print(f"âœ… Writing completed: {writing_result[:50]}...")
    
    return {
        "topic": topic,
        "research": research_result,
        "final_content": writing_result,
        "crew_name": crew["name"],
        "agents_used": [agent["role"] for agent in crew["agents"]],
        "status": "completed"
    }

# Test the crew
if __name__ == "__main__":
    test_topics = [
        "Artificial Intelligence in Healthcare",
        "Sustainable Energy Solutions",
        "Future of Remote Work"
    ]
    
    print("ðŸ¤– Testing CrewAI with Rizk SDK...")
    print("=" * 50)
    
    for i, topic in enumerate(test_topics, 1):
        print(f"\n{i}. Topic: {topic}")
        result = execute_crew_workflow(topic)
        print(f"   Status: {result['status']}")
        print(f"   Agents: {', '.join(result['agents_used'])}")
```

## Advanced CrewAI Patterns

### Multi-Stage Crew Workflow

```python
from typing import List, Dict, Any
from dataclasses import dataclass
from rizk.sdk.decorators import workflow, tool, crew

@dataclass
class TaskResult:
    """Represents the result of a task execution."""
    task_id: str
    agent_role: str
    output: str
    duration: float
    status: str
    dependencies_met: bool = True

@tool(name="market_research", organization_id="enterprise", project_id="crewai_advanced")
def market_research_tool(industry: str, timeframe: str) -> str:
    """Conduct market research for a specific industry."""
    return f"Market research for {industry} ({timeframe}): Growth trends, key players, and opportunities identified."

@tool(name="competitive_analysis", organization_id="enterprise", project_id="crewai_advanced")
def competitive_analysis_tool(competitors: str) -> str:
    """Analyze competitors in the market."""
    return f"Competitive analysis of {competitors}: Strengths, weaknesses, and market positioning evaluated."

@tool(name="strategy_development", organization_id="enterprise", project_id="crewai_advanced")
def strategy_development_tool(research_data: str, analysis_data: str) -> str:
    """Develop strategic recommendations based on research and analysis."""
    combined_length = len(research_data) + len(analysis_data)
    return f"Strategic recommendations based on {combined_length} chars of data: Actionable insights and implementation roadmap."

@tool(name="report_generation", organization_id="enterprise", project_id="crewai_advanced")
def report_generation_tool(strategy: str) -> str:
    """Generate a comprehensive business report."""
    return f"Executive report generated: {strategy[:100]}... [Full report with charts, recommendations, and action items]"

@crew(name="business_intelligence_crew", organization_id="enterprise", project_id="crewai_advanced")
def create_business_intelligence_crew():
    """Create a business intelligence crew with sequential task dependencies."""
    
    return {
        "name": "Business Intelligence Crew",
        "process": "sequential",
        "agents": [
            {
                "role": "market_researcher",
                "goal": "Conduct thorough market research and analysis",
                "backstory": "Senior market analyst with 10+ years experience",
                "tools": ["market_research"]
            },
            {
                "role": "competitive_analyst", 
                "goal": "Analyze competitive landscape and positioning",
                "backstory": "Strategic analyst specializing in competitive intelligence",
                "tools": ["competitive_analysis"]
            },
            {
                "role": "strategy_consultant",
                "goal": "Develop actionable business strategies",
                "backstory": "Management consultant with expertise in strategy development",
                "tools": ["strategy_development"]
            },
            {
                "role": "report_writer",
                "goal": "Create comprehensive business reports",
                "backstory": "Business writer skilled in executive communication",
                "tools": ["report_generation"]
            }
        ],
        "tasks": [
            {
                "id": "market_research_task",
                "description": "Research market trends and opportunities",
                "agent": "market_researcher",
                "dependencies": []
            },
            {
                "id": "competitive_analysis_task", 
                "description": "Analyze competitive landscape",
                "agent": "competitive_analyst",
                "dependencies": []
            },
            {
                "id": "strategy_development_task",
                "description": "Develop strategic recommendations",
                "agent": "strategy_consultant", 
                "dependencies": ["market_research_task", "competitive_analysis_task"]
            },
            {
                "id": "report_generation_task",
                "description": "Generate final business report",
                "agent": "report_writer",
                "dependencies": ["strategy_development_task"]
            }
        ]
    }

@workflow(name="business_intelligence_workflow", organization_id="enterprise", project_id="crewai_advanced")
def execute_business_intelligence_workflow(
    industry: str,
    competitors: List[str],
    timeframe: str = "Q1 2024"
) -> Dict[str, Any]:
    """
    Execute a multi-stage business intelligence workflow.
    
    Args:
        industry: Target industry for analysis
        competitors: List of competitors to analyze
        timeframe: Analysis timeframe
        
    Returns:
        Complete workflow results with task outputs
    """
    
    crew = create_business_intelligence_crew()
    task_results = {}
    
    print(f"ðŸ¢ Starting Business Intelligence Workflow")
    print(f"   Industry: {industry}")
    print(f"   Competitors: {', '.join(competitors)}")
    print(f"   Timeframe: {timeframe}")
    
    # Execute tasks in dependency order
    import time
    
    # Task 1: Market Research
    print("\nðŸ“Š Executing Market Research...")
    start_time = time.time()
    market_data = market_research_tool(industry, timeframe)
    duration = time.time() - start_time
    task_results["market_research_task"] = TaskResult(
        task_id="market_research_task",
        agent_role="market_researcher",
        output=market_data,
        duration=duration,
        status="completed"
    )
    print(f"   âœ… Completed in {duration:.2f}s")
    
    # Task 2: Competitive Analysis (parallel with market research)
    print("\nðŸŽ¯ Executing Competitive Analysis...")
    start_time = time.time()
    competitive_data = competitive_analysis_tool(", ".join(competitors))
    duration = time.time() - start_time
    task_results["competitive_analysis_task"] = TaskResult(
        task_id="competitive_analysis_task",
        agent_role="competitive_analyst",
        output=competitive_data,
        duration=duration,
        status="completed"
    )
    print(f"   âœ… Completed in {duration:.2f}s")
    
    # Task 3: Strategy Development (depends on tasks 1 & 2)
    print("\nðŸŽ¯ Executing Strategy Development...")
    start_time = time.time()
    strategy_data = strategy_development_tool(
        task_results["market_research_task"].output,
        task_results["competitive_analysis_task"].output
    )
    duration = time.time() - start_time
    task_results["strategy_development_task"] = TaskResult(
        task_id="strategy_development_task",
        agent_role="strategy_consultant",
        output=strategy_data,
        duration=duration,
        status="completed"
    )
    print(f"   âœ… Completed in {duration:.2f}s")
    
    # Task 4: Report Generation (depends on task 3)
    print("\nðŸ“ Executing Report Generation...")
    start_time = time.time()
    final_report = report_generation_tool(task_results["strategy_development_task"].output)
    duration = time.time() - start_time
    task_results["report_generation_task"] = TaskResult(
        task_id="report_generation_task",
        agent_role="report_writer",
        output=final_report,
        duration=duration,
        status="completed"
    )
    print(f"   âœ… Completed in {duration:.2f}s")
    
    total_duration = sum(task.duration for task in task_results.values())
    
    return {
        "workflow_id": f"bi_workflow_{industry.lower().replace(' ', '_')}",
        "industry": industry,
        "competitors": competitors,
        "timeframe": timeframe,
        "task_results": task_results,
        "final_output": final_report,
        "crew_name": crew["name"],
        "total_duration": total_duration,
        "tasks_completed": len(task_results),
        "status": "completed"
    }

# Example usage
def demo_business_intelligence_crew():
    """Demonstrate business intelligence crew workflow."""
    
    result = execute_business_intelligence_workflow(
        industry="Artificial Intelligence",
        competitors=["OpenAI", "Anthropic", "Google AI"],
        timeframe="Q1 2024"
    )
    
    print(f"\nðŸ“ˆ Business Intelligence Results:")
    print(f"   Workflow ID: {result['workflow_id']}")
    print(f"   Tasks Completed: {result['tasks_completed']}")
    print(f"   Total Duration: {result['total_duration']:.2f}s")
    print(f"   Final Report: {result['final_output'][:100]}...")

# demo_business_intelligence_crew()
```

### Parallel Crew Execution

```python
import asyncio
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor

@tool(name="content_research", organization_id="enterprise", project_id="crewai_parallel")
def content_research_tool(topic: str, source_type: str) -> str:
    """Research content from specific source types."""
    return f"Content research on {topic} from {source_type}: Relevant information and insights gathered."

@tool(name="social_media_analysis", organization_id="enterprise", project_id="crewai_parallel")
def social_media_analysis_tool(brand: str, platform: str) -> str:
    """Analyze social media presence and engagement."""
    return f"Social media analysis for {brand} on {platform}: Engagement metrics and audience insights."

@tool(name="seo_optimization", organization_id="enterprise", project_id="crewai_parallel")
def seo_optimization_tool(content: str, keywords: str) -> str:
    """Optimize content for search engines."""
    return f"SEO-optimized content with keywords '{keywords}': Enhanced for search visibility and ranking."

@crew(name="parallel_content_crew", organization_id="enterprise", project_id="crewai_parallel")
def create_parallel_content_crew():
    """Create a crew that can execute tasks in parallel."""
    
    return {
        "name": "Parallel Content Marketing Crew",
        "process": "parallel",
        "agents": [
            {
                "role": "content_researcher",
                "goal": "Research content across multiple sources",
                "backstory": "Content research specialist with access to diverse information sources",
                "tools": ["content_research"]
            },
            {
                "role": "social_media_analyst",
                "goal": "Analyze social media trends and engagement",
                "backstory": "Social media expert with deep platform knowledge",
                "tools": ["social_media_analysis"]
            },
            {
                "role": "seo_specialist",
                "goal": "Optimize content for search engines",
                "backstory": "SEO expert with proven track record in content optimization",
                "tools": ["seo_optimization"]
            }
        ]
    }

@workflow(name="parallel_content_workflow", organization_id="enterprise", project_id="crewai_parallel")
async def execute_parallel_content_workflow(
    brand: str,
    content_topics: List[str],
    target_keywords: List[str],
    platforms: List[str] = None
) -> Dict[str, Any]:
    """
    Execute parallel content marketing workflow.
    
    Args:
        brand: Brand name for analysis
        content_topics: List of topics to research
        target_keywords: SEO keywords to target
        platforms: Social media platforms to analyze
        
    Returns:
        Aggregated results from parallel execution
    """
    
    if platforms is None:
        platforms = ["Twitter", "LinkedIn", "Instagram"]
    
    crew = create_parallel_content_crew()
    
    print(f"ðŸš€ Starting Parallel Content Workflow")
    print(f"   Brand: {brand}")
    print(f"   Topics: {', '.join(content_topics)}")
    print(f"   Keywords: {', '.join(target_keywords)}")
    print(f"   Platforms: {', '.join(platforms)}")
    
    # Define parallel tasks
    async def research_task(topic: str) -> Dict[str, Any]:
        """Research content for a specific topic."""
        print(f"ðŸ“š Researching: {topic}")
        result = content_research_tool(topic, "academic_papers")
        return {"topic": topic, "research": result, "agent": "content_researcher"}
    
    async def social_analysis_task(platform: str) -> Dict[str, Any]:
        """Analyze social media for a specific platform."""
        print(f"ðŸ“± Analyzing {platform}")
        result = social_media_analysis_tool(brand, platform)
        return {"platform": platform, "analysis": result, "agent": "social_media_analyst"}
    
    async def seo_task(keyword: str, content: str) -> Dict[str, Any]:
        """Optimize content for specific keywords."""
        print(f"ðŸ” Optimizing for: {keyword}")
        result = seo_optimization_tool(content, keyword)
        return {"keyword": keyword, "optimized_content": result, "agent": "seo_specialist"}
    
    # Execute tasks in parallel
    start_time = asyncio.get_event_loop().time()
    
    # Phase 1: Parallel research and social analysis
    research_tasks = [research_task(topic) for topic in content_topics]
    social_tasks = [social_analysis_task(platform) for platform in platforms]
    
    phase1_results = await asyncio.gather(*research_tasks, *social_tasks)
    
    # Phase 2: SEO optimization (depends on research results)
    research_results = [r for r in phase1_results if "research" in r]
    seo_tasks = [
        seo_task(keyword, research_result["research"])
        for keyword in target_keywords
        for research_result in research_results[:1]  # Use first research result
    ]
    
    seo_results = await asyncio.gather(*seo_tasks)
    
    end_time = asyncio.get_event_loop().time()
    total_duration = end_time - start_time
    
    # Aggregate results
    all_results = phase1_results + seo_results
    
    return {
        "workflow_id": f"parallel_content_{brand.lower().replace(' ', '_')}",
        "brand": brand,
        "crew_name": crew["name"],
        "execution_mode": "parallel",
        "total_duration": total_duration,
        "tasks_executed": len(all_results),
        "research_results": [r for r in all_results if "research" in r],
        "social_analysis": [r for r in all_results if "analysis" in r],
        "seo_optimizations": [r for r in all_results if "optimized_content" in r],
        "status": "completed"
    }

# Example usage
async def demo_parallel_crew():
    """Demonstrate parallel crew execution."""
    
    result = await execute_parallel_content_workflow(
        brand="TechStartup Inc",
        content_topics=["AI Innovation", "Cloud Computing"],
        target_keywords=["artificial intelligence", "cloud solutions"],
        platforms=["LinkedIn", "Twitter"]
    )
    
    print(f"\nâš¡ Parallel Execution Results:")
    print(f"   Workflow ID: {result['workflow_id']}")
    print(f"   Total Duration: {result['total_duration']:.2f}s")
    print(f"   Tasks Executed: {result['tasks_executed']}")
    print(f"   Research Results: {len(result['research_results'])}")
    print(f"   Social Analysis: {len(result['social_analysis'])}")
    print(f"   SEO Optimizations: {len(result['seo_optimizations'])}")

# asyncio.run(demo_parallel_crew())
```

## Production Patterns

### Enterprise CrewAI Management

```python
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from datetime import datetime, timedelta
from rizk.sdk.decorators import workflow, crew

@dataclass
class CrewAIConfig:
    """Enterprise CrewAI configuration."""
    
    # Crew Configuration
    max_agents_per_crew: int = 10
    max_tasks_per_workflow: int = 50
    default_process_type: str = "sequential"
    enable_task_delegation: bool = True
    
    # Performance Configuration
    task_timeout_seconds: int = 300
    crew_timeout_seconds: int = 3600
    max_retries: int = 3
    enable_parallel_execution: bool = True
    
    # Resource Management
    max_concurrent_crews: int = 5
    memory_limit_mb: int = 1024
    cpu_limit_percent: int = 80
    
    # Safety Configuration
    enable_output_validation: bool = True
    content_filtering: bool = True
    rate_limiting: bool = True
    
    # Monitoring Configuration
    enable_detailed_logging: bool = True
    metrics_collection: bool = True
    trace_agent_decisions: bool = True

class EnterpriseCrewAIManager:
    """Manage CrewAI workflows with enterprise features."""
    
    def __init__(self, config: CrewAIConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.active_crews = {}
        self.crew_metrics = {}
    
    @workflow(name="enterprise_crew_execution", organization_id="production", project_id="crewai")
    def execute_crew_safely(
        self,
        crew_definition: Dict[str, Any],
        workflow_input: Dict[str, Any],
        execution_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Execute CrewAI workflow with enterprise safety measures."""
        
        workflow_id = f"crew_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        try:
            # Pre-execution validation
            validation_result = self._validate_crew_definition(crew_definition)
            if not validation_result["valid"]:
                return {
                    "success": False,
                    "workflow_id": workflow_id,
                    "error": f"Validation failed: {validation_result['errors']}",
                    "result": None
                }
            
            # Resource check
            if not self._check_resource_availability():
                return {
                    "success": False,
                    "workflow_id": workflow_id,
                    "error": "Insufficient resources available",
                    "result": None
                }
            
            # Execute crew workflow
            self.logger.info(f"Starting crew execution: {workflow_id}")
            result = self._execute_crew_workflow(
                crew_definition,
                workflow_input,
                execution_context,
                workflow_id
            )
            
            # Post-execution validation
            if self.config.enable_output_validation:
                result = self._validate_and_filter_output(result)
            
            self.logger.info(f"Crew execution completed: {workflow_id}")
            
            return {
                "success": True,
                "workflow_id": workflow_id,
                "error": None,
                "result": result,
                "metadata": {
                    "execution_time": result.get("total_duration", 0),
                    "tasks_completed": result.get("tasks_completed", 0),
                    "agents_used": result.get("agents_used", [])
                }
            }
            
        except Exception as e:
            self.logger.error(f"Crew execution failed: {workflow_id} - {str(e)}")
            return {
                "success": False,
                "workflow_id": workflow_id,
                "error": str(e),
                "result": None
            }
        finally:
            # Cleanup
            if workflow_id in self.active_crews:
                del self.active_crews[workflow_id]
    
    def _validate_crew_definition(self, crew_definition: Dict[str, Any]) -> Dict[str, Any]:
        """Validate crew definition for safety and compliance."""
        
        errors = []
        
        # Check crew structure
        if "agents" not in crew_definition:
            errors.append("Crew must have agents defined")
        elif len(crew_definition["agents"]) > self.config.max_agents_per_crew:
            errors.append(f"Too many agents: {len(crew_definition['agents'])} > {self.config.max_agents_per_crew}")
        
        if "tasks" in crew_definition and len(crew_definition["tasks"]) > self.config.max_tasks_per_workflow:
            errors.append(f"Too many tasks: {len(crew_definition['tasks'])} > {self.config.max_tasks_per_workflow}")
        
        # Validate agents
        for agent in crew_definition.get("agents", []):
            if "role" not in agent:
                errors.append("All agents must have a role defined")
            if "goal" not in agent:
                errors.append("All agents must have a goal defined")
        
        # Validate tasks
        for task in crew_definition.get("tasks", []):
            if "description" not in task:
                errors.append("All tasks must have a description")
            if "agent" not in task:
                errors.append("All tasks must be assigned to an agent")
        
        return {
            "valid": len(errors) == 0,
            "errors": errors
        }
    
    def _check_resource_availability(self) -> bool:
        """Check if resources are available for crew execution."""
        
        # Check concurrent crew limit
        if len(self.active_crews) >= self.config.max_concurrent_crews:
            self.logger.warning("Maximum concurrent crews reached")
            return False
        
        # In a real implementation, check actual system resources
        # For demo, always return True
        return True
    
    def _execute_crew_workflow(
        self,
        crew_definition: Dict[str, Any],
        workflow_input: Dict[str, Any],
        execution_context: Optional[Dict[str, Any]],
        workflow_id: str
    ) -> Dict[str, Any]:
        """Execute the actual crew workflow."""
        
        import time
        start_time = time.time()
        
        # Register active crew
        self.active_crews[workflow_id] = {
            "start_time": start_time,
            "crew_definition": crew_definition,
            "status": "running"
        }
        
        # Simulate crew execution
        crew_name = crew_definition.get("name", "Unknown Crew")
        agents = crew_definition.get("agents", [])
        tasks = crew_definition.get("tasks", [])
        
        print(f"ðŸš€ Executing {crew_name}")
        print(f"   Agents: {len(agents)}")
        print(f"   Tasks: {len(tasks)}")
        
        task_results = []
        
        # Execute tasks
        for i, task in enumerate(tasks):
            task_start = time.time()
            
            # Simulate task execution
            agent_role = task.get("agent", "unknown")
            description = task.get("description", "")
            
            print(f"   ðŸ“‹ Task {i+1}: {description[:50]}...")
            print(f"      Agent: {agent_role}")
            
            # Simulate processing time
            time.sleep(0.1)  # Small delay for demo
            
            task_duration = time.time() - task_start
            
            task_result = {
                "task_id": task.get("id", f"task_{i+1}"),
                "description": description,
                "agent": agent_role,
                "output": f"Completed: {description}",
                "duration": task_duration,
                "status": "completed"
            }
            
            task_results.append(task_result)
            print(f"      âœ… Completed in {task_duration:.2f}s")
        
        total_duration = time.time() - start_time
        
        return {
            "workflow_id": workflow_id,
            "crew_name": crew_name,
            "task_results": task_results,
            "tasks_completed": len(task_results),
            "agents_used": [agent["role"] for agent in agents],
            "total_duration": total_duration,
            "status": "completed"
        }
    
    def _validate_and_filter_output(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and filter crew output for safety."""
        
        if not self.config.content_filtering:
            return result
        
        # Filter sensitive information from task outputs
        for task_result in result.get("task_results", []):
            if "output" in task_result:
                task_result["output"] = self._filter_sensitive_content(task_result["output"])
        
        return result
    
    def _filter_sensitive_content(self, content: str) -> str:
        """Filter sensitive content from outputs."""
        import re
        
        # Remove email addresses
        content = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL]', content)
        
        # Remove phone numbers
        content = re.sub(r'\b\d{3}-\d{3}-\d{4}\b', '[PHONE]', content)
        
        # Remove potential API keys
        content = re.sub(r'\b[A-Za-z0-9]{32,}\b', '[API_KEY]', content)
        
        return content

# Example usage
def demo_enterprise_crewai():
    """Demonstrate enterprise CrewAI management."""
    
    config = CrewAIConfig(
        max_agents_per_crew=5,
        enable_output_validation=True,
        content_filtering=True
    )
    
    manager = EnterpriseCrewAIManager(config)
    
    # Define a test crew
    crew_definition = {
        "name": "Content Creation Crew",
        "agents": [
            {
                "role": "researcher",
                "goal": "Research topic thoroughly",
                "backstory": "Expert researcher with access to multiple sources"
            },
            {
                "role": "writer",
                "goal": "Create engaging content",
                "backstory": "Professional content writer"
            }
        ],
        "tasks": [
            {
                "id": "research_task",
                "description": "Research the given topic",
                "agent": "researcher"
            },
            {
                "id": "writing_task", 
                "description": "Write content based on research",
                "agent": "writer"
            }
        ]
    }
    
    workflow_input = {
        "topic": "Future of Artificial Intelligence",
        "target_audience": "business executives",
        "content_length": "1000 words"
    }
    
    print("ðŸ¢ Enterprise CrewAI Management Demo")
    print("=" * 40)
    
    result = manager.execute_crew_safely(crew_definition, workflow_input)
    
    if result["success"]:
        print(f"âœ… Workflow completed successfully")
        print(f"   Workflow ID: {result['workflow_id']}")
        print(f"   Tasks completed: {result['metadata']['tasks_completed']}")
        print(f"   Execution time: {result['metadata']['execution_time']:.2f}s")
        print(f"   Agents used: {', '.join(result['metadata']['agents_used'])}")
    else:
        print(f"âŒ Workflow failed: {result['error']}")

# demo_enterprise_crewai()
```

## Configuration and Testing

### Environment Configuration

```python
import os
from typing import Optional, List
from dataclasses import dataclass

@dataclass
class CrewAIProductionConfig:
    """Production configuration for CrewAI integration."""
    
    # API Configuration
    openai_api_key: str = ""
    openai_model: str = "gpt-4"
    openai_temperature: float = 0.1
    
    # CrewAI Configuration
    crewai_verbose: bool = False
    crewai_memory: bool = True
    crewai_cache: bool = True
    
    # Rizk SDK Configuration
    rizk_api_key: str = ""
    rizk_app_name: str = "CrewAI-Production"
    rizk_enabled: bool = True
    
    # Performance Configuration
    max_execution_time: int = 3600
    max_agents: int = 10
    max_tasks: int = 50
    
    @classmethod
    def from_environment(cls) -> 'CrewAIProductionConfig':
        """Load configuration from environment variables."""
        return cls(
            openai_api_key=os.getenv("OPENAI_API_KEY", ""),
            openai_model=os.getenv("OPENAI_MODEL", "gpt-4"),
            openai_temperature=float(os.getenv("OPENAI_TEMPERATURE", "0.1")),
            crewai_verbose=os.getenv("CREWAI_VERBOSE", "false").lower() == "true",
            crewai_memory=os.getenv("CREWAI_MEMORY", "true").lower() == "true",
            crewai_cache=os.getenv("CREWAI_CACHE", "true").lower() == "true",
            rizk_api_key=os.getenv("RIZK_API_KEY", ""),
            rizk_app_name=os.getenv("RIZK_APP_NAME", "CrewAI-Production"),
            rizk_enabled=os.getenv("RIZK_ENABLED", "true").lower() == "true",
            max_execution_time=int(os.getenv("MAX_EXECUTION_TIME", "3600")),
            max_agents=int(os.getenv("MAX_AGENTS", "10")),
            max_tasks=int(os.getenv("MAX_TASKS", "50"))
        )
    
    def validate(self) -> List[str]:
        """Validate configuration."""
        errors = []
        
        if not self.openai_api_key:
            errors.append("OpenAI API key is required")
        
        if not self.rizk_api_key:
            errors.append("Rizk API key is required")
        
        if self.max_agents <= 0:
            errors.append("Max agents must be positive")
        
        return errors

# Test framework
def test_crewai_examples():
    """Test all CrewAI integration examples."""
    
    print("ðŸ§ª Testing CrewAI Integration Examples")
    print("=" * 50)
    
    tests_passed = 0
    tests_failed = 0
    
    # Test 1: Basic crew workflow
    try:
        result = execute_crew_workflow("Test Topic")
        assert result["status"] == "completed"
        assert len(result["agents_used"]) == 2
        print("âœ… Basic crew workflow: PASSED")
        tests_passed += 1
    except Exception as e:
        print(f"âŒ Basic crew workflow: FAILED - {str(e)}")
        tests_failed += 1
    
    # Test 2: Business intelligence workflow
    try:
        result = execute_business_intelligence_workflow(
            "Technology",
            ["Company A", "Company B"],
            "Q1 2024"
        )
        assert result["status"] == "completed"
        assert result["tasks_completed"] == 4
        print("âœ… Business intelligence workflow: PASSED")
        tests_passed += 1
    except Exception as e:
        print(f"âŒ Business intelligence workflow: FAILED - {str(e)}")
        tests_failed += 1
    
    # Test 3: Configuration validation
    try:
        config = CrewAIProductionConfig.from_environment()
        errors = config.validate()
        print(f"âœ… Configuration validation: PASSED ({len(errors)} errors)")
        tests_passed += 1
    except Exception as e:
        print(f"âŒ Configuration validation: FAILED - {str(e)}")
        tests_failed += 1
    
    print(f"\nðŸ“Š Test Results:")
    print(f"âœ… Passed: {tests_passed}")
    print(f"âŒ Failed: {tests_failed}")
    print(f"ðŸ“ˆ Success Rate: {(tests_passed / (tests_passed + tests_failed)) * 100:.1f}%")

if __name__ == "__main__":
    test_crewai_examples()
```

## Troubleshooting

### Common Issues and Solutions

| Issue | Symptoms | Solution |
|-------|----------|----------|
| **Agent Communication Errors** | Agents failing to pass information | Check task dependencies and data flow |
| **Task Timeout** | Tasks hanging or taking too long | Adjust timeout settings and optimize agent prompts |
| **Memory Issues** | High memory usage with large crews | Implement memory limits and cleanup |
| **Tool Access Errors** | Agents can't access required tools | Verify tool registration and permissions |
| **Parallel Execution Failures** | Concurrent tasks failing | Check resource limits and implement proper error handling |

## Next Steps

1. **Explore Advanced Patterns**: Check out [Multi-Agent Workflows](../10-examples/multi-agent-workflow.md)
2. **Production Deployment**: See [Production Setup](../advanced-config/production-setup.md)
3. **Custom Tools**: Learn about [Tool Development](../decorators/tool.md)
4. **Performance Optimization**: Review [Performance Best Practices](../advanced-config/performance.md)
5. **Security**: Review [Security Best Practices](../advanced-config/security.md)

---

**Enterprise Support**: For enterprise-specific CrewAI integrations, custom crew development, or advanced multi-agent configurations, contact our enterprise team at [enterprise@rizk.tools](mailto:enterprise@rizk.tools). 

