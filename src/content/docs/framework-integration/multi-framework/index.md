---
title: "Multi-Framework Applications"
description: "Multi-Framework Applications"
---

# Multi-Framework Applications

This guide shows you how to build applications that use multiple LLM frameworks together, manage cross-framework workflows, and maintain unified observability across your entire AI system. Learn patterns for orchestrating LangChain agents with CrewAI teams, integrating LlamaIndex RAG with OpenAI Agents, and more.

## Overview

Multi-framework applications are becoming increasingly common as teams leverage the best features from different AI frameworks:

- **LangChain** for its extensive tool ecosystem and chains
- **CrewAI** for multi-agent collaboration and workflows  
- **LlamaIndex** for advanced RAG and document processing
- **OpenAI Agents** for native OpenAI integration
- **LangGraph** for complex state-based workflows

Rizk SDK provides unified observability and governance across all these frameworks simultaneously.

## Architecture Patterns

### 1. Orchestrated Multi-Framework Pattern

```
Main Controller â†’ Framework A â†’ Framework B â†’ Framework C
      â†“              â†“           â†“           â†“
   Rizk SDK â†â”€â”€â”€â”€â”€â”€â”€â”€â”´â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”´â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
```

### 2. Parallel Multi-Framework Pattern

```
Main Controller
    â”œâ”€ Framework A â”€â”€â†’ Rizk SDK
    â”œâ”€ Framework B â”€â”€â†’ Rizk SDK  
    â””â”€ Framework C â”€â”€â†’ Rizk SDK
```

### 3. Layered Multi-Framework Pattern

```
Presentation Layer (FastAPI/Streamlit)
       â†“
Orchestration Layer (LangGraph/Custom)
       â†“
Processing Layer (LangChain + CrewAI + LlamaIndex)
       â†“
Foundation Layer (OpenAI + Anthropic)
       â†“
Rizk SDK (Unified Observability)
```

## Complete Multi-Framework Examples

### Example 1: Research and Analysis Pipeline

This example combines LlamaIndex for document processing, LangChain for web research, and CrewAI for collaborative analysis.

```python
import asyncio
from typing import List, Dict, Any
from rizk.sdk import Rizk
from rizk.sdk.decorators import workflow, task, agent, tool

# Framework imports
from llama_index.core import VectorStoreIndex, Document
from llama_index.core.query_engine import BaseQueryEngine
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.tools import DuckDuckGoSearchRun
from crewai import Agent, Task, Crew, Process

# Initialize Rizk for multi-framework support
rizk = Rizk.init(
    app_name="Multi-Framework-Research-Pipeline",
    enabled=True
)

class ResearchPipeline:
    """Multi-framework research and analysis pipeline."""
    
    def __init__(self):
        self.llm = ChatOpenAI(temperature=0)
        self.search_tool = DuckDuckGoSearchRun()
        
    @workflow(name="research_pipeline", organization_id="research", project_id="multi_framework")
    async def run_research_pipeline(self, topic: str, documents: List[str] = None) -> Dict[str, Any]:
        """Complete research pipeline using multiple frameworks."""
        
        results = {}
        
        # Step 1: Document Analysis with LlamaIndex
        if documents:
            results['document_analysis'] = await self.analyze_documents(documents, topic)
        
        # Step 2: Web Research with LangChain
        results['web_research'] = await self.web_research(topic)
        
        # Step 3: Collaborative Analysis with CrewAI
        results['collaborative_analysis'] = await self.collaborative_analysis(
            topic, 
            results.get('document_analysis', {}),
            results.get('web_research', {})
        )
        
        # Step 4: Final Synthesis
        results['synthesis'] = await self.synthesize_findings(results)
        
        return results
    
    @task(name="document_analysis", organization_id="research", project_id="multi_framework")
    async def analyze_documents(self, documents: List[str], topic: str) -> Dict[str, Any]:
        """Analyze documents using LlamaIndex."""
        
        try:
            # Create documents
            docs = [Document(text=doc) for doc in documents]
            
            # Build index
            index = VectorStoreIndex.from_documents(docs)
            query_engine = index.as_query_engine()
            
            # Query for topic-specific information
            queries = [
                f"What are the key points about {topic}?",
                f"What are the main challenges related to {topic}?",
                f"What solutions are proposed for {topic}?"
            ]
            
            analysis_results = {}
            for query in queries:
                response = await self._async_query(query_engine, query)
                analysis_results[query] = str(response)
            
            return {
                'status': 'success',
                'document_count': len(documents),
                'analysis': analysis_results
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e),
                'document_count': len(documents) if documents else 0
            }
    
    @task(name="web_research", organization_id="research", project_id="multi_framework")
    async def web_research(self, topic: str) -> Dict[str, Any]:
        """Perform web research using LangChain."""
        
        try:
            # Create research agent
            @agent(name="research_agent", organization_id="research", project_id="multi_framework")
            def create_research_agent():
                from langchain.prompts import ChatPromptTemplate
                
                prompt = ChatPromptTemplate.from_messages([
                    ("system", "You are a research assistant. Use the search tool to find current information."),
                    ("placeholder", "{chat_history}"),
                    ("human", "{input}"),
                    ("placeholder", "{agent_scratchpad}"),
                ])
                
                agent = create_openai_tools_agent(self.llm, [self.search_tool], prompt)
                return AgentExecutor(agent=agent, tools=[self.search_tool])
            
            # Perform research
            research_agent = create_research_agent()
            
            research_queries = [
                f"Find recent developments in {topic}",
                f"What are current trends in {topic}?",
                f"Find expert opinions on {topic}"
            ]
            
            research_results = {}
            for query in research_queries:
                result = await self._async_agent_invoke(research_agent, {"input": query})
                research_results[query] = result['output']
            
            return {
                'status': 'success',
                'research_results': research_results
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e)
            }
    
    @task(name="collaborative_analysis", organization_id="research", project_id="multi_framework")
    async def collaborative_analysis(self, topic: str, doc_analysis: Dict, web_research: Dict) -> Dict[str, Any]:
        """Collaborative analysis using CrewAI."""
        
        try:
            # Create specialized agents
            @agent(name="analyst_agent", organization_id="research", project_id="multi_framework")
            def create_analyst():
                return Agent(
                    role="Senior Research Analyst",
                    goal="Analyze and synthesize research findings",
                    backstory="Expert analyst with deep knowledge in research methodology",
                    verbose=True,
                    llm=self.llm
                )
            
            @agent(name="critic_agent", organization_id="research", project_id="multi_framework")
            def create_critic():
                return Agent(
                    role="Research Critic",
                    goal="Critically evaluate research findings and identify gaps",
                    backstory="Experienced researcher focused on quality and completeness",
                    verbose=True,
                    llm=self.llm
                )
            
            # Create tasks
            analyst = create_analyst()
            critic = create_critic()
            
            # Analysis task
            analysis_task = Task(
                description=f"""
                Analyze the following research findings about {topic}:
                
                Document Analysis: {doc_analysis}
                Web Research: {web_research}
                
                Provide a comprehensive analysis including:
                1. Key insights and patterns
                2. Conflicting information
                3. Knowledge gaps
                4. Recommendations for further research
                """,
                agent=analyst,
                expected_output="Detailed analysis report"
            )
            
            # Critique task
            critique_task = Task(
                description=f"""
                Review the analysis and provide critical feedback:
                1. Assess the quality of sources
                2. Identify potential biases
                3. Suggest improvements
                4. Rate the overall reliability
                """,
                agent=critic,
                expected_output="Critical evaluation report"
            )
            
            # Create and run crew
            @task(name="analysis_crew", organization_id="research", project_id="multi_framework")
            def create_analysis_crew():
                return Crew(
                    agents=[analyst, critic],
                    tasks=[analysis_task, critique_task],
                    process=Process.sequential,
                    verbose=True
                )
            
            crew = create_analysis_crew()
            result = await self._async_crew_kickoff(crew)
            
            return {
                'status': 'success',
                'analysis_result': result
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e)
            }
    
    @task(name="synthesis", organization_id="research", project_id="multi_framework")
    async def synthesize_findings(self, all_results: Dict[str, Any]) -> Dict[str, Any]:
        """Synthesize all findings into final report."""
        
        try:
            # Create synthesis prompt
            synthesis_prompt = f"""
            Based on the following research pipeline results, create a comprehensive final report:
            
            Document Analysis: {all_results.get('document_analysis', 'Not available')}
            Web Research: {all_results.get('web_research', 'Not available')}
            Collaborative Analysis: {all_results.get('collaborative_analysis', 'Not available')}
            
            Please provide:
            1. Executive Summary
            2. Key Findings
            3. Recommendations
            4. Areas for Future Research
            """
            
            # Use OpenAI directly for synthesis
            from openai import OpenAI
            client = OpenAI()
            
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a research synthesis expert."},
                    {"role": "user", "content": synthesis_prompt}
                ],
                max_tokens=2000
            )
            
            synthesis = response.choices[0].message.content
            
            return {
                'status': 'success',
                'final_report': synthesis,
                'pipeline_summary': {
                    'frameworks_used': ['LlamaIndex', 'LangChain', 'CrewAI', 'OpenAI'],
                    'total_steps': 4,
                    'success_rate': self._calculate_success_rate(all_results)
                }
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e)
            }
    
    # Helper methods for async operations
    async def _async_query(self, query_engine: BaseQueryEngine, query: str):
        """Async wrapper for LlamaIndex queries."""
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, query_engine.query, query)
    
    async def _async_agent_invoke(self, agent: AgentExecutor, input_data: Dict):
        """Async wrapper for LangChain agent invocation."""
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, agent.invoke, input_data)
    
    async def _async_crew_kickoff(self, crew: Crew):
        """Async wrapper for CrewAI crew execution."""
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, crew.kickoff)
    
    def _calculate_success_rate(self, results: Dict[str, Any]) -> float:
        """Calculate success rate across all pipeline steps."""
        total_steps = len(results)
        successful_steps = sum(1 for result in results.values() 
                             if isinstance(result, dict) and result.get('status') == 'success')
        return successful_steps / total_steps if total_steps > 0 else 0.0

# Usage example
async def main():
    pipeline = ResearchPipeline()
    
    # Sample documents
    documents = [
        "Artificial Intelligence is transforming industries through automation and intelligent decision-making.",
        "Machine learning algorithms require large datasets for training and validation.",
        "AI ethics considerations include bias, transparency, and accountability."
    ]
    
    # Run the complete pipeline
    results = await pipeline.run_research_pipeline(
        topic="AI in Healthcare",
        documents=documents
    )
    
    print("Multi-Framework Research Pipeline Results:")
    print(f"Final Report: {results.get('synthesis', {}).get('final_report', 'Not available')}")
    print(f"Pipeline Summary: {results.get('synthesis', {}).get('pipeline_summary', {})}")

# Run the example
if __name__ == "__main__":
    asyncio.run(main())
```

### Example 2: Customer Support Multi-Agent System

This example combines OpenAI Agents for conversation handling, LangChain for knowledge retrieval, and CrewAI for escalation management.

```python
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from enum import Enum

class TicketPriority(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class SupportTicket:
    id: str
    user_id: str
    message: str
    priority: TicketPriority
    category: Optional[str] = None
    resolved: bool = False

class MultiFrameworkSupportSystem:
    """Multi-framework customer support system."""
    
    def __init__(self):
        self.llm = ChatOpenAI(temperature=0)
        self.knowledge_base = self._setup_knowledge_base()
        
    @workflow(name="support_system", organization_id="support", project_id="multi_framework")
    async def handle_support_request(self, ticket: SupportTicket) -> Dict[str, Any]:
        """Handle support request using multiple frameworks."""
        
        # Step 1: Initial triage with OpenAI Agents
        triage_result = await self.triage_ticket(ticket)
        
        # Step 2: Knowledge retrieval with LangChain
        knowledge_result = await self.retrieve_knowledge(ticket, triage_result)
        
        # Step 3: Generate response
        if triage_result['requires_escalation']:
            # Use CrewAI for complex cases
            response = await self.escalated_response(ticket, triage_result, knowledge_result)
        else:
            # Use OpenAI Agents for standard responses
            response = await self.standard_response(ticket, knowledge_result)
        
        return {
            'ticket_id': ticket.id,
            'triage': triage_result,
            'knowledge': knowledge_result,
            'response': response,
            'resolved': response.get('resolved', False)
        }
    
    @agent(name="triage_agent", organization_id="support", project_id="multi_framework")
    async def triage_ticket(self, ticket: SupportTicket) -> Dict[str, Any]:
        """Triage support ticket using OpenAI Agents."""
        
        try:
            from openai import OpenAI
            client = OpenAI()
            
            triage_prompt = f"""
            Analyze this support ticket and provide triage information:
            
            User Message: {ticket.message}
            Current Priority: {ticket.priority.value}
            
            Please determine:
            1. Category (technical, billing, general, urgent)
            2. Complexity (simple, moderate, complex)
            3. Requires escalation (yes/no)
            4. Estimated resolution time
            5. Required expertise level
            
            Respond in JSON format.
            """
            
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a support triage specialist. Always respond with valid JSON."},
                    {"role": "user", "content": triage_prompt}
                ],
                max_tokens=500
            )
            
            import json
            triage_data = json.loads(response.choices[0].message.content)
            
            return {
                'status': 'success',
                'category': triage_data.get('category', 'general'),
                'complexity': triage_data.get('complexity', 'moderate'),
                'requires_escalation': triage_data.get('requires_escalation', False),
                'estimated_time': triage_data.get('estimated_resolution_time', '1 hour'),
                'expertise_level': triage_data.get('required_expertise_level', 'standard')
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e),
                'category': 'general',
                'complexity': 'moderate',
                'requires_escalation': False
            }
    
    @task(name="knowledge_retrieval", organization_id="support", project_id="multi_framework")
    async def retrieve_knowledge(self, ticket: SupportTicket, triage: Dict[str, Any]) -> Dict[str, Any]:
        """Retrieve relevant knowledge using LangChain."""
        
        try:
            from langchain.vectorstores import FAISS
            from langchain.embeddings import OpenAIEmbeddings
            from langchain.chains import RetrievalQA
            
            # Create retrieval chain
            embeddings = OpenAIEmbeddings()
            vectorstore = FAISS.from_texts(
                texts=self.knowledge_base,
                embedding=embeddings
            )
            
            qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=vectorstore.as_retriever(search_kwargs={"k": 3})
            )
            
            # Query knowledge base
            query = f"How to help with {triage.get('category', 'general')} issue: {ticket.message}"
            knowledge_response = await self._async_chain_run(qa_chain, query)
            
            return {
                'status': 'success',
                'relevant_knowledge': knowledge_response,
                'sources_found': 3  # Simplified
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e),
                'relevant_knowledge': 'No relevant knowledge found'
            }
    
    @task(name="escalated_response", organization_id="support", project_id="multi_framework")
    async def escalated_response(self, ticket: SupportTicket, triage: Dict, knowledge: Dict) -> Dict[str, Any]:
        """Handle escalated cases using CrewAI."""
        
        try:
            # Create specialist agents
            @agent(name="technical_specialist", organization_id="support", project_id="multi_framework")
            def create_technical_specialist():
                return Agent(
                    role="Technical Support Specialist",
                    goal="Provide expert technical solutions",
                    backstory="Senior technical support engineer with 10+ years experience",
                    verbose=True,
                    llm=self.llm
                )
            
            @agent(name="customer_success", organization_id="support", project_id="multi_framework")
            def create_customer_success():
                return Agent(
                    role="Customer Success Manager",
                    goal="Ensure customer satisfaction and retention",
                    backstory="Customer success expert focused on relationship management",
                    verbose=True,
                    llm=self.llm
                )
            
            # Create agents
            tech_specialist = create_technical_specialist()
            customer_success = create_customer_success()
            
            # Create tasks
            technical_task = Task(
                description=f"""
                Provide technical solution for this escalated support case:
                
                Ticket: {ticket.message}
                Category: {triage.get('category', 'unknown')}
                Complexity: {triage.get('complexity', 'unknown')}
                Knowledge Base Info: {knowledge.get('relevant_knowledge', 'None')}
                
                Provide:
                1. Detailed technical solution
                2. Step-by-step instructions
                3. Potential workarounds
                4. Prevention recommendations
                """,
                agent=tech_specialist,
                expected_output="Comprehensive technical solution"
            )
            
            customer_task = Task(
                description=f"""
                Create customer-friendly response based on technical solution:
                
                Original Issue: {ticket.message}
                Priority: {ticket.priority.value}
                
                Create a response that:
                1. Acknowledges the issue
                2. Explains the solution in simple terms
                3. Sets proper expectations
                4. Offers additional support
                """,
                agent=customer_success,
                expected_output="Customer-friendly response"
            )
            
            # Create and run crew
            @task(name="escalation_crew", organization_id="support", project_id="multi_framework")
            def create_escalation_crew():
                return Crew(
                    agents=[tech_specialist, customer_success],
                    tasks=[technical_task, customer_task],
                    process=Process.sequential,
                    verbose=True
                )
            
            crew = create_escalation_crew()
            result = await self._async_crew_kickoff(crew)
            
            return {
                'status': 'success',
                'response_type': 'escalated',
                'technical_solution': result,
                'resolved': True
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e),
                'response_type': 'escalated',
                'resolved': False
            }
    
    @agent(name="standard_response_agent", organization_id="support", project_id="multi_framework")
    async def standard_response(self, ticket: SupportTicket, knowledge: Dict) -> Dict[str, Any]:
        """Generate standard response using OpenAI Agents."""
        
        try:
            from openai import OpenAI
            client = OpenAI()
            
            response_prompt = f"""
            Create a helpful support response for this customer:
            
            Customer Message: {ticket.message}
            Priority: {ticket.priority.value}
            Relevant Knowledge: {knowledge.get('relevant_knowledge', 'General support guidelines')}
            
            Create a response that is:
            1. Professional and empathetic
            2. Provides clear solution or next steps
            3. Offers additional help if needed
            4. Sets appropriate expectations
            """
            
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful customer support representative."},
                    {"role": "user", "content": response_prompt}
                ],
                max_tokens=800
            )
            
            support_response = response.choices[0].message.content
            
            return {
                'status': 'success',
                'response_type': 'standard',
                'message': support_response,
                'resolved': True
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e),
                'response_type': 'standard',
                'resolved': False
            }
    
    def _setup_knowledge_base(self) -> List[str]:
        """Setup sample knowledge base."""
        return [
            "For login issues, first check if caps lock is on and verify the username spelling.",
            "Password reset links expire after 24 hours. Request a new one if expired.",
            "For billing questions, check the billing section in account settings.",
            "Technical issues should include browser version and error messages.",
            "Account suspension usually occurs due to policy violations or payment issues.",
            "For urgent issues, use the priority support channel with ticket escalation."
        ]
    
    # Async helper methods
    async def _async_chain_run(self, chain, query):
        """Async wrapper for LangChain operations."""
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, chain.run, query)
    
    async def _async_crew_kickoff(self, crew):
        """Async wrapper for CrewAI operations."""
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, crew.kickoff)

# Usage example
async def test_support_system():
    support_system = MultiFrameworkSupportSystem()
    
    # Test tickets
    tickets = [
        SupportTicket(
            id="T001",
            user_id="user123",
            message="I can't log into my account. It says invalid credentials but I'm sure my password is correct.",
            priority=TicketPriority.MEDIUM
        ),
        SupportTicket(
            id="T002", 
            user_id="user456",
            message="My application crashed and I lost all my work. This is urgent!",
            priority=TicketPriority.CRITICAL
        )
    ]
    
    for ticket in tickets:
        result = await support_system.handle_support_request(ticket)
        print(f"\nTicket {ticket.id} Results:")
        print(f"Category: {result['triage'].get('category', 'unknown')}")
        print(f"Escalated: {result['triage'].get('requires_escalation', False)}")
        print(f"Resolved: {result['resolved']}")
        print(f"Response Type: {result['response'].get('response_type', 'unknown')}")

if __name__ == "__main__":
    asyncio.run(test_support_system())
```

## Cross-Framework Communication Patterns

### 1. Event-Driven Architecture

```python
from typing import Dict, Any, List, Callable
import asyncio
from dataclasses import dataclass
from enum import Enum

class EventType(Enum):
    WORKFLOW_STARTED = "workflow_started"
    TASK_COMPLETED = "task_completed"
    AGENT_RESPONSE = "agent_response"
    ERROR_OCCURRED = "error_occurred"
    FRAMEWORK_SWITCHED = "framework_switched"

@dataclass
class FrameworkEvent:
    event_type: EventType
    source_framework: str
    target_framework: str
    data: Dict[str, Any]
    timestamp: float

class MultiFrameworkEventBus:
    """Event bus for coordinating between frameworks."""
    
    def __init__(self):
        self.listeners: Dict[EventType, List[Callable]] = {}
        self.event_history: List[FrameworkEvent] = []
    
    @workflow(name="event_orchestration", organization_id="events", project_id="multi_framework")
    def register_listener(self, event_type: EventType, callback: Callable):
        """Register event listener."""
        if event_type not in self.listeners:
            self.listeners[event_type] = []
        self.listeners[event_type].append(callback)
    
    @task(name="event_emission", organization_id="events", project_id="multi_framework")
    async def emit_event(self, event: FrameworkEvent):
        """Emit event to all registered listeners."""
        self.event_history.append(event)
        
        if event.event_type in self.listeners:
            for callback in self.listeners[event.event_type]:
                try:
                    await callback(event)
                except Exception as e:
                    print(f"Error in event listener: {e}")

# Usage with event coordination
event_bus = MultiFrameworkEventBus()

@workflow(name="coordinated_workflow", organization_id="coordination", project_id="multi_framework")
async def coordinated_multi_framework_workflow(data: Dict[str, Any]):
    """Workflow that coordinates multiple frameworks via events."""
    
    # Start with LangChain
    await event_bus.emit_event(FrameworkEvent(
        event_type=EventType.WORKFLOW_STARTED,
        source_framework="coordinator",
        target_framework="langchain",
        data=data,
        timestamp=time.time()
    ))
    
    # LangChain processing
    langchain_result = await process_with_langchain(data)
    
    await event_bus.emit_event(FrameworkEvent(
        event_type=EventType.TASK_COMPLETED,
        source_framework="langchain",
        target_framework="crewai",
        data=langchain_result,
        timestamp=time.time()
    ))
    
    # CrewAI processing
    crewai_result = await process_with_crewai(langchain_result)
    
    await event_bus.emit_event(FrameworkEvent(
        event_type=EventType.TASK_COMPLETED,
        source_framework="crewai",
        target_framework="llamaindex",
        data=crewai_result,
        timestamp=time.time()
    ))
    
    # Final processing
    final_result = await process_with_llamaindex(crewai_result)
    
    return final_result
```

### 2. Shared State Management

```python
from typing import Any, Dict, Optional
import threading
import json
from datetime import datetime

class SharedFrameworkState:
    """Thread-safe shared state across frameworks."""
    
    def __init__(self):
        self._state: Dict[str, Any] = {}
        self._lock = threading.Lock()
        self._history: List[Dict[str, Any]] = []
    
    @task(name="state_update", organization_id="state", project_id="multi_framework")
    def update_state(self, key: str, value: Any, framework: str):
        """Update shared state with framework tracking."""
        with self._lock:
            old_value = self._state.get(key)
            self._state[key] = value
            
            # Track change history
            self._history.append({
                'timestamp': datetime.now().isoformat(),
                'framework': framework,
                'key': key,
                'old_value': old_value,
                'new_value': value
            })
    
    @task(name="state_retrieval", organization_id="state", project_id="multi_framework")
    def get_state(self, key: str, default: Any = None) -> Any:
        """Get state value safely."""
        with self._lock:
            return self._state.get(key, default)
    
    @task(name="state_context", organization_id="state", project_id="multi_framework") 
    def get_framework_context(self, framework: str) -> Dict[str, Any]:
        """Get framework-specific context."""
        with self._lock:
            return {
                'current_state': dict(self._state),
                'framework_history': [
                    h for h in self._history 
                    if h['framework'] == framework
                ]
            }

# Usage across frameworks
shared_state = SharedFrameworkState()

@workflow(name="stateful_multi_framework", organization_id="state", project_id="multi_framework")
async def stateful_workflow(initial_data: Dict[str, Any]):
    """Workflow with shared state across frameworks."""
    
    # Initialize shared state
    shared_state.update_state('workflow_data', initial_data, 'coordinator')
    shared_state.update_state('current_step', 'langchain_processing', 'coordinator')
    
    # LangChain processing with state
    @task(name="langchain_with_state", organization_id="state", project_id="multi_framework")
    async def langchain_step():
        context = shared_state.get_framework_context('langchain')
        data = shared_state.get_state('workflow_data')
        
        # Process with LangChain
        result = await process_with_langchain(data)
        
        # Update shared state
        shared_state.update_state('langchain_result', result, 'langchain')
        shared_state.update_state('current_step', 'crewai_processing', 'langchain')
        
        return result
    
    # CrewAI processing with state
    @task(name="crewai_with_state", organization_id="state", project_id="multi_framework")
    async def crewai_step():
        context = shared_state.get_framework_context('crewai')
        langchain_result = shared_state.get_state('langchain_result')
        
        # Process with CrewAI
        result = await process_with_crewai(langchain_result)
        
        # Update shared state
        shared_state.update_state('crewai_result', result, 'crewai')
        shared_state.update_state('current_step', 'completed', 'crewai')
        
        return result
    
    # Execute steps
    await langchain_step()
    await crewai_step()
    
    # Return final state
    return shared_state.get_framework_context('coordinator')
```

## Testing Multi-Framework Applications

### Comprehensive Testing Strategy

```python
import unittest
import asyncio
from unittest.mock import patch, MagicMock

class TestMultiFrameworkIntegration(unittest.TestCase):
    """Test multi-framework integration patterns."""
    
    def setUp(self):
        """Set up test environment."""
        self.rizk = Rizk.init(
            app_name="Multi-Framework-Test",
            enabled=True
        )
    
    def test_framework_detection(self):
        """Test that all frameworks are properly detected."""
        from rizk.sdk.utils.framework_detection import detect_framework
        
        # Test different framework objects
        frameworks_detected = []
        
        # Mock framework objects
        langchain_obj = MagicMock()
        langchain_obj.__module__ = "langchain.agents"
        
        crewai_obj = MagicMock()
        crewai_obj.__module__ = "crewai"
        
        # Detection should work
        self.assertIsNotNone(detect_framework())
    
    def test_cross_framework_communication(self):
        """Test communication between frameworks."""
        
        @workflow(name="test_communication", organization_id="test", project_id="multi")
        async def test_workflow():
            # Simulate cross-framework data flow
            langchain_result = {"framework": "langchain", "data": "processed"}
            crewai_result = {"framework": "crewai", "input": langchain_result}
            return crewai_result
        
        # Run test
        result = asyncio.run(test_workflow())
        self.assertEqual(result["framework"], "crewai")
        self.assertIn("input", result)
    
    def test_error_propagation(self):
        """Test error handling across frameworks."""
        
        @workflow(name="test_errors", organization_id="test", project_id="multi")
        async def error_workflow():
            try:
                # Simulate framework error
                raise ValueError("Framework A error")
            except ValueError as e:
                # Error should be properly traced
                return {"error": str(e), "handled": True}
        
        result = asyncio.run(error_workflow())
        self.assertTrue(result["handled"])
    
    def test_performance_across_frameworks(self):
        """Test performance characteristics."""
        import time
        
        @workflow(name="performance_test", organization_id="test", project_id="multi")
        async def performance_workflow():
            start_time = time.time()
            
            # Simulate multiple framework calls
            await asyncio.sleep(0.1)  # Simulate LangChain
            await asyncio.sleep(0.1)  # Simulate CrewAI
            await asyncio.sleep(0.1)  # Simulate LlamaIndex
            
            end_time = time.time()
            return {"duration": end_time - start_time}
        
        result = asyncio.run(performance_workflow())
        self.assertLess(result["duration"], 1.0)  # Should complete quickly

if __name__ == "__main__":
    unittest.main()
```

## Best Practices for Multi-Framework Applications

### 1. Framework Selection Strategy

```python
class FrameworkSelector:
    """Intelligent framework selection based on task requirements."""
    
    FRAMEWORK_STRENGTHS = {
        'langchain': ['tool_integration', 'chain_composition', 'web_search'],
        'crewai': ['multi_agent', 'collaboration', 'role_playing'],
        'llamaindex': ['document_processing', 'rag', 'indexing'],
        'openai_agents': ['native_openai', 'function_calling', 'assistants'],
        'langgraph': ['state_management', 'complex_workflows', 'conditional_logic']
    }
    
    @task(name="framework_selection", organization_id="optimization", project_id="multi_framework")
    def select_optimal_framework(self, task_requirements: List[str]) -> str:
        """Select the best framework for given requirements."""
        
        scores = {}
        for framework, strengths in self.FRAMEWORK_STRENGTHS.items():
            score = sum(1 for req in task_requirements if req in strengths)
            scores[framework] = score
        
        return max(scores, key=scores.get)
    
    @workflow(name="adaptive_framework_usage", organization_id="optimization", project_id="multi_framework")
    async def adaptive_workflow(self, task_description: str, requirements: List[str]):
        """Adaptively use frameworks based on task needs."""
        
        optimal_framework = self.select_optimal_framework(requirements)
        
        if optimal_framework == 'langchain':
            return await self.execute_with_langchain(task_description)
        elif optimal_framework == 'crewai':
            return await self.execute_with_crewai(task_description)
        elif optimal_framework == 'llamaindex':
            return await self.execute_with_llamaindex(task_description)
        else:
            # Fallback to multi-framework approach
            return await self.execute_multi_framework(task_description)
```

### 2. Resource Management

```python
class MultiFrameworkResourceManager:
    """Manage resources across multiple frameworks."""
    
    def __init__(self):
        self.active_connections = {}
        self.resource_limits = {
            'max_concurrent_agents': 10,
            'max_memory_usage': 1024 * 1024 * 1024,  # 1GB
            'max_execution_time': 300  # 5 minutes
        }
    
    @task(name="resource_monitoring", organization_id="resources", project_id="multi_framework")
    async def monitor_resources(self):
        """Monitor resource usage across frameworks."""
        import psutil
        
        memory_usage = psutil.virtual_memory().used
        cpu_usage = psutil.cpu_percent()
        
        return {
            'memory_usage': memory_usage,
            'cpu_usage': cpu_usage,
            'within_limits': memory_usage < self.resource_limits['max_memory_usage']
        }
    
    @workflow(name="resource_aware_execution", organization_id="resources", project_id="multi_framework")
    async def execute_with_resource_management(self, tasks: List[Dict]):
        """Execute tasks with resource awareness."""
        
        results = []
        for task in tasks:
            # Check resources before execution
            resource_status = await self.monitor_resources()
            
            if not resource_status['within_limits']:
                # Wait or skip if resources are constrained
                await asyncio.sleep(1)
                continue
            
            # Execute task
            result = await self.execute_task(task)
            results.append(result)
        
        return results
```

This comprehensive guide provides patterns and examples for building sophisticated multi-framework applications with unified observability and governance through Rizk SDK. The examples demonstrate real-world scenarios where combining multiple frameworks provides superior results compared to using any single framework alone.

## Summary

Multi-framework applications represent the future of AI development, allowing teams to leverage the best capabilities from each framework while maintaining unified observability, governance, and performance monitoring through Rizk SDK. 

