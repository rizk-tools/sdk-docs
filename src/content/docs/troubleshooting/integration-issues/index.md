---
title: "Integration Issues"
description: "Integration Issues"
---

# Integration Issues

This guide helps you diagnose and resolve issues when integrating Rizk SDK with different LLM frameworks and platforms.

## ðŸ”Œ Framework Integration Overview

Rizk SDK supports multiple frameworks through automatic detection and adapter registration:

- **OpenAI Agents SDK**: Native agents and function tools
- **LangChain**: Agents, chains, and tool integrations
- **CrewAI**: Multi-agent workflows and task management
- **LlamaIndex**: Query engines and chat interfaces
- **Custom Frameworks**: Extensible adapter system

## ðŸš¨ Common Integration Issues

### 1. Import Order Problems

**Issue**: Framework not detected or adapters not working
```python
# âŒ Wrong order - framework imported first
import langchain
from rizk.sdk import Rizk

# SDK may not detect LangChain properly
```

**Solution**:
```python
# âœ… Correct order - Rizk first
from rizk.sdk import Rizk
Rizk.init(app_name="MyApp")

# Then import framework
import langchain
from langchain.agents import AgentExecutor
```

### 2. Missing Framework Dependencies

**Issue**: Adapter registration fails due to missing dependencies
```
WARNING: LangChain adapter not available - missing dependencies
```

**Solution**:
```bash
# Install framework-specific dependencies
pip install rizk[langchain]    # For LangChain
pip install rizk[crewai]       # For CrewAI  
pip install rizk[llama-index]  # For LlamaIndex

# Or install frameworks separately
pip install langchain langchain-openai
pip install crewai
pip install llama-index
```

### 3. Framework Detection Issues

**Issue**: SDK defaults to "standard" instead of detecting your framework
```python
from rizk.sdk.utils.framework_detection import detect_framework
print(detect_framework())  # Returns "standard" not "langchain"
```

**Solution**:
```python
# Debug framework detection
from rizk.sdk.utils.framework_detection import debug_framework_detection
debug_framework_detection()

# Explicit framework specification
from rizk.sdk.decorators import workflow

@workflow(name="my_workflow", framework="langchain")
def my_function():
    pass

# Check imported modules
import sys
langchain_modules = [m for m in sys.modules if 'langchain' in m]
print(f"LangChain modules: {langchain_modules}")
```

## ðŸ¦œ LangChain Integration

### Common LangChain Issues

**Issue 1**: Agent execution not being traced
```python
from langchain.agents import AgentExecutor
# Execution not monitored by Rizk
```

**Solution**:
```python
# Proper LangChain integration
from rizk.sdk import Rizk
Rizk.init(app_name="LangChain_App")

from langchain.agents import AgentExecutor
from langchain_openai import ChatOpenAI
from rizk.sdk.decorators import workflow, tool

@tool(name="search_tool")
def search_tool(query: str) -> str:
    return f"Search results for: {query}"

@workflow(name="langchain_agent")
def run_agent(query: str):
    llm = ChatOpenAI(temperature=0)
    agent = AgentExecutor.from_agent_and_tools(
        agent=agent, 
        tools=[search_tool],
        verbose=True
    )
    return agent.run(query)
```

**Issue 2**: Custom callbacks interfering
```python
# Custom callbacks may conflict with Rizk
agent.run(query, callbacks=[custom_callback])
```

**Solution**:
```python
# Combine with Rizk callbacks
from rizk.sdk.adapters.langchain_adapter import RizkGuardrailCallbackHandler

rizk_callback = RizkGuardrailCallbackHandler()
agent.run(query, callbacks=[custom_callback, rizk_callback])
```

### LangChain Verification

```python
def verify_langchain_integration():
    """Verify LangChain integration is working."""
    
    # Check adapter registration
    from rizk.sdk.utils.framework_registry import FrameworkRegistry
    adapter = FrameworkRegistry.get_adapter("langchain")
    print(f"LangChain adapter: {'âœ…' if adapter else 'âŒ'}")
    
    # Check framework detection
    from rizk.sdk.utils.framework_detection import detect_framework
    framework = detect_framework()
    print(f"Detected framework: {framework}")
    
    # Test basic integration
    from rizk.sdk.decorators import workflow
    
    @workflow(name="langchain_test")
    def test_function():
        return "LangChain integration working"
    
    result = test_function()
    print(f"Test result: {result}")

verify_langchain_integration()
```

## ðŸš¢ CrewAI Integration

### Common CrewAI Issues

**Issue 1**: Crew execution not traced
```python
from crewai import Crew, Agent, Task
# Crew kickoff not monitored
```

**Solution**:
```python
# Proper CrewAI integration
from rizk.sdk import Rizk
Rizk.init(app_name="CrewAI_App")

from crewai import Crew, Agent, Task
from rizk.sdk.decorators import crew, agent, task

@agent(name="research_agent")
def create_researcher():
    return Agent(
        role="Researcher",
        goal="Research information",
        backstory="Expert researcher",
        verbose=True
    )

@task(name="research_task") 
def create_research_task(agent):
    return Task(
        description="Research the topic thoroughly",
        agent=agent,
        expected_output="Detailed research report"
    )

@crew(name="research_crew")
def create_crew():
    researcher = create_researcher()
    task = create_research_task(researcher)
    return Crew(
        agents=[researcher],
        tasks=[task],
        verbose=2
    )

# Execute crew
my_crew = create_crew()
result = my_crew.kickoff()
```

**Issue 2**: Task dependencies not tracked
```python
# Sequential tasks without proper tracking
```

**Solution**:
```python
# Use context and dependencies
@task(name="task1", organization_id="org", project_id="proj")
def create_task1(agent):
    return Task(
        description="First task",
        agent=agent,
        context=[]  # No dependencies
    )

@task(name="task2", organization_id="org", project_id="proj")
def create_task2(agent, task1):
    return Task(
        description="Second task",
        agent=agent,
        context=[task1]  # Depends on task1
    )
```

### CrewAI Verification

```python
def verify_crewai_integration():
    """Verify CrewAI integration is working."""
    
    try:
        import crewai
        print("âœ… CrewAI imported successfully")
        
        # Check adapter
        from rizk.sdk.utils.framework_registry import FrameworkRegistry
        adapter = FrameworkRegistry.get_adapter("crewai")
        print(f"CrewAI adapter: {'âœ…' if adapter else 'âŒ'}")
        
        # Test basic crew creation
        from crewai import Agent
        test_agent = Agent(
            role="Test Agent",
            goal="Test goal",
            backstory="Test backstory"
        )
        print("âœ… CrewAI agent creation successful")
        
    except ImportError as e:
        print(f"âŒ CrewAI not available: {e}")
    except Exception as e:
        print(f"âŒ CrewAI integration error: {e}")

verify_crewai_integration()
```

## ðŸ¦™ LlamaIndex Integration

### Common LlamaIndex Issues

**Issue 1**: Query engine not monitored
```python
from llama_index import VectorStoreIndex
# Query execution not traced
```

**Solution**:
```python
# Proper LlamaIndex integration
from rizk.sdk import Rizk
Rizk.init(app_name="LlamaIndex_App")

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from rizk.sdk.decorators import workflow, tool

@workflow(name="llama_query")
def query_documents(query: str):
    # Load documents
    documents = SimpleDirectoryReader("data").load_data()
    
    # Create index
    index = VectorStoreIndex.from_documents(documents)
    
    # Query
    query_engine = index.as_query_engine()
    response = query_engine.query(query)
    
    return str(response)

@tool(name="document_search")
def search_documents(query: str) -> str:
    return query_documents(query)
```

**Issue 2**: Chat engine not integrated
```python
# Chat engine without Rizk monitoring
```

**Solution**:
```python
from rizk.sdk.decorators import workflow

@workflow(name="llama_chat")
def chat_with_documents(message: str):
    # Create chat engine
    chat_engine = index.as_chat_engine()
    response = chat_engine.chat(message)
    return str(response)
```

## ðŸ¤– OpenAI Agents SDK Integration

### Common OpenAI Agents Issues

**Issue 1**: Agent functions not traced
```python
import openai
from agents import Agent

# Functions not monitored
```

**Solution**:
```python
# Proper OpenAI Agents integration
from rizk.sdk import Rizk
Rizk.init(app_name="OpenAI_Agents")

from agents import Agent
from rizk.sdk.decorators import tool, agent

@tool(name="calculator")
def calculate(expression: str) -> str:
    """Calculate mathematical expressions."""
    try:
        result = eval(expression)
        return str(result)
    except:
        return "Invalid expression"

@agent(name="math_agent")
def create_math_agent():
    return Agent(
        name="MathBot",
        instructions="You are a math assistant. Use the calculator tool.",
        tools=[calculate]
    )

# Use the agent
math_agent = create_math_agent()
response = math_agent.run("What is 25 * 4?")
```

**Issue 2**: Streaming responses not handled
```python
# Streaming without proper integration
```

**Solution**:
```python
from rizk.sdk.decorators import workflow

@workflow(name="streaming_agent")
def stream_agent_response(query: str):
    agent = create_math_agent()
    
    # Handle streaming
    for chunk in agent.run(query, stream=True):
        yield chunk
```

## ðŸ› ï¸ Custom Framework Integration

### Creating Custom Adapters

For frameworks not natively supported:

```python
from rizk.sdk.adapters.base import BaseAdapter
from rizk.sdk.utils.framework_registry import FrameworkRegistry

class CustomFrameworkAdapter(BaseAdapter):
    """Adapter for custom framework."""
    
    def adapt_workflow(self, func, name=None, **kwargs):
        """Adapt workflow functions."""
        def wrapper(*args, **kwargs):
            # Add custom tracing logic
            with self.trace_context("workflow", name):
                return func(*args, **kwargs)
        return wrapper
    
    def adapt_task(self, func, name=None, **kwargs):
        """Adapt task functions."""
        def wrapper(*args, **kwargs):
            with self.trace_context("task", name):
                return func(*args, **kwargs)
        return wrapper
    
    def adapt_agent(self, func, name=None, **kwargs):
        """Adapt agent functions."""
        def wrapper(*args, **kwargs):
            with self.trace_context("agent", name):
                return func(*args, **kwargs)
        return wrapper
    
    def adapt_tool(self, func_or_class, name=None, **kwargs):
        """Adapt tool functions."""
        if callable(func_or_class):
            def wrapper(*args, **kwargs):
                with self.trace_context("tool", name):
                    return func_or_class(*args, **kwargs)
            return wrapper
        return func_or_class

# Register custom adapter
FrameworkRegistry.register_adapter("custom_framework", CustomFrameworkAdapter)
```

### Testing Custom Adapters

```python
def test_custom_adapter():
    """Test custom framework adapter."""
    
    from rizk.sdk.decorators import workflow
    
    @workflow(name="custom_test", framework="custom_framework")
    def custom_function():
        return "Custom framework working"
    
    result = custom_function()
    print(f"Custom adapter test: {result}")

test_custom_adapter()
```

## ðŸ” Integration Debugging

### Framework Detection Debugger

```python
def debug_integration_issues():
    """Debug framework integration issues."""
    
    print("ðŸ” Framework Integration Debug")
    print("=" * 40)
    
    # 1. Check Python environment
    import sys
    print(f"Python version: {sys.version}")
    print(f"Python path: {sys.path[:3]}...")  # First 3 paths
    
    # 2. Check installed packages
    try:
        import pkg_resources
        frameworks = ['langchain', 'crewai', 'llama-index', 'agents']
        for framework in frameworks:
            try:
                version = pkg_resources.get_distribution(framework).version
                print(f"âœ… {framework}: {version}")
            except pkg_resources.DistributionNotFound:
                print(f"âŒ {framework}: Not installed")
    except ImportError:
        print("âŒ pkg_resources not available")
    
    # 3. Check Rizk adapters
    from rizk.sdk.utils.framework_registry import FrameworkRegistry
    adapters = FrameworkRegistry.get_all_framework_names()
    print(f"\nRegistered adapters: {adapters}")
    
    # 4. Test framework detection
    from rizk.sdk.utils.framework_detection import detect_framework
    framework = detect_framework()
    print(f"Detected framework: {framework}")
    
    # 5. Check module imports
    framework_modules = {}
    for module_name in sys.modules:
        for fw in ['langchain', 'crewai', 'llama', 'agents']:
            if fw in module_name.lower():
                if fw not in framework_modules:
                    framework_modules[fw] = []
                framework_modules[fw].append(module_name)
    
    print(f"\nImported framework modules:")
    for fw, modules in framework_modules.items():
        print(f"  {fw}: {len(modules)} modules")

debug_integration_issues()
```

## ðŸ“‹ Integration Checklist

### Before Integration
- [ ] Install Rizk SDK: `pip install rizk`
- [ ] Install target framework and dependencies
- [ ] Check Python version compatibility (>=3.10)
- [ ] Set up API keys and environment variables

### During Integration
- [ ] Import Rizk SDK before framework imports
- [ ] Initialize Rizk SDK early in application
- [ ] Use framework-specific decorators
- [ ] Test basic functionality before full integration

### After Integration
- [ ] Verify framework detection works correctly
- [ ] Test decorator application on sample functions
- [ ] Check that tracing and monitoring work
- [ ] Validate guardrails integration
- [ ] Performance test with realistic workloads

### Troubleshooting Steps
- [ ] Enable debug logging
- [ ] Check adapter registration
- [ ] Verify import order
- [ ] Test with minimal examples
- [ ] Check for conflicting dependencies
- [ ] Review error logs and stack traces

## ðŸ“ž Getting Help

If you're still experiencing integration issues:

1. **Enable verbose logging** and check for specific error messages
2. **Test with minimal examples** before complex integrations
3. **Check the GitHub issues** for similar problems
4. **Contact support** with your framework version and error details

Remember that each framework has its own patterns and best practices. Following the framework-specific guidance in this documentation will help ensure smooth integration with Rizk SDK.


