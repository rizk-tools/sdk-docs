---
title: "Framework Detection"
description: "Framework Detection"
---

# Framework Detection

Rizk SDK's automatic framework detection is one of its most powerful features, enabling truly universal LLM framework integration. This document explains how the detection system works, its optimization strategies, and how to work with it effectively.

## Overview

The framework detection system automatically identifies which LLM framework you're using without requiring manual configuration. It analyzes your code, imports, and object types to determine the appropriate adapter to use.

```python
from rizk.sdk import Rizk
from rizk.sdk.decorators import workflow

# No framework specification needed - automatically detected
@workflow(name="my_process")
def my_function():
    # Framework detected based on your code context
    pass
```

## Supported Frameworks

| Framework | Detection Key | Primary Indicators |
|-----------|---------------|-------------------|
| OpenAI Agents SDK | `openai_agents` | `agents` module, `Agent` class |
| LangChain | `langchain` | `langchain` module, `AgentExecutor` class |
| CrewAI | `crewai` | `crewai` module, `Crew` class |
| LlamaIndex | `llama_index` | `llama_index` module, `VectorStoreIndex` class |
| LangGraph | `langgraph` | `langgraph` module, `StateGraph` class |
| Standard/Fallback | `standard` | No specific framework detected |

## Detection Mechanisms

### 1. Module Import Analysis

The system checks `sys.modules` for framework-specific imports:

```python
FRAMEWORK_DETECTION_PATTERNS = {
    "langchain": {
        "modules": ["langchain", "langchain.agents", "langchain.chains"],
        "module_patterns": [r"^langchain(\.|$)", r"^langchain\.agents(\.|$)"]
    },
    "crewai": {
        "modules": ["crewai", "crewai.agent", "crewai.crew"],
        "module_patterns": [r"^crewai(\.|$)", r"^crewai\.agent(\.|$)"]
    }
}
```

**Example Detection:**
```python
import langchain
from langchain.agents import AgentExecutor

# Framework automatically detected as 'langchain'
@workflow(name="langchain_process")
def my_langchain_function():
    pass
```

### 2. Object Type Inspection

The system analyzes function arguments and return types:

```python
def identify_object_framework(obj: Any) -> Optional[str]:
    """Identify framework based on object type and module."""
    
    if hasattr(obj, '__module__'):
        module_name = obj.__module__
        
        # Check if object belongs to a known framework
        if 'langchain' in module_name:
            return 'langchain'
        elif 'crewai' in module_name:
            return 'crewai'
        # ... other checks
```

**Example Detection:**
```python
from crewai import Agent, Crew

def create_crew_agent():
    return Agent(role="researcher", goal="research topics")

# Framework detected as 'crewai' based on return type
@agent(name="crew_agent")
def agent_creator():
    return create_crew_agent()
```

### 3. Class Hierarchy Inspection

The system examines inheritance patterns:

```python
# Detects LangChain based on class inheritance
class MyCustomChain(BaseChain):
    def _call(self, inputs):
        return {"output": "processed"}

@workflow(name="custom_chain")
def run_custom_chain():
    chain = MyCustomChain()  # Framework detected as 'langchain'
    return chain.run("input")
```

### 4. Runtime Context Analysis

The system analyzes the execution context:

```python
def detect_framework_from_context(func, args, kwargs):
    """Detect framework from function call context."""
    
    # Analyze arguments
    for arg in args:
        framework = identify_object_framework(arg)
        if framework:
            return framework
    
    # Analyze keyword arguments
    for value in kwargs.values():
        framework = identify_object_framework(value)
        if framework:
            return framework
    
    return "standard"  # Fallback
```

## Performance Optimizations

### 1. Multi-Level Caching

The detection system uses sophisticated caching to minimize overhead:

```python
class FrameworkDetectionCache:
    """Thread-safe cache with TTL and LRU eviction."""
    
    def __init__(self, max_size=1000, ttl_seconds=3600):
        self._cache = {}  # key -> (framework, timestamp)
        self._lock = threading.RLock()
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
```

**Cache Levels:**
- **Function-level cache**: Results cached per function signature
- **Object-type cache**: Results cached per object type
- **Module-level cache**: Framework imports cached globally

### 2. Pre-compiled Patterns

Regex patterns are pre-compiled for faster matching:

```python
class OptimizedFrameworkDetection:
    def __init__(self):
        self._compiled_patterns: Dict[str, List[Pattern]] = {}
        self._precompile_patterns()
    
    def _precompile_patterns(self):
        """Pre-compile all regex patterns for performance."""
        for framework, config in FRAMEWORK_DETECTION_PATTERNS.items():
            self._compiled_patterns[framework] = [
                re.compile(pattern) for pattern in config["module_patterns"]
            ]
```

### 3. Lazy Framework Import Caching

Framework imports are cached with TTL to avoid repeated sys.modules scanning:

```python
def _get_cached_imported_frameworks(self) -> List[str]:
    """Get cached list of imported frameworks."""
    current_time = time.time()
    
    if (self._imported_frameworks_cache is None or 
        current_time - self._imported_frameworks_cache_time > self._imported_frameworks_cache_ttl):
        
        # Refresh cache
        self._imported_frameworks_cache = self._scan_imported_frameworks()
        self._imported_frameworks_cache_time = current_time
    
    return self._imported_frameworks_cache
```

## Detection API

### Basic Detection

```python
from rizk.sdk.utils.framework_detection import detect_framework

# Detect framework from current context
framework = detect_framework()
print(f"Detected framework: {framework}")

# Detect framework for specific function
def my_langchain_function():
    pass

framework = detect_framework(my_langchain_function)
print(f"Function framework: {framework}")
```

### Advanced Detection

```python
from rizk.sdk.utils.framework_detection import (
    detect_framework_cached,
    is_langchain_agent,
    is_crewai_agent,
    get_detection_cache_stats
)

# Cached detection (recommended for production)
framework = detect_framework_cached(func, *args, **kwargs)

# Specific framework checks
if is_langchain_agent(some_object):
    print("This is a LangChain agent")

if is_crewai_agent(some_object):
    print("This is a CrewAI agent")

# Cache performance monitoring
stats = get_detection_cache_stats()
print(f"Cache hit rate: {stats['hit_rate']:.2f}%")
```

## Framework-Specific Detection Examples

### OpenAI Agents SDK

```python
from agents import Agent, Task, Workflow

# Detection based on imports and object types
agent = Agent(name="assistant")
task = Task(description="help user")

@workflow(name="openai_workflow")  # Detected as 'openai_agents'
def run_agent_workflow():
    return agent.run(task)
```

### LangChain

```python
from langchain.agents import AgentExecutor
from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI

llm = ChatOpenAI()
chain = LLMChain(llm=llm, prompt=prompt)

@workflow(name="langchain_workflow")  # Detected as 'langchain'
def run_langchain_chain():
    return chain.run("input")
```

### CrewAI

```python
from crewai import Agent, Task, Crew, Process

researcher = Agent(role="researcher", goal="research")
task = Task(description="research topic", agent=researcher)
crew = Crew(agents=[researcher], tasks=[task])

@workflow(name="crew_workflow")  # Detected as 'crewai'
def run_crew():
    return crew.kickoff()
```

### LlamaIndex

```python
from llama_index import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core import Settings

documents = SimpleDirectoryReader("data").load_data()
index = VectorStoreIndex.from_documents(documents)

@workflow(name="llama_workflow")  # Detected as 'llama_index'
def query_index():
    query_engine = index.as_query_engine()
    return query_engine.query("What is this about?")
```

## Manual Framework Override

Sometimes you may need to override automatic detection:

```python
from rizk.sdk.decorators import workflow

# Force specific framework adapter
@workflow(name="my_process", framework="langchain")
def my_function():
    # Will use LangChain adapter regardless of detection
    pass

# Or set globally
import os
os.environ["RIZK_FORCE_FRAMEWORK"] = "crewai"

@workflow(name="forced_crewai")
def another_function():
    # Will use CrewAI adapter
    pass
```

## Detection Debugging

### Enable Debug Logging

```python
import logging

# Enable debug logging for framework detection
logging.getLogger("rizk.utils.framework_detection").setLevel(logging.DEBUG)

# Now you'll see detailed detection logs
@workflow(name="debug_workflow")
def my_function():
    pass
```

### Cache Statistics

```python
from rizk.sdk.utils.framework_detection import get_detection_cache_stats, clear_detection_cache

# Get cache performance metrics
stats = get_detection_cache_stats()
print(f"""
Detection Cache Stats:
- Cache Size: {stats['size']}/{stats['max_size']}
- Hit Rate: {stats['hit_rate']:.2f}%
- Total Hits: {stats['hits']}
- Total Misses: {stats['misses']}
- TTL: {stats['ttl_seconds']}s
""")

# Clear cache if needed (for testing)
clear_detection_cache()
```

### Manual Detection Testing

```python
from rizk.sdk.utils.framework_detection import _identify_object_framework

# Test detection on specific objects
import langchain
from langchain.agents import AgentExecutor

agent_executor = AgentExecutor(agent=agent, tools=tools)
framework = _identify_object_framework(agent_executor)
print(f"AgentExecutor detected as: {framework}")  # Should be 'langchain'
```

## Common Detection Scenarios

### Mixed Framework Applications

```python
# Application using multiple frameworks
import langchain
import crewai

@workflow(name="langchain_part")  # Detected as 'langchain'
def process_with_langchain():
    # LangChain code
    pass

@workflow(name="crewai_part")  # Detected as 'crewai'
def process_with_crewai():
    # CrewAI code
    pass
```

### Custom Framework Wrappers

```python
class MyFrameworkWrapper:
    """Custom wrapper around LangChain"""
    
    def __init__(self):
        from langchain.agents import AgentExecutor
        self.executor = AgentExecutor(...)
    
    def run(self, input_text):
        return self.executor.run(input_text)

# Detection works through object inspection
@workflow(name="wrapped_framework")  # Still detected as 'langchain'
def use_wrapper():
    wrapper = MyFrameworkWrapper()
    return wrapper.run("test")
```

### Framework Evolution

```python
# SDK adapts to framework updates automatically
try:
    from langchain_community.agents import AgentExecutor  # New import path
except ImportError:
    from langchain.agents import AgentExecutor  # Fallback

# Detection works with both import paths
@workflow(name="evolved_framework")
def use_updated_framework():
    executor = AgentExecutor(...)
    return executor.run("input")
```

## Performance Considerations

### 1. Detection Overhead

Framework detection adds minimal overhead:

```python
# Typical detection times (cached):
# - First detection: ~1-5ms
# - Cached detection: ~0.01-0.1ms
# - Pattern matching: ~0.1-0.5ms
```

### 2. Memory Usage

The detection system is memory-efficient:

```python
# Cache memory usage:
# - Default cache size: 1000 entries
# - Average entry size: ~100 bytes
# - Total memory: ~100KB
```

### 3. Optimization Tips

```python
# 1. Use cached detection in production
framework = detect_framework_cached(func, *args, **kwargs)

# 2. Pre-warm cache for known functions
for func in critical_functions:
    detect_framework_cached(func)

# 3. Monitor cache hit rates
stats = get_detection_cache_stats()
if stats['hit_rate'] < 80:
    # Consider increasing cache size or TTL
    pass
```

## Troubleshooting Detection Issues

### 1. Framework Not Detected

```python
# Check if framework modules are imported
import sys
print("Imported modules:", [m for m in sys.modules.keys() if 'langchain' in m])

# Manually test detection
from rizk.sdk.utils.framework_detection import detect_framework
framework = detect_framework()
print(f"Detected: {framework}")

# If still 'standard', check object types
def test_function():
    return some_framework_object

framework = detect_framework(test_function)
print(f"Function-specific detection: {framework}")
```

### 2. Wrong Framework Detected

```python
# Clear cache and retry
from rizk.sdk.utils.framework_detection import clear_detection_cache
clear_detection_cache()

# Use manual override
@workflow(name="my_process", framework="correct_framework")
def my_function():
    pass
```

### 3. Performance Issues

```python
# Check cache statistics
stats = get_detection_cache_stats()
if stats['hit_rate'] < 50:
    # Cache might be too small or TTL too short
    print("Consider tuning cache parameters")

# Profile detection performance
import time
start = time.time()
framework = detect_framework_cached(func)
duration = time.time() - start
print(f"Detection took: {duration*1000:.2f}ms")
```

## Best Practices

### 1. Let Detection Work Automatically

```python
# âœ… Good - Let detection work automatically
@workflow(name="my_process")
def my_function():
    pass

# âŒ Avoid - Manual framework specification unless necessary
@workflow(name="my_process", framework="langchain")
def my_function():
    pass
```

### 2. Use Consistent Import Patterns

```python
# âœ… Good - Standard imports
from langchain.agents import AgentExecutor
from crewai import Agent, Crew

# âœ… Also good - Aliased imports
import langchain as lc
from langchain.agents import AgentExecutor as LE

# âš ï¸ Careful - Dynamic imports may not be detected
def dynamic_import():
    module = __import__("langchain.agents")
    return module.AgentExecutor
```

### 3. Monitor Detection Performance

```python
# Add monitoring in production
import logging
from rizk.sdk.utils.framework_detection import get_detection_cache_stats

logger = logging.getLogger(__name__)

def log_detection_stats():
    stats = get_detection_cache_stats()
    logger.info(f"Framework detection cache hit rate: {stats['hit_rate']:.2f}%")

# Call periodically
log_detection_stats()
```

## Advanced Configuration

### Environment Variables

```python
# Configure detection behavior
import os

# Force specific framework globally
os.environ["RIZK_FORCE_FRAMEWORK"] = "langchain"

# Disable caching (for debugging)
os.environ["RIZK_DISABLE_DETECTION_CACHE"] = "true"

# Adjust cache settings
os.environ["RIZK_DETECTION_CACHE_SIZE"] = "2000"
os.environ["RIZK_DETECTION_CACHE_TTL"] = "7200"  # 2 hours
```

### Custom Detection Patterns

```python
# Add custom framework detection (advanced)
from rizk.sdk.utils.framework_detection import FRAMEWORK_DETECTION_PATTERNS

# Add custom framework
FRAMEWORK_DETECTION_PATTERNS["my_framework"] = {
    "modules": ["my_framework", "my_framework.agents"],
    "classes": ["MyAgent", "MyChain"],
    "import_names": ["my_framework"],
    "module_patterns": [r"^my_framework(\.|$)"]
}
```

## Summary

Rizk SDK's framework detection system provides:

âœ… **Automatic Detection** - No manual configuration required  
âœ… **High Performance** - Multi-level caching and optimization  
âœ… **Universal Support** - Works with all major LLM frameworks  
âœ… **Extensible** - Easy to add new frameworks  
âœ… **Debuggable** - Comprehensive logging and statistics  
âœ… **Production Ready** - Thread-safe and memory-efficient  

The detection system is designed to be invisible to developers while providing the foundation for universal framework integration. It adapts automatically as frameworks evolve and new ones emerge, making Rizk SDK truly future-proof. 

