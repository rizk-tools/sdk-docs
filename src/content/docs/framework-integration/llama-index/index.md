---
title: "LlamaIndex Integration Guide"
description: "LlamaIndex Integration Guide"
---

# LlamaIndex Integration Guide

LlamaIndex is the leading framework for building LLM-powered agents and workflows over your data. This guide demonstrates how to integrate Rizk SDK with LlamaIndex applications for comprehensive observability, governance, and performance monitoring.

## Overview

LlamaIndex provides powerful tools for:
- **Context Augmentation**: RAG (Retrieval-Augmented Generation) workflows
- **Agents**: LLM-powered knowledge assistants with tools
- **Workflows**: Multi-step processes combining agents and data connectors
- **Data Connectors**: Ingesting data from various sources (PDFs, APIs, databases)
- **Query Engines**: Natural language interfaces to your data

Rizk SDK enhances LlamaIndex applications with:
- **Automatic Instrumentation**: Trace query engines, chat engines, and agent workflows
- **Performance Monitoring**: Track response times, token usage, and success rates
- **Governance**: Apply policies to queries and responses
- **Context Management**: Hierarchical organization and user tracking

## Quick Start

### Installation

```bash
# Install LlamaIndex and Rizk SDK
pip install llama-index rizk

# For specific LlamaIndex components
pip install llama-index-llms-openai llama-index-embeddings-openai
```

### Basic Setup

```python
import os
from rizk.sdk import Rizk
from rizk.sdk.decorators import workflow, agent, tool, guardrails

# Initialize Rizk SDK
rizk = Rizk.init(
    app_name="LlamaIndex-App",
    organization_id="your_org",
    project_id="llamaindex_project",
    enabled=True
)

# Set OpenAI API key
os.environ["OPENAI_API_KEY"] = "your-openai-api-key"
```

## Core Integration Patterns

### 1. Basic RAG Query Engine

```python
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

@workflow(
    name="rag_query_engine",
    organization_id="finance_org",
    project_id="document_qa"
)
@guardrails()
def create_rag_system(data_path: str = "data/"):
    """Create a RAG system with monitoring and governance."""
    
    try:
        # Load documents
        documents = SimpleDirectoryReader(data_path).load_data()
        
        # Configure LLM and embeddings
        llm = OpenAI(model="gpt-3.5-turbo", temperature=0.1)
        embed_model = OpenAIEmbedding()
        
        # Create index
        index = VectorStoreIndex.from_documents(
            documents,
            embed_model=embed_model
        )
        
        # Create query engine
        query_engine = index.as_query_engine(
            llm=llm,
            similarity_top_k=3,
            response_mode="compact"
        )
        
        return query_engine
        
    except Exception as e:
        print(f"Error creating RAG system: {e}")
        return None

@workflow(
    name="query_documents",
    organization_id="finance_org",
    project_id="document_qa"
)
@guardrails()
def query_documents(query_engine, question: str) -> str:
    """Query documents with monitoring."""
    
    if not query_engine:
        return "RAG system not available"
    
    try:
        response = query_engine.query(question)
        return str(response)
    except Exception as e:
        return f"Query error: {e}"

# Usage
if __name__ == "__main__":
    # Create RAG system
    engine = create_rag_system("./sample_data/")
    
    # Query documents
    questions = [
        "What are the main topics discussed in the documents?",
        "Can you summarize the key findings?",
        "What recommendations are provided?"
    ]
    
    for question in questions:
        print(f"\nQ: {question}")
        answer = query_documents(engine, question)
        print(f"A: {answer}")
```

### 2. Chat Engine with Memory

```python
from llama_index.core.memory import ChatMemoryBuffer

@workflow(
    name="chat_engine_setup",
    organization_id="support_org",
    project_id="customer_chat"
)
@guardrails()
def create_chat_engine(data_path: str = "data/"):
    """Create a conversational chat engine with memory."""
    
    try:
        # Load and index documents
        documents = SimpleDirectoryReader(data_path).load_data()
        index = VectorStoreIndex.from_documents(documents)
        
        # Create memory buffer
        memory = ChatMemoryBuffer.from_defaults(token_limit=3000)
        
        # Create chat engine
        chat_engine = index.as_chat_engine(
            chat_mode="condense_plus_context",
            memory=memory,
            context_prompt=(
                "You are a helpful customer support assistant. "
                "Use the provided context to answer questions accurately. "
                "If you don't know something, say so clearly."
            ),
            verbose=True
        )
        
        return chat_engine
        
    except Exception as e:
        print(f"Error creating chat engine: {e}")
        return None

@workflow(
    name="chat_conversation",
    organization_id="support_org",
    project_id="customer_chat"
)
@guardrails()
def chat_with_engine(chat_engine, message: str, user_id: str = None) -> str:
    """Have a conversation with context awareness."""
    
    if not chat_engine:
        return "Chat engine not available"
    
    try:
        # Add user context if provided
        if user_id:
            from rizk.sdk.utils.context import set_user_context
            set_user_context(user_id=user_id)
        
        response = chat_engine.chat(message)
        return str(response)
        
    except Exception as e:
        return f"Chat error: {e}"

# Usage example
if __name__ == "__main__":
    # Create chat engine
    chat_engine = create_chat_engine("./knowledge_base/")
    
    # Simulate conversation
    conversation = [
        "Hello, I need help with your product features.",
        "What are the pricing options available?",
        "Can you explain the difference between the basic and premium plans?",
        "How do I upgrade my account?"
    ]
    
    print("ðŸ¤– Starting customer support conversation...")
    for i, message in enumerate(conversation, 1):
        print(f"\nðŸ‘¤ User: {message}")
        response = chat_with_engine(
            chat_engine, 
            message, 
            user_id=f"customer_{i}"
        )
        print(f"ðŸ¤– Assistant: {response}")
```

### 3. Agent with Tools

```python
from llama_index.core.agent import ReActAgent
from llama_index.core.tools import FunctionTool
from llama_index.llms.openai import OpenAI

@tool(
    name="calculator",
    organization_id="research_org",
    project_id="agent_tools"
)
def calculator(expression: str) -> str:
    """Calculate mathematical expressions safely."""
    try:
        # Simple calculator - in production, use a proper math parser
        result = eval(expression.replace("^", "**"))
        return f"Result: {result}"
    except Exception as e:
        return f"Calculation error: {e}"

@tool(
    name="document_search",
    organization_id="research_org", 
    project_id="agent_tools"
)
def search_documents(query: str, top_k: int = 3) -> str:
    """Search through indexed documents."""
    try:
        # This would connect to your actual document index
        # For demo purposes, return mock results
        return f"Found {top_k} relevant documents for '{query}'"
    except Exception as e:
        return f"Search error: {e}"

@agent(
    name="research_agent",
    organization_id="research_org",
    project_id="agent_workflows"
)
@guardrails()
def create_research_agent():
    """Create a research agent with tools."""
    
    try:
        # Configure LLM
        llm = OpenAI(model="gpt-3.5-turbo", temperature=0.1)
        
        # Create tools
        calc_tool = FunctionTool.from_defaults(fn=calculator)
        search_tool = FunctionTool.from_defaults(fn=search_documents)
        
        # Create agent
        agent = ReActAgent.from_tools(
            tools=[calc_tool, search_tool],
            llm=llm,
            verbose=True,
            max_iterations=5
        )
        
        return agent
        
    except Exception as e:
        print(f"Error creating agent: {e}")
        return None

@workflow(
    name="agent_task",
    organization_id="research_org",
    project_id="agent_workflows"
)
@guardrails()
def execute_agent_task(agent, task: str, user_id: str = None) -> str:
    """Execute a task using the research agent."""
    
    if not agent:
        return "Agent not available"
    
    try:
        # Set user context
        if user_id:
            from rizk.sdk.utils.context import set_user_context
            set_user_context(user_id=user_id)
        
        response = agent.chat(task)
        return str(response)
        
    except Exception as e:
        return f"Agent execution error: {e}"

# Usage
if __name__ == "__main__":
    # Create research agent
    agent = create_research_agent()
    
    # Execute tasks
    tasks = [
        "Calculate the square root of 144 and then multiply by 5",
        "Search for documents about machine learning and summarize findings",
        "What is 15% of 2500, and can you search for related financial documents?"
    ]
    
    print("ðŸ”¬ Research Agent Tasks:")
    for i, task in enumerate(tasks, 1):
        print(f"\nðŸ“ Task {i}: {task}")
        result = execute_agent_task(agent, task, user_id=f"researcher_{i}")
        print(f"ðŸ¤– Result: {result}")
```

## Enterprise Integration Patterns

### 1. Multi-Tenant Document Management

```python
from typing import Dict, List, Optional
from llama_index.core.storage.storage_context import StorageContext
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.storage.index_store import SimpleIndexStore

class MultiTenantLlamaIndexManager:
    """Enterprise-grade multi-tenant document management."""
    
    def __init__(self, base_storage_path: str = "./storage"):
        self.base_storage_path = base_storage_path
        self.tenant_indexes: Dict[str, VectorStoreIndex] = {}
        self.tenant_chat_engines: Dict[str, any] = {}
    
    @workflow(
        name="tenant_setup",
        organization_id="enterprise_org",
        project_id="multi_tenant"
    )
    @guardrails()
    def setup_tenant(self, tenant_id: str, documents_path: str) -> bool:
        """Setup document index for a specific tenant."""
        
        try:
            # Create tenant-specific storage
            storage_path = f"{self.base_storage_path}/{tenant_id}"
            
            # Load tenant documents
            documents = SimpleDirectoryReader(documents_path).load_data()
            
            # Create storage context
            storage_context = StorageContext.from_defaults(
                docstore=SimpleDocumentStore(),
                index_store=SimpleIndexStore(),
                persist_dir=storage_path
            )
            
            # Create index
            index = VectorStoreIndex.from_documents(
                documents,
                storage_context=storage_context
            )
            
            # Persist index
            index.storage_context.persist(persist_dir=storage_path)
            
            # Cache index
            self.tenant_indexes[tenant_id] = index
            
            # Create chat engine
            self.tenant_chat_engines[tenant_id] = index.as_chat_engine(
                chat_mode="condense_plus_context",
                memory=ChatMemoryBuffer.from_defaults(token_limit=3000)
            )
            
            return True
            
        except Exception as e:
            print(f"Tenant setup failed for {tenant_id}: {e}")
            return False
    
    @workflow(
        name="tenant_query",
        organization_id="enterprise_org",
        project_id="multi_tenant"
    )
    @guardrails()
    def query_tenant_documents(
        self, 
        tenant_id: str, 
        query: str, 
        user_id: Optional[str] = None
    ) -> str:
        """Query documents for a specific tenant."""
        
        try:
            # Set hierarchical context
            from rizk.sdk.utils.context import set_hierarchy_context
            set_hierarchy_context(
                organization_id="enterprise_org",
                project_id="multi_tenant",
                agent_id=f"tenant_{tenant_id}",
                user_id=user_id or "anonymous"
            )
            
            # Get tenant index
            if tenant_id not in self.tenant_indexes:
                return f"Tenant {tenant_id} not found or not initialized"
            
            # Query using chat engine
            chat_engine = self.tenant_chat_engines[tenant_id]
            response = chat_engine.chat(query)
            
            return str(response)
            
        except Exception as e:
            return f"Query failed for tenant {tenant_id}: {e}"

# Usage
if __name__ == "__main__":
    # Initialize multi-tenant manager
    manager = MultiTenantLlamaIndexManager()
    
    # Setup tenants
    tenants = [
        ("acme_corp", "./data/acme_documents/"),
        ("beta_inc", "./data/beta_documents/"),
        ("gamma_ltd", "./data/gamma_documents/")
    ]
    
    for tenant_id, docs_path in tenants:
        success = manager.setup_tenant(tenant_id, docs_path)
        print(f"Tenant {tenant_id} setup: {'âœ…' if success else 'âŒ'}")
    
    # Test queries
    test_queries = [
        ("acme_corp", "What are our main product offerings?", "user_1"),
        ("beta_inc", "Show me the latest financial results", "user_2"),
        ("gamma_ltd", "What compliance requirements do we need to meet?", "user_3")
    ]
    
    for tenant_id, query, user_id in test_queries:
        print(f"\nðŸ¢ {tenant_id} - User {user_id}")
        print(f"Q: {query}")
        response = manager.query_tenant_documents(tenant_id, query, user_id)
        print(f"A: {response[:200]}...")
```

### 2. Performance Monitoring and Analytics

```python
import time
from typing import Dict, Any
from dataclasses import dataclass, field
from collections import defaultdict

@dataclass
class LlamaIndexMetrics:
    """Metrics tracking for LlamaIndex operations."""
    
    query_count: int = 0
    total_response_time: float = 0.0
    token_usage: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    error_count: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    
    @property
    def average_response_time(self) -> float:
        return self.total_response_time / max(self.query_count, 1)
    
    @property
    def error_rate(self) -> float:
        return self.error_count / max(self.query_count, 1)
    
    @property
    def cache_hit_rate(self) -> float:
        total_cache_requests = self.cache_hits + self.cache_misses
        return self.cache_hits / max(total_cache_requests, 1)

class MonitoredLlamaIndexEngine:
    """LlamaIndex engine with comprehensive monitoring."""
    
    def __init__(self, data_path: str):
        self.metrics = LlamaIndexMetrics()
        self.query_engine = None
        self.chat_engine = None
        self._setup_engines(data_path)
    
    def _setup_engines(self, data_path: str):
        """Setup query and chat engines."""
        try:
            documents = SimpleDirectoryReader(data_path).load_data()
            index = VectorStoreIndex.from_documents(documents)
            
            self.query_engine = index.as_query_engine()
            self.chat_engine = index.as_chat_engine()
            
        except Exception as e:
            print(f"Engine setup failed: {e}")
    
    @workflow(
        name="monitored_query",
        organization_id="monitoring_org",
        project_id="performance_tracking"
    )
    @guardrails()
    def query_with_monitoring(
        self, 
        query: str, 
        engine_type: str = "query",
        user_id: str = None
    ) -> Dict[str, Any]:
        """Execute query with comprehensive monitoring."""
        
        start_time = time.time()
        
        try:
            # Set user context
            if user_id:
                from rizk.sdk.utils.context import set_user_context
                set_user_context(user_id=user_id)
            
            # Execute query
            if engine_type == "chat":
                response = self.chat_engine.chat(query)
            else:
                response = self.query_engine.query(query)
            
            # Calculate metrics
            response_time = time.time() - start_time
            self.metrics.query_count += 1
            self.metrics.total_response_time += response_time
            
            # Simulate token tracking (in real implementation, extract from response)
            estimated_tokens = len(str(response).split()) * 1.3  # Rough estimate
            self.metrics.token_usage["completion"] += int(estimated_tokens)
            self.metrics.token_usage["prompt"] += len(query.split()) * 1.3
            
            return {
                "response": str(response),
                "response_time": response_time,
                "tokens_used": int(estimated_tokens),
                "engine_type": engine_type,
                "success": True
            }
            
        except Exception as e:
            # Track error
            self.metrics.error_count += 1
            response_time = time.time() - start_time
            
            return {
                "response": f"Error: {e}",
                "response_time": response_time,
                "tokens_used": 0,
                "engine_type": engine_type,
                "success": False,
                "error": str(e)
            }
    
    @workflow(
        name="get_metrics",
        organization_id="monitoring_org",
        project_id="performance_tracking"
    )
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics."""
        
        return {
            "query_count": self.metrics.query_count,
            "average_response_time": round(self.metrics.average_response_time, 3),
            "error_rate": round(self.metrics.error_rate * 100, 2),
            "total_tokens": sum(self.metrics.token_usage.values()),
            "token_breakdown": dict(self.metrics.token_usage),
            "cache_hit_rate": round(self.metrics.cache_hit_rate * 100, 2),
            "uptime_queries": self.metrics.query_count - self.metrics.error_count
        }

# Usage
if __name__ == "__main__":
    # Initialize monitored engine
    engine = MonitoredLlamaIndexEngine("./sample_data/")
    
    # Test queries with monitoring
    test_queries = [
        ("What are the main topics in the documents?", "query", "analyst_1"),
        ("Can you summarize the key findings?", "query", "analyst_2"),
        ("Hello, I need help understanding this data", "chat", "user_1"),
        ("What recommendations are provided?", "query", "analyst_1"),
        ("Can you explain the methodology used?", "chat", "user_2")
    ]
    
    print("ðŸ” Testing Monitored LlamaIndex Engine:")
    print("=" * 50)
    
    for i, (query, engine_type, user_id) in enumerate(test_queries, 1):
        print(f"\n{i}. Query ({engine_type}): {query}")
        result = engine.query_with_monitoring(query, engine_type, user_id)
        
        print(f"   Response: {result['response'][:100]}...")
        print(f"   Time: {result['response_time']:.3f}s")
        print(f"   Tokens: {result['tokens_used']}")
        print(f"   Success: {'âœ…' if result['success'] else 'âŒ'}")
    
    # Display performance metrics
    print("\nðŸ“Š Performance Metrics:")
    print("=" * 30)
    metrics = engine.get_performance_metrics()
    
    for key, value in metrics.items():
        print(f"{key.replace('_', ' ').title()}: {value}")
```

## Configuration and Best Practices

### 1. Production Configuration

```python
# production_config.py
import os
from typing import Optional
from dataclasses import dataclass

@dataclass
class LlamaIndexConfig:
    """Production configuration for LlamaIndex integration."""
    
    # Core settings
    openai_api_key: str
    data_directory: str
    storage_directory: str
    
    # Performance settings
    chunk_size: int = 1024
    chunk_overlap: int = 20
    similarity_top_k: int = 3
    
    # LLM settings
    model_name: str = "gpt-3.5-turbo"
    temperature: float = 0.1
    max_tokens: int = 500
    
    # Embedding settings
    embedding_model: str = "text-embedding-ada-002"
    embedding_dimensions: int = 1536
    
    # Caching settings
    enable_cache: bool = True
    cache_ttl: int = 3600
    
    # Monitoring settings
    enable_metrics: bool = True
    metrics_interval: int = 300
    
    @classmethod
    def from_environment(cls) -> 'LlamaIndexConfig':
        """Load configuration from environment variables."""
        return cls(
            openai_api_key=os.getenv("OPENAI_API_KEY", ""),
            data_directory=os.getenv("LLAMA_DATA_DIR", "./data"),
            storage_directory=os.getenv("LLAMA_STORAGE_DIR", "./storage"),
            chunk_size=int(os.getenv("LLAMA_CHUNK_SIZE", "1024")),
            chunk_overlap=int(os.getenv("LLAMA_CHUNK_OVERLAP", "20")),
            similarity_top_k=int(os.getenv("LLAMA_TOP_K", "3")),
            model_name=os.getenv("LLAMA_MODEL", "gpt-3.5-turbo"),
            temperature=float(os.getenv("LLAMA_TEMPERATURE", "0.1")),
            max_tokens=int(os.getenv("LLAMA_MAX_TOKENS", "500")),
            embedding_model=os.getenv("LLAMA_EMBEDDING_MODEL", "text-embedding-ada-002"),
            enable_cache=os.getenv("LLAMA_ENABLE_CACHE", "true").lower() == "true",
            cache_ttl=int(os.getenv("LLAMA_CACHE_TTL", "3600")),
            enable_metrics=os.getenv("LLAMA_ENABLE_METRICS", "true").lower() == "true",
            metrics_interval=int(os.getenv("LLAMA_METRICS_INTERVAL", "300"))
        )
    
    def validate(self) -> list:
        """Validate configuration and return any errors."""
        errors = []
        
        if not self.openai_api_key:
            errors.append("OpenAI API key is required")
        
        if not os.path.exists(self.data_directory):
            errors.append(f"Data directory does not exist: {self.data_directory}")
        
        if self.chunk_size < 100:
            errors.append("Chunk size too small (minimum 100)")
        
        if self.similarity_top_k < 1:
            errors.append("similarity_top_k must be at least 1")
        
        return errors

@workflow(
    name="production_setup",
    organization_id="production_org",
    project_id="llamaindex_prod"
)
@guardrails()
def setup_production_llamaindex(config: Optional[LlamaIndexConfig] = None):
    """Setup LlamaIndex for production use."""
    
    if config is None:
        config = LlamaIndexConfig.from_environment()
    
    # Validate configuration
    errors = config.validate()
    if errors:
        raise ValueError(f"Configuration errors: {', '.join(errors)}")
    
    # Setup OpenAI
    os.environ["OPENAI_API_KEY"] = config.openai_api_key
    
    # Configure LlamaIndex
    from llama_index.core import Settings
    from llama_index.llms.openai import OpenAI
    from llama_index.embeddings.openai import OpenAIEmbedding
    
    Settings.llm = OpenAI(
        model=config.model_name,
        temperature=config.temperature,
        max_tokens=config.max_tokens
    )
    
    Settings.embed_model = OpenAIEmbedding(
        model=config.embedding_model
    )
    
    Settings.chunk_size = config.chunk_size
    Settings.chunk_overlap = config.chunk_overlap
    
    print("âœ… LlamaIndex production setup complete")
    return config

# Usage
if __name__ == "__main__":
    try:
        config = setup_production_llamaindex()
        print(f"ðŸ“ Data directory: {config.data_directory}")
        print(f"ðŸ”§ Model: {config.model_name}")
        print(f"ðŸ“Š Chunk size: {config.chunk_size}")
        print(f"ðŸŽ¯ Top K: {config.similarity_top_k}")
    except Exception as e:
        print(f"âŒ Setup failed: {e}")
```

## Troubleshooting Guide

### Common Issues and Solutions

#### 1. Installation Issues

```bash
# Issue: ModuleNotFoundError for llama-index
pip install --upgrade llama-index llama-index-llms-openai llama-index-embeddings-openai

# Issue: Dependency conflicts
pip install --upgrade pip
pip install llama-index --no-deps
pip install -r requirements.txt
```

#### 2. OpenAI API Issues

```python
# Issue: API key not found
import os
print("OpenAI API Key:", os.getenv("OPENAI_API_KEY", "Not set"))

# Issue: Rate limiting
from llama_index.llms.openai import OpenAI
llm = OpenAI(
    model="gpt-3.5-turbo",
    request_timeout=60,
    max_retries=3
)
```

#### 3. Memory Issues with Large Documents

```python
# Solution: Optimize chunk size and processing
from llama_index.core.node_parser import SentenceSplitter

parser = SentenceSplitter(
    chunk_size=512,  # Smaller chunks
    chunk_overlap=10,
    paragraph_separator="\n\n"
)

# Process documents in batches
def process_large_documents(file_paths: list, batch_size: int = 10):
    for i in range(0, len(file_paths), batch_size):
        batch = file_paths[i:i + batch_size]
        # Process batch
        yield batch
```

#### 4. Performance Issues

```python
# Solution: Enable caching and optimize settings
from llama_index.core import Settings
from llama_index.core.storage.storage_context import StorageContext

# Enable caching
Settings.cache = True

# Optimize retrieval
query_engine = index.as_query_engine(
    similarity_top_k=3,  # Reduce if too slow
    response_mode="compact",  # Faster response mode
    streaming=True  # Enable streaming for large responses
)
```

#### 5. Debug Mode

```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Enable LlamaIndex debug mode
from llama_index.core import Settings
Settings.debug = True

# Test with verbose output
query_engine = index.as_query_engine(verbose=True)
```

## Next Steps

### Advanced Integration

1. **[Custom Adapters](../07-custom-adapters/creating-adapters.md)** - Build custom framework adapters
2. **[Policy Configuration](../guardrails/creating-policies.md)** - Create LlamaIndex-specific policies
3. **[Performance Optimization](../advanced-config/performance-tuning.md)** - Optimize for production workloads

### Related Frameworks

1. **[LangChain Integration](langchain.md)** - Compare with LangChain patterns
2. **[LangGraph Integration](langgraph.md)** - Workflow orchestration
3. **[Custom Frameworks](custom-frameworks.md)** - Build your own integration

### Production Deployment

1. **[Scaling Strategies](../09-deployment/scaling.md)** - Handle high-volume workloads
2. **[Monitoring Setup](../05-observability/custom-metrics.md)** - Production monitoring
3. **[Security Configuration](../advanced-config/security.md)** - Secure your deployment

---

LlamaIndex provides powerful capabilities for building RAG applications, agents, and workflows. With Rizk SDK integration, you get enterprise-grade observability, governance, and performance monitoring out of the box. The combination enables you to build production-ready AI applications with confidence.

