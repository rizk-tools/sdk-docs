---
title: "Getting Started with Rizk SDK"
description: "Getting Started with Rizk SDK"
---

# Getting Started with Rizk SDK

**The Universal LLM Observability & Governance Platform**

Rizk SDK is the industry-standard solution for LLM observability, tracing, and governance. Trusted by leading AI teams, Rizk provides comprehensive monitoring and policy enforcement across any LLM framework with zero configuration required.

## Why Rizk SDK?

### Universal Framework Integration
Unlike point solutions that require framework-specific implementations, Rizk SDK provides a single, unified API that works seamlessly across:

- **OpenAI Agents** - Native integration with OpenAI's agent framework
- **LangChain** - Full ecosystem support including LangGraph and LangSmith
- **CrewAI** - Multi-agent workflow monitoring and governance
- **LlamaIndex** - RAG application observability and policy enforcement
- **Custom Frameworks** - Extensible adapter system for any LLM implementation

### Enterprise-Grade Observability
Built on OpenTelemetry standards, Rizk SDK delivers production-ready observability:

```python
from rizk.sdk import Rizk
from rizk.sdk.decorators import workflow, guardrails

# Initialize once, monitor everything
rizk = Rizk.init(app_name="production-app", enabled=True)

@workflow(name="user_interaction", organization_id="acme", project_id="ai-assistant")
@guardrails()
def handle_user_request(request: str) -> str:
    # Your LLM logic - automatically monitored and governed
    return llm_response
```

**Result**: Complete visibility into your LLM operations with distributed tracing, performance metrics, and policy enforcement.

### Multi-Layer Governance
Rizk's policy enforcement system provides comprehensive governance through three evaluation layers:

1. **Fast Rules Engine** - Regex-based pattern matching for immediate policy decisions
2. **Policy Augmentation** - Context-aware prompt enhancement with compliance guidelines  
3. **LLM Fallback** - Advanced LLM-based evaluation for complex policy scenarios

## Quick Start Paths

### âš¡ [5-Minute Quick Start](quickstart.md)
Get Rizk SDK running in your application immediately. Perfect for evaluation and proof-of-concept implementations.

**Ideal for**: Initial evaluation, demos, rapid prototyping

### ðŸ“¦ [Complete Installation Guide](installation.md)
Comprehensive setup covering production environments, framework-specific configurations, and enterprise deployment patterns.

**Ideal for**: Production deployments, enterprise environments, complex integrations

### ðŸ‘¨â€ðŸ’» [End-to-End Tutorial](first-example.md)
Build a complete LLM application with monitoring and governance from scratch. Includes best practices, error handling, and production considerations.

**Ideal for**: Learning core concepts, understanding best practices, building production applications

### âš™ï¸ [Configuration Reference](configuration.md)
Essential configuration patterns for different deployment scenarios, from development to enterprise production.

**Ideal for**: DevOps teams, production deployments, compliance requirements

## Architecture Overview

Rizk SDK implements a clean, layered architecture that integrates seamlessly with your existing infrastructure:

```
â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
â”‚                    Your LLM Application                        â”‚
â”‚            (Any Framework, Any LLM Provider)                   â”‚
â”œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¤
â”‚                       Rizk SDK Layer                          â”‚
â”‚  â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”  â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”  â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â” â”‚
â”‚  â”‚   Decorators    â”‚  â”‚   Guardrails    â”‚  â”‚  Observability  â”‚ â”‚
â”‚  â”‚   Universal     â”‚  â”‚     Engine      â”‚  â”‚    & Tracing    â”‚ â”‚
â”‚  â”‚   Monitoring    â”‚  â”‚   Multi-Layer   â”‚  â”‚   OpenTelemetry â”‚ â”‚
â”‚  â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜  â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜  â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜ â”‚
â”œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¤
â”‚              Auto-Detected Framework Adapters                  â”‚
â”‚  â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â” â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â” â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â” â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â” â”‚
â”‚  â”‚   OpenAI    â”‚ â”‚ LangChain   â”‚ â”‚   CrewAI    â”‚ â”‚ LlamaIndexâ”‚ â”‚
â”‚  â”‚   Agents    â”‚ â”‚ Ecosystem   â”‚ â”‚ Multi-Agent â”‚ â”‚    RAG    â”‚ â”‚
â”‚  â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜ â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜ â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜ â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜ â”‚
â”œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¤
â”‚                  OpenTelemetry Foundation                      â”‚
â”‚              (Industry Standard Observability)                â”‚
â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
```

## Key Capabilities

### Automatic Framework Detection
Rizk SDK automatically detects and configures appropriate adapters for your LLM framework:

```python
# Works with any framework - no manual configuration required
@workflow(name="langchain_workflow")
def process_with_langchain():
    # LangChain implementation
    return agent_executor.invoke({"input": query})

@workflow(name="crewai_workflow") 
def process_with_crewai():
    # CrewAI implementation
    return crew.kickoff({"topic": topic})
```

### Hierarchical Context Management
Organize your LLM operations with enterprise-grade context hierarchy:

```
Organization â†’ Project â†’ Agent â†’ Conversation â†’ User
```

This structure enables:
- **Cost Attribution** - Track spending by business unit
- **Performance Analysis** - Identify bottlenecks across teams
- **Compliance Reporting** - Generate audit trails by organization
- **Access Control** - Implement role-based policy enforcement

### Production-Ready Governance
Implement comprehensive AI governance with minimal code changes:

```python
@workflow(name="financial_advisor")
@guardrails()  # Automatic compliance enforcement
def provide_financial_guidance(query: str) -> str:
    # Automatic policy enforcement for financial regulations
    return financial_llm_response(query)
```

## Enterprise Use Cases

### Financial Services
- **Compliance Monitoring** - Ensure responses meet regulatory requirements
- **Risk Management** - Detect and prevent inappropriate financial advice
- **Audit Trails** - Complete traceability for regulatory reporting

### Healthcare
- **HIPAA Compliance** - Protect patient data in LLM interactions
- **Medical Accuracy** - Prevent medical advice from general-purpose models
- **Access Control** - Role-based policy enforcement

### Legal Technology
- **Confidentiality Protection** - Prevent disclosure of sensitive information
- **Professional Standards** - Ensure responses meet legal professional standards
- **Client Attribution** - Track usage and costs by client matter

## Integration Patterns

### Microservices Architecture
```python
# Service A - Customer Support
@workflow(name="customer_support", organization_id="support", project_id="chatbot")
@guardrails()
def handle_support_request(request: str) -> str:
    return support_response(request)

# Service B - Sales Assistant  
@workflow(name="sales_assistant", organization_id="sales", project_id="lead_qualification")
@guardrails()
def qualify_lead(conversation: str) -> Dict:
    return lead_analysis(conversation)
```

### Event-Driven Systems
```python
@workflow(name="process_user_event")
@guardrails()
async def handle_user_event(event: UserEvent) -> None:
    # Async processing with full observability
    response = await llm_processor.process(event.content)
    await event_bus.publish(ProcessedEvent(response))
```

### Batch Processing
```python
@workflow(name="batch_document_processing")
@guardrails()
def process_document_batch(documents: List[Document]) -> List[ProcessedDocument]:
    # Batch processing with individual item tracing
    return [process_document(doc) for doc in documents]
```

## Performance & Scale

### Benchmarked Performance
- **Latency Overhead**: < 2ms per operation
- **Memory Footprint**: < 10MB baseline
- **Throughput**: Tested at 10,000+ requests/second
- **Concurrency**: Full async/await support

### Scalability Features
- **Distributed Caching** - Redis integration for multi-instance deployments
- **Load Balancing** - Stateless design for horizontal scaling
- **Resource Management** - Configurable resource limits and throttling

## Next Steps

Choose your integration path based on your needs:

| Use Case | Recommended Path | Time Investment |
|----------|------------------|-----------------|
| **Evaluation** | [Quick Start](quickstart.md) | 5 minutes |
| **Development** | [Complete Tutorial](first-example.md) | 30 minutes |
| **Production** | [Installation Guide](installation.md) + [Configuration](configuration.md) | 2 hours |
| **Enterprise** | All guides + [Advanced Configuration](../advanced-config/) | 1 day |

### Advanced Topics
After completing the getting started guides, explore:

1. **[Core Concepts](../core-concepts/)** - Deep dive into SDK architecture
2. **[Framework Integration](../framework-integration/)** - Framework-specific patterns
3. **[Policy Engineering](../guardrails/)** - Advanced governance strategies
4. **[Production Deployment](../advanced-config/)** - Enterprise deployment patterns

## Support & Community

### Enterprise Support
- **Technical Support** - Direct access to engineering team
- **Implementation Services** - Guided integration and optimization
- **Custom Adapters** - Framework-specific integration development
- **Compliance Consulting** - Regulatory requirement implementation

### Community Resources
- **Documentation** - Comprehensive guides and API reference
- **GitHub** - Open source contributions and issue tracking
- **Discord** - Real-time community support and discussions
- **Blog** - Best practices and case studies

---

**Ready to implement world-class LLM observability?**

[Get Started in 5 Minutes â†’](quickstart.md)

*Rizk SDK - Trusted by leading AI teams worldwide* 

