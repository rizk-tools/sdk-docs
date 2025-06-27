---
title: "Your First Complete Example"
description: "Your First Complete Example"
---

# Your First Complete Example

This guide walks you through building a complete LLM application with Rizk SDK monitoring and governance from scratch. You'll learn the fundamentals while building a practical customer support chatbot.

## What We're Building

We'll create a **customer support chatbot** that:
- Answers common questions using an LLM
- Has policy enforcement to ensure appropriate responses
- Includes comprehensive monitoring and tracing
- Handles errors gracefully
- Demonstrates best practices

## Prerequisites

- Python 3.10+ installed
- Basic familiarity with Python
- OpenAI API key (free tier works fine)
- Rizk API key (get one at [rizk.tools](https://rizk.tools))

## Step 1: Project Setup

### Create Project Directory

```bash
# Create project directory
mkdir customer-support-bot
cd customer-support-bot

# Create virtual environment (recommended)
python -m venv venv

# Activate virtual environment
# Windows PowerShell:
.\venv\Scripts\Activate.ps1
# Linux/macOS:
source venv/bin/activate
```

### Install Dependencies

```bash
# Install Rizk SDK with OpenAI support
pip install rizk[openai]

# Install python-dotenv for environment management
pip install python-dotenv
```

### Environment Configuration

Create a `.env` file in your project root:

```env
# .env
RIZK_API_KEY=your-rizk-api-key-here
OPENAI_API_KEY=your-openai-api-key-here
RIZK_APP_NAME=customer-support-bot
RIZK_ENABLED=true
```

## Step 2: Basic Application Structure

Create the main application file `chatbot.py`:

```python
# chatbot.py
import os
from typing import Dict, Any
from dotenv import load_dotenv
import openai
from rizk.sdk import Rizk
from rizk.sdk.decorators import workflow, task, guardrails

# Load environment variables
load_dotenv()

# Initialize OpenAI
openai.api_key = os.getenv("OPENAI_API_KEY")

# Initialize Rizk SDK
rizk = Rizk.init(
    app_name=os.getenv("RIZK_APP_NAME", "customer-support-bot"),
    api_key=os.getenv("RIZK_API_KEY"),
    enabled=True
)

print("âœ… Customer Support Bot initialized with Rizk SDK monitoring")
```

**What's happening here:**
- We load environment variables for secure API key management
- Initialize OpenAI client for LLM interactions
- Initialize Rizk SDK with our application name and API key
- The `enabled=True` parameter activates monitoring and guardrails

## Step 3: Create Core Functions

### Knowledge Base Function

```python
# Add to chatbot.py

@task(
    name="query_knowledge_base",
    organization_id="customer_support",
    project_id="chatbot_v1"
)
def query_knowledge_base(question: str) -> Dict[str, Any]:
    """
    Query our knowledge base for relevant information.
    In a real application, this would query a database or vector store.
    """
    
    # Mock knowledge base - in production, use a real database
    knowledge_base = {
        "pricing": {
            "answer": "Our basic plan starts at $29/month, with premium plans at $99/month.",
            "confidence": 0.9
        },
        "support hours": {
            "answer": "Our support team is available Monday-Friday, 9 AM to 6 PM EST.",
            "confidence": 0.95
        },
        "refund policy": {
            "answer": "We offer a 30-day money-back guarantee for all plans.",
            "confidence": 0.9
        },
        "technical issues": {
            "answer": "For technical issues, please check our status page or contact technical support.",
            "confidence": 0.8
        }
    }
    
    # Simple keyword matching - in production, use semantic search
    question_lower = question.lower()
    for topic, info in knowledge_base.items():
        if topic in question_lower:
            return {
                "found": True,
                "answer": info["answer"],
                "confidence": info["confidence"],
                "source": "knowledge_base"
            }
    
    return {
        "found": False,
        "answer": None,
        "confidence": 0.0,
        "source": "knowledge_base"
    }
```

**Key Points:**
- The `@task` decorator adds monitoring to this function
- We use hierarchical organization: `organization_id` â†’ `project_id` â†’ function
- The function returns structured data for better observability
- Mock knowledge base demonstrates the pattern for real implementations

### LLM Response Function

```python
# Add to chatbot.py

@task(
    name="generate_llm_response", 
    organization_id="customer_support",
    project_id="chatbot_v1"
)
@guardrails()  # Adds policy enforcement
def generate_llm_response(question: str, context: str = None) -> str:
    """
    Generate a response using OpenAI's LLM with optional context.
    """
    
    # Build system prompt
    system_prompt = """You are a helpful customer support assistant. 
    Follow these guidelines:
    - Be professional and friendly
    - Provide accurate information
    - If you don't know something, say so and offer to escalate
    - Keep responses concise but helpful
    - Never provide financial or legal advice
    """
    
    # Build user message
    user_message = question
    if context:
        user_message = f"Context: {context}\n\nQuestion: {question}"
    
    try:
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            max_tokens=200,
            temperature=0.7
        )
        
        return response.choices[0].message.content.strip()
        
    except Exception as e:
        # Error handling with context
        error_msg = f"LLM Error: {str(e)}"
        print(f"âŒ {error_msg}")
        return "I'm sorry, I'm experiencing technical difficulties. Please try again or contact support directly."
```

**Key Points:**
- `@guardrails()` decorator adds automatic policy enforcement
- System prompt defines behavior boundaries
- Error handling provides graceful fallback
- Structured error logging helps with debugging

### Main Workflow Function

```python
# Add to chatbot.py

@workflow(
    name="handle_customer_query",
    organization_id="customer_support", 
    project_id="chatbot_v1"
)
def handle_customer_query(question: str) -> Dict[str, Any]:
    """
    Main workflow for handling customer queries.
    Combines knowledge base lookup with LLM generation.
    """
    
    print(f"ðŸ” Processing question: {question}")
    
    # Step 1: Check knowledge base first
    kb_result = query_knowledge_base(question)
    
    if kb_result["found"] and kb_result["confidence"] > 0.8:
        # High confidence knowledge base answer
        print("âœ… Found high-confidence answer in knowledge base")
        return {
            "answer": kb_result["answer"],
            "source": "knowledge_base",
            "confidence": kb_result["confidence"],
            "processing_path": "direct_kb_lookup"
        }
    
    # Step 2: Use LLM with optional context
    context = kb_result["answer"] if kb_result["found"] else None
    llm_response = generate_llm_response(question, context)
    
    print("âœ… Generated LLM response")
    return {
        "answer": llm_response,
        "source": "llm_with_context" if context else "llm_only",
        "confidence": 0.7,  # Default LLM confidence
        "processing_path": "kb_then_llm" if context else "llm_only",
        "kb_context_used": bool(context)
    }
```

**Key Points:**
- `@workflow` decorator marks this as a high-level business process
- Implements a fallback strategy: knowledge base â†’ LLM
- Returns structured metadata for observability
- Demonstrates decision-making logic that's fully traced

## Step 4: Add Interactive Interface

```python
# Add to chatbot.py

def run_interactive_chat():
    """
    Interactive chat interface for testing.
    """
    print("\n" + "="*60)
    print("ðŸ¤– Customer Support Bot (Powered by Rizk SDK)")
    print("="*60)
    print("Type your questions below. Type 'quit' to exit.")
    print("All interactions are monitored and governed by Rizk SDK.\n")
    
    while True:
        try:
            # Get user input
            question = input("You: ").strip()
            
            if question.lower() in ['quit', 'exit', 'bye']:
                print("ðŸ‘‹ Thank you for using our support bot!")
                break
                
            if not question:
                continue
            
            # Process the question
            print("Bot: Processing...")
            result = handle_customer_query(question)
            
            # Display response
            print(f"Bot: {result['answer']}")
            print(f"     (Source: {result['source']}, Confidence: {result['confidence']:.1f})")
            print()
            
        except KeyboardInterrupt:
            print("\nðŸ‘‹ Goodbye!")
            break
        except Exception as e:
            print(f"âŒ Error: {str(e)}")
            print("Please try again or contact support.\n")

if __name__ == "__main__":
    run_interactive_chat()
```

## Step 5: Add Custom Policies

Create a policies directory and custom policy file:

```bash
mkdir policies
```

Create `policies/support_policies.yaml`:

```yaml
# policies/support_policies.yaml
version: "1.0.0"
policies:
  - id: "customer_support_policy"
    name: "Customer Support Content Policy"
    domains: ["customer_support"]
    description: "Ensures customer support responses are appropriate and helpful"
    action: "allow"
    guidelines:
      - "Always maintain a professional and helpful tone"
      - "Never provide financial or legal advice"
      - "If uncertain, offer to escalate to human support"
      - "Protect customer privacy and data"
      - "Avoid making promises about specific timelines or outcomes"
    patterns:
      # Block inappropriate content
      - pattern: "(?i)(financial advice|legal advice|investment|lawsuit)"
        action: "block"
        reason: "Cannot provide financial or legal advice"
      
      # Flag for review
      - pattern: "(?i)(refund|billing|payment|charge)"
        action: "flag"
        reason: "Financial inquiry - may need human review"
```

Update your chatbot to use custom policies:

```python
# Update the Rizk initialization in chatbot.py
rizk = Rizk.init(
    app_name=os.getenv("RIZK_APP_NAME", "customer-support-bot"),
    api_key=os.getenv("RIZK_API_KEY"),
    enabled=True,
    policies_path="./policies"  # Add custom policies
)
```

## Step 6: Add Error Handling and Logging

```python
# Add to chatbot.py (after imports)
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('chatbot.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# Update the workflow function with better error handling
@workflow(
    name="handle_customer_query",
    organization_id="customer_support", 
    project_id="chatbot_v1"
)
def handle_customer_query(question: str) -> Dict[str, Any]:
    """
    Main workflow for handling customer queries with comprehensive error handling.
    """
    
    try:
        logger.info(f"Processing customer query: {question[:50]}...")
        
        # Validate input
        if not question or len(question.strip()) < 3:
            return {
                "answer": "I need a bit more information to help you. Could you please rephrase your question?",
                "source": "validation_error",
                "confidence": 1.0,
                "processing_path": "input_validation_failed"
            }
        
        # Step 1: Knowledge base lookup
        kb_result = query_knowledge_base(question)
        logger.info(f"Knowledge base lookup: {'found' if kb_result['found'] else 'not found'}")
        
        if kb_result["found"] and kb_result["confidence"] > 0.8:
            logger.info("Using high-confidence knowledge base answer")
            return {
                "answer": kb_result["answer"],
                "source": "knowledge_base",
                "confidence": kb_result["confidence"],
                "processing_path": "direct_kb_lookup"
            }
        
        # Step 2: LLM generation
        context = kb_result["answer"] if kb_result["found"] else None
        logger.info(f"Generating LLM response with context: {bool(context)}")
        
        llm_response = generate_llm_response(question, context)
        
        logger.info("Successfully generated response")
        return {
            "answer": llm_response,
            "source": "llm_with_context" if context else "llm_only",
            "confidence": 0.7,
            "processing_path": "kb_then_llm" if context else "llm_only",
            "kb_context_used": bool(context)
        }
        
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}", exc_info=True)
        return {
            "answer": "I apologize, but I'm experiencing technical difficulties. Please try again in a moment or contact our support team directly.",
            "source": "error_fallback",
            "confidence": 0.0,
            "processing_path": "error_handling",
            "error": str(e)
        }
```

## Step 7: Test Your Application

Run your chatbot:

```bash
python chatbot.py
```

Try these test queries:

1. **Knowledge Base Query**: "What are your pricing plans?"
2. **General Question**: "How do I reset my password?"
3. **Policy Test**: "Can you give me financial advice?" (should be blocked)
4. **Edge Case**: "" (empty input)

Expected output:
```
âœ… Customer Support Bot initialized with Rizk SDK monitoring
============================================================
ðŸ¤– Customer Support Bot (Powered by Rizk SDK)
============================================================
Type your questions below. Type 'quit' to exit.
All interactions are monitored and governed by Rizk SDK.

You: What are your pricing plans?
Bot: Processing...
ðŸ” Processing question: What are your pricing plans?
âœ… Found high-confidence answer in knowledge base
Bot: Our basic plan starts at $29/month, with premium plans at $99/month.
     (Source: knowledge_base, Confidence: 0.9)

You: How do I reset my password?
Bot: Processing...
ðŸ” Processing question: How do I reset my password?
âœ… Generated LLM response
Bot: To reset your password, you can usually find a "Forgot Password" link on the login page. Click that link, enter your email address, and you'll receive instructions to reset your password. If you don't see the email, please check your spam folder. If you continue to have trouble, I'd be happy to escalate this to our technical support team for further assistance.
     (Source: llm_only, Confidence: 0.7)
```

## Step 8: View Your Monitoring Data

### Check Local Logs

Your application generates detailed logs:

```bash
# View application logs
cat chatbot.log

# Or follow logs in real-time
tail -f chatbot.log
```

### Rizk Dashboard

1. Visit [dashboard.rizk.tools](https://dashboard.rizk.tools)
2. Log in with your Rizk account
3. Navigate to your application: "customer-support-bot"
4. Explore:
   - **Traces**: See the complete flow of each query
   - **Metrics**: Performance and usage statistics
   - **Policies**: Guardrail enforcement events

### OpenTelemetry Data

If using a custom OTLP endpoint, check your observability platform for:
- Distributed traces showing the complete request flow
- Spans for each function call with timing and metadata
- Custom attributes like confidence scores and processing paths

## Step 9: Production Enhancements

### Configuration Management

Create `config.py`:

```python
# config.py
import os
from dataclasses import dataclass
from typing import Optional

@dataclass
class Config:
    # Rizk Configuration
    rizk_api_key: str
    rizk_app_name: str = "customer-support-bot"
    rizk_enabled: bool = True
    rizk_policies_path: str = "./policies"
    
    # OpenAI Configuration
    openai_api_key: str
    openai_model: str = "gpt-3.5-turbo"
    openai_max_tokens: int = 200
    openai_temperature: float = 0.7
    
    # Application Configuration
    log_level: str = "INFO"
    kb_confidence_threshold: float = 0.8
    
    @classmethod
    def from_env(cls) -> 'Config':
        return cls(
            rizk_api_key=os.getenv("RIZK_API_KEY", ""),
            rizk_app_name=os.getenv("RIZK_APP_NAME", "customer-support-bot"),
            rizk_enabled=os.getenv("RIZK_ENABLED", "true").lower() == "true",
            rizk_policies_path=os.getenv("RIZK_POLICIES_PATH", "./policies"),
            
            openai_api_key=os.getenv("OPENAI_API_KEY", ""),
            openai_model=os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"),
            openai_max_tokens=int(os.getenv("OPENAI_MAX_TOKENS", "200")),
            openai_temperature=float(os.getenv("OPENAI_TEMPERATURE", "0.7")),
            
            log_level=os.getenv("LOG_LEVEL", "INFO"),
            kb_confidence_threshold=float(os.getenv("KB_CONFIDENCE_THRESHOLD", "0.8"))
        )
    
    def validate(self) -> list[str]:
        """Validate configuration and return list of errors."""
        errors = []
        
        if not self.rizk_api_key:
            errors.append("RIZK_API_KEY is required")
        
        if not self.openai_api_key:
            errors.append("OPENAI_API_KEY is required")
            
        if not 0 <= self.kb_confidence_threshold <= 1:
            errors.append("KB_CONFIDENCE_THRESHOLD must be between 0 and 1")
            
        return errors
```

### Health Check Endpoint

For production deployments, add a health check:

```python
# health_check.py
from rizk.sdk import Rizk
from config import Config

def health_check() -> dict:
    """Health check endpoint for production monitoring."""
    
    config = Config.from_env()
    
    # Validate configuration
    config_errors = config.validate()
    if config_errors:
        return {
            "status": "unhealthy",
            "errors": config_errors,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    # Check Rizk SDK status
    try:
        rizk_status = "initialized" if Rizk._instance else "not_initialized"
    except:
        rizk_status = "error"
    
    return {
        "status": "healthy",
        "services": {
            "rizk_sdk": rizk_status,
            "openai": "configured" if config.openai_api_key else "not_configured"
        },
        "config": {
            "app_name": config.rizk_app_name,
            "policies_enabled": config.rizk_enabled,
            "model": config.openai_model
        },
        "timestamp": datetime.utcnow().isoformat()
    }

if __name__ == "__main__":
    import json
    from datetime import datetime
    print(json.dumps(health_check(), indent=2))
```

## What You've Accomplished

Congratulations! You've built a production-ready customer support chatbot with:

âœ… **Complete Monitoring**: Every function call is traced and monitored  
âœ… **Policy Enforcement**: Custom guardrails prevent inappropriate responses  
âœ… **Error Handling**: Graceful fallbacks for all error scenarios  
âœ… **Structured Logging**: Comprehensive logging for debugging and monitoring  
âœ… **Configuration Management**: Environment-based configuration for different deployments  
âœ… **Knowledge Base Integration**: Hybrid approach combining structured data with LLM flexibility  
âœ… **Production Readiness**: Health checks and validation for production deployment  

## Key Learnings

### 1. Decorator Hierarchy
- `@workflow` for business processes
- `@task` for individual operations  
- `@guardrails` for policy enforcement

### 2. Observability Best Practices
- Use hierarchical organization: `organization_id` â†’ `project_id`
- Return structured data with metadata
- Include confidence scores and processing paths
- Log key decision points

### 3. Error Handling Patterns
- Validate inputs early
- Provide graceful fallbacks
- Log errors with context
- Return user-friendly error messages

### 4. Policy Design
- Layer policies by domain
- Use regex patterns for fast rules
- Include clear guidelines for LLM evaluation
- Test policies with edge cases

## Next Steps

### Immediate Improvements
1. **Add more knowledge base entries** for better coverage
2. **Implement semantic search** instead of keyword matching
3. **Add conversation context** for multi-turn conversations
4. **Create unit tests** for all functions

### Advanced Features
1. **[Streaming Responses](../observability/streaming.md)** - Real-time response streaming
2. **[Custom Adapters](../framework-integration/custom-frameworks.md)** - Integrate with other frameworks
3. **[Advanced Policies](../guardrails/creating-policies.md)** - More sophisticated governance rules
4. **[Performance Optimization](../advanced-config/performance-tuning.md)** - Scale for high traffic

### Production Deployment
1. **[Production Setup](../advanced-config/production-setup.md)** - Deploy to production
2. **[Security Hardening](../advanced-config/security.md)** - Secure your deployment
3. **[Monitoring Dashboards](../observability/dashboards.md)** - Set up comprehensive monitoring

---

**ðŸŽ‰ Great job!** You now understand the core concepts of Rizk SDK and have built a real application. The patterns you've learned here apply to any LLM application, regardless of the underlying framework.

**Need help?** Check out the [troubleshooting guide](../troubleshooting/) or join our [Discord community](https://discord.gg/rizk). 

