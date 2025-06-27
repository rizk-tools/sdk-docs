---
title: "Installation & Setup"
description: "Installation & Setup"
---

# Installation & Setup

This guide covers installing the Rizk SDK across different environments and setting up your first integration.

## Quick Installation

For most users, the simplest installation method is using pip:

```bash
pip install rizk
```

## Installation Options

### Standard Installation

The standard installation includes the core SDK with basic functionality:

```bash
# Standard installation
pip install rizk

# With specific version
pip install rizk==0.1.0

# Upgrade to latest version
pip install --upgrade rizk
```

### Framework-Specific Installations

Install with optional dependencies for specific frameworks:

```bash
# For LangChain integration
pip install rizk[langchain]

# For CrewAI integration  
pip install rizk[crewai]

# For LlamaIndex integration
pip install rizk[llama-index]

# For all framework integrations
pip install rizk[all]
```

### Development Installation

For contributors or advanced users who want the latest features:

```bash
# Clone the repository
git clone https://github.com/rizk-tools/rizk-sdk.git
cd rizk-sdk

# Install in development mode
pip install -e .

# Install with development dependencies
pip install -e .[dev]
```

## System Requirements

### Python Version
- **Python 3.10 or higher** (tested on 3.10, 3.11, 3.12, 3.13)
- **Operating Systems**: Windows, macOS, Linux

### Core Dependencies

The SDK automatically installs these core dependencies:

- `traceloop-sdk` - OpenTelemetry integration
- `opentelemetry-*` - Observability instrumentation  
- `pydantic` - Data validation and serialization
- `colorama` - Cross-platform colored terminal output
- `pyyaml` - YAML configuration file support

### Optional Dependencies

Framework-specific dependencies (installed with extras):

```bash
# LangChain ecosystem
pip install rizk[langchain]  # Installs: langchain, langchain-openai, langchain-community

# CrewAI ecosystem  
pip install rizk[crewai]     # Installs: crewai, crewai-tools

# LlamaIndex ecosystem
pip install rizk[llama-index] # Installs: llama-index, llama-index-core

# OpenAI integration
pip install rizk[openai]     # Installs: openai>=1.0.0

# Anthropic integration
pip install rizk[anthropic]  # Installs: anthropic

# All integrations
pip install rizk[all]        # Installs all optional dependencies
```

## Environment Setup

### 1. API Key Configuration

Rizk requires an API key for operation. Set it using one of these methods:

**Environment Variable (Recommended)**:
```bash
# Windows PowerShell
$env:RIZK_API_KEY="your-api-key-here"

# Windows Command Prompt
set RIZK_API_KEY=your-api-key-here

# Linux/macOS
export RIZK_API_KEY="your-api-key-here"
```

**Configuration File**:
Create a `.env` file in your project root:
```env
RIZK_API_KEY=your-api-key-here
RIZK_APP_NAME=my-application
RIZK_ENABLED=true
```

**Programmatic Configuration**:
```python
from rizk.sdk import Rizk

rizk = Rizk.init(
    app_name="MyApp",
    api_key="your-api-key-here",  # Direct usage of API Keys not recommended for production or development
    enabled=True
)
```

### 2. OpenTelemetry Configuration (Optional)

Configure custom OpenTelemetry endpoints if needed:

```bash
# Custom OTLP endpoint
export RIZK_OPENTELEMETRY_ENDPOINT="https://your-otlp-endpoint.com"

# Disable telemetry entirely
export RIZK_TRACING_ENABLED=false
```

### 3. Policy Configuration (Optional)

Set custom policy directories:

```bash
# Custom policies directory
export RIZK_POLICIES_PATH="/path/to/your/policies"

# Disable policy enforcement
export RIZK_POLICY_ENFORCEMENT=false
```

## Installation Verification

Verify your installation with this simple test:

```python
# test_installation.py
from rizk.sdk import Rizk
from rizk.sdk.decorators import workflow

# Initialize SDK
rizk = Rizk.init(
    app_name="InstallationTest",
    api_key="test-key",
    enabled=True
)

# Test decorator
@workflow(name="test_workflow", organization_id="test", project_id="test")
def test_function():
    return "Rizk SDK is working!"

# Run test
result = test_function()
print(f"âœ… Installation successful: {result}")
```

Run the test:
```bash
python test_installation.py
```

Expected output:
```
âœ… Installation successful: Rizk SDK is working!
```

## Framework-Specific Setup

### OpenAI Agents SDK

```bash
# Install Rizk with OpenAI support
pip install rizk[openai]

# Set OpenAI API key (if using OpenAI LLMs)
export OPENAI_API_KEY="your-openai-api-key"
```

```python
from rizk.sdk import Rizk
from rizk.sdk.decorators import tool, workflow
from agents import Agent, Runner

# Initialize Rizk
rizk = Rizk.init(app_name="OpenAI-Agents-App", enabled=True)

@tool(name="calculator")
def calculate(expression: str) -> str:
    return str(eval(expression))

agent = Agent(
    name="MathBot",
    instructions="You are a helpful math assistant",
    tools=[calculate]
)

@workflow(name="math_workflow")
async def run_agent(query: str):
    result = await Runner.run(agent, query)
    return result.final_output
```

### LangChain

```bash
# Install Rizk with LangChain support
pip install rizk[langchain]
```

```python
from rizk.sdk import Rizk
from rizk.sdk.decorators import workflow, tool
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_openai import ChatOpenAI

# Initialize Rizk
rizk = Rizk.init(app_name="LangChain-App", enabled=True)

@tool(name="weather_tool")
def get_weather(city: str) -> str:
    return f"Weather in {city}: Sunny, 75Â°F"

llm = ChatOpenAI()
tools = [get_weather]
agent = create_openai_tools_agent(llm, tools, "You are a weather assistant")
agent_executor = AgentExecutor(agent=agent, tools=tools)

@workflow(name="weather_workflow")
def run_weather_agent(query: str):
    return agent_executor.invoke({"input": query})["output"]
```

### CrewAI

```bash
# Install Rizk with CrewAI support
pip install rizk[crewai]
```

```python
from rizk.sdk import Rizk
from rizk.sdk.decorators import crew, task, agent
from crewai import Agent, Task, Crew, Process

# Initialize Rizk
rizk = Rizk.init(app_name="CrewAI-App", enabled=True)

@agent(name="researcher")
def create_researcher():
    return Agent(
        role="Researcher",
        goal="Research topics thoroughly",
        backstory="You are an expert researcher",
        verbose=True
    )

@task(name="research_task")
def create_research_task(agent, topic):
    return Task(
        description=f"Research {topic} comprehensively",
        agent=agent,
        expected_output="A detailed research report"
    )

@crew(name="research_crew")
def create_crew():
    researcher = create_researcher()
    task = create_research_task(researcher, "AI trends")
    
    return Crew(
        agents=[researcher],
        tasks=[task],
        process=Process.sequential,
        verbose=True
    )
```

## Virtual Environment Setup

For production deployments, use virtual environments:

### Using venv (Python standard library)

```bash
# Create virtual environment
python -m venv rizk-env

# Activate (Windows)
rizk-env\Scripts\activate

# Activate (Linux/macOS)
source rizk-env/bin/activate

# Install Rizk
pip install rizk

# Deactivate when done
deactivate
```

### Using conda

```bash
# Create conda environment
conda create -n rizk-env python=3.11

# Activate environment
conda activate rizk-env

# Install Rizk
pip install rizk

# Deactivate when done
conda deactivate
```

### Using Poetry (Recommended for Development)

```bash
# Initialize Poetry project
poetry init

# Add Rizk dependency
poetry add rizk

# Install dependencies
poetry install

# Run in Poetry environment
poetry run python your_script.py
```

Example `pyproject.toml`:
```toml
[tool.poetry]
name = "my-rizk-app"
version = "0.1.0"
description = "My application using Rizk SDK"

[tool.poetry.dependencies]
python = "^3.10"
rizk = "^0.1.0"
langchain = { version = "^0.1.0", optional = true }
crewai = { version = "^0.1.0", optional = true }

[tool.poetry.extras]
langchain = ["langchain"]
crewai = ["crewai"]
all = ["langchain", "crewai"]
```

## Docker Setup

For containerized deployments:

### Basic Dockerfile

```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Set environment variables
ENV RIZK_API_KEY=""
ENV RIZK_APP_NAME="dockerized-app"

# Run application
CMD ["python", "main.py"]
```

### requirements.txt

```txt
rizk==0.1.0
# Add other dependencies as needed
langchain>=0.1.0
openai>=1.0.0
```

### Docker Compose

```yaml
# docker-compose.yml
version: '3.8'

services:
  rizk-app:
    build: .
    environment:
      - RIZK_API_KEY=${RIZK_API_KEY}
      - RIZK_APP_NAME=my-dockerized-app
      - RIZK_TRACING_ENABLED=true
    ports:
      - "8000:8000"
    volumes:
      - ./policies:/app/policies:ro  # Mount custom policies
```

## Common Installation Issues

### Issue 1: Python Version Compatibility

**Error**: `ERROR: Package 'rizk' requires a different Python`

**Solution**: Ensure you're using Python 3.10 or higher:
```bash
python --version  # Should be 3.10+
pip --version     # Should use the correct Python version
```

### Issue 2: Dependency Conflicts

**Error**: `ERROR: pip's dependency resolver does not currently have a backtracking strategy`

**Solutions**:
```bash
# Option 1: Upgrade pip
pip install --upgrade pip

# Option 2: Use --force-reinstall
pip install --force-reinstall rizk

# Option 3: Create fresh virtual environment
python -m venv fresh-env
fresh-env\Scripts\activate  # Windows
pip install rizk
```

### Issue 3: Network/Proxy Issues

**Error**: `ERROR: Could not fetch URL`

**Solutions**:
```bash
# Configure proxy (if needed)
pip install --proxy http://proxy.company.com:8080 rizk

# Use trusted hosts (if SSL issues)
pip install --trusted-host pypi.org --trusted-host pypi.python.org rizk

# Upgrade certificates
pip install --upgrade certifi
```

### Issue 4: Permission Issues

**Error**: `ERROR: Could not install packages due to an EnvironmentError: [Errno 13] Permission denied`

**Solutions**:
```bash
# Option 1: Install for user only
pip install --user rizk

# Option 2: Use virtual environment (recommended)
python -m venv rizk-env
rizk-env\Scripts\activate
pip install rizk
```

## Next Steps

After installation, proceed to:

1. **[Quick Start Guide](quickstart.md)** - Get up and running in 5 minutes
2. **[First Example](first-example.md)** - Complete walkthrough with working code  
3. **[Configuration](configuration.md)** - Detailed configuration options

## Getting Help

If you encounter issues:

1. **Check the [Troubleshooting Guide](../troubleshooting/common-issues.md)**
2. **Search [GitHub Issues](https://github.com/rizk-ai/rizk-sdk/issues)**
3. **Join our [Discord Community](https://discord.gg/rizk)**
4. **Contact [Support](mailto:hello@rizk.tools)**

---

**Supported Python Versions**: 3.10, 3.11, 3.12  
**Supported Platforms**: Windows, macOS, Linux  
**Installation Size**: ~50MB (core), ~200MB (with all extras) 

