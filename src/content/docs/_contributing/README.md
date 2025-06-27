---
title: "Contributing to Rizk SDK"
description: "Guide for contributing to the Rizk SDK project, including development setup, code standards, and contribution workflow."
---

# Contributing to Rizk SDK

Thank you for your interest in contributing to Rizk SDK! This section will help you understand how to contribute effectively to the project.

## ðŸ“‹ Overview

Rizk SDK is an open-source project focused on providing universal LLM observability and governance. We welcome contributions in the form of:

- **Bug reports and fixes**
- **Feature requests and implementations** 
- **Documentation improvements**
- **Framework adapter contributions**
- **Example code and tutorials**
- **Performance optimizations**
- **Test coverage improvements**

## ðŸš€ Quick Start for Contributors

### 1. Development Setup

```bash
# Clone the repository
git clone https://github.com/rizk-ai/rizk-sdk.git
cd rizk-sdk

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# or
.venv\Scripts\activate  # Windows

# Install development dependencies
pip install -e ".[dev]"
```

### 2. Running Tests

```bash
# Run all tests
pytest

# Run specific test categories
pytest tests/test_guardrails/
pytest tests/test_framework_adapters_comprehensive.py
pytest tests/test_streaming_system.py

# Run with coverage
pytest --cov=rizk --cov-report=html
```

### 3. Code Quality

```bash
# Format code
black rizk/
isort rizk/

# Type checking
mypy rizk/

# Linting
flake8 rizk/
```

## ðŸ“ Contribution Areas

### 1. Framework Adapters

**Adding support for new LLM frameworks:**

```python
# Example: Adding support for a new framework
from rizk.sdk.adapters.base import BaseAdapter

class NewFrameworkAdapter(BaseAdapter):
    FRAMEWORK_NAME = "new_framework"
    
    def adapt_workflow(self, func, name=None, **kwargs):
        # Implementation for workflow adaptation
        pass
        
    def adapt_task(self, func, name=None, **kwargs):
        # Implementation for task adaptation
        pass
```

**Required for new adapters:**
- Complete BaseAdapter implementation
- Framework detection patterns
- Comprehensive test coverage
- Documentation with examples
- Performance benchmarks

### 2. LLM Client Adapters

**Adding support for new LLM providers:**

```python
# Example: Adding support for a new LLM provider
from rizk.sdk.adapters.llm_base_adapter import BaseLLMAdapter

class NewLLMAdapter(BaseLLMAdapter):
    is_available = True  # Check if dependencies are installed
    
    def patch_client(self):
        # Patch the LLM client for guardrails integration
        pass
        
    def apply_outbound_guardrails(self, result):
        # Apply outbound guardrails to LLM responses
        pass
```

### 3. Policy and Guardrails

**Contributing to policy enforcement:**

- Adding new policy types
- Improving fast rules performance
- Enhancing LLM fallback mechanisms
- Adding policy templates for specific domains

### 4. Documentation

**Documentation improvements needed:**

- Framework-specific integration guides
- Advanced configuration examples
- Performance tuning guides
- Security best practices
- API reference completeness

## ðŸ”„ Development Workflow

### 1. Issue Assignment

1. Check [GitHub Issues](https://github.com/rizk-ai/rizk-sdk/issues) for open items
2. Comment on issues you'd like to work on
3. Wait for assignment confirmation
4. Create a new issue if working on something not listed

### 2. Branch Naming

```bash
# Feature branches
git checkout -b feature/add-new-framework-adapter
git checkout -b feature/improve-streaming-performance

# Bug fix branches  
git checkout -b fix/guardrails-memory-leak
git checkout -b fix/adapter-registration-race-condition

# Documentation branches
git checkout -b docs/update-configuration-guide
git checkout -b docs/add-examples-section
```

### 3. Commit Messages

Follow the [Conventional Commits](https://www.conventionalcommits.org/) specification:

```bash
# Features
git commit -m "feat(adapters): add support for new LLM framework"
git commit -m "feat(guardrails): implement streaming guardrails validation"

# Bug fixes
git commit -m "fix(streaming): resolve memory leak in stream processor"
git commit -m "fix(config): handle missing environment variables gracefully"

# Documentation
git commit -m "docs(examples): add comprehensive LangChain examples"
git commit -m "docs(api): update decorator API documentation"

# Tests
git commit -m "test(guardrails): add comprehensive policy enforcement tests"
git commit -m "test(adapters): improve framework adapter test coverage"
```

### 4. Pull Request Process

1. **Create descriptive PR titles and descriptions**
2. **Include test coverage for new features**
3. **Update documentation for API changes**
4. **Add examples for new functionality**
5. **Ensure all CI checks pass**
6. **Request review from maintainers**

## ðŸ“‹ Code Standards

### 1. Python Code Style

- **Black** for code formatting
- **isort** for import sorting
- **Type hints** for all public APIs
- **Docstrings** for all public functions and classes
- **Error handling** with appropriate logging

### 2. Testing Requirements

- **Unit tests** for all new functionality
- **Integration tests** for framework adapters
- **Performance tests** for critical paths
- **Mock external dependencies** appropriately
- **Minimum 80% code coverage** for new code

### 3. Documentation Standards

- **Clear examples** for all features
- **API documentation** with parameters and return types
- **Configuration options** with default values
- **Error scenarios** and troubleshooting
- **Performance considerations** where relevant

## ðŸ› Reporting Issues

### Bug Reports

Include in your bug report:

1. **Rizk SDK version**
2. **Python version and platform**
3. **Framework versions** (if applicable)
4. **Minimal reproduction code**
5. **Expected vs actual behavior**
6. **Error logs and stack traces**

### Feature Requests  

Include in your feature request:

1. **Use case description**
2. **Proposed solution**
3. **Alternative solutions considered**
4. **Framework compatibility requirements**
5. **Breaking change implications**

## ðŸ† Recognition

Contributors will be recognized in:

- **README contributors section**
- **Release notes** for significant contributions
- **Special recognition** for major features or fixes

## ðŸ“ž Getting Help

- **GitHub Discussions** for general questions
- **GitHub Issues** for bug reports and feature requests
- **Email** hello@rizk.tools for urgent security issues

## ðŸ“„ License

By contributing to Rizk SDK, you agree that your contributions will be licensed under the same license as the project.

---

**Ready to contribute?** Check out our [Good First Issues](https://github.com/rizk-ai/rizk-sdk/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22) to get started! 
