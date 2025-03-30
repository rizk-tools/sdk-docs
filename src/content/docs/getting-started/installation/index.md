---
title: "Installation"
description: "Documentation for Installation"
---

This guide will help you install and set up the Rizk SDK in your project.

## Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

## Installing the SDK

You can install the Rizk SDK using pip:

```bash
pip install rizk-sdk
```

## Environment Setup

The Rizk SDK requires certain environment variables to be set up. You can set them in your `.env` file or directly in your environment:

```env
RIZK_API_KEY=your_api_key_here
RIZK_OPENTELEMETRY_ENDPOINT=https://api.rizk.tools
```

## Dependencies

The Rizk SDK has the following main dependencies:

- `opentelemetry-api`
- `opentelemetry-sdk`
- `traceloop-sdk`
- `pyyaml`
- `aiohttp`

These will be automatically installed when you install the Rizk SDK.

## Verifying Installation

You can verify your installation by running a simple test:

```python
from rizk.sdk import Rizk

# Initialize the SDK
client = Rizk.init(
    app_name="my_app",
    api_key="your_api_key_here"
)

# If no exception is raised, the installation was successful
print("Rizk SDK installed successfully!")
```

## Next Steps

- [Quick Start Guide](./quickstart.md)
- [Configuration Guide](./configuration.md)
- [API Reference](../api/rizk.md) 