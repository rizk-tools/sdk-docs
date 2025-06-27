---
title: "Security Configuration"
description: "Security Configuration"
---

# Security Configuration

Comprehensive security configuration for Rizk SDK in production environments. This guide covers API key management, content privacy, network security, and compliance considerations.

## API Key Management

### Best Practices

**Never hardcode API keys** in your source code:

```python
# âŒ DON'T: Hardcode API keys
rizk = Rizk.init(
    app_name="MyApp",
    api_key="rizk_live_abc123..."  # Never do this!
)

# âœ… DO: Use environment variables
rizk = Rizk.init(
    app_name="MyApp",
    api_key=os.getenv("RIZK_API_KEY")
)
```

### Environment Variable Configuration

```bash
# Development
export RIZK_API_KEY="rizk_dev_..."

# Production
export RIZK_API_KEY="rizk_live_..."

# Use different keys for different environments
export RIZK_API_KEY_DEV="rizk_dev_..."
export RIZK_API_KEY_STAGING="rizk_staging_..."
export RIZK_API_KEY_PROD="rizk_live_..."
```

### Docker Secrets

For containerized deployments, use Docker secrets:

```dockerfile
# Dockerfile
FROM python:3.11-slim

# Install dependencies
RUN pip install rizk

# Create non-root user
RUN groupadd -r rizk && useradd -r -g rizk rizk
USER rizk

# Use secrets for API keys
CMD ["python", "app.py"]
```

```yaml
# docker-compose.yml
version: '3.8'
services:
  app:
    build: .
    secrets:
      - rizk_api_key
    environment:
      - RIZK_API_KEY_FILE=/run/secrets/rizk_api_key

secrets:
  rizk_api_key:
    file: ./secrets/rizk_api_key.txt
```

### Kubernetes Secrets

```yaml
# Create secret
apiVersion: v1
kind: Secret
metadata:
  name: rizk-secrets
type: Opaque
stringData:
  api-key: "rizk_live_your_api_key_here"
---
# Use in deployment
apiVersion: apps/v1
kind: Deployment
metadata:
  name: rizk-app
spec:
  template:
    spec:
      containers:
      - name: app
        env:
        - name: RIZK_API_KEY
          valueFrom:
            secretKeyRef:
              name: rizk-secrets
              key: api-key
```

### Key Rotation

Implement automatic key rotation:

```python
import os
import time
from datetime import datetime, timedelta

class RizkKeyManager:
    def __init__(self):
        self.current_key = os.getenv("RIZK_API_KEY")
        self.backup_key = os.getenv("RIZK_API_KEY_BACKUP")
        self.last_rotation = datetime.now()
        
    def should_rotate(self) -> bool:
        """Check if key should be rotated (every 30 days)"""
        return datetime.now() - self.last_rotation > timedelta(days=30)
    
    def rotate_key(self):
        """Rotate to backup key and request new primary"""
        if self.backup_key:
            # Switch to backup key
            os.environ["RIZK_API_KEY"] = self.backup_key
            self.current_key = self.backup_key
            
            # Reinitialize Rizk with new key
            Rizk.init(app_name="MyApp", api_key=self.current_key)
            
            self.last_rotation = datetime.now()
            print("âœ… API key rotated successfully")
```

## Content Privacy

### Disable Content Tracing

For sensitive applications, you could disable content tracing:

```python
# Disable content in traces
rizk = Rizk.init(
    app_name="SensitiveApp",
    api_key=os.getenv("RIZK_API_KEY"),
    trace_content=False  # Don't send content to telemetry
)
```
Please reach out to the Rizk team for local deployments.

```bash
# Via environment variables
export RIZK_TRACE_CONTENT=false
```

### Content Filtering

Implement content filtering before sending to Rizk:

```python
import re
from typing import Any, Dict

class ContentFilter:
    """Filter sensitive content before telemetry"""
    
    SENSITIVE_PATTERNS = [
        r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',  # Credit cards
        r'\b\d{3}-\d{2}-\d{4}\b',  # SSN
        r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Email
        r'\b(?:\d{1,3}\.){3}\d{1,3}\b',  # IP addresses
    ]
    
    @classmethod
    def filter_content(cls, content: str) -> str:
        """Remove sensitive patterns from content"""
        filtered = content
        for pattern in cls.SENSITIVE_PATTERNS:
            filtered = re.sub(pattern, '[REDACTED]', filtered, flags=re.IGNORECASE)
        return filtered
    
    @classmethod
    def filter_attributes(cls, attributes: Dict[str, Any]) -> Dict[str, Any]:
        """Filter sensitive data from span attributes"""
        filtered = {}
        for key, value in attributes.items():
            if isinstance(value, str):
                filtered[key] = cls.filter_content(value)
            else:
                filtered[key] = value
        return filtered

# Use with custom span processor
from opentelemetry.sdk.trace.export import SpanProcessor

class FilteringSpanProcessor(SpanProcessor):
    def __init__(self, wrapped_processor: SpanProcessor):
        self.wrapped = wrapped_processor
    
    def on_start(self, span, parent_context=None):
        # Filter attributes before processing
        if hasattr(span, 'attributes'):
            filtered_attrs = ContentFilter.filter_attributes(span.attributes)
            span.attributes.clear()
            span.attributes.update(filtered_attrs)
        
        self.wrapped.on_start(span, parent_context)
```

### Local-Only Mode

For maximum privacy, run Rizk in local-only mode:

```python
# Disable all external telemetry
rizk = Rizk.init(
    app_name="LocalOnlyApp",
    api_key=None,  # No API key = local only
    opentelemetry_endpoint=None,  # No external endpoint
    telemetry_enabled=False,  # Disable telemetry
    enabled=True  # Keep local tracing for debugging
)
```

## Network Security

### TLS Configuration

Ensure all connections use TLS:

```python
# Standard secure configuration (uses https://api.rizk.tools by default)
rizk = Rizk.init(
    app_name="SecureApp",
    api_key=os.getenv("RIZK_API_KEY"),
    headers={
        "User-Agent": "MySecureApp/1.0",
        "X-Request-ID": "unique-request-id"
    }
)
```

> **Note**: All connections to Rizk's endpoint (https://api.rizk.tools) use HTTPS by default. To force HTTPS for a custom endpoint, set `opentelemetry_endpoint="https://your-secure-endpoint.com"`.

### Proxy Configuration

Configure proxy settings for corporate environments:

```python
import os

# Set proxy environment variables
os.environ["HTTPS_PROXY"] = "https://corporate-proxy:8080"
os.environ["HTTP_PROXY"] = "http://corporate-proxy:8080"
os.environ["NO_PROXY"] = "localhost,127.0.0.1,.internal.domain"

# Initialize Rizk (will use proxy settings)
rizk = Rizk.init(
    app_name="CorporateApp",
    api_key=os.getenv("RIZK_API_KEY")
)
```

### Certificate Validation

For custom certificates:

```python
import ssl
import certifi
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

# Custom SSL context
ssl_context = ssl.create_default_context(cafile=certifi.where())
ssl_context.check_hostname = True
ssl_context.verify_mode = ssl.CERT_REQUIRED

# Custom exporter with SSL
custom_exporter = OTLPSpanExporter(
    endpoint="https://api.rizk.tools:443",
    credentials=ssl_context,
    headers=(("authorization", f"Bearer {os.getenv('RIZK_API_KEY')}"),)
)

rizk = Rizk.init(
    app_name="SecureApp",
    exporter=custom_exporter
)
```

## Access Control

### Role-Based Configuration

Different configurations for different roles:

```python
import os
from enum import Enum

class UserRole(Enum):
    ADMIN = "admin"
    DEVELOPER = "developer"
    READ_ONLY = "readonly"

def get_rizk_config(role: UserRole) -> dict:
    """Get Rizk configuration based on user role"""
    base_config = {
        "app_name": "RoleBasedApp",
        "api_key": os.getenv("RIZK_API_KEY")
    }
    
    if role == UserRole.ADMIN:
        return {
            **base_config,
            "trace_content": True,  # Full access
            "policies_path": "./admin_policies",
            "debug_mode": True
        }
    elif role == UserRole.DEVELOPER:
        return {
            **base_config,
            "trace_content": True,  # Limited access
            "policies_path": "./dev_policies",
            "debug_mode": False
        }
    else:  # READ_ONLY
        return {
            **base_config,
            "trace_content": False,  # No content access
            "telemetry_enabled": False,
            "policies_path": "./readonly_policies"
        }

# Usage
user_role = UserRole(os.getenv("USER_ROLE", "readonly"))
config = get_rizk_config(user_role)
rizk = Rizk.init(**config)
```

### IP Allowlisting

For highly secure environments:

```python
import socket
from typing import List

class NetworkSecurity:
    ALLOWED_IPS = [
        "192.168.1.0/24",  # Internal network
        "10.0.0.0/8",      # Corporate network
        "203.0.113.0/24"   # Specific allowed range
    ]
    
    @classmethod
    def is_allowed_ip(cls, ip: str) -> bool:
        """Check if IP is in allowed ranges"""
        # Implement IP range checking logic
        # This is a simplified example
        return any(ip.startswith(allowed.split('/')[0][:7]) 
                  for allowed in cls.ALLOWED_IPS)
    
    @classmethod
    def get_current_ip(cls) -> str:
        """Get current machine IP"""
        hostname = socket.gethostname()
        return socket.gethostbyname(hostname)
    
    @classmethod
    def check_network_access(cls) -> bool:
        """Verify network access before initializing Rizk"""
        current_ip = cls.get_current_ip()
        if not cls.is_allowed_ip(current_ip):
            print(f"âŒ Network access denied from IP: {current_ip}")
            return False
        return True

# Use before Rizk initialization
if NetworkSecurity.check_network_access():
    rizk = Rizk.init(app_name="SecureApp")
else:
    # Fall back to local-only mode
    rizk = Rizk.init(app_name="SecureApp", enabled=False)
```

## Compliance and Audit

### Audit Logging

Enable comprehensive audit logging:

```python
import logging
import json
from datetime import datetime

# Configure audit logger
audit_logger = logging.getLogger("rizk.audit")
audit_handler = logging.FileHandler("rizk_audit.log")
audit_formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
audit_handler.setFormatter(audit_formatter)
audit_logger.addHandler(audit_handler)
audit_logger.setLevel(logging.INFO)

class AuditLogger:
    @staticmethod
    def log_initialization(app_name: str, user_id: str = None):
        """Log Rizk initialization"""
        audit_logger.info(json.dumps({
            "event": "rizk_init",
            "app_name": app_name,
            "user_id": user_id,
            "timestamp": datetime.now().isoformat(),
            "ip_address": NetworkSecurity.get_current_ip()
        }))
    
    @staticmethod
    def log_policy_violation(violation_details: dict):
        """Log policy violations"""
        audit_logger.warning(json.dumps({
            "event": "policy_violation",
            "details": violation_details,
            "timestamp": datetime.now().isoformat()
        }))
    
    @staticmethod
    def log_api_key_usage(key_prefix: str):
        """Log API key usage (without exposing full key)"""
        audit_logger.info(json.dumps({
            "event": "api_key_used",
            "key_prefix": key_prefix[:10] + "...",  # Only log prefix
            "timestamp": datetime.now().isoformat()
        }))

# Use audit logging
api_key = os.getenv("RIZK_API_KEY")
if api_key:
    AuditLogger.log_api_key_usage(api_key)

AuditLogger.log_initialization("MyApp", user_id="user123")
rizk = Rizk.init(app_name="MyApp")
```

### Data Retention Policies

Configure data retention:

```python
from datetime import datetime, timedelta

class DataRetentionPolicy:
    # Retention periods (in days)
    TRACE_RETENTION = 30
    AUDIT_LOG_RETENTION = 365
    POLICY_LOG_RETENTION = 90
    
    @classmethod
    def should_purge_traces(cls, trace_date: datetime) -> bool:
        """Check if traces should be purged"""
        cutoff = datetime.now() - timedelta(days=cls.TRACE_RETENTION)
        return trace_date < cutoff
    
    @classmethod
    def get_retention_config(cls) -> dict:
        """Get retention configuration for Rizk"""
        return {
            "trace_retention_days": cls.TRACE_RETENTION,
            "audit_retention_days": cls.AUDIT_LOG_RETENTION,
            "policy_retention_days": cls.POLICY_LOG_RETENTION
        }

# Apply retention policy
retention_config = DataRetentionPolicy.get_retention_config()
rizk = Rizk.init(
    app_name="ComplianceApp",
    resource_attributes={
        "data.retention.traces": str(retention_config["trace_retention_days"]),
        "data.retention.audit": str(retention_config["audit_retention_days"])
    }
)
```

## Security Checklist

### Pre-Production Checklist

- [ ] **API Keys**: No hardcoded keys in source code
- [ ] **Environment Variables**: API keys set via secure env vars
- [ ] **Content Tracing**: Disabled for sensitive applications
- [ ] **TLS**: All connections use HTTPS
- [ ] **Network**: Proxy and firewall configured
- [ ] **Access Control**: Role-based permissions implemented
- [ ] **Audit Logging**: Comprehensive logging enabled
- [ ] **Data Retention**: Retention policies configured
- [ ] **Key Rotation**: Rotation strategy implemented
- [ ] **Monitoring**: Security alerts configured

### Runtime Security Monitoring

```python
import time
from collections import defaultdict

class SecurityMonitor:
    def __init__(self):
        self.failed_attempts = defaultdict(int)
        self.last_attempt = defaultdict(float)
    
    def check_rate_limit(self, identifier: str, max_attempts: int = 10, 
                        window_seconds: int = 300) -> bool:
        """Simple rate limiting"""
        now = time.time()
        
        # Reset counter if window expired
        if now - self.last_attempt[identifier] > window_seconds:
            self.failed_attempts[identifier] = 0
        
        self.last_attempt[identifier] = now
        
        if self.failed_attempts[identifier] >= max_attempts:
            AuditLogger.log_policy_violation({
                "type": "rate_limit_exceeded",
                "identifier": identifier,
                "attempts": self.failed_attempts[identifier]
            })
            return False
        
        return True
    
    def record_failure(self, identifier: str):
        """Record failed attempt"""
        self.failed_attempts[identifier] += 1

# Use security monitor
security_monitor = SecurityMonitor()

def secure_rizk_init(app_name: str, user_id: str = None):
    """Securely initialize Rizk with monitoring"""
    identifier = user_id or "anonymous"
    
    if not security_monitor.check_rate_limit(identifier):
        raise SecurityError("Rate limit exceeded")
    
    try:
        rizk = Rizk.init(app_name=app_name)
        AuditLogger.log_initialization(app_name, user_id)
        return rizk
    except Exception as e:
        security_monitor.record_failure(identifier)
        AuditLogger.log_policy_violation({
            "type": "initialization_failure",
            "error": str(e),
            "user_id": user_id
        })
        raise
```

## Emergency Procedures

### Security Incident Response

```python
class SecurityIncidentResponse:
    @staticmethod
    def disable_rizk_immediately():
        """Emergency shutdown of Rizk functionality"""
        os.environ["RIZK_ENABLED"] = "false"
        os.environ["RIZK_TRACING_ENABLED"] = "false"
        os.environ["RIZK_TELEMETRY_ENABLED"] = "false"
        
        print("ðŸš¨ SECURITY ALERT: Rizk functionality disabled")
        AuditLogger.log_policy_violation({
            "type": "emergency_shutdown",
            "reason": "security_incident",
            "timestamp": datetime.now().isoformat()
        })
    
    @staticmethod
    def revoke_api_key():
        """Revoke current API key"""
        if "RIZK_API_KEY" in os.environ:
            del os.environ["RIZK_API_KEY"]
        
        print("ðŸ”‘ API key revoked - contact security team for new key")
    
    @staticmethod
    def switch_to_local_mode():
        """Switch to local-only mode"""
        os.environ["RIZK_API_KEY"] = ""
        os.environ["RIZK_OPENTELEMETRY_ENDPOINT"] = ""
        
        # Reinitialize in local mode
        rizk = Rizk.init(
            app_name="EmergencyMode",
            api_key=None,
            enabled=True  # Keep local functionality
        )
        
        print("ðŸ  Switched to local-only mode")
        return rizk

# Emergency hotkey (development only)
def emergency_shutdown():
    """Emergency shutdown procedure"""
    SecurityIncidentResponse.disable_rizk_immediately()
    SecurityIncidentResponse.revoke_api_key()
    print("âœ… Emergency shutdown complete")
```

---

**âš ï¸ Important**: Always test security configurations in a non-production environment first. Security requirements vary by organization - adapt these examples to your specific needs and compliance requirements.

**ðŸ”’ Remember**: Security is a shared responsibility. While Rizk provides secure defaults and configuration options, you must implement appropriate security measures for your specific environment and use case. 

