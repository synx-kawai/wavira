---
name: security-review
description: Use this skill when reviewing code for security issues, checking for credential leaks, validating input handling, or auditing dependencies. Automatically invoked for PRs with security-sensitive changes.
---

# Security Review Skill

This skill provides security analysis for the Wavira project.

## Security Checklist

### 1. Credential Management

- [ ] No hardcoded passwords or API keys
- [ ] WiFi credentials use placeholders in docs
- [ ] Serial ports use machine-agnostic examples
- [ ] .gitignore covers sensitive files

```bash
# Check for potential secrets
grep -rn "password\|secret\|api_key\|token" --include="*.py" --include="*.yaml" --include="*.md"
```

### 2. Input Validation

- [ ] Serial port paths validated
- [ ] File paths sanitized (no path traversal)
- [ ] CSI data format validated before parsing
- [ ] Network inputs validated

```python
# Bad: Direct path usage
port = user_input

# Good: Validated path
if not port.startswith('/dev/'):
    raise ValueError("Invalid port path")
```

### 3. Dependency Security

```bash
# Check for known vulnerabilities
pip install safety
safety check -r requirements.txt

# Update dependencies
pip list --outdated
```

### 4. Data Privacy

- [ ] No PII in CSI data files
- [ ] Metadata doesn't expose sensitive info
- [ ] Logs don't contain credentials

## Common Vulnerabilities

### Path Traversal
```python
# Vulnerable
filename = request.args.get('file')
with open(f'data/{filename}', 'r') as f:
    ...

# Fixed
filename = os.path.basename(request.args.get('file'))
safe_path = os.path.join('data', filename)
```

### Command Injection
```python
# Vulnerable
os.system(f"esptool --port {port} flash")

# Fixed
import subprocess
subprocess.run(['esptool', '--port', port, 'flash'], check=True)
```

### Insecure Deserialization
```python
# Vulnerable (with untrusted data)
import pickle
data = pickle.load(open('data.pkl', 'rb'))

# Safer (for ML models, use with caution)
import torch
model = torch.load('model.pt', weights_only=True)
```

## Project-Specific Concerns

### ESP32 Firmware
- Firmware binary is trusted (from verified build)
- WiFi credentials baked into firmware (use config menu)
- Serial communication is local-only

### CSI Data
- CSI data contains MAC addresses (could be PII)
- Recommend anonymizing MAC addresses in production
- Location data may reveal sensitive information

### Model Checkpoints
- PyTorch checkpoints can contain arbitrary code
- Only load checkpoints from trusted sources
- Use `weights_only=True` when possible

## Security Review Commands

```bash
# Find potential secrets
git secrets --scan

# Check Python security
bandit -r wavira/ scripts/

# Audit dependencies
pip-audit

# Check for outdated packages
pip list --outdated --format=json
```

## Reporting Issues

If security issues are found:
1. Do NOT commit the fix immediately
2. Document the issue privately
3. Assess impact and exploitability
4. Create fix in a private branch
5. Review fix before merging
