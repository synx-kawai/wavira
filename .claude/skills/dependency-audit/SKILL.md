---
name: dependency-audit
description: Use this skill to audit Python dependencies for security vulnerabilities, outdated packages, and license compliance. Run periodically or before releases.
---

# Dependency Audit Skill

This skill audits project dependencies for security, updates, and compliance.

## Quick Audit

```bash
# Full audit (security + outdated + licenses)
pip install pip-audit safety pip-licenses
pip-audit
pip list --outdated
pip-licenses --format=markdown
```

## Security Vulnerability Check

### Using pip-audit (Recommended)

```bash
# Install
pip install pip-audit

# Audit installed packages
pip-audit

# Audit requirements file
pip-audit -r requirements.txt

# Output as JSON for CI/CD
pip-audit --format=json -o audit-report.json
```

### Using safety

```bash
# Install
pip install safety

# Check installed packages
safety check

# Check requirements file
safety check -r requirements.txt

# Ignore specific vulnerabilities
safety check --ignore 12345
```

## Outdated Packages

```bash
# List outdated packages
pip list --outdated

# JSON format for parsing
pip list --outdated --format=json

# Show only direct dependencies
pip list --outdated --not-required
```

### Update Strategy

| Update Type | Risk Level | Action |
|-------------|------------|--------|
| Patch (x.x.1 -> x.x.2) | Low | Auto-update |
| Minor (x.1.x -> x.2.x) | Medium | Review changelog |
| Major (1.x.x -> 2.x.x) | High | Test thoroughly |

## License Compliance

```bash
# Install
pip install pip-licenses

# Generate license report
pip-licenses --format=markdown --output-file=LICENSES.md

# Check for specific licenses
pip-licenses --fail-on="GPL;LGPL"

# Summary view
pip-licenses --summary
```

### Allowed Licenses for Wavira

- MIT
- BSD-2-Clause
- BSD-3-Clause
- Apache-2.0
- PSF-2.0
- ISC

### Restricted Licenses

- GPL (requires source disclosure)
- AGPL (network copyleft)
- Commercial (requires purchase)

## Dependency Tree Analysis

```bash
# Install
pip install pipdeptree

# Show dependency tree
pipdeptree

# Show reverse dependencies (who uses this package)
pipdeptree --reverse --packages numpy

# JSON output
pipdeptree --json-tree
```

## Project-Specific Checks

### Core Dependencies

```python
# wavira core dependencies
CRITICAL_PACKAGES = [
    'torch',      # ML framework
    'numpy',      # Numerical computing
    'pyserial',   # ESP32 communication
]
```

### Known Compatible Versions

| Package | Min Version | Max Version | Notes |
|---------|-------------|-------------|-------|
| torch | 2.0.0 | - | CUDA 11.8+ for GPU |
| numpy | 1.21.0 | 1.26.x | 2.0 has breaking changes |
| pyserial | 3.5 | - | - |

## CI/CD Integration

### GitHub Actions Example

```yaml
name: Dependency Audit

on:
  schedule:
    - cron: '0 0 * * 0'  # Weekly
  push:
    paths:
      - 'requirements*.txt'
      - 'pyproject.toml'

jobs:
  audit:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.9'

      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install pip-audit safety

      - name: Security audit
        run: pip-audit

      - name: Check outdated
        run: pip list --outdated
```

## Recommended Workflow

### Before Each Release

1. Run `pip-audit` - fix critical vulnerabilities
2. Run `pip list --outdated` - update patch versions
3. Run `pip-licenses` - verify license compliance
4. Update `requirements.txt` with pinned versions

### Weekly Maintenance

1. Check for security advisories
2. Review minor version updates
3. Test with updated dependencies
4. Update lock files if using poetry/pipenv

### Quarterly Review

1. Evaluate major version updates
2. Remove unused dependencies
3. Consolidate overlapping packages
4. Update minimum Python version if needed

## Fixing Vulnerabilities

```bash
# Update specific package
pip install --upgrade package-name

# Update all packages (careful!)
pip install --upgrade -r requirements.txt

# Pin to specific safe version
echo "package-name==1.2.3" >> requirements.txt
pip install -r requirements.txt
```

## Reporting

Generate audit report for documentation:

```bash
echo "# Dependency Audit Report" > AUDIT.md
echo "Generated: $(date)" >> AUDIT.md
echo "" >> AUDIT.md
echo "## Security" >> AUDIT.md
pip-audit --format=markdown >> AUDIT.md 2>&1 || echo "No vulnerabilities found" >> AUDIT.md
echo "" >> AUDIT.md
echo "## Outdated Packages" >> AUDIT.md
pip list --outdated --format=markdown >> AUDIT.md
echo "" >> AUDIT.md
echo "## Licenses" >> AUDIT.md
pip-licenses --format=markdown >> AUDIT.md
```
