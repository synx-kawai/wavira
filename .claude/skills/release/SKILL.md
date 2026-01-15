---
name: release
description: Use this skill to create releases, including version bumping, changelog updates, Git tagging, and package building. Supports semantic versioning.
---

# Release Skill

This skill manages the release process for Wavira, including versioning, tagging, and packaging.

## Quick Release

```bash
# Patch release (0.1.0 -> 0.1.1)
/release patch

# Minor release (0.1.0 -> 0.2.0)
/release minor

# Major release (0.1.0 -> 1.0.0)
/release major
```

## Release Checklist

### Pre-Release

- [ ] All tests passing: `pytest tests/ -v`
- [ ] No security vulnerabilities: `pip-audit`
- [ ] Documentation updated
- [ ] CHANGELOG.md updated
- [ ] Version bumped in source files

### Release

- [ ] Create Git tag
- [ ] Push tag to remote
- [ ] Build distribution packages
- [ ] Create GitHub release

### Post-Release

- [ ] Verify package installation
- [ ] Announce release
- [ ] Update development version

## Version Management

### Version Locations

Update version in these files:

```
wavira/__init__.py:  __version__ = "0.1.0"
pyproject.toml:      version = "0.1.0"
setup.py:            version="0.1.0"  (if exists)
```

### Semantic Versioning Rules

| Change Type | Version Bump | Example |
|-------------|--------------|---------|
| Bug fixes, patches | PATCH | 0.1.0 -> 0.1.1 |
| New features (backward compatible) | MINOR | 0.1.0 -> 0.2.0 |
| Breaking changes | MAJOR | 0.1.0 -> 1.0.0 |

### Version Bump Script

```python
import re
import sys

def bump_version(current: str, bump_type: str) -> str:
    """Bump semantic version."""
    major, minor, patch = map(int, current.split('.'))

    if bump_type == 'major':
        return f"{major + 1}.0.0"
    elif bump_type == 'minor':
        return f"{major}.{minor + 1}.0"
    elif bump_type == 'patch':
        return f"{major}.{minor}.{patch + 1}"
    else:
        raise ValueError(f"Unknown bump type: {bump_type}")

# Usage: python bump.py 0.1.0 minor -> 0.2.0
```

## Git Tagging

```bash
# Create annotated tag
git tag -a v0.2.0 -m "Release v0.2.0: Description of release"

# Push tag
git push origin v0.2.0

# Push all tags
git push --tags

# Delete tag (if needed)
git tag -d v0.2.0
git push origin :refs/tags/v0.2.0
```

## Building Packages

### Python Package

```bash
# Install build tools
pip install build twine

# Build source and wheel distributions
python -m build

# Check package
twine check dist/*

# Upload to PyPI (if applicable)
twine upload dist/*

# Upload to TestPyPI first
twine upload --repository testpypi dist/*
```

### Directory Structure After Build

```
dist/
├── wavira-0.2.0.tar.gz      # Source distribution
└── wavira-0.2.0-py3-none-any.whl  # Wheel
```

## GitHub Release

### Using gh CLI

```bash
# Create release from tag
gh release create v0.2.0 \
  --title "v0.2.0" \
  --notes-file RELEASE_NOTES.md

# Create release with assets
gh release create v0.2.0 \
  --title "v0.2.0" \
  --notes-file RELEASE_NOTES.md \
  dist/wavira-0.2.0.tar.gz \
  dist/wavira-0.2.0-py3-none-any.whl

# Create draft release
gh release create v0.2.0 --draft

# Generate release notes automatically
gh release create v0.2.0 --generate-notes
```

## Release Notes Template

```markdown
# v0.2.0

## Highlights

Brief description of the most important changes.

## What's New

### Features
- Feature 1 description
- Feature 2 description

### Improvements
- Improvement 1
- Improvement 2

### Bug Fixes
- Fix 1
- Fix 2

## Breaking Changes

- Description of any breaking changes

## Upgrade Guide

Instructions for upgrading from previous version.

## Contributors

Thanks to all contributors for this release!

## Full Changelog

https://github.com/user/wavira/compare/v0.1.0...v0.2.0
```

## Complete Release Workflow

```bash
# 1. Ensure clean working directory
git status

# 2. Run tests
pytest tests/ -v

# 3. Run security audit
pip-audit

# 4. Update version (in all locations)
# Edit wavira/__init__.py, pyproject.toml

# 5. Update CHANGELOG.md
# Add new version section

# 6. Commit version bump
git add -A
git commit -m "chore: bump version to 0.2.0"

# 7. Create tag
git tag -a v0.2.0 -m "Release v0.2.0"

# 8. Push changes and tag
git push origin main
git push origin v0.2.0

# 9. Build packages
python -m build

# 10. Create GitHub release
gh release create v0.2.0 \
  --title "v0.2.0" \
  --generate-notes \
  dist/*

# 11. Verify installation
pip install dist/wavira-0.2.0-py3-none-any.whl
python -c "import wavira; print(wavira.__version__)"
```

## Hotfix Release

For urgent fixes on released versions:

```bash
# 1. Create hotfix branch from tag
git checkout -b hotfix/0.1.1 v0.1.0

# 2. Apply fix
# ... make changes ...

# 3. Bump patch version
# Edit version files

# 4. Commit and tag
git commit -am "fix: critical bug description"
git tag -a v0.1.1 -m "Hotfix v0.1.1"

# 5. Push
git push origin hotfix/0.1.1
git push origin v0.1.1

# 6. Merge back to main
git checkout main
git merge hotfix/0.1.1
git push origin main
```

## ESP32 Firmware Releases

For firmware releases, include:

```bash
# Build firmware (requires ESP-IDF)
cd esp-csi
idf.py build

# Copy binary to release
cp build/esp32_csi.bin ../dist/esp32_csi_firmware_v0.2.0.bin

# Include in GitHub release
gh release upload v0.2.0 dist/esp32_csi_firmware_v0.2.0.bin
```
