---
name: changelog-generator
description: Use this skill to generate or update CHANGELOG.md from Git commit history. Follows Conventional Commits and Keep a Changelog format.
---

# Changelog Generator Skill

This skill generates structured changelogs from Git commit history following the Keep a Changelog format.

## Usage

```bash
# Generate changelog for latest release
/changelog-generator

# Generate changelog between tags
/changelog-generator v0.1.0..v0.2.0
```

## Output Format

```markdown
# Changelog

## [Unreleased]

### Added
- New feature descriptions

### Changed
- Modification descriptions

### Fixed
- Bug fix descriptions

### Removed
- Removed feature descriptions
```

## Commit Type Mapping

| Commit Prefix | Changelog Section |
|---------------|-------------------|
| `feat:` | Added |
| `fix:` | Fixed |
| `docs:` | Documentation |
| `refactor:` | Changed |
| `perf:` | Changed |
| `test:` | (skip) |
| `chore:` | (skip) |
| `BREAKING CHANGE:` | Breaking Changes |

## Workflow

1. **Gather commits**: Fetch commits since last tag or specified range
2. **Parse commits**: Extract type, scope, and description
3. **Group by type**: Organize into changelog sections
4. **Format output**: Generate markdown following Keep a Changelog
5. **Update file**: Prepend new section to existing CHANGELOG.md

## Commands

```bash
# Get commits since last tag
git log $(git describe --tags --abbrev=0)..HEAD --oneline

# Get all tags sorted by version
git tag --sort=-v:refname

# Get commit details
git log --pretty=format:"%h %s" --no-merges
```

## Example Output

```markdown
## [0.2.0] - 2024-01-15

### Added
- CSI visualizer dashboard with real-time plotting
- Multi-device support for ESP32 CSI collection

### Fixed
- Serial port detection on macOS Sonoma
- Memory leak in continuous data collection

### Changed
- Improved MQTT connection stability
- Updated PyTorch dependency to 2.0+
```

## Configuration

The skill respects the following conventions:
- Semantic versioning (MAJOR.MINOR.PATCH)
- Conventional Commits format
- Keep a Changelog structure
- Links to GitHub compare views when possible

## Integration

Works with:
- `git tag` for version tracking
- GitHub releases
- CI/CD pipelines for automated changelog updates
