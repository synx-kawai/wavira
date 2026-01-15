# Contributing to Wavira

Thank you for your interest in contributing to Wavira! This document provides guidelines for contributing to the project.

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/YOUR_USERNAME/wavira.git`
3. Set up development environment (see [DEVELOPMENT.md](docs/DEVELOPMENT.md))
4. Create a feature branch: `git checkout -b feature/your-feature-name`

## Development Workflow

### Branch Naming

- `feature/<description>` - New features
- `fix/<description>` - Bug fixes
- `docs/<description>` - Documentation changes
- `refactor/<description>` - Code refactoring
- `test/<description>` - Test additions/modifications

### Commit Messages

We use [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation
- `test`: Tests
- `refactor`: Code refactoring
- `chore`: Maintenance tasks
- `ci`: CI/CD changes

Examples:
```
feat(model): add attention mechanism to encoder
fix(serial): handle timeout on ESP32 disconnect
docs(readme): update installation instructions
```

### Code Style

- Follow [PEP 8](https://pep8.org/) for Python code
- Maximum line length: 100 characters
- Use type hints for function signatures
- Write docstrings in Google style

Before submitting:
```bash
# Format code
black wavira/ scripts/ tests/
isort wavira/ scripts/ tests/

# Check linting
flake8 wavira/ scripts/ tests/

# Run tests
pytest tests/ -v
```

## Pull Request Process

1. Ensure all tests pass locally
2. Update documentation if needed
3. Add tests for new functionality
4. Create a PR with a clear description
5. Link related issues using `Closes #<issue-number>`
6. Request review from maintainers

### PR Title Format

Use the same format as commit messages:
```
feat(model): add transformer encoder option
```

### PR Description Template

```markdown
## Summary
Brief description of changes.

## Related Issues
Closes #<issue-number>

## Changes
- Change 1
- Change 2

## Testing
- How was this tested?

## Checklist
- [ ] Tests pass
- [ ] Documentation updated
- [ ] Code follows style guidelines
```

## Testing

- All new features must include tests
- Maintain or improve code coverage
- Use pytest fixtures for common setup
- Test edge cases and error conditions

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=wavira --cov-report=term-missing

# Run specific test file
pytest tests/test_encoder.py -v
```

## Documentation

- Update README.md for user-facing changes
- Add docstrings to all public functions
- Update API documentation if endpoints change
- Include code examples where helpful

## Security

- Never commit credentials or API keys
- Report security vulnerabilities privately
- See [docs/SECURITY.md](docs/SECURITY.md) for security guidelines

## Questions?

- Open an issue for questions
- Check existing issues before creating new ones
- Tag issues appropriately

## License

By contributing, you agree that your contributions will be licensed under the same license as the project.
