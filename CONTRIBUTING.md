# Contributing to DopamineCore

Thank you for your interest in contributing to DopamineCore! This document provides guidelines and instructions for contributing.

## How to Contribute

### Reporting Bugs

- Use the [GitHub Issues](https://github.com/anbarchik/dopamine-core/issues) page
- Check existing issues first to avoid duplicates
- Use the bug report template and include:
  - Steps to reproduce
  - Expected vs. actual behavior
  - Python version and OS
  - Relevant code snippets or logs

### Suggesting Features

- Open a feature request issue with the template
- Describe the use case and expected behavior
- If possible, reference relevant research or prior art

### Pull Requests

1. Fork the repository
2. Create a feature branch from `main`
3. Make your changes
4. Add or update tests as needed
5. Ensure all tests pass
6. Submit a PR using the pull request template

## Development Setup

### Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) package manager

### Getting Started

```bash
# Clone the repository
git clone https://github.com/anbarchik/dopamine-core.git
cd dopamine-core

# Install dependencies
uv sync --dev

# Run tests
uv run pytest

# Run linter
uv run ruff check src/

# Run type checker
uv run mypy src/
```

### Running Tests

```bash
# All tests
uv run pytest

# With coverage
uv run pytest --cov=dopamine_core

# Specific test file
uv run pytest tests/test_engine.py -v
```

## Code Style

- We use [Ruff](https://docs.astral.sh/ruff/) for linting and formatting
- Line length: 100 characters
- Type hints are required for all public APIs
- Follow existing patterns in the codebase

### Key Conventions

- Use `snake_case` for functions and variables
- Use `PascalCase` for classes
- Prefix private methods with `_`
- Dataclasses for data containers, Pydantic models for configuration
- No external dependencies beyond `pydantic` in core

## Project Structure

```
src/dopamine_core/
├── __init__.py          # Public API
├── types.py             # Core data types
├── config.py            # Configuration schemas
├── engine.py            # Main orchestrator
├── signals/             # Signal extraction and RPE
│   ├── extractor.py     # CoT behavioral analysis
│   └── rpe.py           # Reward prediction error
└── injection/           # Context injection
    ├── templates.py     # Naturalistic prompt templates
    └── context.py       # Injection logic
```

## Guidelines

### For Signal Extraction

When adding new behavioral signal patterns:
- Patterns must be language-agnostic where possible
- Include both positive and negative indicators
- Test with diverse agent response styles
- Keep pattern weights in reasonable ranges

### For Context Templates

When adding injection templates:
- **Never** use internal terminology (dopamine, reward signal, RPE, tonic, phasic)
- Templates should read as natural environmental commentary
- Include multiple phrasings for randomization
- Test that templates don't leak implementation details

### For Safety

- All signals must be bounded
- New features should not bypass safety clamping
- Consider adversarial scenarios (reward hacking)

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
