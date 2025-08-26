# Contributing to NSM Universal Translator

Thank you for your interest in contributing to the NSM Universal Translator! This document provides guidelines for contributing to the project.

## Getting Started

### Prerequisites

- Python 3.9+
- spaCy with language models
- Git

### Setup Development Environment

```bash
# Clone the repository
git clone https://github.com/lavelock67/universal_symantics.git
cd universal_symantics

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install spaCy models
python -m spacy download en_core_web_sm
python -m spacy download es_core_news_sm
python -m spacy download fr_core_news_sm

# Copy configuration
cp config.example.env .env
```

## Development Workflow

### 1. Create a Feature Branch

```bash
git checkout -b feature/your-feature-name
```

### 2. Make Changes

Follow these guidelines:

- **Code Style**: Follow PEP 8 and use type hints
- **Documentation**: Add docstrings for new functions/classes
- **Testing**: Add tests for new functionality
- **Configuration**: Use environment variables for configurable values

### 3. Run Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test suite
pytest tests/test_curated_suite.py -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

### 4. Commit Changes

```bash
git add .
git commit -m "feat: add new feature description"
```

Use conventional commit messages:
- `feat:` for new features
- `fix:` for bug fixes
- `docs:` for documentation changes
- `test:` for test additions
- `refactor:` for code refactoring

### 5. Push and Create Pull Request

```bash
git push origin feature/your-feature-name
```

## Areas for Contribution

### 1. Detection Improvements

- **New UD patterns** for better prime detection
- **Enhanced MWE rules** for specific languages
- **Improved lexical patterns** for edge cases

### 2. Router Enhancements

- **Better safety-critical feature detection**
- **Improved risk scoring algorithms**
- **Enhanced decision logic**

### 3. Language Support

- **New language implementations** (German, Italian, etc.)
- **Language-specific optimizations**
- **Cultural adaptation**

### 4. Testing and Evaluation

- **New test cases** for edge cases
- **Performance benchmarks**
- **Evaluation metrics**

### 5. Documentation

- **API documentation** improvements
- **Usage examples** for specific scenarios
- **Tutorial guides**

## Code Standards

### Python Code Style

```python
#!/usr/bin/env python3
"""Module docstring with clear description."""

from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)


def example_function(param1: str, param2: Optional[int] = None) -> List[str]:
    """Function docstring with clear description.
    
    Args:
        param1: Description of parameter
        param2: Optional parameter description
        
    Returns:
        List of strings with description
        
    Raises:
        ValueError: When invalid parameters provided
    """
    if not param1:
        raise ValueError("param1 cannot be empty")
    
    result = []
    # Implementation here
    
    return result
```

### Test Standards

```python
def test_example_function():
    """Test example function with clear test cases."""
    # Arrange
    param1 = "test"
    param2 = 42
    
    # Act
    result = example_function(param1, param2)
    
    # Assert
    assert isinstance(result, list)
    assert len(result) > 0
```

## Adding New Language Support

### 1. Create Language Module

```python
# src/detect/languages/your_language.py
from typing import List, Dict, Any

def detect_primes_your_language(text: str) -> List[str]:
    """Detect NSM primes in your language."""
    # Implementation here
    pass
```

### 2. Add MWE Rules

```yaml
# data/mwe_your_language.yml
quantifiers:
  - phrase: "your_phrase"
    prime: "PRIME"
    scope: "quant"

intensifiers:
  - phrase: "your_intensifier"
    prime: "VERY"
    attach: "head"
```

### 3. Add Test Cases

```json
{"text": "Your test sentence", "lang": "your_lang", "expected_primes": ["PRIME1", "PRIME2"], "expected_router": "translate", "category": "test", "description": "Your language test"}
```

### 4. Update Configuration

Add language-specific settings to `config.example.env`:

```bash
# Your Language
YOUR_LANG_LEGALITY_THRESHOLD=0.85
YOUR_LANG_DRIFT_THRESHOLD=0.2
YOUR_LANG_CONFIDENCE_THRESHOLD=0.65
```

## Reporting Issues

### Bug Reports

When reporting bugs, please include:

1. **Environment**: OS, Python version, spaCy version
2. **Steps to reproduce**: Clear, step-by-step instructions
3. **Expected behavior**: What should happen
4. **Actual behavior**: What actually happens
5. **Error messages**: Full error traceback
6. **Sample input**: Text that causes the issue

### Feature Requests

When requesting features, please include:

1. **Use case**: Why this feature is needed
2. **Proposed solution**: How you think it should work
3. **Alternatives**: Other approaches considered
4. **Impact**: How it affects existing functionality

## Review Process

### Pull Request Guidelines

1. **Title**: Clear, descriptive title
2. **Description**: Detailed description of changes
3. **Tests**: All tests pass
4. **Documentation**: Updated documentation
5. **Code review**: Address reviewer comments

### Review Checklist

- [ ] Code follows style guidelines
- [ ] Tests are included and pass
- [ ] Documentation is updated
- [ ] No sensitive data in commits
- [ ] Configuration changes documented
- [ ] Performance impact considered

## Getting Help

### Questions and Discussion

- **GitHub Issues**: For bugs and feature requests
- **GitHub Discussions**: For questions and general discussion
- **Documentation**: Check USAGE.md and README.md first

### Development Resources

- **spaCy Documentation**: https://spacy.io/usage
- **FastAPI Documentation**: https://fastapi.tiangolo.com/
- **Pydantic Documentation**: https://pydantic-docs.helpmanual.io/

## License

By contributing to this project, you agree that your contributions will be licensed under the MIT License.

## Code of Conduct

Please be respectful and inclusive in all interactions. We welcome contributors from all backgrounds and experience levels.
