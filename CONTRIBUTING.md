# Contributing to ML Trading Signals

Thank you for your interest in contributing to ML Trading Signals! This document provides guidelines for contributing to this machine learning project.

## 🤝 How to Contribute

### Reporting Bugs

If you find a bug, please create an issue with:
- Clear description of the problem
- Steps to reproduce
- Expected vs actual behavior
- Environment details (OS, Python version, library versions)
- Error messages and stack traces
- Sample data if applicable

### Suggesting Enhancements

We welcome suggestions for:
- New machine learning algorithms
- Additional technical indicators
- Feature engineering ideas
- Performance optimizations
- API improvements

### Pull Requests

1. **Fork the repository**
2. **Create a feature branch** (`git checkout -b feature/new-indicator`)
3. **Make your changes**
4. **Add tests** for new functionality
5. **Ensure all tests pass** (`pytest`)
6. **Update documentation**
7. **Commit your changes** (`git commit -m 'feat: add new indicator'`)
8. **Push to the branch** (`git push origin feature/new-indicator`)
9. **Open a Pull Request**

## 📋 Development Guidelines

### Code Style

- Follow **PEP 8** style guide
- Use **type hints** for all functions
- Write **docstrings** (Google style)
- Keep functions focused and small
- Use meaningful variable names

### Machine Learning Best Practices

- **Reproducibility**: Set random seeds
- **Data Leakage**: Avoid future information in features
- **Time-Series Aware**: Use proper train/val/test splits
- **Feature Scaling**: Normalize/standardize features
- **Model Validation**: Use cross-validation for temporal data
- **Hyperparameter Tuning**: Document parameter choices

### Testing

- Write unit tests for all new code
- Test feature engineering functions
- Test model training pipeline
- Mock external data sources
- Maintain >80% test coverage

### Documentation

- Update README.md for new features
- Add docstrings to all functions
- Include usage examples
- Document model parameters
- Explain feature engineering logic

### Commit Messages

Follow conventional commits:

```
<type>(<scope>): <subject>
```

**Types:**
- `feat`: New feature (algorithm, indicator, etc.)
- `fix`: Bug fix
- `docs`: Documentation
- `perf`: Performance improvement
- `refactor`: Code refactoring
- `test`: Tests
- `chore`: Maintenance

**Examples:**
```
feat(indicators): add Ichimoku Cloud indicator
fix(training): correct data leakage in feature engineering
perf(model): optimize XGBoost hyperparameters
docs(api): add prediction endpoint examples
```

## 🧪 Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test
pytest tests/unit/test_features.py -v

# Run integration tests
pytest tests/integration/ -v
```

## 📊 Adding New Features

### Adding a New Technical Indicator

1. Add function to `src/features/technical_indicators.py`
2. Write unit tests in `tests/unit/test_features.py`
3. Update documentation
4. Add to feature list in README

Example:
```python
@staticmethod
def add_custom_indicator(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add custom technical indicator.
    
    Args:
        df: DataFrame with OHLCV data
        
    Returns:
        DataFrame with custom indicator
    """
    data = df.copy()
    # Your implementation
    return data
```

### Adding a New ML Algorithm

1. Add model to `src/models/classifier.py`
2. Update `_create_model()` method
3. Write tests
4. Update documentation
5. Benchmark performance

## 🔍 Code Review Process

1. **Automated Checks**: CI/CD runs tests and linting
2. **Code Review**: Review for quality and ML best practices
3. **Feedback**: Address requested changes
4. **Approval**: Merge after approval

## 📝 License

By contributing, you agree that your contributions will be licensed under the MIT License.

## 💬 Questions?

Open an issue for any questions about contributing!

---

**Author**: Gabriel Demetrios Lafis
