---
name: test-runner
description: Use this skill when running tests, debugging test failures, or adding new test cases. Automatically invoked after code changes that require test validation.
---

# Test Runner Skill

This skill manages pytest-based testing for the Wavira project.

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_dataset.py -v

# Run specific test
pytest tests/test_loss.py::test_inbatch_loss_basic -v

# Run with coverage
pytest tests/ -v --cov=wavira --cov-report=html
```

## Test Structure

```
tests/
├── __init__.py
├── test_dataset.py    # CSIDataset tests
├── test_loss.py       # InBatchNegativeLoss tests
└── test_model.py      # WhoFi model tests (if exists)
```

## Writing Tests

### Test Template

```python
import pytest
import torch
from wavira import WhoFi, CSIDataset

class TestFeatureName:
    """Tests for feature description."""

    def test_basic_functionality(self):
        """Test basic expected behavior."""
        # Arrange
        input_data = torch.randn(2, 3, 114, 200)

        # Act
        result = some_function(input_data)

        # Assert
        assert result.shape == expected_shape

    def test_edge_case(self):
        """Test edge case handling."""
        with pytest.raises(ValueError):
            some_function(invalid_input)
```

### Fixtures

```python
@pytest.fixture
def sample_csi_data():
    """Generate sample CSI data for testing."""
    return torch.randn(2, 3, 114, 200)

@pytest.fixture
def trained_model():
    """Load a pre-trained model for testing."""
    model = WhoFi(n_channels=3, n_subcarriers=114)
    return model
```

## Common Test Patterns

### Model Output Shape
```python
def test_model_output_shape():
    model = WhoFi(n_channels=3, n_subcarriers=114, signature_dim=256)
    x = torch.randn(4, 3, 114, 200)
    output = model(x)
    assert output.shape == (4, 256)
```

### Loss Computation
```python
def test_loss_computation():
    loss_fn = InBatchNegativeLoss()
    embeddings = torch.randn(8, 256)
    labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
    loss = loss_fn(embeddings, labels)
    assert loss.item() >= 0
```

## Debugging Failed Tests

1. Run with verbose output: `pytest -vvs`
2. Use pdb on failure: `pytest --pdb`
3. Show local variables: `pytest -l`
4. Run last failed: `pytest --lf`
