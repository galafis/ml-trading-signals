"""
Integration tests for FastAPI application.
"""
import pytest
from fastapi.testclient import TestClient
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import os

from src.api.main import app
from src.models.classifier import TradingClassifier


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


@pytest.fixture
def temp_model(tmp_path):
    """Create a temporary trained model."""
    # Create sample data
    np.random.seed(42)
    n_samples = 100
    n_features = 10

    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f"feature_{i}" for i in range(n_features)],
    )
    y = pd.Series(np.random.randint(0, 2, n_samples))

    # Train model
    model = TradingClassifier(model_type="xgboost", n_estimators=10)
    model.train(X, y)

    # Save model
    model_path = tmp_path / "test_model.pkl"
    model.save_model(str(model_path))

    return str(model_path)


class TestAPIEndpoints:
    """Test API endpoints."""

    def test_root_endpoint(self, client):
        """Test root endpoint."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "ML Trading Signals API"
        assert data["version"] == "1.0.0"
        assert "docs" in data

    def test_health_check(self, client):
        """Test health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data

    def test_get_supported_symbols(self, client):
        """Test getting supported symbols."""
        response = client.get("/symbols")
        assert response.status_code == 200
        data = response.json()
        assert "symbols" in data
        assert isinstance(data["symbols"], list)
        assert len(data["symbols"]) > 0
        assert "^BVSP" in data["symbols"]
        assert "PETR4.SA" in data["symbols"]

    def test_predict_missing_model(self, client):
        """Test prediction with missing model file."""
        request_data = {
            "symbol": "PETR4.SA",
            "model_path": "/nonexistent/model.pkl",
            "model_type": "xgboost",
            "lookback_days": 100,
        }

        response = client.post("/predict", json=request_data)
        # Could be 404 (model not found) or 500 (network error with yfinance)
        assert response.status_code in [404, 500]

    def test_predict_invalid_symbol(self, client, temp_model):
        """Test prediction with invalid symbol."""
        request_data = {
            "symbol": "INVALID_SYMBOL_DOES_NOT_EXIST",
            "model_path": temp_model,
            "model_type": "xgboost",
            "lookback_days": 100,
        }

        response = client.post("/predict", json=request_data)
        # Should get either 404 or 500 depending on yfinance response
        assert response.status_code in [404, 500]

    def test_feature_importance_missing_model(self, client):
        """Test feature importance with missing model."""
        response = client.get(
            "/feature-importance?model_path=/nonexistent/model.pkl&model_type=xgboost"
        )
        assert response.status_code == 404
        assert "Model file not found" in response.json()["detail"]

    def test_feature_importance_success(self, client, temp_model):
        """Test feature importance with valid model."""
        # First, we need to load and check the model
        from src.models.classifier import TradingClassifier

        model = TradingClassifier.load_model(temp_model, "xgboost")

        # Model should have feature importance after training
        assert model.feature_importance is not None

        response = client.get(
            f"/feature-importance?model_path={temp_model}&model_type=xgboost&top_n=5"
        )
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) <= 5
        if len(data) > 0:
            assert "feature" in data[0]
            assert "importance" in data[0]


class TestAPIValidation:
    """Test API request validation."""

    def test_predict_missing_fields(self, client):
        """Test prediction with missing required fields."""
        # Missing symbol
        response = client.post(
            "/predict",
            json={"model_path": "/path/to/model.pkl", "model_type": "xgboost"},
        )
        assert response.status_code == 422  # Validation error

    def test_predict_invalid_model_type(self, client):
        """Test prediction with invalid model type."""
        request_data = {
            "symbol": "PETR4.SA",
            "model_path": "/path/to/model.pkl",
            "model_type": "invalid_type",
            "lookback_days": 100,
        }

        response = client.post("/predict", json=request_data)
        # Should fail during model loading
        assert response.status_code in [404, 500]

    def test_feature_importance_invalid_top_n(self, client, temp_model):
        """Test feature importance with invalid top_n."""
        response = client.get(
            f"/feature-importance?model_path={temp_model}&model_type=xgboost&top_n=-1"
        )
        # Should work, pandas head() handles negative values
        assert response.status_code == 200


class TestAPIResponseFormat:
    """Test API response formats."""

    def test_root_response_format(self, client):
        """Test root endpoint response format."""
        response = client.get("/")
        data = response.json()

        # Check all expected fields are present
        assert "message" in data
        assert "version" in data
        assert "docs" in data

        # Check field types
        assert isinstance(data["message"], str)
        assert isinstance(data["version"], str)
        assert isinstance(data["docs"], str)

    def test_health_response_format(self, client):
        """Test health check response format."""
        response = client.get("/health")
        data = response.json()

        assert "status" in data
        assert "timestamp" in data
        assert data["status"] == "healthy"

        # Timestamp should be ISO format string
        assert isinstance(data["timestamp"], str)
        assert "T" in data["timestamp"]  # ISO 8601 format

    def test_symbols_response_format(self, client):
        """Test symbols endpoint response format."""
        response = client.get("/symbols")
        data = response.json()

        assert "symbols" in data
        assert isinstance(data["symbols"], list)

        # All symbols should be strings
        for symbol in data["symbols"]:
            assert isinstance(symbol, str)
            assert len(symbol) > 0
