"""
Integration tests for FastAPI application.
"""
import pytest
from fastapi.testclient import TestClient
import pandas as pd
import numpy as np
from pathlib import Path
import os

from src.api.main import app, MODELS_DIR
from src.models.classifier import TradingClassifier


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


@pytest.fixture
def temp_model(tmp_path, monkeypatch):
    """Create a temporary trained model and point MODELS_DIR to tmp_path."""
    # Point the API to our temp directory
    monkeypatch.setattr("src.api.main.MODELS_DIR", str(tmp_path))

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

    return "test_model.pkl"


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
            "model_name": "nonexistent.pkl",
            "model_type": "xgboost",
            "lookback_days": 100,
        }

        response = client.post("/predict", json=request_data)
        assert response.status_code in [404, 500]

    def test_predict_path_traversal(self, client):
        """Test that path traversal in model_name is rejected."""
        request_data = {
            "symbol": "PETR4.SA",
            "model_name": "../../etc/passwd",
            "model_type": "xgboost",
            "lookback_days": 100,
        }

        response = client.post("/predict", json=request_data)
        assert response.status_code == 400

    def test_feature_importance_missing_model(self, client):
        """Test feature importance with missing model."""
        response = client.get(
            "/feature-importance?model_name=nonexistent.pkl&model_type=xgboost"
        )
        assert response.status_code == 404

    def test_feature_importance_success(self, client, temp_model):
        """Test feature importance with valid model."""
        response = client.get(
            f"/feature-importance?model_name={temp_model}&model_type=xgboost&top_n=5"
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
        response = client.post(
            "/predict",
            json={"model_name": "model.pkl", "model_type": "xgboost"},
        )
        assert response.status_code == 422  # Validation error

    def test_predict_lookback_out_of_range(self, client):
        """Test prediction with lookback_days outside allowed range."""
        request_data = {
            "symbol": "PETR4.SA",
            "model_name": "model.pkl",
            "model_type": "xgboost",
            "lookback_days": 10,  # below minimum of 50
        }
        response = client.post("/predict", json=request_data)
        assert response.status_code == 422


class TestAPIResponseFormat:
    """Test API response formats."""

    def test_root_response_format(self, client):
        """Test root endpoint response format."""
        response = client.get("/")
        data = response.json()

        assert "message" in data
        assert "version" in data
        assert "docs" in data
        assert isinstance(data["message"], str)
        assert isinstance(data["version"], str)

    def test_health_response_format(self, client):
        """Test health check response format."""
        response = client.get("/health")
        data = response.json()

        assert "status" in data
        assert "timestamp" in data
        assert data["status"] == "healthy"
        assert isinstance(data["timestamp"], str)

    def test_symbols_response_format(self, client):
        """Test symbols endpoint response format."""
        response = client.get("/symbols")
        data = response.json()

        assert "symbols" in data
        assert isinstance(data["symbols"], list)
        for symbol in data["symbols"]:
            assert isinstance(symbol, str)
            assert len(symbol) > 0
