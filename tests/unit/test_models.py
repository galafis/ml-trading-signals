"""
Unit tests for ML models.
"""
import pytest
import pandas as pd
import numpy as np
from src.models.classifier import TradingClassifier


@pytest.fixture
def sample_training_data():
    """Create sample training data."""
    np.random.seed(42)
    n_samples = 1000
    n_features = 10
    
    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f'feature_{i}' for i in range(n_features)]
    )
    y = pd.Series(np.random.randint(0, 2, n_samples))
    
    return X, y


class TestTradingClassifier:
    """Tests for trading classifier."""
    
    def test_initialization_xgboost(self):
        """Test XGBoost initialization."""
        model = TradingClassifier(model_type='xgboost')
        assert model.model_type == 'xgboost'
        assert model.model is not None
    
    def test_initialization_lightgbm(self):
        """Test LightGBM initialization."""
        model = TradingClassifier(model_type='lightgbm')
        assert model.model_type == 'lightgbm'
        assert model.model is not None
    
    def test_initialization_random_forest(self):
        """Test Random Forest initialization."""
        model = TradingClassifier(model_type='random_forest')
        assert model.model_type == 'random_forest'
        assert model.model is not None
    
    def test_initialization_invalid(self):
        """Test invalid model type."""
        with pytest.raises(ValueError):
            TradingClassifier(model_type='invalid_model')
    
    def test_train(self, sample_training_data):
        """Test model training."""
        X, y = sample_training_data
        
        # Split data
        split_idx = int(len(X) * 0.8)
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        model = TradingClassifier(model_type='xgboost', n_estimators=10)
        metrics = model.train(X_train, y_train, X_val, y_val)
        
        assert 'train_accuracy' in metrics
        assert 'accuracy' in metrics
        assert 0 <= metrics['train_accuracy'] <= 1
        assert 0 <= metrics['accuracy'] <= 1
    
    def test_predict(self, sample_training_data):
        """Test prediction."""
        X, y = sample_training_data
        
        model = TradingClassifier(model_type='xgboost', n_estimators=10)
        model.train(X, y)
        
        predictions = model.predict(X)
        
        assert len(predictions) == len(X)
        assert all(p in [0, 1] for p in predictions)
    
    def test_predict_proba(self, sample_training_data):
        """Test probability prediction."""
        X, y = sample_training_data
        
        model = TradingClassifier(model_type='xgboost', n_estimators=10)
        model.train(X, y)
        
        probabilities = model.predict_proba(X)
        
        assert probabilities.shape[0] == len(X)
        assert probabilities.shape[1] == 2
        assert all(0 <= p <= 1 for row in probabilities for p in row)
    
    def test_evaluate(self, sample_training_data):
        """Test model evaluation."""
        X, y = sample_training_data
        
        # Split data
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        model = TradingClassifier(model_type='xgboost', n_estimators=10)
        model.train(X_train, y_train)
        
        metrics = model.evaluate(X_test, y_test)
        
        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1' in metrics
        assert 'auc' in metrics
    
    def test_feature_importance(self, sample_training_data):
        """Test feature importance."""
        X, y = sample_training_data
        
        model = TradingClassifier(model_type='xgboost', n_estimators=10)
        model.train(X, y)
        
        importance = model.get_feature_importance(top_n=5)
        
        assert len(importance) <= 5
        assert 'feature' in importance.columns
        assert 'importance' in importance.columns
    
    def test_save_load_model(self, sample_training_data, tmp_path):
        """Test model saving and loading."""
        X, y = sample_training_data
        
        # Train model
        model = TradingClassifier(model_type='xgboost', n_estimators=10)
        model.train(X, y)
        
        # Save model
        model_path = tmp_path / "test_model.pkl"
        model.save_model(str(model_path))
        
        # Load model
        loaded_model = TradingClassifier.load_model(str(model_path), 'xgboost')
        
        # Test predictions are the same
        original_pred = model.predict(X)
        loaded_pred = loaded_model.predict(X)
        
        assert np.array_equal(original_pred, loaded_pred)
