"""
Integration tests for training pipeline.
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

from src.training.train_pipeline import TrainingPipeline
from src.features.technical_indicators import TechnicalIndicators
from src.features.data_preparation import DataPreparation

# Mark all tests in this module as requiring network
pytestmark = [pytest.mark.integration, pytest.mark.network]


@pytest.fixture
def sample_pipeline():
    """Create a sample training pipeline with short date range for testing."""
    # Use recent dates to avoid data availability issues
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
    
    pipeline = TrainingPipeline(
        symbol='^BVSP',
        start_date=start_date,
        end_date=end_date,
        model_type='xgboost',
        target_type='direction',
        horizon=1
    )
    return pipeline


class TestTrainingPipeline:
    """Test complete training pipeline."""
    
    def test_pipeline_initialization(self, sample_pipeline):
        """Test pipeline initialization."""
        assert sample_pipeline.symbol == '^BVSP'
        assert sample_pipeline.model_type == 'xgboost'
        assert sample_pipeline.target_type == 'direction'
        assert sample_pipeline.horizon == 1
        assert sample_pipeline.data is None
        assert sample_pipeline.model is None
    
    def test_fetch_data(self, sample_pipeline):
        """Test data fetching."""
        df = sample_pipeline.fetch_data()
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert 'close' in df.columns
        assert 'open' in df.columns
        assert 'high' in df.columns
        assert 'low' in df.columns
        assert 'volume' in df.columns
        
        # Check that data is sorted by date
        assert df.index.is_monotonic_increasing
    
    def test_engineer_features(self, sample_pipeline):
        """Test feature engineering."""
        # First fetch data
        df = sample_pipeline.fetch_data()
        
        # Engineer features
        df_with_features = sample_pipeline.engineer_features(df)
        
        assert len(df_with_features.columns) > len(df.columns)
        
        # Check some expected features
        assert 'sma_20' in df_with_features.columns
        assert 'rsi' in df_with_features.columns
        assert 'macd' in df_with_features.columns
    
    def test_prepare_data(self, sample_pipeline):
        """Test data preparation."""
        # Fetch and engineer features
        df = sample_pipeline.fetch_data()
        df = sample_pipeline.engineer_features(df)
        
        # Prepare data
        train_df, val_df, test_df = sample_pipeline.prepare_data(df)
        
        # Check splits exist
        assert len(train_df) > 0
        assert len(val_df) > 0
        assert len(test_df) > 0
        
        # Check target column exists
        assert 'target' in train_df.columns
        assert 'target' in val_df.columns
        assert 'target' in test_df.columns
        
        # Check data preparation was initialized
        assert sample_pipeline.data_prep is not None
        assert len(sample_pipeline.data_prep.feature_columns) > 0
    
    def test_train_model(self, sample_pipeline):
        """Test model training."""
        # Prepare data
        df = sample_pipeline.fetch_data()
        df = sample_pipeline.engineer_features(df)
        train_df, val_df, test_df = sample_pipeline.prepare_data(df)
        
        # Train model
        metrics = sample_pipeline.train_model(train_df, val_df)
        
        assert isinstance(metrics, dict)
        assert 'train_accuracy' in metrics
        assert 'accuracy' in metrics
        assert 0 <= metrics['train_accuracy'] <= 1
        assert 0 <= metrics['accuracy'] <= 1
        
        # Check model was created
        assert sample_pipeline.model is not None
    
    def test_evaluate_model(self, sample_pipeline):
        """Test model evaluation."""
        # Run through full pipeline
        df = sample_pipeline.fetch_data()
        df = sample_pipeline.engineer_features(df)
        train_df, val_df, test_df = sample_pipeline.prepare_data(df)
        sample_pipeline.train_model(train_df, val_df)
        
        # Evaluate
        metrics = sample_pipeline.evaluate_model(test_df)
        
        assert isinstance(metrics, dict)
        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1' in metrics
        assert 'auc' in metrics
        
        # All metrics should be between 0 and 1
        for key, value in metrics.items():
            assert 0 <= value <= 1
    
    def test_complete_pipeline(self, sample_pipeline):
        """Test complete end-to-end pipeline."""
        results = sample_pipeline.run(use_mlflow=False)
        
        assert isinstance(results, dict)
        assert 'train_metrics' in results
        assert 'test_metrics' in results
        assert 'feature_importance' in results
        assert 'model' in results
        
        # Check that all required metrics exist
        assert 'accuracy' in results['test_metrics']
        assert len(results['feature_importance']) > 0
    
    def test_save_model(self, sample_pipeline, tmp_path):
        """Test model saving."""
        # Run pipeline
        sample_pipeline.run(use_mlflow=False)
        
        # Save model
        model_path = tmp_path / "test_model.pkl"
        sample_pipeline.save_model(str(model_path))
        
        assert model_path.exists()
        assert model_path.stat().st_size > 0


class TestPipelineWithDifferentModels:
    """Test pipeline with different model types."""
    
    @pytest.mark.parametrize("model_type", [
        'xgboost', 'lightgbm', 'random_forest', 'gradient_boosting', 'logistic'
    ])
    def test_different_models(self, model_type):
        """Test pipeline with different model types."""
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
        
        pipeline = TrainingPipeline(
            symbol='^BVSP',
            start_date=start_date,
            end_date=end_date,
            model_type=model_type,
            target_type='direction',
            horizon=1
        )
        
        results = pipeline.run(use_mlflow=False)
        
        assert 'test_metrics' in results
        assert results['model'].model_type == model_type


class TestPipelineWithDifferentTargets:
    """Test pipeline with different target types."""
    
    @pytest.mark.parametrize("target_type", ['direction', 'returns', 'binary'])
    def test_different_targets(self, target_type):
        """Test pipeline with different target types."""
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
        
        pipeline = TrainingPipeline(
            symbol='^BVSP',
            start_date=start_date,
            end_date=end_date,
            model_type='xgboost',
            target_type=target_type,
            horizon=1
        )
        
        # Fetch and prepare data
        df = pipeline.fetch_data()
        df = pipeline.engineer_features(df)
        train_df, val_df, test_df = pipeline.prepare_data(df)
        
        # Check target was created correctly
        assert 'target' in train_df.columns
        
        if target_type in ['direction', 'binary']:
            # Binary targets should be 0 or 1
            assert train_df['target'].isin([0, 1]).all()
        else:  # returns
            # Returns should be continuous
            assert train_df['target'].dtype == float


class TestPipelineErrorHandling:
    """Test pipeline error handling."""
    
    def test_invalid_symbol(self):
        """Test pipeline with invalid symbol."""
        pipeline = TrainingPipeline(
            symbol='INVALID_SYMBOL_XYZ123',
            start_date='2023-01-01',
            end_date='2023-12-31',
            model_type='xgboost'
        )
        
        # Should raise an error or return empty data
        df = pipeline.fetch_data()
        # If symbol is invalid, data might be empty
        # This behavior depends on yfinance
        assert isinstance(df, pd.DataFrame)
    
    def test_save_model_before_training(self, tmp_path):
        """Test saving model before training."""
        pipeline = TrainingPipeline(
            symbol='^BVSP',
            start_date='2023-01-01',
            end_date='2023-12-31',
            model_type='xgboost'
        )
        
        model_path = tmp_path / "test_model.pkl"
        
        with pytest.raises(ValueError, match="No model to save"):
            pipeline.save_model(str(model_path))
