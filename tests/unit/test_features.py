"""
Unit tests for feature engineering.
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.features.technical_indicators import TechnicalIndicators
from src.features.data_preparation import DataPreparation


@pytest.fixture
def sample_data():
    """Create sample OHLCV data."""
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
    np.random.seed(42)
    
    data = pd.DataFrame({
        'open': 100 + np.random.randn(len(dates)).cumsum(),
        'high': 102 + np.random.randn(len(dates)).cumsum(),
        'low': 98 + np.random.randn(len(dates)).cumsum(),
        'close': 100 + np.random.randn(len(dates)).cumsum(),
        'volume': np.random.randint(1000000, 10000000, len(dates))
    }, index=dates)
    
    data['high'] = data[['open', 'high', 'close']].max(axis=1)
    data['low'] = data[['open', 'low', 'close']].min(axis=1)
    
    return data


class TestTechnicalIndicators:
    """Tests for technical indicators."""
    
    def test_add_trend_indicators(self, sample_data):
        """Test trend indicators."""
        result = TechnicalIndicators.add_trend_indicators(sample_data)
        
        assert 'sma_5' in result.columns
        assert 'sma_10' in result.columns
        assert 'sma_20' in result.columns
        assert 'ema_5' in result.columns
        assert 'macd' in result.columns
        assert 'adx' in result.columns
    
    def test_add_momentum_indicators(self, sample_data):
        """Test momentum indicators."""
        result = TechnicalIndicators.add_momentum_indicators(sample_data)
        
        assert 'rsi' in result.columns
        assert 'stoch' in result.columns
        assert 'williams_r' in result.columns
        assert 'roc' in result.columns
    
    def test_add_volatility_indicators(self, sample_data):
        """Test volatility indicators."""
        result = TechnicalIndicators.add_volatility_indicators(sample_data)
        
        assert 'bb_high' in result.columns
        assert 'bb_low' in result.columns
        assert 'atr' in result.columns
        assert 'kc_high' in result.columns
    
    def test_add_volume_indicators(self, sample_data):
        """Test volume indicators."""
        result = TechnicalIndicators.add_volume_indicators(sample_data)
        
        assert 'obv' in result.columns
        assert 'mfi' in result.columns
        assert 'vpt' in result.columns
    
    def test_add_price_features(self, sample_data):
        """Test price features."""
        result = TechnicalIndicators.add_price_features(sample_data)
        
        assert 'returns' in result.columns
        assert 'log_returns' in result.columns
        assert 'price_change' in result.columns
        assert 'high_low_range' in result.columns
    
    def test_add_all_indicators(self, sample_data):
        """Test adding all indicators."""
        result = TechnicalIndicators.add_all_indicators(sample_data)
        
        # Should have many more columns than original
        assert len(result.columns) > len(sample_data.columns)
        
        # Check some key indicators
        assert 'sma_20' in result.columns
        assert 'rsi' in result.columns
        assert 'bb_high' in result.columns
        assert 'obv' in result.columns


class TestDataPreparation:
    """Tests for data preparation."""
    
    def test_initialization(self):
        """Test data preparation initialization."""
        prep = DataPreparation(scaler_type='standard')
        assert prep.scaler_type == 'standard'
        
        prep = DataPreparation(scaler_type='minmax')
        assert prep.scaler_type == 'minmax'
    
    def test_create_target_direction(self, sample_data):
        """Test target creation for direction."""
        prep = DataPreparation()
        result = prep.create_target_variable(sample_data, target_type='direction', horizon=1)
        
        assert 'target' in result.columns
        assert result['target'].isin([0, 1]).all()
    
    def test_create_target_returns(self, sample_data):
        """Test target creation for returns."""
        prep = DataPreparation()
        result = prep.create_target_variable(sample_data, target_type='returns', horizon=1)
        
        assert 'target' in result.columns
        assert result['target'].dtype == float
    
    def test_prepare_features(self, sample_data):
        """Test feature preparation."""
        # Add some features
        data = TechnicalIndicators.add_all_indicators(sample_data)
        
        prep = DataPreparation()
        result = prep.prepare_features(data)
        
        assert len(prep.feature_columns) > 0
        assert 'close' not in prep.feature_columns
        assert 'target' not in prep.feature_columns
    
    def test_split_data(self, sample_data):
        """Test data splitting."""
        prep = DataPreparation()
        train, val, test = prep.split_data(sample_data, train_size=0.7, val_size=0.15)
        
        assert len(train) > 0
        assert len(val) > 0
        assert len(test) > 0
        assert len(train) + len(val) + len(test) == len(sample_data)
    
    def test_scale_features(self, sample_data):
        """Test feature scaling."""
        data = TechnicalIndicators.add_all_indicators(sample_data)
        
        prep = DataPreparation()
        data = prep.prepare_features(data)
        train, val, test = prep.split_data(data)
        
        train_scaled, val_scaled, test_scaled = prep.scale_features(train, val, test)
        
        assert len(train_scaled) == len(train)
        assert len(val_scaled) == len(val)
        assert len(test_scaled) == len(test)
    
    def test_prepare_for_modeling(self, sample_data):
        """Test complete preparation pipeline."""
        data = TechnicalIndicators.add_all_indicators(sample_data)
        
        prep = DataPreparation()
        train, val, test = prep.prepare_for_modeling(data)
        
        assert 'target' in train.columns
        assert 'target' in val.columns
        assert 'target' in test.columns
        assert len(prep.feature_columns) > 0
