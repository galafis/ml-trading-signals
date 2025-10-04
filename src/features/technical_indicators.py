"""
Technical indicators for feature engineering.
"""
import pandas as pd
import numpy as np
from typing import Optional
import ta


class TechnicalIndicators:
    """
    Calculate technical indicators for machine learning features.
    """
    
    @staticmethod
    def add_trend_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """
        Add trend indicators to dataframe.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with trend indicators
        """
        data = df.copy()
        
        # Moving Averages
        data['sma_5'] = ta.trend.sma_indicator(data['close'], window=5)
        data['sma_10'] = ta.trend.sma_indicator(data['close'], window=10)
        data['sma_20'] = ta.trend.sma_indicator(data['close'], window=20)
        data['sma_50'] = ta.trend.sma_indicator(data['close'], window=50)
        
        data['ema_5'] = ta.trend.ema_indicator(data['close'], window=5)
        data['ema_10'] = ta.trend.ema_indicator(data['close'], window=10)
        data['ema_20'] = ta.trend.ema_indicator(data['close'], window=20)
        
        # MACD
        data['macd'] = ta.trend.macd(data['close'])
        data['macd_signal'] = ta.trend.macd_signal(data['close'])
        data['macd_diff'] = ta.trend.macd_diff(data['close'])
        
        # ADX
        data['adx'] = ta.trend.adx(data['high'], data['low'], data['close'])
        
        return data
    
    @staticmethod
    def add_momentum_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """
        Add momentum indicators to dataframe.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with momentum indicators
        """
        data = df.copy()
        
        # RSI
        data['rsi'] = ta.momentum.rsi(data['close'], window=14)
        
        # Stochastic Oscillator
        data['stoch'] = ta.momentum.stoch(data['high'], data['low'], data['close'])
        data['stoch_signal'] = ta.momentum.stoch_signal(data['high'], data['low'], data['close'])
        
        # Williams %R
        data['williams_r'] = ta.momentum.williams_r(data['high'], data['low'], data['close'])
        
        # ROC (Rate of Change)
        data['roc'] = ta.momentum.roc(data['close'], window=12)
        
        return data
    
    @staticmethod
    def add_volatility_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """
        Add volatility indicators to dataframe.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with volatility indicators
        """
        data = df.copy()
        
        # Bollinger Bands
        bollinger = ta.volatility.BollingerBands(data['close'])
        data['bb_high'] = bollinger.bollinger_hband()
        data['bb_mid'] = bollinger.bollinger_mavg()
        data['bb_low'] = bollinger.bollinger_lband()
        data['bb_width'] = bollinger.bollinger_wband()
        
        # ATR (Average True Range)
        data['atr'] = ta.volatility.average_true_range(data['high'], data['low'], data['close'])
        
        # Keltner Channel
        keltner = ta.volatility.KeltnerChannel(data['high'], data['low'], data['close'])
        data['kc_high'] = keltner.keltner_channel_hband()
        data['kc_mid'] = keltner.keltner_channel_mband()
        data['kc_low'] = keltner.keltner_channel_lband()
        
        return data
    
    @staticmethod
    def add_volume_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """
        Add volume indicators to dataframe.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with volume indicators
        """
        data = df.copy()
        
        # OBV (On-Balance Volume)
        data['obv'] = ta.volume.on_balance_volume(data['close'], data['volume'])
        
        # Volume Moving Average
        data['volume_sma_20'] = data['volume'].rolling(window=20).mean()
        
        # MFI (Money Flow Index)
        data['mfi'] = ta.volume.money_flow_index(
            data['high'], data['low'], data['close'], data['volume']
        )
        
        # Volume Price Trend
        data['vpt'] = ta.volume.volume_price_trend(data['close'], data['volume'])
        
        return data
    
    @staticmethod
    def add_price_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Add price-based features to dataframe.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with price features
        """
        data = df.copy()
        
        # Returns
        data['returns'] = data['close'].pct_change()
        data['log_returns'] = np.log(data['close'] / data['close'].shift(1))
        
        # Price changes
        data['price_change'] = data['close'] - data['open']
        data['price_change_pct'] = (data['close'] - data['open']) / data['open']
        
        # High-Low range
        data['high_low_range'] = data['high'] - data['low']
        data['high_low_range_pct'] = (data['high'] - data['low']) / data['close']
        
        # Gap
        data['gap'] = data['open'] - data['close'].shift(1)
        data['gap_pct'] = (data['open'] - data['close'].shift(1)) / data['close'].shift(1)
        
        return data
    
    @staticmethod
    def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """
        Add all technical indicators to dataframe.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with all indicators
        """
        data = df.copy()
        
        data = TechnicalIndicators.add_trend_indicators(data)
        data = TechnicalIndicators.add_momentum_indicators(data)
        data = TechnicalIndicators.add_volatility_indicators(data)
        data = TechnicalIndicators.add_volume_indicators(data)
        data = TechnicalIndicators.add_price_features(data)
        
        return data
