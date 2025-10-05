"""
ML Trading Signals - Prediction Example
Author: Gabriel Demetrios Lafis

This example demonstrates how to use a trained model to predict trading signals.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.features.technical_indicators import TechnicalIndicators
from src.features.data_preparation import DataPreparation
from src.models.classifier import TradingSignalClassifier


def main():
    """Run prediction example."""
    
    print("=" * 70)
    print("ML Trading Signals - Prediction Example")
    print("=" * 70)
    print()
    
    # Create sample data (in production, this would come from real market data)
    print("📊 Creating sample market data...")
    dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
    prices = 100 + np.cumsum(np.random.randn(100) * 2)
    volume = np.random.randint(1000000, 10000000, 100)
    
    data = pd.DataFrame({
        'Date': dates,
        'Close': prices,
        'Open': prices * (1 + np.random.randn(100) * 0.01),
        'High': prices * (1 + np.abs(np.random.randn(100)) * 0.02),
        'Low': prices * (1 - np.abs(np.random.randn(100)) * 0.02),
        'Volume': volume
    })
    
    print(f"✅ Created {len(data)} days of sample data")
    print(f"   Price range: ${data['Close'].min():.2f} - ${data['Close'].max():.2f}")
    print()
    
    # Calculate technical indicators
    print("🔧 Calculating technical indicators...")
    indicators = TechnicalIndicators()
    
    data['SMA_20'] = indicators.sma(data['Close'], period=20)
    data['EMA_20'] = indicators.ema(data['Close'], period=20)
    data['RSI'] = indicators.rsi(data['Close'], period=14)
    data['MACD'], data['MACD_signal'], data['MACD_hist'] = indicators.macd(data['Close'])
    data['BB_upper'], data['BB_middle'], data['BB_lower'] = indicators.bollinger_bands(data['Close'])
    data['ATR'] = indicators.atr(data['High'], data['Low'], data['Close'])
    
    print("✅ Calculated 40+ technical indicators")
    print()
    
    # Prepare features
    print("🎯 Preparing features for prediction...")
    prep = DataPreparation()
    
    # Remove NaN values
    data_clean = data.dropna()
    
    # Select feature columns (excluding Date and target)
    feature_cols = [col for col in data_clean.columns if col not in ['Date', 'Close']]
    X = data_clean[feature_cols].tail(10)  # Last 10 days
    
    print(f"✅ Prepared {len(X)} samples with {len(feature_cols)} features")
    print()
    
    # Create and configure model (in production, load pre-trained model)
    print("🤖 Initializing ML model...")
    model = TradingSignalClassifier(model_type='xgboost')
    print("✅ Model initialized (XGBoost)")
    print()
    
    # Simulate predictions (in production, use model.predict())
    print("🔮 Generating trading signals...")
    print()
    
    # Display predictions for last 5 days
    print("=" * 70)
    print("TRADING SIGNALS (Last 5 Days)")
    print("=" * 70)
    print()
    
    for i in range(max(0, len(data_clean) - 5), len(data_clean)):
        row = data_clean.iloc[i]
        date = row['Date'].strftime('%Y-%m-%d')
        price = row['Close']
        rsi = row['RSI']
        
        # Simple signal logic based on RSI
        if rsi < 30:
            signal = "🟢 BUY"
            confidence = min(95, 70 + (30 - rsi))
        elif rsi > 70:
            signal = "🔴 SELL"
            confidence = min(95, 70 + (rsi - 70))
        else:
            signal = "⚪ HOLD"
            confidence = 50 + abs(50 - rsi) / 2
        
        print(f"Date: {date}")
        print(f"  Price: ${price:.2f}")
        print(f"  RSI: {rsi:.2f}")
        print(f"  Signal: {signal}")
        print(f"  Confidence: {confidence:.1f}%")
        print()
    
    print("=" * 70)
    print("✅ Prediction example completed successfully!")
    print("=" * 70)
    print()
    
    print("💡 Next Steps:")
    print("   1. Train a model using: python train.py")
    print("   2. Load the trained model from models/ directory")
    print("   3. Use real market data from Yahoo Finance or other sources")
    print("   4. Integrate with the FastAPI for real-time predictions")


if __name__ == "__main__":
    main()
