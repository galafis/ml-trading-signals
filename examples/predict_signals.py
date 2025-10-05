"""
ML Trading Signals - Prediction Example
Author: Gabriel Demetrios Lafis

This example demonstrates how to use technical indicators for trading signals.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.features.technical_indicators import TechnicalIndicators
from src.models.classifier import TradingClassifier


def main():
    """Run prediction example."""
    
    print("=" * 70)
    print("ML Trading Signals - Prediction Example")
    print("=" * 70)
    print()
    
    # Create sample data
    print("📊 Creating sample market data...")
    np.random.seed(42)
    dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
    prices = 100 + np.cumsum(np.random.randn(100) * 2)
    volume = np.random.randint(1000000, 10000000, 100)
    
    data = pd.DataFrame({
        'date': dates,
        'close': prices,
        'open': prices * (1 + np.random.randn(100) * 0.01),
        'high': prices * (1 + np.abs(np.random.randn(100)) * 0.02),
        'low': prices * (1 - np.abs(np.random.randn(100)) * 0.02),
        'volume': volume
    })
    
    print(f"✅ Created {len(data)} days of sample data")
    print(f"   Price range: ${data['close'].min():.2f} - ${data['close'].max():.2f}")
    print()
    
    # Calculate technical indicators
    print("🔧 Calculating technical indicators...")
    indicators = TechnicalIndicators()
    
    # Add all indicators
    data = indicators.add_trend_indicators(data)
    data = indicators.add_momentum_indicators(data)
    data = indicators.add_volatility_indicators(data)
    data = indicators.add_volume_indicators(data)
    
    # Count non-null indicators
    indicator_cols = [col for col in data.columns if col not in ['date', 'open', 'high', 'low', 'close', 'volume']]
    print(f"✅ Calculated {len(indicator_cols)} technical indicators")
    print(f"   Examples: {', '.join(indicator_cols[:5])}")
    print()
    
    # Remove NaN values
    data_clean = data.dropna()
    print(f"🎯 Prepared {len(data_clean)} samples for analysis")
    print()
    
    # Create model (in production, load pre-trained model)
    print("🤖 Initializing ML model...")
    model = TradingClassifier(model_type='xgboost')
    print("✅ Model initialized (XGBoost)")
    print()
    
    # Generate signals for last 5 days
    print("🔮 Generating trading signals...")
    print()
    
    print("=" * 70)
    print("TRADING SIGNALS (Last 5 Days)")
    print("=" * 70)
    print()
    
    for i in range(max(0, len(data_clean) - 5), len(data_clean)):
        row = data_clean.iloc[i]
        date = row['date'].strftime('%Y-%m-%d')
        price = row['close']
        
        # Get RSI if available
        rsi = row.get('rsi', 50)
        macd = row.get('macd', 0)
        
        # Simple signal logic
        if rsi < 30 and macd > 0:
            signal = "🟢 STRONG BUY"
            confidence = min(95, 75 + (30 - rsi))
        elif rsi < 40:
            signal = "🟢 BUY"
            confidence = min(85, 65 + (40 - rsi))
        elif rsi > 70 and macd < 0:
            signal = "🔴 STRONG SELL"
            confidence = min(95, 75 + (rsi - 70))
        elif rsi > 60:
            signal = "🔴 SELL"
            confidence = min(85, 65 + (rsi - 60))
        else:
            signal = "⚪ HOLD"
            confidence = 50 + abs(50 - rsi) / 4
        
        print(f"Date: {date}")
        print(f"  Price: ${price:.2f}")
        print(f"  RSI: {rsi:.2f}")
        print(f"  MACD: {macd:.4f}")
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
    print("   3. Use real market data from Yahoo Finance")
    print("   4. Integrate with the FastAPI for real-time predictions")
    print()
    print(f"📊 Available Indicators: {len(indicator_cols)}")
    print(f"   Trend: SMA, EMA, MACD, ADX")
    print(f"   Momentum: RSI, Stochastic, Williams %R")
    print(f"   Volatility: Bollinger Bands, ATR, Keltner")
    print(f"   Volume: OBV, MFI, Volume SMA")


if __name__ == "__main__":
    main()
