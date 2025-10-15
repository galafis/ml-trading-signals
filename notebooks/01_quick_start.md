# ML Trading Signals - Quick Start Tutorial

This notebook demonstrates how to use the ML Trading Signals library for technical analysis and trading signal generation.

## Table of Contents
1. [Setup and Imports](#setup)
2. [Fetch Market Data](#data)
3. [Calculate Technical Indicators](#indicators)
4. [Train ML Model](#train)
5. [Generate Predictions](#predict)
6. [Evaluate Performance](#evaluate)

## 1. Setup and Imports <a name="setup"></a>

```python
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

# Add project root to path
sys.path.insert(0, '..')

from src.features.technical_indicators import TechnicalIndicators
from src.features.data_preparation import DataPreparation
from src.models.classifier import TradingClassifier
from src.training.train_pipeline import TrainingPipeline

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

print("✓ Setup complete!")
```

## 2. Fetch Market Data <a name="data"></a>

```python
import yfinance as yf

# Fetch historical data for Bovespa Index
symbol = "^BVSP"
end_date = datetime.now()
start_date = end_date - timedelta(days=730)  # 2 years

print(f"Fetching data for {symbol}...")
ticker = yf.Ticker(symbol)
df = ticker.history(start=start_date.strftime('%Y-%m-%d'), 
                    end=end_date.strftime('%Y-%m-%d'))

# Standardize column names
df.columns = [col.lower() for col in df.columns]
df = df[['open', 'high', 'low', 'close', 'volume']]

print(f"✓ Fetched {len(df)} days of data")
print(f"  Date range: {df.index[0].date()} to {df.index[-1].date()}")
print(f"  Price range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")

# Plot price history
plt.figure(figsize=(14, 6))
plt.plot(df.index, df['close'], label='Close Price', linewidth=2)
plt.title(f'{symbol} - Price History', fontsize=14, fontweight='bold')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.grid(alpha=0.3)
plt.show()
```

## 3. Calculate Technical Indicators <a name="indicators"></a>

```python
# Add all technical indicators
indicators = TechnicalIndicators()
df_with_indicators = indicators.add_all_indicators(df)

print(f"✓ Added {len(df_with_indicators.columns) - 5} technical indicators")
print(f"\nIndicator categories:")
print(f"  - Trend: SMA, EMA, MACD, ADX")
print(f"  - Momentum: RSI, Stochastic, Williams %R")
print(f"  - Volatility: Bollinger Bands, ATR, Keltner")
print(f"  - Volume: OBV, MFI, Volume trends")

# Display sample indicators
sample_indicators = df_with_indicators[['close', 'sma_20', 'rsi', 'macd', 'bb_high', 'bb_low']].tail(10)
print("\nSample indicators (last 10 days):")
print(sample_indicators)

# Plot key indicators
fig, axes = plt.subplots(3, 1, figsize=(14, 10))

# Price with moving averages
axes[0].plot(df_with_indicators.index, df_with_indicators['close'], label='Close', linewidth=2)
axes[0].plot(df_with_indicators.index, df_with_indicators['sma_20'], label='SMA 20', linewidth=1.5)
axes[0].plot(df_with_indicators.index, df_with_indicators['ema_20'], label='EMA 20', linewidth=1.5)
axes[0].set_title('Price with Moving Averages', fontweight='bold')
axes[0].legend()
axes[0].grid(alpha=0.3)

# RSI
axes[1].plot(df_with_indicators.index, df_with_indicators['rsi'], label='RSI', color='purple', linewidth=2)
axes[1].axhline(y=70, color='r', linestyle='--', label='Overbought (70)')
axes[1].axhline(y=30, color='g', linestyle='--', label='Oversold (30)')
axes[1].set_title('RSI (Relative Strength Index)', fontweight='bold')
axes[1].legend()
axes[1].grid(alpha=0.3)

# MACD
axes[2].plot(df_with_indicators.index, df_with_indicators['macd'], label='MACD', linewidth=2)
axes[2].plot(df_with_indicators.index, df_with_indicators['macd_signal'], label='Signal', linewidth=2)
axes[2].bar(df_with_indicators.index, df_with_indicators['macd_diff'], label='Histogram', alpha=0.3)
axes[2].set_title('MACD (Moving Average Convergence Divergence)', fontweight='bold')
axes[2].legend()
axes[2].grid(alpha=0.3)

plt.tight_layout()
plt.show()
```

## 4. Train ML Model <a name="train"></a>

```python
# Create training pipeline
pipeline = TrainingPipeline(
    symbol=symbol,
    start_date=start_date.strftime('%Y-%m-%d'),
    end_date=end_date.strftime('%Y-%m-%d'),
    model_type='xgboost',
    target_type='direction',
    horizon=1
)

print("Training model...")
print("=" * 60)

# Run training pipeline
results = pipeline.run(use_mlflow=False)

print("\n" + "=" * 60)
print("Training Complete!")
print("=" * 60)

# Display results
print("\nTest Set Performance:")
for metric, value in results['test_metrics'].items():
    print(f"  {metric.upper()}: {value:.4f}")

print("\nTop 10 Most Important Features:")
print(results['feature_importance'].head(10).to_string(index=False))
```

## 5. Generate Predictions <a name="predict"></a>

```python
# Get the trained model
model = results['model']
data_prep = pipeline.data_prep

# Make predictions on test set
test_df = pipeline.test_df
feature_cols = data_prep.feature_columns
X_test = test_df[feature_cols]
y_test = test_df['target']

# Predict
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)

# Add predictions to test data
test_results = test_df.copy()
test_results['prediction'] = predictions
test_results['probability'] = probabilities[:, 1]
test_results['confidence'] = test_results['probability'].apply(
    lambda x: 'High' if x >= 0.8 or x <= 0.2 else 'Medium' if x >= 0.6 or x <= 0.4 else 'Low'
)

print("Sample Predictions (last 10 days):")
display_cols = ['close', 'target', 'prediction', 'probability', 'confidence']
print(test_results[display_cols].tail(10))

# Calculate accuracy
accuracy = (test_results['target'] == test_results['prediction']).mean()
print(f"\nTest Accuracy: {accuracy:.2%}")
```

## 6. Evaluate Performance <a name="evaluate"></a>

```python
from sklearn.metrics import confusion_matrix, classification_report

# Confusion Matrix
cm = confusion_matrix(y_test, predictions)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot confusion matrix
im = axes[0].imshow(cm, interpolation='nearest', cmap='Blues')
axes[0].figure.colorbar(im, ax=axes[0])
axes[0].set_title('Confusion Matrix', fontweight='bold')
axes[0].set_xlabel('Predicted Label')
axes[0].set_ylabel('True Label')

# Add text annotations
thresh = cm.max() / 2.
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        axes[0].text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=16, fontweight='bold')

# Plot feature importance
top_features = results['feature_importance'].head(10)
axes[1].barh(range(len(top_features)), top_features['importance'])
axes[1].set_yticks(range(len(top_features)))
axes[1].set_yticklabels(top_features['feature'])
axes[1].set_xlabel('Importance')
axes[1].set_title('Top 10 Feature Importance', fontweight='bold')
axes[1].invert_yaxis()

plt.tight_layout()
plt.show()

# Print classification report
print("\nDetailed Classification Report:")
print(classification_report(y_test, predictions, 
                           target_names=['Down/Hold', 'Up']))
```

## Conclusion

This notebook demonstrated:
- ✓ Fetching market data from Yahoo Finance
- ✓ Calculating 40+ technical indicators
- ✓ Training an XGBoost classifier
- ✓ Making predictions with confidence scores
- ✓ Evaluating model performance

## Next Steps

1. **Experiment with different models**: Try LightGBM, Random Forest
2. **Tune hyperparameters**: Optimize n_estimators, max_depth, learning_rate
3. **Add custom features**: Create your own technical indicators
4. **Test different symbols**: Try stocks, ETFs, cryptocurrencies
5. **Deploy the model**: Use the FastAPI server for real-time predictions

For more information, see the [README.md](../README.md) and [CONTRIBUTING.md](../CONTRIBUTING.md) files.
