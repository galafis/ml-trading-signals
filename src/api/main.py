"""
FastAPI application for ML trading signals.
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
from datetime import datetime
import yfinance as yf

from src.features.technical_indicators import TechnicalIndicators
from src.models.classifier import TradingClassifier

app = FastAPI(
    title="ML Trading Signals API",
    version="1.0.0",
    description="Machine Learning API for generating trading signals",
)


class PredictionRequest(BaseModel):
    """Request model for predictions."""

    symbol: str = Field(..., description="Trading symbol")
    model_path: str = Field(..., description="Path to trained model")
    model_type: str = Field("xgboost", description="Type of model")
    lookback_days: int = Field(100, description="Number of days to fetch for features")


class PredictionResponse(BaseModel):
    """Response model for predictions."""

    symbol: str
    timestamp: datetime
    signal: int
    probability: float
    confidence: str
    current_price: float


class FeatureImportanceResponse(BaseModel):
    """Response model for feature importance."""

    feature: str
    importance: float


@app.get("/")
def root():
    """Root endpoint."""
    return {"message": "ML Trading Signals API", "version": "1.0.0", "docs": "/docs"}


@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


@app.post("/predict", response_model=PredictionResponse)
async def predict_signal(request: PredictionRequest):
    """
    Generate trading signal for a symbol.
    """
    try:
        # Fetch recent data
        ticker = yf.Ticker(request.symbol)
        df = ticker.history(period=f"{request.lookback_days}d")

        if df.empty:
            raise HTTPException(
                status_code=404, detail=f"No data found for {request.symbol}"
            )

        # Standardize columns
        df.columns = [col.lower() for col in df.columns]
        df = df[["open", "high", "low", "close", "volume"]]

        # Engineer features
        df = TechnicalIndicators.add_all_indicators(df)

        # Load model
        model = TradingClassifier.load_model(request.model_path, request.model_type)

        # Get latest features
        latest_data = df.iloc[[-1]]
        feature_cols = [
            col
            for col in df.columns
            if col not in ["open", "high", "low", "close", "volume"]
        ]
        X = latest_data[feature_cols].fillna(0).replace([np.inf, -np.inf], 0)

        # Make prediction
        signal = int(model.predict(X)[0])
        probability = float(model.predict_proba(X)[0][signal])

        # Determine confidence
        if probability >= 0.8:
            confidence = "high"
        elif probability >= 0.6:
            confidence = "medium"
        else:
            confidence = "low"

        return PredictionResponse(
            symbol=request.symbol,
            timestamp=datetime.now(),
            signal=signal,
            probability=probability,
            confidence=confidence,
            current_price=float(latest_data["close"].iloc[0]),
        )

    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Model file not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/feature-importance", response_model=List[FeatureImportanceResponse])
async def get_feature_importance(
    model_path: str, model_type: str = "xgboost", top_n: int = 20
):
    """
    Get feature importance from trained model.
    """
    try:
        model = TradingClassifier.load_model(model_path, model_type)
        importance_df = model.get_feature_importance(top_n)

        return [
            FeatureImportanceResponse(
                feature=row["feature"], importance=float(row["importance"])
            )
            for _, row in importance_df.iterrows()
        ]

    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Model file not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/symbols")
def get_supported_symbols():
    """
    Get list of supported symbols.
    """
    return {
        "symbols": [
            "^BVSP",  # Bovespa Index
            "PETR4.SA",  # Petrobras
            "VALE3.SA",  # Vale
            "ITUB4.SA",  # Itaú
            "BBDC4.SA",  # Bradesco
            "ABEV3.SA",  # Ambev
            "B3SA3.SA",  # B3
            "WEGE3.SA",  # WEG
            "RENT3.SA",  # Localiza
            "MGLU3.SA",  # Magazine Luiza
        ]
    }
