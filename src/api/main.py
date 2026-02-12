"""
FastAPI application for ML trading signals.
"""
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List
import pandas as pd
import numpy as np
from datetime import datetime
import yfinance as yf
import joblib

from src.features.technical_indicators import TechnicalIndicators
from src.features.data_preparation import DataPreparation
from src.models.classifier import TradingClassifier

# Directory where saved models are stored (configurable via env var)
MODELS_DIR = os.environ.get("MODELS_DIR", "models")

app = FastAPI(
    title="ML Trading Signals API",
    version="1.0.0",
    description="Machine Learning API for generating trading signals",
)


class PredictionRequest(BaseModel):
    """Request model for predictions."""

    symbol: str = Field(..., description="Trading symbol (e.g. ^BVSP, PETR4.SA)")
    model_name: str = Field(
        ..., description="Name of the saved model file (e.g. xgboost_bvsp.pkl)"
    )
    model_type: str = Field("xgboost", description="Type of model")
    lookback_days: int = Field(
        100, ge=50, le=500, description="Number of days to fetch for features"
    )


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


def _resolve_model_path(model_name: str) -> str:
    """
    Resolve model name to a safe file path inside MODELS_DIR.

    Prevents path traversal by rejecting names with directory separators
    and ensuring the resolved path stays inside MODELS_DIR.
    """
    if os.sep in model_name or "/" in model_name or "\\" in model_name:
        raise HTTPException(
            status_code=400,
            detail="Model name must not contain path separators",
        )

    path = os.path.normpath(os.path.join(MODELS_DIR, model_name))

    # Ensure resolved path is inside MODELS_DIR
    if not path.startswith(os.path.normpath(MODELS_DIR)):
        raise HTTPException(status_code=400, detail="Invalid model name")

    if not os.path.isfile(path):
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")

    return path


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

    The model must be trained and saved in the models/ directory first
    (see train.py). Features are scaled using the same StandardScaler
    parameters saved alongside the model.
    """
    try:
        model_path = _resolve_model_path(request.model_name)

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

        # Determine feature columns (same logic as DataPreparation)
        exclude_cols = {"open", "high", "low", "close", "volume", "target"}
        feature_cols = [col for col in df.columns if col not in exclude_cols]

        # Clean features (same as DataPreparation.prepare_features)
        df[feature_cols] = df[feature_cols].ffill().bfill()
        df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan)
        df[feature_cols] = df[feature_cols].fillna(0)

        # Load model
        model = TradingClassifier.load_model(model_path, request.model_type)

        # Get latest row
        latest_data = df.iloc[[-1]]
        X = latest_data[feature_cols]

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

    except HTTPException:
        raise
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Model file not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/feature-importance", response_model=List[FeatureImportanceResponse])
async def get_feature_importance(
    model_name: str, model_type: str = "xgboost", top_n: int = 20
):
    """
    Get feature importance from trained model.
    """
    try:
        model_path = _resolve_model_path(model_name)
        model = TradingClassifier.load_model(model_path, model_type)
        importance_df = model.get_feature_importance(top_n)

        return [
            FeatureImportanceResponse(
                feature=row["feature"], importance=float(row["importance"])
            )
            for _, row in importance_df.iterrows()
        ]

    except HTTPException:
        raise
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Model file not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/symbols")
def get_supported_symbols():
    """
    Get list of suggested Brazilian stock symbols.
    """
    return {
        "symbols": [
            "^BVSP",
            "PETR4.SA",
            "VALE3.SA",
            "ITUB4.SA",
            "BBDC4.SA",
            "ABEV3.SA",
            "B3SA3.SA",
            "WEGE3.SA",
            "RENT3.SA",
            "MGLU3.SA",
        ]
    }
