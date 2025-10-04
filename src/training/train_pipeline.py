"""
Training pipeline for ML trading models.
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
import yfinance as yf
from datetime import datetime, timedelta
import mlflow
import mlflow.sklearn

from src.features.technical_indicators import TechnicalIndicators
from src.features.data_preparation import DataPreparation
from src.models.classifier import TradingClassifier


class TrainingPipeline:
    """
    Complete training pipeline for ML trading models.
    """
    
    def __init__(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        model_type: str = 'xgboost',
        target_type: str = 'direction',
        horizon: int = 1
    ):
        """
        Initialize training pipeline.
        
        Args:
            symbol: Trading symbol
            start_date: Start date for data
            end_date: End date for data
            model_type: Type of ML model
            target_type: Type of target variable
            horizon: Prediction horizon
        """
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date
        self.model_type = model_type
        self.target_type = target_type
        self.horizon = horizon
        
        self.data: Optional[pd.DataFrame] = None
        self.train_df: Optional[pd.DataFrame] = None
        self.val_df: Optional[pd.DataFrame] = None
        self.test_df: Optional[pd.DataFrame] = None
        self.model: Optional[TradingClassifier] = None
        self.data_prep: Optional[DataPreparation] = None
    
    def fetch_data(self) -> pd.DataFrame:
        """
        Fetch market data.
        
        Returns:
            DataFrame with OHLCV data
        """
        print(f"Fetching data for {self.symbol}...")
        ticker = yf.Ticker(self.symbol)
        df = ticker.history(start=self.start_date, end=self.end_date)
        
        # Standardize column names
        df.columns = [col.lower() for col in df.columns]
        df = df[['open', 'high', 'low', 'close', 'volume']]
        
        print(f"Fetched {len(df)} rows of data")
        return df
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer features from raw data.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with engineered features
        """
        print("Engineering features...")
        data = TechnicalIndicators.add_all_indicators(df)
        print(f"Created {len(data.columns)} features")
        return data
    
    def prepare_data(self, df: pd.DataFrame) -> tuple:
        """
        Prepare data for modeling.
        
        Args:
            df: DataFrame with features
            
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        print("Preparing data...")
        self.data_prep = DataPreparation(scaler_type='standard')
        
        train_df, val_df, test_df = self.data_prep.prepare_for_modeling(
            df,
            target_type=self.target_type,
            horizon=self.horizon
        )
        
        print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
        return train_df, val_df, test_df
    
    def train_model(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        model_params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, float]:
        """
        Train ML model.
        
        Args:
            train_df: Training data
            val_df: Validation data
            model_params: Model parameters
            
        Returns:
            Dictionary of training metrics
        """
        print(f"Training {self.model_type} model...")
        
        # Prepare features and target
        feature_cols = self.data_prep.feature_columns
        X_train = train_df[feature_cols]
        y_train = train_df['target']
        X_val = val_df[feature_cols]
        y_val = val_df['target']
        
        # Create and train model
        if model_params is None:
            model_params = {}
        
        self.model = TradingClassifier(model_type=self.model_type, **model_params)
        metrics = self.model.train(X_train, y_train, X_val, y_val)
        
        print("Training completed!")
        print(f"Train Accuracy: {metrics['train_accuracy']:.4f}")
        print(f"Val Accuracy: {metrics['accuracy']:.4f}")
        
        return metrics
    
    def evaluate_model(self, test_df: pd.DataFrame) -> Dict[str, float]:
        """
        Evaluate model on test set.
        
        Args:
            test_df: Test data
            
        Returns:
            Dictionary of test metrics
        """
        print("Evaluating model on test set...")
        
        feature_cols = self.data_prep.feature_columns
        X_test = test_df[feature_cols]
        y_test = test_df['target']
        
        metrics = self.model.evaluate(X_test, y_test)
        
        print(f"Test Accuracy: {metrics['accuracy']:.4f}")
        print(f"Test Precision: {metrics['precision']:.4f}")
        print(f"Test Recall: {metrics['recall']:.4f}")
        print(f"Test F1: {metrics['f1']:.4f}")
        print(f"Test AUC: {metrics['auc']:.4f}")
        
        return metrics
    
    def run(
        self,
        model_params: Optional[Dict[str, Any]] = None,
        use_mlflow: bool = False
    ) -> Dict[str, Any]:
        """
        Run complete training pipeline.
        
        Args:
            model_params: Model parameters
            use_mlflow: Whether to log to MLflow
            
        Returns:
            Dictionary with results
        """
        if use_mlflow:
            mlflow.start_run()
            mlflow.log_param("symbol", self.symbol)
            mlflow.log_param("model_type", self.model_type)
            mlflow.log_param("target_type", self.target_type)
            mlflow.log_param("horizon", self.horizon)
        
        try:
            # Fetch data
            self.data = self.fetch_data()
            
            # Engineer features
            self.data = self.engineer_features(self.data)
            
            # Prepare data
            self.train_df, self.val_df, self.test_df = self.prepare_data(self.data)
            
            # Train model
            train_metrics = self.train_model(self.train_df, self.val_df, model_params)
            
            # Evaluate model
            test_metrics = self.evaluate_model(self.test_df)
            
            # Get feature importance
            feature_importance = self.model.get_feature_importance()
            
            results = {
                'train_metrics': train_metrics,
                'test_metrics': test_metrics,
                'feature_importance': feature_importance,
                'model': self.model
            }
            
            if use_mlflow:
                # Log metrics
                for key, value in train_metrics.items():
                    mlflow.log_metric(key, value)
                for key, value in test_metrics.items():
                    mlflow.log_metric(f"test_{key}", value)
                
                # Log model
                mlflow.sklearn.log_model(self.model.model, "model")
            
            return results
        
        finally:
            if use_mlflow:
                mlflow.end_run()
    
    def save_model(self, filepath: str):
        """
        Save trained model.
        
        Args:
            filepath: Path to save model
        """
        if self.model is None:
            raise ValueError("No model to save. Train model first.")
        
        self.model.save_model(filepath)
        print(f"Model saved to {filepath}")
