"""
Machine learning classifiers for trading signal generation.
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import xgboost as xgb
import lightgbm as lgb
import joblib


class TradingClassifier:
    """
    Classifier for predicting trading signals (buy/sell/hold).
    """
    
    def __init__(self, model_type: str = 'xgboost', **kwargs):
        """
        Initialize classifier.
        
        Args:
            model_type: Type of model ('xgboost', 'lightgbm', 'random_forest', 'gradient_boosting', 'logistic')
            **kwargs: Model-specific parameters
        """
        self.model_type = model_type
        self.model = self._create_model(**kwargs)
        self.feature_importance: Optional[pd.DataFrame] = None
    
    def _create_model(self, **kwargs):
        """Create model based on type."""
        if self.model_type == 'xgboost':
            return xgb.XGBClassifier(
                n_estimators=kwargs.get('n_estimators', 100),
                max_depth=kwargs.get('max_depth', 6),
                learning_rate=kwargs.get('learning_rate', 0.1),
                random_state=kwargs.get('random_state', 42),
                use_label_encoder=False,
                eval_metric='logloss'
            )
        
        elif self.model_type == 'lightgbm':
            return lgb.LGBMClassifier(
                n_estimators=kwargs.get('n_estimators', 100),
                max_depth=kwargs.get('max_depth', 6),
                learning_rate=kwargs.get('learning_rate', 0.1),
                random_state=kwargs.get('random_state', 42),
                verbose=-1
            )
        
        elif self.model_type == 'random_forest':
            return RandomForestClassifier(
                n_estimators=kwargs.get('n_estimators', 100),
                max_depth=kwargs.get('max_depth', 10),
                random_state=kwargs.get('random_state', 42)
            )
        
        elif self.model_type == 'gradient_boosting':
            return GradientBoostingClassifier(
                n_estimators=kwargs.get('n_estimators', 100),
                max_depth=kwargs.get('max_depth', 6),
                learning_rate=kwargs.get('learning_rate', 0.1),
                random_state=kwargs.get('random_state', 42)
            )
        
        elif self.model_type == 'logistic':
            return LogisticRegression(
                random_state=kwargs.get('random_state', 42),
                max_iter=kwargs.get('max_iter', 1000)
            )
        
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None
    ) -> Dict[str, float]:
        """
        Train the classifier.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            
        Returns:
            Dictionary of training metrics
        """
        # Train model
        if self.model_type in ['xgboost', 'lightgbm'] and X_val is not None:
            self.model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False
            )
        else:
            self.model.fit(X_train, y_train)
        
        # Calculate feature importance
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importance = pd.DataFrame({
                'feature': X_train.columns,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
        
        # Calculate training metrics
        y_pred = self.model.predict(X_train)
        y_pred_proba = self.model.predict_proba(X_train)[:, 1]
        
        metrics = {
            'train_accuracy': accuracy_score(y_train, y_pred),
            'train_precision': precision_score(y_train, y_pred, zero_division=0),
            'train_recall': recall_score(y_train, y_pred, zero_division=0),
            'train_f1': f1_score(y_train, y_pred, zero_division=0),
            'train_auc': roc_auc_score(y_train, y_pred_proba)
        }
        
        # Calculate validation metrics if provided
        if X_val is not None and y_val is not None:
            val_metrics = self.evaluate(X_val, y_val)
            metrics.update(val_metrics)
        
        return metrics
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict class labels.
        
        Args:
            X: Features
            
        Returns:
            Predicted labels
        """
        return self.model.predict(X)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict class probabilities.
        
        Args:
            X: Features
            
        Returns:
            Predicted probabilities
        """
        return self.model.predict_proba(X)
    
    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """
        Evaluate model performance.
        
        Args:
            X: Features
            y: True labels
            
        Returns:
            Dictionary of evaluation metrics
        """
        y_pred = self.predict(X)
        y_pred_proba = self.predict_proba(X)[:, 1]
        
        return {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred, zero_division=0),
            'recall': recall_score(y, y_pred, zero_division=0),
            'f1': f1_score(y, y_pred, zero_division=0),
            'auc': roc_auc_score(y, y_pred_proba)
        }
    
    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """
        Get top N most important features.
        
        Args:
            top_n: Number of top features to return
            
        Returns:
            DataFrame with feature importance
        """
        if self.feature_importance is None:
            raise ValueError("Model must be trained first")
        
        return self.feature_importance.head(top_n)
    
    def save_model(self, filepath: str):
        """
        Save model to file.
        
        Args:
            filepath: Path to save model
        """
        joblib.dump(self.model, filepath)
    
    @classmethod
    def load_model(cls, filepath: str, model_type: str) -> 'TradingClassifier':
        """
        Load model from file.
        
        Args:
            filepath: Path to load model from
            model_type: Type of model
            
        Returns:
            TradingClassifier instance
        """
        classifier = cls(model_type=model_type)
        classifier.model = joblib.load(filepath)
        return classifier
