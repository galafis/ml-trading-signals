"""
Data preparation and feature engineering for ML models.
"""
import pandas as pd
import numpy as np
from typing import Tuple, List, Optional
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit


class DataPreparation:
    """
    Prepare data for machine learning models.
    """

    def __init__(self, scaler_type: str = "standard"):
        """
        Initialize data preparation.

        Args:
            scaler_type: Type of scaler ('standard' or 'minmax')
        """
        self.scaler_type = scaler_type
        self.scaler = StandardScaler() if scaler_type == "standard" else MinMaxScaler()
        self.feature_columns: List[str] = []

    def create_target_variable(
        self,
        df: pd.DataFrame,
        target_type: str = "direction",
        horizon: int = 1,
        threshold: float = 0.0,
    ) -> pd.DataFrame:
        """
        Create target variable for prediction.

        Args:
            df: DataFrame with price data
            target_type: Type of target ('direction', 'returns', 'binary')
            horizon: Prediction horizon in periods
            threshold: Threshold for binary classification

        Returns:
            DataFrame with target variable
        """
        data = df.copy()

        if target_type == "direction":
            # Predict price direction (1=up, 0=down)
            data["target"] = (data["close"].shift(-horizon) > data["close"]).astype(int)

        elif target_type == "returns":
            # Predict future returns
            data["target"] = data["close"].pct_change(horizon).shift(-horizon)

        elif target_type == "binary":
            # Binary classification with threshold
            future_returns = data["close"].pct_change(horizon).shift(-horizon)
            data["target"] = (future_returns > threshold).astype(int)

        else:
            raise ValueError(f"Unknown target type: {target_type}")

        return data

    def prepare_features(
        self, df: pd.DataFrame, feature_columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Prepare features for modeling.

        Args:
            df: DataFrame with features
            feature_columns: List of feature columns to use

        Returns:
            DataFrame with prepared features
        """
        data = df.copy()

        # Select feature columns
        if feature_columns is None:
            # Exclude OHLCV and target columns
            exclude_cols = ["open", "high", "low", "close", "volume", "target"]
            feature_columns = [col for col in data.columns if col not in exclude_cols]

        self.feature_columns = feature_columns

        # Handle missing values
        data[feature_columns] = data[feature_columns].ffill().bfill()
        # Handle infinite values
        data[feature_columns] = data[feature_columns].replace([np.inf, -np.inf], np.nan)
        data[feature_columns] = data[feature_columns].fillna(0)

        return data

    def split_data(
        self, df: pd.DataFrame, train_size: float = 0.7, val_size: float = 0.15
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data into train, validation, and test sets (time-series aware).

        Args:
            df: DataFrame with features and target
            train_size: Proportion of data for training
            val_size: Proportion of data for validation

        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        n = len(df)
        train_end = int(n * train_size)
        val_end = int(n * (train_size + val_size))

        train_df = df.iloc[:train_end]
        val_df = df.iloc[train_end:val_end]
        test_df = df.iloc[val_end:]

        return train_df, val_df, test_df

    def scale_features(
        self, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Scale features using fitted scaler.

        Args:
            train_df: Training dataframe
            val_df: Validation dataframe
            test_df: Test dataframe

        Returns:
            Tuple of scaled dataframes
        """
        # Fit scaler on training data
        self.scaler.fit(train_df[self.feature_columns])

        # Transform all sets
        train_scaled = train_df.copy()
        val_scaled = val_df.copy()
        test_scaled = test_df.copy()

        train_scaled[self.feature_columns] = self.scaler.transform(
            train_df[self.feature_columns]
        )
        val_scaled[self.feature_columns] = self.scaler.transform(
            val_df[self.feature_columns]
        )
        test_scaled[self.feature_columns] = self.scaler.transform(
            test_df[self.feature_columns]
        )

        return train_scaled, val_scaled, test_scaled

    def get_time_series_splits(
        self, df: pd.DataFrame, n_splits: int = 5
    ) -> TimeSeriesSplit:
        """
        Get time series cross-validation splits.

        Args:
            df: DataFrame with data
            n_splits: Number of splits

        Returns:
            TimeSeriesSplit object
        """
        return TimeSeriesSplit(n_splits=n_splits)

    def prepare_for_modeling(
        self,
        df: pd.DataFrame,
        target_type: str = "direction",
        horizon: int = 1,
        feature_columns: Optional[List[str]] = None,
        train_size: float = 0.7,
        val_size: float = 0.15,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Complete data preparation pipeline.

        Args:
            df: DataFrame with features
            target_type: Type of target variable
            horizon: Prediction horizon
            feature_columns: List of feature columns
            train_size: Training set size
            val_size: Validation set size

        Returns:
            Tuple of (train_df, val_df, test_df) ready for modeling
        """
        # Create target variable
        data = self.create_target_variable(df, target_type, horizon)

        # Prepare features
        data = self.prepare_features(data, feature_columns)

        # Remove rows with NaN target
        data = data.dropna(subset=["target"])

        # Split data
        train_df, val_df, test_df = self.split_data(data, train_size, val_size)

        # Scale features
        train_df, val_df, test_df = self.scale_features(train_df, val_df, test_df)

        return train_df, val_df, test_df
