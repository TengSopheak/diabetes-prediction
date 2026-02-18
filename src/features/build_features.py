"""
Feature engineering logic for the diabetes prediction pipeline.
Handles encoding, scaling, feature selection, and balancing.
"""

import logging
from typing import List, Optional, Tuple, Dict

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from imblearn.over_sampling import SMOTE

from config.settings import (
    NUMERICAL_COLUMNS,
    CATEGORICAL_COLUMNS,
    CORRELATION_THRESHOLD,
    SMOTE_RANDOM_STATE,
    SCALER_PATH,
    PROCESSED_DIR,
    X_TRAIN_PATH,
    Y_TRAIN_PATH,
    X_VAL_PATH,
    Y_VAL_PATH,
    X_TEST_PATH,
    Y_TEST_PATH,
)

logger = logging.getLogger(__name__)


class FeatureEncoder:
    """
    Handles one-hot encoding of categorical variables.
    
    Fits encoders on training data only, then transforms all sets consistently.
    """
    
    def __init__(self, drop: str = "first", handle_unknown: str = "ignore"):
        """
        Initialize the encoder configuration.
        
        Args:
            drop: Strategy for dropping categories ('first', 'if_binary', or None)
            handle_unknown: How to handle unknown categories ('error' or 'ignore')
        """
        self.drop = drop
        self.handle_unknown = handle_unknown
        self.encoders_: Dict[str, OneHotEncoder] = {}
        self.feature_names_: Dict[str, List[str]] = {}
        
    def fit(self, train_df: pd.DataFrame) -> "FeatureEncoder":
        """
        Fit one-hot encoders on training data.
        
        Args:
            train_df: Training dataframe with categorical columns
            
        Returns:
            Self for method chaining
        """
        logger.info("Fitting categorical encoders")
        
        for col in CATEGORICAL_COLUMNS:
            if col not in train_df.columns:
                logger.warning(f"Categorical column {col} not found, skipping")
                continue
                
            # Special handling for gender to drop 'Male'
            categories = None
            drop_param = self.drop
            
            if col == "gender":
                categories = [["Male", "Female"]]
                drop_param = "first"
            
            if col == "smoking_history":
                categories = [["never", "unknown", "current", "former"]]
                drop_param = None
            
            encoder = OneHotEncoder(
                handle_unknown=self.handle_unknown,
                sparse_output=False,
                categories=categories,
                drop=drop_param
            )
            
            encoder.fit(train_df[[col]])
            self.encoders_[col] = encoder
            self.feature_names_[col] = encoder.get_feature_names_out([col]).tolist()
            
            logger.info(f"Fitted encoder for {col}: {self.feature_names_[col]}")
            
        return self
    
    def transform(self, df: pd.DataFrame, dataset_name: str = "dataset") -> pd.DataFrame:
        """
        Transform dataframe using fitted encoders.
        
        Args:
            df: DataFrame to transform
            dataset_name: Name identifier for logging
            
        Returns:
            DataFrame with encoded categorical variables and original categoricals dropped
        """
        df = df.copy()
        
        for col, encoder in self.encoders_.items():
            if col not in df.columns:
                continue
                
            # Transform
            encoded = encoder.transform(df[[col]])
            
            # Create DataFrame with proper column names
            encoded_df = pd.DataFrame(
                encoded,
                columns=self.feature_names_[col],
                index=df.index
            )
            
            # Drop original column and join encoded
            df = df.drop(columns=[col])
            df = pd.concat([df, encoded_df], axis=1)
            
            logger.debug(f"{dataset_name}: encoded {col} -> {self.feature_names_[col]}")
            
        return df
    
    def fit_transform(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Fit on training data and transform all sets.
        
        Args:
            train_df: Training dataframe
            val_df: Validation dataframe
            test_df: Test dataframe
            
        Returns:
            Tuple of encoded (train_df, val_df, test_df)
        """
        self.fit(train_df)
        return (
            self.transform(train_df, "train"),
            self.transform(val_df, "val"),
            self.transform(test_df, "test")
        )


class FeatureScaler:
    """
    Handles standardization of numerical features.
    
    Fits scaler on training data only, then applies to all sets.
    """
    
    def __init__(self, columns: Optional[List[str]] = None):
        """
        Initialize the scaler.
        
        Args:
            columns: List of numerical columns to scale
        """
        self.columns = columns or NUMERICAL_COLUMNS
        self.scaler_ = StandardScaler()
        self.is_fitted_ = False
        
    def fit(self, train_df: pd.DataFrame) -> "FeatureScaler":
        """
        Fit scaler on training data.
        
        Args:
            train_df: Training dataframe
            
        Returns:
            Self for method chaining
        """
        logger.info(f"Fitting scaler on columns: {self.columns}")
        
        available_cols = [col for col in self.columns if col in train_df.columns]
        if not available_cols:
            raise ValueError(f"None of the specified columns found in dataframe: {self.columns}")
            
        self.scaler_.fit(train_df[available_cols])
        self.is_fitted_ = True
        
        logger.info(f"Scaler fitted on {len(available_cols)} features")
        return self
    
    def transform(self, df: pd.DataFrame, dataset_name: str = "dataset") -> pd.DataFrame:
        """
        Scale numerical features in dataframe.
        
        Args:
            df: DataFrame to transform
            dataset_name: Name identifier for logging
            
        Returns:
            DataFrame with scaled numerical features
        """
        if not self.is_fitted_:
            raise RuntimeError("Scaler must be fitted before transform")
            
        df = df.copy()
        available_cols = [col for col in self.columns if col in df.columns]
        
        if available_cols:
            df[available_cols] = self.scaler_.transform(df[available_cols])
            logger.debug(f"{dataset_name}: scaled {available_cols}")
            
        return df
    
    def fit_transform(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Fit on training data and transform all sets.
        
        Args:
            train_df: Training dataframe
            val_df: Validation dataframe
            test_df: Test dataframe
            
        Returns:
            Tuple of scaled (train_df, val_df, test_df)
        """
        self.fit(train_df)
        return (
            self.transform(train_df, "train"),
            self.transform(val_df, "val"),
            self.transform(test_df, "test")
        )
    
    def save(self, path: str = str(SCALER_PATH)) -> None:
        """Save fitted scaler to disk."""
        if not self.is_fitted_:
            raise RuntimeError("Cannot save unfitted scaler")
        joblib.dump(self.scaler_, path)
        logger.info(f"Scaler saved to {path}")
    
    def load(self, path: str = str(SCALER_PATH)) -> "FeatureScaler":
        """Load fitted scaler from disk."""
        self.scaler_ = joblib.load(path)
        self.is_fitted_ = True
        logger.info(f"Scaler loaded from {path}")
        return self


class FeatureSelector:
    """
    Handles feature selection based on correlation with target.
    
    Selects features with absolute correlation above threshold.
    """
    
    def __init__(self, threshold: float = CORRELATION_THRESHOLD):
        """
        Initialize feature selector.
        
        Args:
            threshold: Minimum absolute correlation for feature selection
        """
        self.threshold = threshold
        self.selected_features_: List[str] = []
        self.correlations_: pd.Series = None
        
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "FeatureSelector":
        """
        Calculate correlations and select features.
        
        Args:
            X: Feature matrix
            y: Target vector
            
        Returns:
            Self for method chaining
        """
        logger.info(f"Selecting features with |correlation| > {self.threshold}")
        
        # Calculate correlations
        df = X.copy()
        df['target'] = y.values
        
        self.correlations_ = df.corr(numeric_only=True)['target'].drop('target').sort_values(
            ascending=False, key=abs
        )
        
        # Select features above threshold
        self.selected_features_ = self.correlations_[
            abs(self.correlations_) > self.threshold
        ].index.tolist()
        
        # Maintain original order of columns
        self.selected_features_ = [col for col in X.columns if col in self.selected_features_]
        
        logger.info(f"Selected {len(self.selected_features_)} features: {self.selected_features_}")
        logger.info(f"Top correlations:\n{self.correlations_.head()}")
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Select features from dataframe.
        
        Args:
            X: Feature matrix
            
        Returns:
            DataFrame with selected features only
        """
        if not self.selected_features_:
            raise RuntimeError("FeatureSelector must be fitted before transform")
            
        return X[self.selected_features_]
    
    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """
        Fit and transform training data.
        
        Args:
            X: Feature matrix
            y: Target vector
            
        Returns:
            DataFrame with selected features
        """
        self.fit(X, y)
        return self.transform(X)


class DataBalancer:
    """
    Handles class imbalance using SMOTE oversampling.
    
    Only applied to training data.
    """
    
    def __init__(self, random_state: int = SMOTE_RANDOM_STATE):
        """
        Initialize the balancer.
        
        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        self.smote_ = SMOTE(random_state=random_state)
        
    def fit_resample(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Apply SMOTE to balance training data.
        
        Args:
            X: Training features
            y: Training target
            
        Returns:
            Tuple of balanced (X, y)
        """
        logger.info("Applying SMOTE to balance training data")
        logger.info(f"Original class distribution:\n{y.value_counts()}")
        
        X_balanced, y_balanced = self.smote_.fit_resample(X, y)
        
        # Convert back to DataFrame/Series to maintain column names
        X_balanced = pd.DataFrame(X_balanced, columns=X.columns)
        y_balanced = pd.Series(y_balanced, name=y.name)
        
        logger.info(f"Balanced class distribution:\n{y_balanced.value_counts()}")
        
        return X_balanced, y_balanced


def save_processed_data(
    X_train: pd.DataFrame, y_train: pd.Series,
    X_val: pd.DataFrame, y_val: pd.Series,
    X_test: pd.DataFrame, y_test: pd.Series,
    selected_features: Optional[List[str]] = None
) -> None:
    """
    Save all processed datasets to disk.
    
    Args:
        X_train: Processed training features
        y_train: Training targets
        X_val: Processed validation features
        y_val: Validation targets
        X_test: Processed test features
        y_test: Test targets
        selected_features: List of selected feature names (for column ordering)
    """

    # Use selected features if provided, otherwise use all
    cols = selected_features if selected_features else X_train.columns.tolist()
    
    # Ensure all dataframes have the same columns
    X_train[cols].to_csv(X_TRAIN_PATH, index=False)
    y_train.to_csv(Y_TRAIN_PATH, index=False)
    X_val[cols].to_csv(X_VAL_PATH, index=False)
    y_val.to_csv(Y_VAL_PATH, index=False)
    X_test[cols].to_csv(X_TEST_PATH, index=False)
    y_test.to_csv(Y_TEST_PATH, index=False)
    
    logger.info(f"All processed datasets saved to {PROCESSED_DIR}")


def build_features_pipeline(
    X_train: pd.DataFrame, y_train: pd.Series,
    X_val: pd.DataFrame, y_val: pd.Series,
    X_test: pd.DataFrame, y_test: pd.Series,
    apply_smote: bool = True,
    save_data: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """
    Complete feature engineering pipeline.
    
    Executes encoding, scaling, selection, and balancing in sequence.
    
    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data
        X_test, y_test: Test data
        apply_smote: Whether to apply SMOTE balancing to training data
        save_data: Whether to save processed datasets to disk
        
    Returns:
        Tuple of processed (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    logger.info("Starting feature engineering pipeline")
    
    # Step 1: Encode categorical features
    encoder = FeatureEncoder()
    X_train_enc, X_val_enc, X_test_enc = encoder.fit_transform(X_train, X_val, X_test)
    
    # Step 2: Scale numerical features
    scaler = FeatureScaler()
    X_train_scl, X_val_scl, X_test_scl = scaler.fit_transform(X_train_enc, X_val_enc, X_test_enc)
    scaler.save()  # Save scaler for inference
    
    # Step 3: Feature selection (fit on training only)
    selector = FeatureSelector()
    X_train_sel = selector.fit_transform(X_train_scl, y_train)
    X_val_sel = selector.transform(X_val_scl)
    X_test_sel = selector.transform(X_test_scl)
    
    # Step 4: Balance training data with SMOTE
    if apply_smote:
        balancer = DataBalancer()
        X_train_bal, y_train_bal = balancer.fit_resample(X_train_sel, y_train)
    else:
        X_train_bal, y_train_bal = X_train_sel, y_train
    
    # Save processed data
    if save_data:
        save_processed_data(
            X_train_bal, y_train_bal,
            X_val_sel, y_val,
            X_test_sel, y_test,
            selected_features=selector.selected_features_
        )
    
    logger.info("Feature engineering pipeline completed")
    return X_train_bal, X_val_sel, X_test_sel, y_train_bal, y_val, y_test


if __name__ == "__main__":
    # Example usage
    from data.load_data import DataLoader, load_processed_data
    from data.preprocess import DataCleaner, OutlierHandler
    
    # Load and split
    loader = DataLoader()
    raw_df = loader.load_raw_data()
    X_train_raw, X_val_raw, X_test_raw, y_train, y_val, y_test = loader.split_data(raw_df)
    
    # Clean and preprocess
    cleaner = DataCleaner()
    X_train_clean, X_val_clean, X_test_clean = cleaner.fit_transform(X_train_raw, X_val_raw, X_test_raw)
    
    # Handle outliers
    outlier_handler = OutlierHandler()
    X_train_proc, X_val_proc, X_test_proc = outlier_handler.fit_transform(
        X_train_clean, X_val_clean, X_test_clean
    )
    
    # Build features
    build_features_pipeline(
        X_train_proc, y_train,
        X_val_proc, y_val,
        X_test_proc, y_test
    )
