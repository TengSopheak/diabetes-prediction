"""
Data preprocessing and cleaning operations for the diabetes prediction pipeline.
Handles missing values, categorical mapping, and data type conversions.
"""

import logging
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from config.settings import (
    GENDER_MAPPING,
    SMOKING_MAPPING,
    NUMERICAL_COLUMNS,
)

logger = logging.getLogger(__name__)


class DataCleaner:
    """
    Handles data cleaning operations including categorical mapping and missing value handling.
    
    This class applies consistent cleaning operations across train, validation, and test sets
    using parameters learned from the training data only to prevent data leakage.
    """
    
    def __init__(
        self,
        gender_mapping: Optional[Dict[str, str]] = None,
        smoking_mapping: Optional[Dict[str, str]] = None,
    ):
        """
        Initialize the DataCleaner with mapping configurations.
        
        Args:
            gender_mapping: Dictionary mapping gender categories to standard values
            smoking_mapping: Dictionary mapping smoking history categories to standard values
        """
        self.gender_mapping = gender_mapping or GENDER_MAPPING
        self.smoking_mapping = smoking_mapping or SMOKING_MAPPING
        
    def map_categorical_variables(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize categorical variable values using configured mappings.
        
        Maps gender and smoking_history columns to consistent categories.
        Handles unknown categories by leaving them as-is (to be handled by encoder).
        
        Args:
            df: DataFrame with 'gender' and 'smoking_history' columns
            
        Returns:
            DataFrame with standardized categorical values
            
        Raises:
            KeyError: If required columns are missing
        """
        df = df.copy()
        
        # Map gender
        if "gender" in df.columns:
            logger.debug("Mapping gender categories")
            df["gender"] = df["gender"].map(self.gender_mapping)
            # Handle any unmapped values (e.g., 'Other' already handled in mapping)
            unmapped = df["gender"].isna()
            if unmapped.any():
                logger.warning(f"Found {unmapped.sum()} unmapped gender values, setting to 'Male'")
                df.loc[unmapped, "gender"] = "Male"
        
        # Map smoking history
        if "smoking_history" in df.columns:
            logger.debug("Mapping smoking history categories")
            df["smoking_history"] = df["smoking_history"].map(self.smoking_mapping)
            unmapped = df["smoking_history"].isna()
            if unmapped.any():
                unknown_count = unmapped.sum()
                logger.warning(f"Found {unknown_count} unmapped smoking_history values, setting to 'unknown'")
                df.loc[unmapped, "smoking_history"] = "unknown"
                
        return df
    
    def check_missing_values(self, df: pd.DataFrame, dataset_name: str = "dataset") -> pd.Series:
        """
        Check and report missing values in the dataframe.
        
        Args:
            df: DataFrame to check
            dataset_name: Name identifier for logging purposes
            
        Returns:
            Series with count of missing values per column
        """
        missing = df.isnull().sum()
        if missing.any():
            logger.warning(f"Missing values found in {dataset_name}:\n{missing[missing > 0]}")
        else:
            logger.info(f"No missing values in {dataset_name}")
        return missing
    
    def fit_transform(
        self, 
        train_df: pd.DataFrame, 
        val_df: pd.DataFrame, 
        test_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Apply all cleaning operations to train, val, and test sets.
        
        Uses training data to determine any fitted parameters, then applies
        consistently to all sets.
        
        Args:
            train_df: Training dataframe
            val_df: Validation dataframe
            test_df: Test dataframe
            
        Returns:
            Tuple of cleaned (train_df, val_df, test_df)
        """
        logger.info("Starting data cleaning")
        
        # Check for missing values in all sets
        self.check_missing_values(train_df, "training set")
        self.check_missing_values(val_df, "validation set")
        self.check_missing_values(test_df, "test set")
        
        # Apply categorical mapping (same mapping for all sets)
        train_clean = self.map_categorical_variables(train_df)
        val_clean = self.map_categorical_variables(val_df)
        test_clean = self.map_categorical_variables(test_df)
        
        # Log category distributions
        if "gender" in train_clean.columns:
            logger.info(f"Gender distribution in train: {train_clean['gender'].value_counts().to_dict()}")
        if "smoking_history" in train_clean.columns:
            logger.info(f"Smoking history distribution in train: {train_clean['smoking_history'].value_counts().to_dict()}")
        
        logger.info("Data cleaning completed")
        return train_clean, val_clean, test_clean


class OutlierHandler:
    """
    Handles outlier detection and capping using IQR method.
    
    Fits bounds on training data only, then applies same bounds to validation and test.
    """
    
    def __init__(self, columns: Optional[list] = None, iqr_multiplier: float = 1.5):
        """
        Initialize the OutlierHandler.
        
        Args:
            columns: List of numerical columns to check for outliers
            iqr_multiplier: Multiplier for IQR to determine outlier bounds
        """
        self.columns = columns or NUMERICAL_COLUMNS
        self.iqr_multiplier = iqr_multiplier
        self.bounds_: Dict[str, Dict[str, float]] = {}
        
    def fit(self, df: pd.DataFrame) -> "OutlierHandler":
        """
        Calculate outlier bounds from training data.
        
        Args:
            df: Training dataframe
            
        Returns:
            Self for method chaining
        """
        logger.info("Fitting outlier bounds on training data")
        
        for col in self.columns:
            if col not in df.columns:
                logger.warning(f"Column {col} not found in dataframe, skipping")
                continue
                
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - self.iqr_multiplier * IQR
            upper_bound = Q3 + self.iqr_multiplier * IQR
            
            self.bounds_[col] = {
                "lower": lower_bound,
                "upper": upper_bound,
                "Q1": Q1,
                "Q3": Q3,
                "IQR": IQR
            }
            
            outlier_count = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
            logger.info(f"{col}: bounds [{lower_bound:.2f}, {upper_bound:.2f}], "
                       f"outliers in train: {outlier_count}")
                       
        return self
    
    def transform(self, df: pd.DataFrame, dataset_name: str = "dataset") -> pd.DataFrame:
        """
        Apply outlier capping using fitted bounds.
        
        Args:
            df: DataFrame to transform
            dataset_name: Name identifier for logging
            
        Returns:
            DataFrame with outliers capped
        """
        df = df.copy()
        
        for col, bounds in self.bounds_.items():
            if col not in df.columns:
                continue
                
            lower_bound = bounds["lower"]
            upper_bound = bounds["upper"]
            
            # Count outliers before capping
            lower_outliers = (df[col] < lower_bound).sum()
            upper_outliers = (df[col] > upper_bound).sum()
            
            # Cap outliers
            df[col] = np.where(df[col] < lower_bound, lower_bound, df[col])
            df[col] = np.where(df[col] > upper_bound, upper_bound, df[col])
            
            if lower_outliers > 0 or upper_outliers > 0:
                logger.debug(f"{dataset_name} - {col}: capped {lower_outliers} lower and "
                           f"{upper_outliers} upper outliers")
                           
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
            Tuple of transformed (train_df, val_df, test_df)
        """
        self.fit(train_df)
        return (
            self.transform(train_df, "train"),
            self.transform(val_df, "val"),
            self.transform(test_df, "test")
        )


if __name__ == "__main__":
    # Example usage
    from data.load_data import DataLoader
    
    loader = DataLoader()
    raw_df = loader.load_raw_data()
    
    # Map categorical variables before splitting to ensure consistent mapping across sets
    mapper = DataCleaner()
    raw_df = mapper.map_categorical_variables(raw_df)

    X_train, X_val, X_test, y_train, y_val, y_test = loader.split_data(raw_df)
    
    # Clean data
    cleaner = DataCleaner()
    X_train_clean, X_val_clean, X_test_clean = cleaner.fit_transform(X_train, X_val, X_test)
    
    # Handle outliers
    outlier_handler = OutlierHandler()
    X_train_processed, X_val_processed, X_test_processed = outlier_handler.fit_transform(
        X_train_clean, X_val_clean, X_test_clean
    )
    
    print("Preprocessing complete")
