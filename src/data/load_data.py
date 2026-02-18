"""
Data loading functionality for the diabetes prediction pipeline.
Handles reading raw data and loading preprocessed datasets.
"""

import logging
from pathlib import Path
from typing import Tuple, Optional

import pandas as pd
from sklearn.model_selection import train_test_split

from config.settings import (
    RAW_DATA_PATH,
    X_TRAIN_PATH,
    Y_TRAIN_PATH,
    X_VAL_PATH,
    Y_VAL_PATH,
    X_TEST_PATH,
    Y_TEST_PATH,
    TEST_SIZE,
    VALIDATION_SIZE,
    RANDOM_STATE,
    STRATIFY,
    TARGET_COLUMN,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataLoader:
    """
    Handles loading and initial splitting of diabetes data.
    
    This class ensures data is split before any preprocessing to prevent
    data leakage between train, validation, and test sets.
    """
    
    def __init__(
        self,
        test_size: float = TEST_SIZE,
        validation_size: float = VALIDATION_SIZE,
        random_state: int = RANDOM_STATE,
        stratify: bool = STRATIFY,
    ):
        """
        Initialize the DataLoader with split parameters.
        
        Args:
            test_size: Proportion of data to reserve for validation + test
            validation_size: Proportion of temp set to use for validation (rest for test)
            random_state: Random seed for reproducibility
            stratify: Whether to stratify splits based on target variable
        """
        self.test_size = test_size
        self.validation_size = validation_size
        self.random_state = random_state
        self.stratify = stratify
        
    def load_raw_data(self, file_path: Optional[Path] = None) -> pd.DataFrame:
        """
        Load the raw diabetes dataset from CSV.
        
        Args:
            file_path: Path to the CSV file. Uses default if None.
            
        Returns:
            Raw DataFrame with all data
            
        Raises:
            FileNotFoundError: If the file does not exist
            pd.errors.EmptyDataError: If the file is empty
        """
        path = file_path or RAW_DATA_PATH
        
        try:
            logger.info(f"Loading raw data from {path}")
            df = pd.read_csv(path)
            logger.info(f"Successfully loaded {len(df)} rows and {len(df.columns)} columns")
            return df
        except FileNotFoundError:
            logger.error(f"File not found: {path}")
            raise
        except pd.errors.EmptyDataError:
            logger.error(f"File is empty: {path}")
            raise
            
    def split_data(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
        """
        Split data into train, validation, and test sets.
        
        Performs stratified split to maintain class distribution across sets.
        First splits into train and temp, then splits temp into val and test.
        
        Args:
            df: Raw dataframe containing features and target
            
        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        logger.info("Starting train/validation/test split")
        
        # Separate features and target
        X = df.drop(columns=[TARGET_COLUMN])
        y = df[TARGET_COLUMN]
        
        stratify_param = y if self.stratify else None
        
        # First split: train vs temp (val + test)
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=stratify_param
        )
        
        # Second split: validation vs test
        stratify_temp = y_temp if self.stratify else None
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp,
            test_size=self.validation_size,
            random_state=self.random_state,
            stratify=stratify_temp
        )
        
        logger.info(f"Training set: {X_train.shape[0]} samples")
        logger.info(f"Validation set: {X_val.shape[0]} samples")
        logger.info(f"Test set: {X_test.shape[0]} samples")
        
        return X_train, X_val, X_test, y_train, y_val, y_test


def load_processed_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, 
                                   pd.Series, pd.Series, pd.Series]:
    """
    Load preprocessed datasets from disk.
    
    Returns:
        Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
        
    Raises:
        FileNotFoundError: If any processed file is missing
    """
    logger.info("Loading preprocessed datasets")
    
    try:
        X_train = pd.read_csv(X_TRAIN_PATH)
        y_train = pd.read_csv(Y_TRAIN_PATH).squeeze()
        X_val = pd.read_csv(X_VAL_PATH)
        y_val = pd.read_csv(Y_VAL_PATH).squeeze()
        X_test = pd.read_csv(X_TEST_PATH)
        y_test = pd.read_csv(Y_TEST_PATH).squeeze()
        
        logger.info("All datasets loaded successfully")
        return X_train, X_val, X_test, y_train, y_val, y_test
        
    except FileNotFoundError as e:
        logger.error(f"Processed data file not found: {e}")
        raise


def save_split_data(
    X_train: pd.DataFrame, X_val: pd.DataFrame, X_test: pd.DataFrame,
    y_train: pd.Series, y_val: pd.Series, y_test: pd.Series
) -> None:
    """
    Save split datasets to disk before preprocessing.
    
    Args:
        X_train: Training features
        X_val: Validation features
        X_test: Test features
        y_train: Training targets
        y_val: Validation targets
        y_test: Test targets
    """
    PROCESSED_DIR = X_TRAIN_PATH.parent
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    
    X_train.to_csv(PROCESSED_DIR / "X_train.csv", index=False)
    y_train.to_csv(PROCESSED_DIR / "y_train.csv", index=False)
    X_val.to_csv(PROCESSED_DIR / "X_val.csv", index=False)
    y_val.to_csv(PROCESSED_DIR / "y_val.csv", index=False)
    X_test.to_csv(PROCESSED_DIR / "X_test.csv", index=False)
    y_test.to_csv(PROCESSED_DIR / "y_test.csv", index=False)
    
    logger.info("Split data saved successfully")


if __name__ == "__main__":
    # Example usage
    loader = DataLoader()
    raw_df = loader.load_raw_data()
    
    # X_train, X_val, X_test, y_train, y_val, y_test = loader.split_data(raw_df)
    # save_split_data(X_train, X_val, X_test, y_train, y_val, y_test)
