"""
Prediction and inference functionality for the diabetes prediction pipeline.
Handles loading models and making predictions on new data.
"""

import logging
from pathlib import Path
from typing import Union, Optional, List, Dict, Any

import joblib
import numpy as np
import pandas as pd

from config.settings import (
    BEST_MODEL_PATH,
    SCALER_PATH,
    NUMERICAL_COLUMNS
)

logger = logging.getLogger(__name__)


class DiabetesPredictor:
    """
    Production-ready predictor for diabetes classification.
    
    Handles loading trained models and preprocessing artifacts,
    and making predictions on new data.
    """
    
    def __init__(
        self,
        model_path: Union[str, Path] = BEST_MODEL_PATH,
        scaler_path: Union[str, Path] = SCALER_PATH,
    ):
        """
        Initialize predictor with paths to saved artifacts.
        
        Args:
            model_path: Path to saved model
            scaler_path: Path to saved scaler
        """
        self.model_path = Path(model_path)
        self.scaler_path = Path(scaler_path)
        self.model_ = None
        self.scaler_ = None
        self.feature_names_: Optional[List[str]] = None
        
    def load_artifacts(self) -> "DiabetesPredictor":
        """
        Load model and preprocessing artifacts from disk.
        
        Returns:
            Self for method chaining
            
        Raises:
            FileNotFoundError: If artifacts are not found
        """
        logger.info("Loading model artifacts")
        
        # Load model
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found at {self.model_path}")
        self.model_ = joblib.load(self.model_path)
        logger.info(f"Model loaded from {self.model_path}")
        
        # Load scaler
        if not self.scaler_path.exists():
            raise FileNotFoundError(f"Scaler not found at {self.scaler_path}")
        self.scaler_ = joblib.load(self.scaler_path)
        logger.info(f"Scaler loaded from {self.scaler_path}")
        
        # Try to get feature names from model if available
        if hasattr(self.model_, 'feature_name_'):
            self.feature_names_ = self.model_.feature_name_
        elif hasattr(self.model_, 'n_features_in_'):
            # LightGBM and sklearn store this
            pass
            
        return self
    
    def preprocess(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Apply preprocessing transformations to input data.
        
        Note: Assumes categorical encoding and feature selection already done.
        Only applies scaling here. For full preprocessing, use the pipeline.
        
        Args:
            X: Raw or partially processed features
            
        Returns:
            Preprocessed features ready for prediction
        """
        if self.scaler_ is None:
            raise RuntimeError("Scaler not loaded. Call load_artifacts() first.")
            
        X_processed = X.copy()
        
        # Apply scaling to numerical columns if they exist
        numerical_cols = [col for col in NUMERICAL_COLUMNS if col in X_processed.columns]
        if numerical_cols:
            X_processed[numerical_cols] = self.scaler_.transform(X_processed[numerical_cols])
            
        return X_processed
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make binary predictions on new data.
        
        Args:
            X: Feature matrix (should be preprocessed or raw depending on setup)
            
        Returns:
            Array of binary predictions (0 or 1)
        """
        if self.model_ is None:
            raise RuntimeError("Model not loaded. Call load_artifacts() first.")
            
        predictions = self.model_.predict(X)
        return predictions
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make probability predictions on new data.
        
        Args:
            X: Feature matrix
            
        Returns:
            Array of probability predictions for positive class
        """
        if self.model_ is None:
            raise RuntimeError("Model not loaded. Call load_artifacts() first.")
            
        if hasattr(self.model_, "predict_proba"):
            probabilities = self.model_.predict_proba(X)[:, 1]
        elif hasattr(self.model_, "decision_function"):
            # For models like SVM that don't have predict_proba
            decision_scores = self.model_.decision_function(X)
            # Convert to probabilities using sigmoid
            probabilities = 1 / (1 + np.exp(-decision_scores))
        else:
            raise AttributeError("Model does not support probability predictions")
            
        return probabilities
    
    def predict_batch(
        self, 
        X: pd.DataFrame,
        return_probabilities: bool = True
    ) -> Dict[str, Any]:
        """
        Make predictions on a batch of data with full results.
        
        Args:
            X: Feature matrix
            return_probabilities: Whether to include probability scores
            
        Returns:
            Dictionary with predictions and metadata
        """
        predictions = self.predict(X)
        
        result = {
            "predictions": predictions,
            "predicted_labels": ["Diabetes" if p == 1 else "No Diabetes" for p in predictions],
            "total_samples": len(X),
            "positive_cases": int(np.sum(predictions)),
            "negative_cases": int(len(predictions) - np.sum(predictions)),
        }
        
        if return_probabilities:
            probabilities = self.predict_proba(X)
            result["probabilities"] = probabilities
            result["risk_scores"] = [
                "High Risk" if p > 0.7 else "Medium Risk" if p > 0.3 else "Low Risk"
                for p in probabilities
            ]
            
        return result
    
    def predict_single(
        self,
        age: float,
        bmi: float,
        hba1c_level: float,
        blood_glucose_level: float,
        hypertension: int = 0,
        heart_disease: int = 0,
        gender_female: int = 0,
        smoking_history_current: int = 0,
        smoking_history_former: int = 0,
        smoking_history_never: int = 0,
        smoking_history_unknown: int = 0,
    ) -> Dict[str, Any]:
        """
        Make prediction on a single patient record.
        
        Args:
            age: Patient age (normalized/scaled)
            bmi: Body mass index (normalized/scaled)
            hba1c_level: HbA1c level (normalized/scaled)
            blood_glucose_level: Blood glucose level (normalized/scaled)
            hypertension: Binary indicator (0/1)
            heart_disease: Binary indicator (0/1)
            gender_female: Binary indicator (1 if female, 0 otherwise)
            smoking_history_current: Binary indicator
            smoking_history_former: Binary indicator
            smoking_history_never: Binary indicator
            smoking_history_unknown: Binary indicator
            
        Returns:
            Dictionary with prediction results
        """
        # Create DataFrame with single row
        data = {
            "age": [age],
            "bmi": [bmi],
            "HbA1c_level": [hba1c_level],
            "blood_glucose_level": [blood_glucose_level],
            "hypertension": [hypertension],
            "heart_disease": [heart_disease],
            "gender_Female": [gender_female],
            "smoking_history_current": [smoking_history_current],
            "smoking_history_former": [smoking_history_former],
            "smoking_history_never": [smoking_history_never],
            "smoking_history_unknown": [smoking_history_unknown],
        }
        
        X = pd.DataFrame(data)
        
        # Make prediction
        prediction = self.predict(X)[0]
        probability = self.predict_proba(X)[0]
        
        return {
            "diabetes_prediction": int(prediction),
            "diabetes_probability": float(probability),
            "prediction_label": "Diabetes" if prediction == 1 else "No Diabetes",
            "risk_level": "High Risk" if probability > 0.7 else "Medium Risk" if probability > 0.3 else "Low Risk",
            "input_features": data,
        }


class BatchPredictor:
    """
    Handles batch predictions with full preprocessing pipeline.
    
    For production use where raw data needs complete preprocessing.
    """
    
    def __init__(
        self,
        model_path: Union[str, Path] = BEST_MODEL_PATH,
        scaler_path: Union[str, Path] = SCALER_PATH,
    ):
        self.predictor = DiabetesPredictor(model_path, scaler_path)
        
    def load_artifacts(self) -> "BatchPredictor":
        """Load all artifacts."""
        self.predictor.load_artifacts()
        return self
    
    def predict_from_raw(
        self,
        df: pd.DataFrame,
        preprocess_fn: Optional[Any] = None
    ) -> Dict[str, Any]:
        """
        Make predictions from raw data with preprocessing.
        
        Args:
            df: Raw dataframe
            preprocess_fn: Optional preprocessing function/class to apply
            
        Returns:
            Prediction results dictionary
        """
        if preprocess_fn:
            X_processed = preprocess_fn(df)
        else:
            # Assume data is already processed
            X_processed = df
            
        return self.predictor.predict_batch(X_processed)


def load_and_predict(
    X: pd.DataFrame,
    model_path: Optional[str] = None
) -> np.ndarray:
    """
    Convenience function for quick predictions.
    
    Args:
        X: Feature matrix
        model_path: Optional path to model
        
    Returns:
        Predictions array
    """
    predictor = DiabetesPredictor(model_path or BEST_MODEL_PATH)
    predictor.load_artifacts()
    return predictor.predict(X)


if __name__ == "__main__":
    # Example usage
    from data.load_data import load_processed_data
    
    # Load test data
    _, _, X_test, _, _, y_test = load_processed_data()
    
    # Initialize predictor
    predictor = DiabetesPredictor()
    predictor.load_artifacts()
    
    # Make predictions
    results = predictor.predict_batch(X_test.iloc[:10])
    
    print("Prediction Results:")
    print(f"Total samples: {results['total_samples']}")
    print(f"Positive cases: {results['positive_cases']}")
    print(f"Negative cases: {results['negative_cases']}")
    print(f"Probabilities: {results.get('probabilities', 'N/A')[:5]}")
