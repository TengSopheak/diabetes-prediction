"""
Model training procedures for the diabetes prediction pipeline.
Handles training multiple models and selecting the best performer.
"""

import logging
from typing import Dict, Optional, Tuple, Any
import warnings

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from config.settings import MODELS_CONFIG, PRIMARY_METRIC, BEST_MODEL_PATH

logger = logging.getLogger(__name__)

# Suppress specific warnings
warnings.filterwarnings('ignore', category=UserWarning)


class ModelTrainer:
    """
    Handles training of multiple classification models.
    
    Supports training various algorithms with configured hyperparameters
    and selecting the best model based on validation performance.
    """
    
    def __init__(self, models_config: Optional[Dict] = None):
        """
        Initialize trainer with model configurations.
        
        Args:
            models_config: Dictionary mapping model names to class and parameters.
                         Uses settings.MODELS_CONFIG if not provided.
        """
        self.models_config = models_config or MODELS_CONFIG
        self.models_: Dict[str, Any] = {}
        self.training_results_: Dict[str, Dict] = {}
        
    def _get_model_class(self, class_path: str):
        """
        Dynamically import and return model class.
        
        Args:
            class_path: String path to class (e.g., 'sklearn.linear_model.LogisticRegression')
            
        Returns:
            Model class
        """
        module_path, class_name = class_path.rsplit('.', 1)
        module = __import__(module_path, fromlist=[class_name])
        return getattr(module, class_name)
    
    def initialize_models(self) -> Dict[str, Any]:
        """
        Initialize all model instances from configuration.
        
        Returns:
            Dictionary of model name to initialized model instance
        """
        models = {}
        
        for name, config in self.models_config.items():
            try:
                model_class = self._get_model_class(config["class"])
                model = model_class(**config["params"])
                models[name] = model
                logger.debug(f"Initialized {name}")
            except Exception as e:
                logger.error(f"Failed to initialize {name}: {e}")
                raise
                
        self.models_ = models
        logger.info(f"Initialized {len(models)} models")
        return models
    
    def train_single_model(
        self, 
        model_name: str, 
        model: Any, 
        X_train: pd.DataFrame, 
        y_train: pd.Series
    ) -> Any:
        """
        Train a single model.
        
        Args:
            model_name: Name identifier for the model
            model: Model instance to train
            X_train: Training features
            y_train: Training targets
            
        Returns:
            Trained model instance
        """
        logger.info(f"Training {model_name}...")
        try:
            model.fit(X_train, y_train)
            logger.info(f"Completed training {model_name}")
            return model
        except Exception as e:
            logger.error(f"Error training {model_name}: {e}")
            raise
    
    def train_all_models(
        self, 
        X_train: pd.DataFrame, 
        y_train: pd.Series
    ) -> Dict[str, Any]:
        """
        Train all configured models.
        
        Args:
            X_train: Training features
            y_train: Training targets
            
        Returns:
            Dictionary of trained models
        """
        if not self.models_:
            self.initialize_models()
            
        trained_models = {}
        
        for name, model in self.models_.items():
            trained_model = self.train_single_model(name, model, X_train, y_train)
            trained_models[name] = trained_model
            
        self.models_ = trained_models
        logger.info("All models trained successfully")
        return trained_models
    
    def get_trained_models(self) -> Dict[str, Any]:
        """Return dictionary of trained models."""
        if not self.models_:
            raise RuntimeError("Models have not been trained yet")
        return self.models_
    
    def save_model(self, model: Any, path: str = str(BEST_MODEL_PATH)) -> None:
        """
        Save a trained model to disk.
        
        Args:
            model: Trained model instance
            path: Path to save the model
        """
        # Ensure directory exists
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        joblib.dump(model, path)
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str = str(BEST_MODEL_PATH)) -> Any:
        """
        Load a trained model from disk.
        
        Args:
            path: Path to the saved model
            
        Returns:
            Loaded model instance
        """
        model = joblib.load(path)
        logger.info(f"Model loaded from {path}")
        return model


class ModelSelector:
    """
    Selects the best model based on validation performance.
    
    Compares models using specified metrics and identifies the top performer.
    """
    
    def __init__(self, primary_metric: str = PRIMARY_METRIC):
        """
        Initialize selector with metric configuration.
        
        Args:
            primary_metric: Metric to use for ranking models (e.g., 'roc_auc', 'f1')
        """
        self.primary_metric = primary_metric
        
    def select_best_model(
        self, 
        results_df: pd.DataFrame, 
        trained_models: Dict[str, Any]
    ) -> Tuple[str, Any]:
        """
        Select best model based on validation results.
        
        Args:
            results_df: DataFrame with validation metrics for each model
            trained_models: Dictionary of trained model instances
            
        Returns:
            Tuple of (best_model_name, best_model_instance)
        """
        if results_df.empty:
            raise ValueError("Results dataframe is empty")
            
        # Sort by primary metric (descending)
        sorted_results = results_df.sort_values(
            by=f"val_{self.primary_metric}",
            ascending=False
        )
        
        best_model_name = sorted_results.iloc[0]["model_name"]
        best_model = trained_models[best_model_name]
        
        logger.info(f"Best model selected: {best_model_name} "
                   f"({self.primary_metric}={sorted_results.iloc[0][f'val_{self.primary_metric}']:.4f})")
        
        return best_model_name, best_model
    
    def get_model_ranking(self, results_df: pd.DataFrame) -> pd.DataFrame:
        """
        Get ranking of all models by primary metric.
        
        Args:
            results_df: DataFrame with validation metrics
            
        Returns:
            DataFrame sorted by primary metric
        """
        return results_df.sort_values(
            by=f"val_{self.primary_metric}", 
            ascending=False
        ).reset_index(drop=True)


def train_and_select_best_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    evaluate_models: bool = True
) -> Tuple[str, Any, pd.DataFrame]:
    """
    Complete training pipeline: train all models, evaluate, and select best.
    
    Args:
        X_train: Training features
        y_train: Training targets
        X_val: Validation features
        y_val: Validation targets
        evaluate_models: Whether to evaluate models on validation set
        
    Returns:
        Tuple of (best_model_name, best_model_instance, results_dataframe)
    """
    from models.evaluate import ModelEvaluator
    
    # Initialize and train models
    trainer = ModelTrainer()
    trainer.initialize_models()
    trained_models = trainer.train_all_models(X_train, y_train)
    
    # Evaluate on validation set
    if evaluate_models:
        evaluator = ModelEvaluator()
        results = []
        
        for name, model in trained_models.items():
            metrics = evaluator.evaluate_model(model, X_val, y_val, dataset_name="val")
            metrics["model_name"] = name
            results.append(metrics)
            
        results_df = pd.DataFrame(results)
        
        # Select best model
        selector = ModelSelector()
        best_name, best_model = selector.select_best_model(results_df, trained_models)
        
        # Save best model
        trainer.save_model(best_model)
        
        return best_name, best_model, results_df
    else:
        # Just return first model if not evaluating
        first_model_name = list(trained_models.keys())[0]
        return first_model_name, trained_models[first_model_name], pd.DataFrame()


if __name__ == "__main__":
    # Example usage
    from data.load_data import load_processed_data
    
    X_train, X_val, X_test, y_train, y_val, y_test = load_processed_data()
    
    best_name, best_model, results = train_and_select_best_model(
        X_train, y_train, X_val, y_val
    )
    
    print(f"\nBest Model: {best_name}")
    print(results)
