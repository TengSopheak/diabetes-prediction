"""
Model evaluation metrics and validation for the diabetes prediction pipeline.
Provides comprehensive evaluation capabilities and visualization.
"""

import logging
from typing import Dict, List, Optional, Any, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

from config.settings import METRICS_TO_TRACK

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """
    Handles comprehensive model evaluation across multiple metrics.
    
    Calculates classification metrics and generates evaluation reports.
    """
    
    def __init__(self, metrics: Optional[List[str]] = None):
        """
        Initialize evaluator with metrics to track.
        
        Args:
            metrics: List of metric names to calculate. Uses settings.METRICS_TO_TRACK if None.
        """
        self.metrics = metrics or METRICS_TO_TRACK
        self.metric_functions_ = {
            "accuracy": accuracy_score,
            "precision": precision_score,
            "recall": recall_score,
            "f1": f1_score,
            "roc_auc": roc_auc_score,
        }
        
    def evaluate_model(
        self, 
        model: Any, 
        X: pd.DataFrame, 
        y: pd.Series,
        dataset_name: str = "test"
    ) -> Dict[str, float]:
        """
        Evaluate a single model on a dataset.
        
        Args:
            model: Trained model instance
            X: Feature matrix
            y: True labels
            dataset_name: Name identifier for logging
            
        Returns:
            Dictionary of metric names to values
        """
        # Make predictions
        y_pred = model.predict(X)
        
        # Get probabilities if available (for ROC-AUC)
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X)[:, 1]
        elif hasattr(model, "decision_function"):
            y_prob = model.decision_function(X)
        else:
            y_prob = y_pred  # Fallback to predictions if no probabilities
            
        # Calculate metrics
        results = {}
        for metric in self.metrics:
            try:
                if metric == "roc_auc":
                    results[f"{dataset_name}_{metric}"] = self.metric_functions_[metric](y, y_prob)
                else:
                    results[f"{dataset_name}_{metric}"] = self.metric_functions_[metric](y, y_pred)
            except Exception as e:
                logger.warning(f"Could not calculate {metric}: {e}")
                results[f"{dataset_name}_{metric}"] = np.nan
                
        logger.debug(f"Evaluated on {dataset_name}: {results}")
        return results
    
    def evaluate_all_models(
        self,
        models: Dict[str, Any],
        X: pd.DataFrame,
        y: pd.Series,
        dataset_name: str = "test"
    ) -> pd.DataFrame:
        """
        Evaluate multiple models and compile results.
        
        Args:
            models: Dictionary of model name to model instance
            X: Feature matrix
            y: True labels
            dataset_name: Name identifier
            
        Returns:
            DataFrame with metrics for all models
        """
        results = []
        
        for name, model in models.items():
            metrics = self.evaluate_model(model, X, y, dataset_name)
            metrics["model_name"] = name
            results.append(metrics)
            
        return pd.DataFrame(results)
    
    def get_classification_report(
        self, 
        model: Any, 
        X: pd.DataFrame, 
        y: pd.Series
    ) -> str:
        """
        Generate detailed classification report.
        
        Args:
            model: Trained model
            X: Features
            y: True labels
            
        Returns:
            Classification report string
        """
        y_pred = model.predict(X)
        return classification_report(y, y_pred)
    
    def get_confusion_matrix(
        self, 
        model: Any, 
        X: pd.DataFrame, 
        y: pd.Series
    ) -> np.ndarray:
        """
        Calculate confusion matrix.
        
        Args:
            model: Trained model
            X: Features
            y: True labels
            
        Returns:
            Confusion matrix array
        """
        y_pred = model.predict(X)
        return confusion_matrix(y, y_pred)


class EvaluationVisualizer:
    """
    Creates visualizations for model evaluation.
    
    Generates plots for metrics comparison and confusion matrices.
    """
    
    def __init__(self, figsize: Tuple[int, int] = (10, 6)):
        """
        Initialize visualizer.
        
        Args:
            figsize: Default figure size for plots
        """
        self.figsize = figsize
        
    def plot_metrics_comparison(
        self,
        results_df: pd.DataFrame,
        metric: str = "roc_auc",
        dataset: str = "test",
        title: Optional[str] = None,
        xlim: Optional[Tuple[float, float]] = None
    ) -> plt.Figure:
        """
        Create bar plot comparing models on a specific metric.
        
        Args:
            results_df: DataFrame with model results
            metric: Metric to plot
            dataset: Dataset type (train/val/test)
            title: Plot title
            xlim: X-axis limits
            
        Returns:
            Matplotlib figure
        """
        col_name = f"{dataset}_{metric}"
        
        if col_name not in results_df.columns:
            raise ValueError(f"Column {col_name} not found in results")
            
        # Sort by metric
        df_sorted = results_df.sort_values(by=col_name, ascending=False)
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        sns.barplot(
            data=df_sorted,
            x=col_name,
            y="model_name",
            palette="viridis",
            ax=ax
        )
        
        ax.set_xlabel(f"{dataset.capitalize()} {metric.upper()}")
        ax.set_ylabel("Model")
        ax.set_title(title or f"Model Comparison - {metric.upper()}")
        
        if xlim:
            ax.set_xlim(xlim)
            
        plt.tight_layout()
        return fig
    
    def plot_confusion_matrices(
        self,
        models: Dict[str, Any],
        X: pd.DataFrame,
        y: pd.Series,
        figsize: Optional[Tuple[int, int]] = None,
        cols: int = 3
    ) -> plt.Figure:
        """
        Create subplot grid of confusion matrices for all models.
        
        Args:
            models: Dictionary of trained models
            X: Features
            y: True labels
            figsize: Figure size (auto-calculated if None)
            cols: Number of columns in subplot grid
            
        Returns:
            Matplotlib figure
        """
        n_models = len(models)
        rows = (n_models + cols - 1) // cols
        
        if figsize is None:
            figsize = (5 * cols, 4 * rows)
            
        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        axes = axes.flatten() if n_models > 1 else [axes]
        
        for idx, (name, model) in enumerate(models.items()):
            y_pred = model.predict(X)
            cm = confusion_matrix(y, y_pred)
            
            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                ax=axes[idx],
                cbar=True if idx == 0 else False
            )
            axes[idx].set_title(f"Confusion Matrix - {name}")
            axes[idx].set_xlabel("Predicted")
            axes[idx].set_ylabel("Actual")
            
        # Hide unused subplots
        for idx in range(n_models, len(axes)):
            fig.delaxes(axes[idx])
            
        plt.tight_layout()
        return fig
    
    def plot_roc_curves(
        self,
        models: Dict[str, Any],
        X: pd.DataFrame,
        y: pd.Series,
        figsize: Tuple[int, int] = (10, 8)
    ) -> plt.Figure:
        """
        Plot ROC curves for all models.
        
        Args:
            models: Dictionary of trained models
            X: Features
            y: True labels
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        from sklearn.metrics import roc_curve
        
        fig, ax = plt.subplots(figsize=figsize)
        
        for name, model in models.items():
            if hasattr(model, "predict_proba"):
                y_prob = model.predict_proba(X)[:, 1]
            elif hasattr(model, "decision_function"):
                y_prob = model.decision_function(X)
            else:
                continue
                
            fpr, tpr, _ = roc_curve(y, y_prob)
            auc = roc_auc_score(y, y_prob)
            ax.plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})")
            
        ax.plot([0, 1], [0, 1], "k--", label="Random")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curves")
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def save_plot(self, fig: plt.Figure, path: str) -> None:
        """
        Save plot to file.
        
        Args:
            fig: Matplotlib figure to save
            path: File path for saving
        """
        fig.savefig(path, dpi=300, bbox_inches="tight")
        logger.info(f"Plot saved to {path}")
        plt.close(fig)


def comprehensive_evaluation(
    models: Dict[str, Any],
    X_test: pd.DataFrame,
    y_test: pd.Series,
    X_val: Optional[pd.DataFrame] = None,
    y_val: Optional[pd.Series] = None,
    save_plots: bool = False,
    output_dir: Optional[str] = None
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Run comprehensive evaluation on all models.
    
    Args:
        models: Dictionary of trained models
        X_test: Test features
        y_test: Test targets
        X_val: Optional validation features
        y_val: Optional validation targets
        save_plots: Whether to save plots to disk
        output_dir: Directory to save plots
        
    Returns:
        Tuple of (results_dataframe, evaluation_figures)
    """
    evaluator = ModelEvaluator()
    visualizer = EvaluationVisualizer()
    
    figures = {}
    
    # Evaluate on test set
    test_results = evaluator.evaluate_all_models(models, X_test, y_test, "test")
    
    # Evaluate on validation set if provided
    if X_val is not None and y_val is not None:
        val_results = evaluator.evaluate_all_models(models, X_val, y_val, "val")
        # Merge results
        results = pd.merge(
            test_results, 
            val_results, 
            on="model_name", 
            suffixes=("", "_val")
        )
    else:
        results = test_results
    
    # Generate plots
    # 1. Metrics comparison
    fig_metrics = visualizer.plot_metrics_comparison(
        results, metric="roc_auc", dataset="test"
    )
    figures["metrics_comparison"] = fig_metrics
    
    # 2. Confusion matrices
    fig_cm = visualizer.plot_confusion_matrices(models, X_test, y_test)
    figures["confusion_matrices"] = fig_cm
    
    # 3. ROC curves
    fig_roc = visualizer.plot_roc_curves(models, X_test, y_test)
    figures["roc_curves"] = fig_roc
    
    # Save plots if requested
    if save_plots and output_dir:
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        for name, fig in figures.items():
            visualizer.save_plot(fig, f"{output_dir}/{name}.png")
    
    return results, figures


if __name__ == "__main__":
    # Example usage
    from data.load_data import load_processed_data
    from models.train import ModelTrainer
    
    X_train, X_val, X_test, y_train, y_val, y_test = load_processed_data()
    
    # Load trained models or retrain
    trainer = ModelTrainer()
    trainer.initialize_models()
    models = trainer.train_all_models(X_train, y_train)
    
    # Comprehensive evaluation
    results, figures = comprehensive_evaluation(
        models, X_test, y_test, X_val, y_val
    )
    
    print("\nEvaluation Results:")
    print(results)
