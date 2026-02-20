#!/usr/bin/env python3
"""
Main entry point for the diabetes prediction pipeline.

Orchestrates the complete workflow from data loading to model evaluation.
Can be run as a script or imported as a module.
"""

import argparse
import logging
from pathlib import Path
from typing import Optional

import pandas as pd

# Add project root to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent))

from config.settings import RAW_DATA_PATH
from data.load_data import DataLoader, load_processed_data, save_split_data
from data.preprocess import DataCleaner, OutlierHandler
from features.build_features import build_features_pipeline
from models.train import train_and_select_best_model, ModelTrainer
from models.evaluate import comprehensive_evaluation
from models.predict import DiabetesPredictor
from utils.helpers import setup_logging

logger = logging.getLogger(__name__)


def run_full_pipeline(
    raw_data_path: Optional[Path] = None,
    skip_initial_mapping: bool = False,
    skip_preprocessing: bool = False,
    skip_training: bool = False,
    evaluate_all: bool = True,
    save_artifacts: bool = True
) -> dict:
    """
    Execute the complete ML pipeline.
    
    Args:
        raw_data_path: Path to raw data CSV
        skip_initial_mapping: Whether to skip initial mapping of categorical values
        skip_preprocessing: Whether to skip preprocessing and load processed data
        skip_training: Whether to skip training and load saved model
        evaluate_all: Whether to evaluate all models or just best
        save_artifacts: Whether to save models and preprocessing artifacts
        
    Returns:
        Dictionary with pipeline results and artifacts
    """
    logger.info("=" * 50)
    logger.info("Starting Diabetes Prediction Pipeline")
    logger.info("=" * 50)
    
    results = {}
    
    # Stage 1: Data Loading and Initial Mapping
    if not skip_initial_mapping:
        logger.info("Stage 1: Data Loading and Initial Mapping")
        loader = DataLoader()
        raw_df = loader.load_raw_data(raw_data_path or RAW_DATA_PATH)

        mapper = DataCleaner()
        raw_df = mapper.map_categorical_variables(raw_df)

        # Stage 2: Splitting and Saving Raw Data
        logger.info("Stage 2: Splitting and Saving Raw Data")
        X_train_raw, X_val_raw, X_test_raw, y_train, y_val, y_test = loader.split_data(raw_df)

    if not skip_preprocessing:       
        # Stage 3: Data Preprocessing
        logger.info("Stage 3: Data Preprocessing")
        cleaner = DataCleaner()
        X_train_clean, X_val_clean, X_test_clean = cleaner.fit_transform(
            X_train_raw, X_val_raw, X_test_raw
        )
        
        outlier_handler = OutlierHandler()
        X_train_proc, X_val_proc, X_test_proc = outlier_handler.fit_transform(
            X_train_clean, X_val_clean, X_test_clean
        )
        
        # Stage 4: Feature Engineering
        logger.info("Stage 4: Feature Engineering")
        X_train, X_val, X_test, y_train, y_val, y_test = build_features_pipeline(
            X_train_proc, y_train,
            X_val_proc, y_val,
            X_test_proc, y_test,
            apply_smote=True,
            save_data=save_artifacts
        )

        if save_artifacts:
            save_split_data(
                X_train, X_val, X_test,
                y_train, y_val, y_test
            )
            
    else:
        logger.info("Skipping preprocessing, loading processed data")
        X_train, X_val, X_test, y_train, y_val, y_test = load_processed_data()
    
    # Stage 5: Model Training
    if not skip_training:
        logger.info("Stage 5: Model Training")
        best_name, best_model, train_results = train_and_select_best_model(
            X_train, y_train, X_val, y_val, evaluate_models=True
        )
        results["best_model_name"] = best_name
        results["validation_results"] = train_results
        logger.info(f"Best model: {best_name}")
    else:
        logger.info("Skipping training, loading saved model")
        trainer = ModelTrainer()
        best_model = trainer.load_model()
        best_name = "loaded_model"
    
    # Stage 6: Evaluation
    if evaluate_all and not skip_training:
        logger.info("Stage 6: Comprehensive Evaluation")
        # Train all models for comparison
        trainer = ModelTrainer()
        trainer.initialize_models()
        all_models = trainer.train_all_models(X_train, y_train)
        
        eval_results, figures = comprehensive_evaluation(
            all_models, X_test, y_test, X_val, y_val,
            save_plots=save_artifacts,
            output_dir="reports/figures" if save_artifacts else None
        )
        results["test_results"] = eval_results
        results["figures"] = figures
        
        logger.info("\nTest Set Results:")
        print(eval_results[["model_name", "test_roc_auc", "test_f1", "test_precision", "test_recall"]])
    
    # Stage 7: Final Best Model Evaluation
    logger.info("Stage 7: Final Best Model Test Evaluation")
    from models.evaluate import ModelEvaluator
    evaluator = ModelEvaluator()
    final_metrics = evaluator.evaluate_model(best_model, X_test, y_test, "final_test")
    results["final_test_metrics"] = final_metrics
    
    logger.info(f"Final Test Metrics for {best_name}:")
    for metric, value in final_metrics.items():
        logger.info(f"  {metric}: {value:.4f}")
    
    logger.info("Pipeline completed successfully!")
    return results


def predict_single_record(
    age: float,
    bmi: float,
    hba1c_level: float,
    blood_glucose_level: float,
    **kwargs
) -> dict:
    """
    Make prediction on a single patient record.
    
    Args:
        age: Patient age (raw value, will be preprocessed)
        bmi: Body mass index (raw value)
        hba1c_level: HbA1c level (raw value)
        blood_glucose_level: Blood glucose level (raw value)
        **kwargs: Additional features (hypertension, heart_disease, etc.)
        
    Returns:
        Prediction results dictionary
    """
    # For production use, you'd implement full preprocessing here
    # For now, assume values are preprocessed
    predictor = DiabetesPredictor()
    predictor.load_artifacts()
    
    return predictor.predict_single(
        age=age,
        bmi=bmi,
        hba1c_level=hba1c_level,
        blood_glucose_level=blood_glucose_level,
        **kwargs
    )


def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Diabetes Prediction Pipeline"
    )
    parser.add_argument(
        "--mode",
        choices=["full", "preprocess", "train", "evaluate", "predict"],
        default="full",
        help="Pipeline execution mode"
    )
    parser.add_argument(
        "--data-path",
        type=str,
        help="Path to raw data CSV"
    )
    parser.add_argument(
        "--skip-preprocessing",
        action="store_true",
        help="Use already preprocessed data"
    )
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Use already trained model"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Directory for outputs"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(level=logging.INFO)
    
    if args.mode == "full":
        results = run_full_pipeline(
            raw_data_path=Path(args.data_path) if args.data_path else None,
            skip_preprocessing=args.skip_preprocessing,
            skip_training=args.skip_training
        )
        print("\nPipeline Results Summary:")
        print(f"Best Model: {results.get('best_model_name')}")
        if 'final_test_metrics' in results:
            print("Final Test Metrics:", results['final_test_metrics'])
            
    elif args.mode == "preprocess":
        # Run only preprocessing
        loader = DataLoader()
        raw_df = loader.load_raw_data(args.data_path or RAW_DATA_PATH)
        splits = loader.split_data(raw_df)
        X_train_raw, X_val_raw, X_test_raw, y_train, y_val, y_test = splits
        
        cleaner = DataCleaner()
        X_train_clean, X_val_clean, X_test_clean = cleaner.fit_transform(
            X_train_raw, X_val_raw, X_test_raw
        )
        
        outlier_handler = OutlierHandler()
        X_train_proc, X_val_proc, X_test_proc = outlier_handler.fit_transform(
            X_train_clean, X_val_clean, X_test_clean
        )
        
        build_features_pipeline(
            X_train_proc, y_train,
            X_val_proc, y_val,
            X_test_proc, y_test
        )
        
    elif args.mode == "train":
        # Run only training
        X_train, X_val, X_test, y_train, y_val, y_test = load_processed_data()
        best_name, best_model, results = train_and_select_best_model(
            X_train, y_train, X_val, y_val
        )
        print(f"Best model: {best_name}")
        print(results)
        
    elif args.mode == "evaluate":
        # Run only evaluation
        X_train, X_val, X_test, y_train, y_val, y_test = load_processed_data()
        trainer = ModelTrainer()
        trainer.initialize_models()
        models = trainer.train_all_models(X_train, y_train)
        
        eval_results, _ = comprehensive_evaluation(
            models, X_test, y_test, X_val, y_val
        )
        print(eval_results)
        
    elif args.mode == "predict":
        # Interactive prediction mode
        print("Prediction mode - enter patient data:")
        # Simplified for example
        predictor = DiabetesPredictor()
        predictor.load_artifacts()
        print("Predictor ready. Use predict_single_record() function for actual predictions.")


if __name__ == "__main__":
    main()
