"""
Configuration settings for the diabetes prediction pipeline.
Centralizes all hyperparameters, file paths, and constants.
"""

from pathlib import Path
from typing import Dict, List

# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Data paths
# DATA_DIR = "./dataset/raw"
# PROCESSED_DIR = "./dataset/test_folder"
# MODELS_DIR = "./model"

DATA_DIR = PROJECT_ROOT / "dataset" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "dataset" / "test_folder"
MODELS_DIR = PROJECT_ROOT / "model"

# Ensure directories exist
for dir_path in [DATA_DIR, PROCESSED_DIR, MODELS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# File paths
RAW_DATA_PATH = DATA_DIR / "diabetes_data.csv"
X_TRAIN_PATH = PROCESSED_DIR / "X_train_balanced.csv"
Y_TRAIN_PATH = PROCESSED_DIR / "y_train_balanced.csv"
X_VAL_PATH = PROCESSED_DIR / "X_val.csv"
Y_VAL_PATH = PROCESSED_DIR / "y_val.csv"
X_TEST_PATH = PROCESSED_DIR / "X_test.csv"
Y_TEST_PATH = PROCESSED_DIR / "y_test.csv"
SCALER_PATH = MODELS_DIR / "scaler.pkl"
BEST_MODEL_PATH = MODELS_DIR / "best_diabetes_model_LightGBM.pkl"

# Data split configuration
TEST_SIZE: float = 0.30
VALIDATION_SIZE: float = 0.50  # Of the temp set (0.30 * 0.50 = 0.15 of total)
RANDOM_STATE: int = 42
STRATIFY: bool = True

# Feature engineering configuration
NUMERICAL_COLUMNS: List[str] = ["age", "bmi", "HbA1c_level", "blood_glucose_level"]
CATEGORICAL_COLUMNS: List[str] = ["gender", "smoking_history"]
TARGET_COLUMN: str = "diabetes"

# Gender mapping configuration
GENDER_MAPPING: Dict[str, str] = {
    "Other": "Male",
    "Female": "Female",
    "Male": "Male"
}

# Smoking history mapping configuration
SMOKING_MAPPING: Dict[str, str] = {
    "never": "never",
    "No Info": "unknown",
    "current": "current",
    "former": "former",
    "not current": "former",
    "ever": "former"
}

# Outlier handling configuration
OUTLIER_COLUMNS: List[str] = ["bmi", "HbA1c_level", "blood_glucose_level"]
IQR_MULTIPLIER: float = 1.5

# Feature selection configuration
CORRELATION_THRESHOLD: float = 0.01

# Model configuration
MODELS_CONFIG: Dict[str, Dict] = {
    "Logistic Regression": {
        "class": "sklearn.linear_model.LogisticRegression",
        "params": {"max_iter": 1000, "random_state": 42}
    },
    "Decision Tree": {
        "class": "sklearn.tree.DecisionTreeClassifier",
        "params": {"random_state": 42}
    },
    "Random Forest": {
        "class": "sklearn.ensemble.RandomForestClassifier",
        "params": {"n_estimators": 100, "random_state": 42}
    },
    "Gradient Boosting": {
        "class": "sklearn.ensemble.GradientBoostingClassifier",
        "params": {"random_state": 42}
    },
    "K-Neighbors": {
        "class": "sklearn.neighbors.KNeighborsClassifier",
        "params": {}
    },
    "Gaussian NB": {
        "class": "sklearn.naive_bayes.GaussianNB",
        "params": {}
    },
    "XGBoost": {
        "class": "xgboost.XGBClassifier",
        "params": {"random_state": 42, "eval_metric": "logloss"}
    },
    "LightGBM": {
        "class": "lightgbm.LGBMClassifier",
        "params": {"random_state": 42, "verbose": -1}
    }
}

# SMOTE configuration
SMOTE_RANDOM_STATE: int = 42

# Evaluation configuration
METRICS_TO_TRACK: List[str] = [
    "accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc"
]

# Primary metric for model selection
PRIMARY_METRIC: str = "roc_auc"
