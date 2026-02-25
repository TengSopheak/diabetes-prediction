# 👩🏻‍⚕️ Patient Diabetes Prediction System

This is my class project that builds a diabetes prediction pipeline.
It covers EDA, feature work, model training, and a small FastAPI app connected to AWS to host the model.
The goal is simple: Predict whether a patient is likely to have diabetes and provide a confidence score for that prediction.
This is to show a full workflow that a developer can reproduce.

---

## 📂 Dataset

* Source: CSV file provided by the professor for the class project.
* Target: `diabetes` (binary).
* Original class counts before balancing:

  * Class 0: 64,050
  * Class 1: 5,950

---

## 📓 Methodology

### ⚙️ Pre-processing

1. Load the CSV dataset.
2. Map ambiguous values in `gender` and `smoking_history`.
3. Encode the target `diabetes` before splitting to avoid label issues.
4. Split data early: 70% train, 15% validation, 15% test. This helps avoid data leakage.

### 🛠️ Feature engineering

* Check for missing values. None were found.
* Fix outliers in 3 of 4 numeric columns using the IQR method.
* Encode categorical features (`gender`, `smoking_history`) with OneHotEncoder.
* Scale numeric features with `StandardScaler`. Scaling fit on train and applied to val/test.
* Feature selection: keep features with absolute correlation > 0.01 to target.
* Balance training data with SMOTE. Original class balance was very skewed, so SMOTE was used to balance.
* Save preprocessed train and test datasets for reuse.

### 🧪 Model experiments

* Load preprocessed train, validation, and test sets.
* Train these models:

  * Logistic Regression
  * Decision Tree
  * Random Forest
  * Gradient Boosting
  * K-Neighbors
  * Gaussian NB
  * XGBoost
  * LightGBM
* Evaluate on validation then on test. Metrics used: accuracy, precision, recall, F1, ROC-AUC.
* A confusion matrix was generated for every model.
* The best model (LightGBM) was saved as `best_diabetes_model_LightGBM.pkl`.

---

## 🎯 Results

### Validation Set Results (sorted by ROC-AUC)

| Model               | Validation Accuracy | Validation Precision | Validation Recall | Validation F1 Score | Validation ROC-AUC |
| ------------------- | ------------------- | -------------------- | ----------------- | ------------------- | ------------------ |
| LightGBM            | 0.970467            | 0.916834             | 0.717647          | 0.805103            | 0.979052           |
| XGBoost             | 0.969400            | 0.897661             | 0.722353          | 0.800522            | 0.978008           |
| Gradient Boosting   | 0.953867            | 0.702571             | 0.792941          | 0.745026            | 0.977870           |
| Random Forest       | 0.959400            | 0.765127             | 0.753725          | 0.759384            | 0.967889           |
| Logistic Regression | 0.887867            | 0.423754             | 0.887059          | 0.573529            | 0.963201           |
| K-Neighbors         | 0.916333            | 0.504907             | 0.807059          | 0.621189            | 0.919540           |
| Gaussian NB         | 0.872800            | 0.383253             | 0.814902          | 0.521325            | 0.917877           |
| Decision Tree       | 0.948067            | 0.673913             | 0.753725          | 0.711588            | 0.860907           |

---

### Test Set Results (sorted by ROC-AUC)

| Model               | Test Accuracy | Test Precision | Test Recall | Test F1 Score | Test ROC-AUC |
| ------------------- | ------------- | -------------- | ----------- | ------------- | ------------ |
| LightGBM            | 0.971067      | 0.940314       | 0.704314    | 0.805381      | 0.977927     |
| XGBoost             | 0.969867      | 0.912738       | 0.713725    | 0.801056      | 0.976534     |
| Gradient Boosting   | 0.951933      | 0.692897       | 0.780392    | 0.734046      | 0.974233     |
| Random Forest       | 0.957733      | 0.760357       | 0.734118    | 0.747007      | 0.964816     |
| Logistic Regression | 0.886400      | 0.419331       | 0.874510    | 0.566853      | 0.959507     |
| Gaussian NB         | 0.876067      | 0.388037       | 0.793725    | 0.521246      | 0.915701     |
| K-Neighbors         | 0.912667      | 0.491558       | 0.799216    | 0.608722      | 0.912843     |
| Decision Tree       | 0.951133      | 0.699558       | 0.745098    | 0.721610      | 0.859735     |

---

## Best and worst performing models

**Best model: LightGBM**

* Highest ROC-AUC on validation and test.
* High precision and balanced recall.
* Saved as the final model file `best_diabetes_model_LightGBM.pkl`.

**Worst model: Decision Tree (as evaluated here)**

* Lowest ROC-AUC among tested models on both val and test.
* Likely overfit to training data, causing lower general performance.
* Still decent accuracy but weaker discrimination power vs boosting models.

---

## 🚀 Install and run

### Local setup (clone repo)

```bash
git clone https://github.com/TengSopheak/diabetes-prediction.git
cd diabetes-prediction
```

### Using uv

```bash
uv venv --python 3.11.9
.venv\Scripts\activate
uv pip install -r requirements.txt
uv run python -m src.main
uv run uvicorn api.app:app --reload --port 8000
```

### Using pip

```bash
python3.11 -m venv myenv
.venv\Scripts\activate
pip install -r requirements.txt
python -m src.main
uvicorn api.app:app --reload --port 8000
```

### ☁️ AWS deployment (demo)

> These are the steps used to copy files to an EC2 instance and run the demo app.

1. Prepare key file permissions in PowerShell:

```powershell
icacls [aws_key].pem /inheritance:r
icacls [aws_key].pem /grant:r "%username%:R"
```

2. Copy files and connect from Command Prompt:

```bash
scp -i [aws_key].pem app.py requirements.txt best_diabetes_model_LightGBM.pkl ubuntu@<EC2_IPv4>:/home/ubuntu/
ssh -i "[aws_key].pem" ubuntu@<EC2_IPv4>
```

3. On EC2, update and install runtime:

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y python3-pip python3-venv git
```

4. Create virtual env and install:

```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

5. Run FastAPI:

```bash
uvicorn api.app:app --reload --port 8000
```

---

## ⚙ Usage

* After starting the app, open the FastAPI docs in a browser:

```
http://localhost:8000/docs
```

* The docs page shows available endpoints and lets you try the API.
* For a quick test, post a JSON with the required feature keys to the prediction endpoint. The docs will show the exact request schema.

---

## Notes and next steps

* Confusion matrices were produced for each model and saved for analysis.
* The pipeline saves preprocessed datasets and the final model to speed up demo runs.
* Next ideas:

  * Add a compact input schema and example payload to README.
  * Add unit tests for preprocessing and inference.
  * Try calibration or threshold tuning for better precision/recall trade off.

---
