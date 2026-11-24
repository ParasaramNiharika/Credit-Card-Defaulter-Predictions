
import os
import joblib
import optuna
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd

from pathlib import Path

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

import warnings
warnings.filterwarnings("ignore")

# --------------------------------------------------
# 1. PROJECT PATHS AND SETTINGS
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parents[1]

DATA_PATH = PROJECT_DIR / "data" / "Credit Card Defaulter Prediction.csv"

MODEL_DIR = PROJECT_DIR / "models"
REPORT_DIR = PROJECT_DIR / "reports"

MODEL_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)

N_TRIALS = 20
RANDOM_STATE = 42

# Use the same MLflow database for this project
mlflow_db = PROJECT_DIR / "mlflow.db"
mlflow.set_tracking_uri(
    f"sqlite:///{mlflow_db.as_posix()}"
)

mlflow.set_experiment("Credit Card Defaulter Prediction")


# --------------------------------------------------
# 2. LOAD AND CLEAN DATASET
# --------------------------------------------------

print("Loading Credit Card Defaulter Dataset...")

data = pd.read_csv(DATA_PATH)

# Remove spaces from column names
data.columns = data.columns.str.strip()

# Identify target column
target_candidates = [
    "default",
    "default payment next month"
]

target_col = next(
    (col for col in target_candidates if col in data.columns),
    None
)

if target_col is None:
    raise ValueError(
        "Target column not found. Please check dataset columns."
    )

# Remove rows with missing target
data = data.dropna(subset=[target_col]).copy()

# Convert target into integer
y = pd.to_numeric(
    data[target_col],
    errors="raise"
).astype(int)

# Remove target and ID from features
drop_cols = [target_col]

if "ID" in data.columns:
    drop_cols.append("ID")

X = data.drop(columns=drop_cols)

# Identify categorical columns
categorical_cols = [
    col for col in ["SEX", "EDUCATION", "MARRIAGE"]
    if col in X.columns
]

# Remaining columns are numerical
numerical_cols = [
    col for col in X.columns
    if col not in categorical_cols
]

# Convert numerical columns safely
for col in numerical_cols:
    X[col] = pd.to_numeric(X[col], errors="coerce")

# Convert categorical columns to object dtype
for col in categorical_cols:
    X[col] = X[col].astype(object)

print("Dataset shape:", data.shape)
print("Numerical columns:", numerical_cols)
print("Categorical columns:", categorical_cols)


# --------------------------------------------------
# 3. TRAIN TEST SPLIT
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y
)

print("Training data:", X_train.shape)
print("Testing data:", X_test.shape)


# --------------------------------------------------
# 4. PREPROCESSING PIPELINE
# --------------------------------------------------

numerical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False
    ))
])

preprocessor = ColumnTransformer([
    ("num", numerical_pipeline, numerical_cols),
    ("cat", categorical_pipeline, categorical_cols)
])


# --------------------------------------------------
# 5. DEFINE MODELS AND OPTUNA PARAMETERS
# --------------------------------------------------

def create_model(trial, model_name):

    if model_name == "Logistic Regression":

        C = trial.suggest_float(
            "C", 0.001, 100, log=True
        )

        class_weight = trial.suggest_categorical(
            "class_weight", [None, "balanced"]
        )

        return LogisticRegression(
            C=C,
            class_weight=class_weight,
            max_iter=2000,
            random_state=RANDOM_STATE
        )

    elif model_name == "Decision Tree":

        max_depth = trial.suggest_int(
            "max_depth", 3, 30
        )

        min_samples_split = trial.suggest_int(
            "min_samples_split", 2, 30
        )

        min_samples_leaf = trial.suggest_int(
            "min_samples_leaf", 1, 20
        )

        class_weight = trial.suggest_categorical(
            "class_weight", [None, "balanced"]
        )

        return DecisionTreeClassifier(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            class_weight=class_weight,
            random_state=RANDOM_STATE
        )

    elif model_name == "Random Forest":

        n_estimators = trial.suggest_int(
            "n_estimators", 100, 300, step=100
        )

        max_depth = trial.suggest_int(
            "max_depth", 5, 30
        )

        min_samples_split = trial.suggest_int(
            "min_samples_split", 2, 20
        )

        min_samples_leaf = trial.suggest_int(
            "min_samples_leaf", 1, 10
        )

        max_features = trial.suggest_categorical(
            "max_features", ["sqrt", "log2"]
        )

        class_weight = trial.suggest_categorical(
            "class_weight", [None, "balanced"]
        )

        return RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            class_weight=class_weight,
            random_state=RANDOM_STATE,
            n_jobs=-1
        )

    elif model_name == "SVM":

        C = trial.suggest_float(
            "C", 0.001, 100, log=True
        )

        class_weight = trial.suggest_categorical(
            "class_weight", [None, "balanced"]
        )

        return LinearSVC(
            C=C,
            class_weight=class_weight,
            max_iter=5000,
            random_state=RANDOM_STATE
        )

    elif model_name == "Naive Bayes":

        var_smoothing = trial.suggest_float(
            "var_smoothing", 1e-12, 1e-7, log=True
        )

        return GaussianNB(
            var_smoothing=var_smoothing
        )


# --------------------------------------------------
# 6. OPTUNA OBJECTIVE FUNCTION
# --------------------------------------------------

def objective(trial, model_name):

    model = create_model(trial, model_name)

    pipeline = Pipeline([
        ("preprocessing", preprocessor),
        ("model", model)
    ])

    cv = StratifiedKFold(
        n_splits=3,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    # Nested MLflow run for each Optuna trial
    with mlflow.start_run(
        run_name=f"{model_name}_Trial_{trial.number}",
        nested=True
    ):

        mlflow.log_param("model_name", model_name)
        mlflow.log_params(trial.params)

        scores = cross_val_score(
            pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring="f1",
            n_jobs=1,
            error_score="raise"
        )

        mean_f1 = scores.mean()

        mlflow.log_metric("cv_f1_score", mean_f1)

        print(
            model_name,
            "Trial:",
            trial.number,
            "CV F1:",
            round(mean_f1, 4)
        )

    return mean_f1


# --------------------------------------------------
# 7. OPTUNA TUNING FOR ALL FIVE MODELS
# --------------------------------------------------

model_names = [
    "Logistic Regression",
    "Decision Tree",
    "Random Forest",
    "SVM",
    "Naive Bayes"
]

all_results = []
trained_models = {}

for model_name in model_names:

    print("\n====================================")
    print("OPTIMIZING:", model_name)
    print("====================================")

    with mlflow.start_run(run_name=model_name):

        mlflow.log_param("model_name", model_name)
        mlflow.log_param("number_of_trials", N_TRIALS)
        mlflow.log_param("cv_folds", 3)
        mlflow.log_param("optimization_metric", "F1 Score")

        study = optuna.create_study(
            direction="maximize",
            study_name=model_name.replace(" ", "_")
        )

        study.optimize(
            lambda trial: objective(trial, model_name),
            n_trials=N_TRIALS
        )

        best_params = study.best_trial.params
        best_cv_f1 = study.best_value

        print("\nBest Parameters:", best_params)
        print("Best CV F1 Score:", best_cv_f1)

        mlflow.log_metric("best_cv_f1_score", best_cv_f1)
        mlflow.log_params({
            f"best_{key}": str(value)
            for key, value in best_params.items()
        })

        # Train final model with best parameters
        best_trial = optuna.trial.FixedTrial(best_params)

        best_estimator = create_model(
            best_trial,
            model_name
        )

        best_pipeline = Pipeline([
            ("preprocessing", preprocessor),
            ("model", best_estimator)
        ])

        best_pipeline.fit(X_train, y_train)

        # Test predictions
        y_pred = best_pipeline.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)

        precision = precision_score(
            y_test, y_pred, zero_division=0
        )

        recall = recall_score(
            y_test, y_pred, zero_division=0
        )

        f1 = f1_score(
            y_test, y_pred, zero_division=0
        )

        if hasattr(best_pipeline, "predict_proba"):
            y_score = best_pipeline.predict_proba(X_test)[:, 1]
        else:
            y_score = best_pipeline.decision_function(X_test)

        auc = roc_auc_score(y_test, y_score)

        # Log final test metrics
        mlflow.log_metric("test_accuracy", accuracy)
        mlflow.log_metric("test_precision", precision)
        mlflow.log_metric("test_recall", recall)
        mlflow.log_metric("test_f1_score", f1)
        mlflow.log_metric("test_roc_auc", auc)

        # Log trained model
        mlflow.sklearn.log_model(
            best_pipeline,
            artifact_path="best_model"
        )

        trained_models[model_name] = {
            "pipeline": best_pipeline,
            "cv_f1": best_cv_f1,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "roc_auc": auc
        }

        all_results.append({
            "Model": model_name,
            "Best CV F1": best_cv_f1,
            "Test Accuracy": accuracy,
            "Test Precision": precision,
            "Test Recall": recall,
            "Test F1": f1,
            "Test ROC AUC": auc
        })

        print("\n", model_name, "completed!")
        print("Test Accuracy:", accuracy)
        print("Test Precision:", precision)
        print("Test Recall:", recall)
        print("Test F1 Score:", f1)
        print("Test ROC AUC:", auc)


# --------------------------------------------------
# 8. COMPARE ALL FIVE MODELS
# --------------------------------------------------

results_df = pd.DataFrame(all_results)

results_df = results_df.sort_values(
    by="Best CV F1",
    ascending=False
)

print("\n====================================")
print("ALL MODEL RESULTS")
print("====================================")

print(results_df.to_string(index=False))

results_df.to_csv(
    REPORT_DIR / "optuna_model_comparison.csv",
    index=False
)


# --------------------------------------------------
# 9. SELECT THE BEST MODEL
# --------------------------------------------------

best_model_name = results_df.iloc[0]["Model"]

best_model_pipeline = trained_models[
    best_model_name
]["pipeline"]

print("\n====================================")
print("BEST MODEL:", best_model_name)
print("====================================")

print(
    "Best CV F1:",
    trained_models[best_model_name]["cv_f1"]
)

# Save best model locally
best_model_path = MODEL_DIR / "best_credit_card_pipeline.pkl"

joblib.dump(
    best_model_pipeline,
    best_model_path
)

# Log overall best model in MLflow
with mlflow.start_run(run_name="BEST_MODEL_OVERALL"):

    mlflow.log_param("best_model_name", best_model_name)

    mlflow.log_metric(
        "best_cv_f1_score",
        trained_models[best_model_name]["cv_f1"]
    )

    mlflow.log_metric(
        "test_accuracy",
        trained_models[best_model_name]["accuracy"]
    )

    mlflow.log_metric(
        "test_precision",
        trained_models[best_model_name]["precision"]
    )

    mlflow.log_metric(
        "test_recall",
        trained_models[best_model_name]["recall"]
    )

    mlflow.log_metric(
        "test_f1_score",
        trained_models[best_model_name]["f1_score"]
    )

    mlflow.log_metric(
        "test_roc_auc",
        trained_models[best_model_name]["roc_auc"]
    )

    mlflow.sklearn.log_model(
        best_model_pipeline,
        artifact_path="best_model"
    )

print("\nBest model saved successfully!")
print("Model path:", best_model_path)
print("MLflow tracking completed!")
