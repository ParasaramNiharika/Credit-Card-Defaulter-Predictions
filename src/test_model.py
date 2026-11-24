
import pandas as pd
import joblib
import __main__

from src.preprocessing import feature_engineering

# Make the preprocessing function available to the saved pipeline
__main__.feature_engineering = feature_engineering

# Load the saved ML pipeline
model = joblib.load("models/credit_card_pipeline.pkl")

# Load the dataset
data = pd.read_csv("data/Credit Card Defaulter Prediction.csv")

# Remove the target column and ID
data.columns = data.columns.str.strip()
X = data.drop(columns=["default", "ID"], errors="ignore")

# Select one customer record
sample = X.iloc[[0]]

# Predict
prediction = model.predict(sample)

print("Prediction:", prediction[0])

if hasattr(model, "predict_proba"):
    probability = model.predict_proba(sample)
    print("Prediction probabilities:", probability[0])
