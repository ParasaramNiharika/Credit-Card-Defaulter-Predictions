
import sys
from pathlib import Path
import __main__

import joblib
import pandas as pd
import streamlit as st

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# Import the original preprocessing function
from src.preprocessing import feature_engineering
from src.prediction_logger import log_prediction

# Required for loading the saved pipeline
__main__.feature_engineering = feature_engineering

# Load saved model
MODEL_PATH = PROJECT_ROOT / "models" / "credit_card_pipeline.pkl"
model = joblib.load(MODEL_PATH)

# Page settings
st.set_page_config(
    page_title="Credit Card Default Prediction",
    page_icon="💳",
    layout="wide"
)

st.title("💳 Credit Card Default Prediction")
st.write("Enter customer details to predict the likelihood of credit card default.")

st.subheader("Customer Information")

with st.form("prediction_form"):

    col1, col2, col3 = st.columns(3)

    with col1:
        limit_bal = st.number_input("Credit Limit", min_value=0, value=50000)
        sex = st.selectbox("Gender", [1, 2], format_func=lambda x: "Male" if x == 1 else "Female")
        education = st.selectbox("Education", [1, 2, 3, 4, 5, 6, 0])
        marriage = st.selectbox("Marital Status", [1, 2, 3, 0])
        age = st.number_input("Age", min_value=18, max_value=100, value=30)

    with col2:
        pay_0 = st.selectbox("Repayment Status (September)", [-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        pay_2 = st.selectbox("Repayment Status (August)", [-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        pay_3 = st.selectbox("Repayment Status (July)", [-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        pay_4 = st.selectbox("Repayment Status (June)", [-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        pay_5 = st.selectbox("Repayment Status (May)", [-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        pay_6 = st.selectbox("Repayment Status (April)", [-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

    with col3:
        bill_amt1 = st.number_input("September Bill Amount", value=0)
        bill_amt2 = st.number_input("August Bill Amount", value=0)
        bill_amt3 = st.number_input("July Bill Amount", value=0)
        bill_amt4 = st.number_input("June Bill Amount", value=0)
        bill_amt5 = st.number_input("May Bill Amount", value=0)
        bill_amt6 = st.number_input("April Bill Amount", value=0)

    st.subheader("Previous Payments")

    pay_col1, pay_col2, pay_col3 = st.columns(3)

    with pay_col1:
        pay_amt1 = st.number_input("September Payment", value=0)
        pay_amt2 = st.number_input("August Payment", value=0)

    with pay_col2:
        pay_amt3 = st.number_input("July Payment", value=0)
        pay_amt4 = st.number_input("June Payment", value=0)

    with pay_col3:
        pay_amt5 = st.number_input("May Payment", value=0)
        pay_amt6 = st.number_input("April Payment", value=0)

    submitted = st.form_submit_button("Predict Default")


if submitted:
    customer = pd.DataFrame([{
        "LIMIT_BAL": limit_bal,
        "SEX": sex,
        "EDUCATION": education,
        "MARRIAGE": marriage,
        "AGE": age,
        "PAY_0": pay_0,
        "PAY_2": pay_2,
        "PAY_3": pay_3,
        "PAY_4": pay_4,
        "PAY_5": pay_5,
        "PAY_6": pay_6,
        "BILL_AMT1": bill_amt1,
        "BILL_AMT2": bill_amt2,
        "BILL_AMT3": bill_amt3,
        "BILL_AMT4": bill_amt4,
        "BILL_AMT5": bill_amt5,
        "BILL_AMT6": bill_amt6,
        "PAY_AMT1": pay_amt1,
        "PAY_AMT2": pay_amt2,
        "PAY_AMT3": pay_amt3,
        "PAY_AMT4": pay_amt4,
        "PAY_AMT5": pay_amt5,
        "PAY_AMT6": pay_amt6
    }])

    prediction = model.predict(customer)[0]

    probabilities = model.predict_proba(customer)[0]
    classes = list(model.classes_)
    default_index = classes.index(1)
    default_probability = probabilities[default_index]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error("The model predicts that this customer may default.")
    else:
        st.success("The model predicts that this customer may not default.")

    st.write(
        f"Estimated default probability: "
        f"{default_probability * 100:.2f}%"
    )

    # Save prediction in log file
    log_prediction(customer, prediction, default_probability)

    st.success("Prediction recorded successfully!")

    st.caption(
        "This is a machine-learning prediction, not a guarantee "
        "of future customer behaviour."
    )
