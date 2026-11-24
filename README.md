# Credit Card Defaulter Prediction

## Project Overview
The Credit Card Defaulter Prediction project uses Machine Learning techniques to predict whether a credit card customer is likely to default on their payment in the next month.

The application provides a user-friendly interface built with Streamlit, allowing users to enter customer details and obtain a default prediction.

## Objectives
- Predict credit card payment defaults using Machine Learning.
- Perform data preprocessing and feature engineering.
- Train and evaluate multiple classification models.
- Track experiments and model performance using MLflow.
- Deploy an interactive prediction application using Streamlit.

## Machine Learning Models
The project explores the following classification algorithms:

- Logistic Regression
- Decision Tree
- Random Forest
- Support Vector Machine (SVM)
- Naive Bayes

## Technologies Used
- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- MLflow
- Optuna
- Streamlit

## Project Structure

```text
Credit_Card_Defaulter_Prediction/
│
├── app/
│   └── app.py
├── data/
├── models/
├── notebooks/
├── reports/
├── src/
├── logs/
├── requirements.txt
├── README.md
└── .gitignore
```

## Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/Credit_Card_Defaulter_Prediction.git
```

### 2. Navigate to the Project Folder

```bash
cd Credit_Card_Defaulter_Prediction
```

### 3. Create a Virtual Environment

```bash
python -m venv venv
```

### 4. Activate the Virtual Environment

Windows:

```bash
venv\Scripts\activate
```

### 5. Install Dependencies

```bash
pip install -r requirements.txt
```

### 6. Run the Streamlit Application

```bash
streamlit run app/app.py
```

## Model Evaluation
The models are evaluated using the following classification metrics:

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix
- Classification Report

MLflow is used for experiment tracking and model performance comparison.

## Application Features
- User-friendly Streamlit interface.
- Customer information input.
- Credit card default prediction.
- Machine Learning-based classification.

## Future Enhancements
- Improve model performance through hyperparameter tuning.
- Deploy the application on Streamlit Community Cloud.
- Enhance prediction explainability.
- Add real-time model monitoring.

## Disclaimer
This project is developed for educational and research purposes only. Predictions should not be considered as financial advice or used as the sole basis for credit approval decisions.

## Author
Parasaram Niharika

Machine Learning Project – Credit Card Defaulter Prediction
