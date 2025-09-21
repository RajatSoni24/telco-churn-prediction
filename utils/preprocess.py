import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import joblib

def preprocess_input(data):
    """
    Preprocess the input data to match the model's training format
    """
    # Create a copy to avoid modifying the original data
    df = data.copy()
    
    # Convert TotalCharges to numeric, coerce errors to NaN
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    
    # Handle missing values (if any)
    if df['TotalCharges'].isnull().any():
        # Fill missing TotalCharges with 0 for new customers (tenure=0)
        df['TotalCharges'] = df['TotalCharges'].fillna(0)
    
    # Convert SeniorCitizen to string for one-hot encoding
    df['SeniorCitizen'] = df['SeniorCitizen'].astype(str)
    
    # Categorical columns to one-hot encode
    categorical_cols = [
        'gender', 'SeniorCitizen', 'Partner', 'Dependents',
        'PhoneService', 'MultipleLines', 'InternetService',
        'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
        'TechSupport', 'StreamingTV', 'StreamingMovies',
        'Contract', 'PaperlessBilling', 'PaymentMethod'
    ]
    
    # Numerical columns
    numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    
    # One-hot encode categorical variables
    df_encoded = pd.get_dummies(df[categorical_cols], drop_first=True)
    
    # Add numerical columns
    for col in numerical_cols:
        df_encoded[col] = df[col]
    
    # Ensure all expected columns are present (in case some categories are missing in the input)
    expected_columns = [
        'gender_Male', 'SeniorCitizen_1', 'Partner_Yes', 'Dependents_Yes',
        'PhoneService_Yes', 'MultipleLines_No phone service', 'MultipleLines_Yes',
        'InternetService_Fiber optic', 'InternetService_No',
        'OnlineSecurity_No internet service', 'OnlineSecurity_Yes',
        'OnlineBackup_No internet service', 'OnlineBackup_Yes',
        'DeviceProtection_No internet service', 'DeviceProtection_Yes',
        'TechSupport_No internet service', 'TechSupport_Yes',
        'StreamingTV_No internet service', 'StreamingTV_Yes',
        'StreamingMovies_No internet service', 'StreamingMovies_Yes',
        'Contract_One year', 'Contract_Two year',
        'PaperlessBilling_Yes',
        'PaymentMethod_Credit card (automatic)',
        'PaymentMethod_Electronic check',
        'PaymentMethod_Mailed check',
        'tenure', 'MonthlyCharges', 'TotalCharges'
    ]
    
    # Add missing columns with 0
    for col in expected_columns:
        if col not in df_encoded.columns:
            df_encoded[col] = 0
    
    # Reorder columns to match training data
    df_encoded = df_encoded[expected_columns]
    
    return df_encoded

def load_model(model_path='models/churn_model.pkl'):
    """
    Load the trained model
    """
    try:
        model = joblib.load(model_path)
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def predict_churn(model, input_data):
    """
    Make predictions using the loaded model
    """
    try:
        # Preprocess the input data
        processed_data = preprocess_input(input_data)
        
        # Make prediction
        prediction = model.predict(processed_data)
        prediction_proba = model.predict_proba(processed_data)
        
        # Convert prediction to human-readable format
        result = 'Churn' if prediction[0] == 1 else 'No Churn'
        confidence = prediction_proba[0][1] if prediction[0] == 1 else prediction_proba[0][0]
        
        return result, confidence
    except Exception as e:
        print(f"Error making prediction: {e}")
        return None, None
