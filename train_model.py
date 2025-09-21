import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

def load_data():
    """Load and preprocess the telco churn dataset"""
    try:
        # Load the data
        df = pd.read_csv('data/telco_churn.csv')
        
        # Convert TotalCharges to numeric, coerce errors to NaN
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        
        # Handle missing values
        df['TotalCharges'] = df['TotalCharges'].fillna(0)
        
        # Drop customerID as it's not a feature
        df = df.drop('customerID', axis=1)
        
        # Convert Churn to binary (1 for 'Yes', 0 for 'No')
        df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
        
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def preprocess_data(df):
    """Preprocess the data for model training"""
    # Convert SeniorCitizen to string for one-hot encoding
    df['SeniorCitizen'] = df['SeniorCitizen'].astype(str)
    
    # Define categorical and numerical columns
    categorical_cols = [
        'gender', 'SeniorCitizen', 'Partner', 'Dependents',
        'PhoneService', 'MultipleLines', 'InternetService',
        'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
        'TechSupport', 'StreamingTV', 'StreamingMovies',
        'Contract', 'PaperlessBilling', 'PaymentMethod'
    ]
    
    numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    
    # One-hot encode categorical variables
    df_encoded = pd.get_dummies(df[categorical_cols], drop_first=True)
    
    # Add numerical columns
    for col in numerical_cols:
        df_encoded[col] = df[col]
    
    # Target variable
    y = df['Churn']
    
    return df_encoded, y

def train_model(X, y):
    """Train a Random Forest Classifier"""
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Initialize and train the model
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        class_weight='balanced'
    )
    
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Print evaluation metrics
    print("\nModel Evaluation:")
    print("-" * 50)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    return model, X_test, y_test

def save_model(model, filename='models/churn_model.pkl'):
    """Save the trained model to disk"""
    # Create models directory if it doesn't exist
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    # Save the model
    joblib.dump(model, filename)
    print(f"\nModel saved to {filename}")

if __name__ == "__main__":
    print("Loading and preprocessing data...")
    df = load_data()
    
    if df is not None:
        X, y = preprocess_data(df)
        
        print("\nTraining model...")
        model, X_test, y_test = train_model(X, y)
        
        # Save the model
        save_model(model)
        
        print("\nModel training and saving completed successfully!")
