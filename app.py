import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
from utils.preprocess import preprocess_input, load_model, predict_churn

# Set page config
st.set_page_config(
    page_title="Telco Customer Churn Prediction",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {font-size: 32px; font-weight: 700; color: #1f77b4; text-align: center; margin-bottom: 20px;}
    .sub-header {font-size: 20px; font-weight: 600; color: #2c3e50; margin-top: 20px;}
    .stButton>button {background-color: #1f77b4; color: white; border-radius: 5px; padding: 0.5rem 1rem;}
    .stButton>button:hover {background-color: #155a8a;}
    .prediction-box {padding: 20px; border-radius: 10px; margin: 20px 0; text-align: center;}
    .churn {background-color: #ffdddd; border-left: 5px solid #ff3333;}
    .no-churn {background-color: #ddffdd; border-left: 5px solid #33cc33;}
    .feature-importance {margin-top: 30px;}
    </style>
""", unsafe_allow_html=True)

# Title and description
st.markdown('<div class="main-header">📱 Telco Customer Churn Prediction</div>', unsafe_allow_html=True)
st.write("""
    This application predicts whether a customer will churn or not based on their account information, 
    services subscribed, and customer account details.
""")

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Make Prediction", "Explore Data", "About"])

# Load model (you'll need to train and save this model first)
model = load_model('models/churn_model.pkl')

if page == "Make Prediction":
    st.markdown('<div class="sub-header">🔮 Make a Prediction</div>', unsafe_allow_html=True)
    
    # Create a form for user input
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Customer Information")
            gender = st.selectbox("Gender", ["Female", "Male"])
            senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
            partner = st.selectbox("Has Partner", ["No", "Yes"])
            dependents = st.selectbox("Has Dependents", ["No", "Yes"])
            tenure = st.number_input("Tenure (months)", min_value=0, max_value=100, value=1)
            
        with col2:
            st.subheader("Services Subscribed")
            phone_service = st.selectbox("Phone Service", ["No", "Yes"])
            multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
            internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
            online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
            online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
            
            st.subheader("Account Information")
            contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
            paperless_billing = st.selectbox("Paperless Billing", ["No", "Yes"])
            payment_method = st.selectbox("Payment Method", [
                "Electronic check", "Mailed check", 
                "Bank transfer (automatic)", "Credit card (automatic)"
            ])
            monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, value=50.0, step=0.01)
            total_charges = st.number_input("Total Charges ($)", min_value=0.0, value=500.0, step=0.01)
        
        submitted = st.form_submit_button("Predict Churn")
    
    # When the form is submitted
    if submitted:
        # Create a dictionary with the input data
        input_data = {
            'gender': [gender],
            'SeniorCitizen': [1 if senior_citizen == "Yes" else 0],
            'Partner': [partner],
            'Dependents': [dependents],
            'tenure': [tenure],
            'PhoneService': [phone_service],
            'MultipleLines': [multiple_lines],
            'InternetService': [internet_service],
            'OnlineSecurity': [online_security],
            'OnlineBackup': [online_backup],
            'DeviceProtection': ["No"],  # Default value
            'TechSupport': ["No"],  # Default value
            'StreamingTV': ["No"],  # Default value
            'StreamingMovies': ["No"],  # Default value
            'Contract': [contract],
            'PaperlessBilling': [paperless_billing],
            'PaymentMethod': [payment_method],
            'MonthlyCharges': [monthly_charges],
            'TotalCharges': [total_charges]
        }
        
        # Convert to DataFrame
        input_df = pd.DataFrame(input_data)
        
        # Make prediction
        if model is not None:
            prediction, confidence = predict_churn(model, input_df)
            
            # Display prediction
            if prediction is not None:
                st.markdown("### Prediction Result")
                if prediction == "Churn":
                    st.markdown(f'<div class="prediction-box churn">' 
                              f'<h3>Prediction: {prediction} 🔴</h3>' 
                              f'<p>Confidence: {confidence*100:.2f}%</p>' 
                              f'<p>This customer is likely to churn. Consider retention strategies.</p>' 
                              f'</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="prediction-box no-churn">' 
                              f'<h3>Prediction: {prediction} ✅</h3>' 
                              f'<p>Confidence: {confidence*100:.2f}%</p>' 
                              f'<p>This customer is not likely to churn.</p>' 
                              f'</div>', unsafe_allow_html=True)
                
                # Show feature importance (if available)
                if hasattr(model, 'feature_importances_'):
                    st.markdown("### Feature Importance")
                    feature_importance = pd.DataFrame({
                        'Feature': input_df.columns,
                        'Importance': model.feature_importances_
                    }).sort_values('Importance', ascending=False)
                    
                    fig = px.bar(
                        feature_importance.head(10), 
                        x='Importance', 
                        y='Feature', 
                        orientation='h',
                        title='Top 10 Most Important Features',
                        labels={'Importance': 'Importance Score', 'Feature': ''}
                    )
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.error("Error making prediction. Please check the input data and try again.")
        else:
            st.error("Model not found. Please train and save the model first.")

elif page == "Explore Data":
    st.markdown('<div class="sub-header">📊 Explore the Data</div>', unsafe_allow_html=True)
    
    # Load sample data for visualization
    @st.cache_data
    def load_data():
        # In a real app, you would load your actual data here
        # For now, we'll use sample data
        data = {
            'Churn': ['No']*2000 + ['Yes']*500,
            'tenure': list(np.random.randint(1, 72, 2500)),
            'MonthlyCharges': list(np.random.normal(64, 30, 2500)),
            'Contract': np.random.choice(['Month-to-month', 'One year', 'Two year'], 2500, p=[0.6, 0.25, 0.15]),
            'InternetService': np.random.choice(['DSL', 'Fiber optic', 'No'], 2500, p=[0.4, 0.4, 0.2])
        }
        return pd.DataFrame(data)
    
    df = load_data()
    
    # Show data summary
    st.subheader("Data Overview")
    st.dataframe(df.head())
    
    # Show churn distribution
    st.subheader("Churn Distribution")
    fig1 = px.pie(df, names='Churn', title='Churn Distribution', 
                 color_discrete_sequence=px.colors.qualitative.Set2)
    st.plotly_chart(fig1, use_container_width=True)
    
    # Show tenure vs monthly charges
    st.subheader("Tenure vs Monthly Charges")
    fig2 = px.scatter(
        df, x='tenure', y='MonthlyCharges', 
        color='Churn',
        title='Tenure vs Monthly Charges by Churn Status',
        labels={'tenure': 'Tenure (months)', 'MonthlyCharges': 'Monthly Charges ($)'}
    )
    st.plotly_chart(fig2, use_container_width=True)
    
    # Show contract type distribution
    st.subheader("Contract Type Distribution")
    fig3 = px.histogram(
        df, x='Contract', color='Churn',
        title='Contract Type Distribution by Churn Status',
        barmode='group',
        color_discrete_sequence=px.colors.qualitative.Set2[1:]
    )
    st.plotly_chart(fig3, use_container_width=True)

else:  # About page
    st.markdown('<div class="sub-header">ℹ️ About This App</div>', unsafe_allow_html=True)
    
    st.write("""
    ### Telco Customer Churn Prediction
    
    This application helps predict customer churn for a telecommunications company. 
    Churn prediction is essential for customer retention and business growth.
    
    #### Features:
    - Predict customer churn probability
    - Explore data visualizations
    - Understand feature importance
    
    #### How to Use:
    1. Navigate to "Make Prediction"
    2. Fill in the customer details
    3. Click "Predict Churn" to see the prediction
    
    #### Data Source:
    The model is trained on the Telco Customer Churn dataset, which includes information about:
    - Customer account information
    - Services subscribed
    - Customer account details
    
    #### Model Information:
    - Model: Random Forest Classifier
    - Accuracy: ~80% (example)
    - Last Updated: September 2023
    
    For more information, contact the development team.
    """)
