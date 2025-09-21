# Telco Customer Churn Prediction

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32.0-FF4B4B)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3.0-orange)

A Streamlit web application for predicting customer churn in a telecommunications company. This application helps identify customers who are likely to churn, enabling proactive retention strategies.

## 🚀 Features

- **📊 Interactive Dashboard**: User-friendly interface for churn prediction
- **🔍 Data Exploration**: Visualize customer data and churn patterns
- **🤖 Machine Learning**: Random Forest model for churn prediction
- **📈 Model Insights**: Feature importance visualization
- **📱 Responsive Design**: Works on desktop and mobile devices

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/telco-churn-prediction.git
   cd telco-churn-prediction
   ```

2. **Create and activate a virtual environment** (recommended):
   ```bash
   # Windows
   python -m venv venv
   .\venv\Scripts\activate
   
   # macOS/Linux
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## 🚀 Quick Start

1. **Train the model**:
   ```bash
   python train_model.py
   ```
   This will train the model and save it to the `models` directory.

2. **Run the application**:
   ```bash
   streamlit run app.py
   ```

3. **Open your browser** and navigate to `http://localhost:8501`

## 📂 Project Structure

```
telco-churn-prediction/
├── app.py                  # Main Streamlit application
├── train_model.py          # Model training script
├── utils/
│   └── preprocess.py       # Data preprocessing utilities
├── models/                 # Saved models
├── data/                   # Dataset directory
│   └── telco_churn.csv     # Sample dataset (not included in repo)
├── .gitignore
├── requirements.txt        # Python dependencies
└── README.md
```

## 📊 Dataset

The application uses the [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) dataset, which includes:

- **Customer Account Information**: Customer ID, gender, senior citizen status, partner/dependents
- **Services Enrolled**: Phone, multiple lines, internet service, online security, etc.
- **Account Information**: Tenure, contract type, paperless billing, payment method
- **Charges**: Monthly charges, total charges
- **Target Variable**: Churn status (Yes/No)

## 🤖 Model Details

- **Algorithm**: Random Forest Classifier
- **Preprocessing**:
  - One-hot encoding for categorical variables
  - Handling missing values
  - Feature scaling
- **Evaluation Metrics**:
  - Accuracy: ~77%
  - Precision (Churn): 0.55
  - Recall (Churn): 0.71
  - F1-Score (Churn): 0.62

## 📝 How to Use

1. **Make Prediction**:
   - Fill in the customer details in the form
   - Click "Predict Churn" to see the prediction
   - View the prediction probability and feature importance

2. **Explore Data**:
   - View distribution of churn
   - Explore relationships between features
   - Analyze contract types and payment methods

## 📦 Dependencies

- Python 3.8+
- Streamlit
- Pandas
- NumPy
- scikit-learn
- Plotly
- Matplotlib
- Seaborn

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Dataset provided by [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- Built with [Streamlit](https://streamlit.io/)
- Icons by [Shields.io](https://shields.io/)
