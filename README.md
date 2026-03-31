# Loan Default Prediction Streamlit App

This is a Streamlit web application for predicting loan default risk using machine learning models with bias handling.

## Features

- **Data Overview**: Load and explore the loan application dataset
- **Model Training**: Train 4 different models (Logistic Regression, Decision Tree, Random Forest, XGBoost)
- **Prediction**: Make predictions using any of the trained models
- **Model Evaluation**: Compare and analyze performance of all models

## Setup

1. Ensure you have Python installed
2. Install required packages:
   ```bash
   pip install streamlit pandas scikit-learn xgboost matplotlib seaborn joblib
   ```

3. Unzip the data file:
   ```bash
   # Extract application_train.csv.zip
   ```

## Running the App

```bash
streamlit run app.py
```

The app will open in your browser (typically at `http://localhost:8501`)

## Usage

1. **Data Overview**: Click "Load and Process Data" to see dataset statistics
2. **Model Training**: Click "Train All Models" to train LR, DT, RF, and XGB models (takes ~3-4 minutes)
3. **Prediction**: Select any model from the dropdown and enter loan application details to get default risk prediction
4. **Model Evaluation**: 
   - Use the dropdown to select "All Models" for comparison table, or choose individual models (Logistic Regression, Decision Tree, Random Forest, XGBoost) for detailed analysis
   - View ROC-AUC scores, confusion matrices, and feature importance plots

## Model Details

- **Logistic Regression**: With class weights and L1 regularization
- **Decision Tree**: Optimized depth and minimum samples
- **Random Forest**: Ensemble of 300 trees with class balancing
- **XGBoost**: Gradient boosting with scale_pos_weight for imbalance

All models use:
- Feature engineering from raw loan application data
- Optimal threshold selection for F1 score maximization
- 3-zone decision system: Approve, Review, Reject

## Files

- `app.py`: Main Streamlit application
- `application_train.csv`: Training data (extract from zip)
- `all_models.pkl`: Saved trained models (created after training)
- `label_encoders.pkl`: Categorical feature encoders
- `scaler.pkl`: Feature scaler for Logistic Regression
- `test_data.pkl`: Test data for evaluation
- `xgb_model.pkl`: Trained model (created after training)
- `label_encoders.pkl`: Encoders for categorical features