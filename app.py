import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay, precision_recall_curve
from xgboost import XGBClassifier
import joblib
import os

# Page config
st.set_page_config(page_title="Loan Default Prediction", page_icon="🏦", layout="wide")

# Title
st.title("🏦 Loan Default Prediction System")
st.markdown("Predict loan default risk using machine learning models with bias handling.")

# Sidebar
st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to", ["Data Overview", "Model Training", "Prediction", "Model Evaluation"])

# Load data function
@st.cache_data
def load_data():
    df = pd.read_csv('application_train.csv')
    return df

# Preprocessing function
def preprocess_data(df):
    # Drop columns with >50% missing
    missing = df.isnull().sum() / len(df) * 100
    cols_to_drop = missing[missing > 50].index
    df = df.drop(columns=cols_to_drop)

    # Fill missing values
    for col in df.select_dtypes(include=['float64', 'int64']):
        df[col] = df[col].fillna(df[col].median())
    for col in df.select_dtypes(include=['object']):
        df[col] = df[col].fillna(df[col].mode()[0])

    # Remove extreme income outliers
    df = df[df['AMT_INCOME_TOTAL'] < df['AMT_INCOME_TOTAL'].quantile(0.99)]

    return df

# Feature engineering
def feature_engineering(df):
    # Age & Employment
    df['AGE'] = df['DAYS_BIRTH'] / -365
    df['EMPLOYED_YEARS'] = df['DAYS_EMPLOYED'].clip(upper=0) / -365
    df['EMPLOYED_PERCENT'] = df['EMPLOYED_YEARS'] / (df['AGE'] + 1)
    df['AGE_EMPLOYED_DIFF'] = df['AGE'] - df['EMPLOYED_YEARS']

    # Credit & Income Ratios
    df['CREDIT_INCOME_RATIO'] = df['AMT_CREDIT'] / (df['AMT_INCOME_TOTAL'] + 1)
    df['ANNUITY_INCOME_RATIO'] = df['AMT_ANNUITY'] / (df['AMT_INCOME_TOTAL'] + 1)
    df['CREDIT_TERM'] = df['AMT_CREDIT'] / (df['AMT_ANNUITY'] + 1)
    df['PAYMENT_RATE'] = df['AMT_ANNUITY'] / (df['AMT_CREDIT'] + 1)

    # Family & Per-Person
    df['INCOME_PER_PERSON'] = df['AMT_INCOME_TOTAL'] / (df['CNT_FAM_MEMBERS'] + 1)
    df['CREDIT_PER_PERSON'] = df['AMT_CREDIT'] / (df['CNT_FAM_MEMBERS'] + 1)
    df['ADULTS'] = df['CNT_FAM_MEMBERS'] - df['CNT_CHILDREN']
    df['CHILD_TO_ADULT_RATIO'] = df['CNT_CHILDREN'] / (df['ADULTS'] + 1)

    # External Scores
    ext_cols = [c for c in ['EXT_SOURCE_1','EXT_SOURCE_2','EXT_SOURCE_3'] if c in df.columns]
    if len(ext_cols) >= 2:
        df['EXT_MEAN'] = df[ext_cols].mean(axis=1)
        df['EXT_MIN'] = df[ext_cols].min(axis=1)
        df['EXT_MAX'] = df[ext_cols].max(axis=1)
        df['EXT_PRODUCT'] = df[ext_cols].prod(axis=1)

    # Document & Contact
    doc_cols = [c for c in df.columns if 'FLAG_DOCUMENT' in c]
    df['DOC_COUNT'] = df[doc_cols].sum(axis=1)

    contact_cols = [c for c in ['FLAG_MOBIL','FLAG_EMP_PHONE','FLAG_WORK_PHONE',
                                  'FLAG_CONT_MOBILE','FLAG_PHONE','FLAG_EMAIL'] if c in df.columns]
    df['CONTACT_SUM'] = df[contact_cols].sum(axis=1)

    # Social Circle
    if 'DEF_30_CNT_SOCIAL_CIRCLE' in df.columns:
        df['DEF_30_RATIO'] = df['DEF_30_CNT_SOCIAL_CIRCLE'] / (df['OBS_30_CNT_SOCIAL_CIRCLE'] + 1)
        df['DEF_60_RATIO'] = df['DEF_60_CNT_SOCIAL_CIRCLE'] / (df['OBS_60_CNT_SOCIAL_CIRCLE'] + 1)

    # Bureau
    bureau_cols = [c for c in df.columns if 'AMT_REQ_CREDIT_BUREAU' in c]
    if bureau_cols:
        df['BUREAU_TOTAL'] = df[bureau_cols].sum(axis=1)

    # High Income
    df['IS_HIGH_INCOME'] = (df['AMT_INCOME_TOTAL'] > df['AMT_INCOME_TOTAL'].median()).astype(int)

    # Fix infinities
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    return df

# Prepare X, y
def prepare_data(df):
    y = df['TARGET']
    drop_cols = ['TARGET'] + [c for c in ['SK_ID_CURR'] if c in df.columns]
    X = df.drop(columns=drop_cols)

    # Encode categoricals
    cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    le_dict = {}
    for col in cat_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        le_dict[col] = le

    return X, y, le_dict

# Prepare scaled data for LR
def prepare_scaled_data(X, y):
    numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns

    scaler = StandardScaler()
    X_scaled = X.copy()
    X_scaled[numeric_cols] = scaler.fit_transform(X[numeric_cols].fillna(0))

    # For LR — use scaled + fill NaN with 0
    X_lr = X_scaled.fillna(0)

    # For tree-based models — use unscaled, just fill NaN with -999
    X_tree = X.fillna(-999)

    return X_lr, X_tree, scaler

# Train a single model
@st.cache_resource
def train_single_model(model_key, X_train_lr, X_train_tree, y_train):
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    if model_key == 'LR':
        st.write("Training Logistic Regression...")
        model = LogisticRegression(
            class_weight='balanced',
            max_iter=3000,
            solver='saga',
            C=0.1,
            random_state=42
        )
        model.fit(X_train_lr, y_train)
        return model

    if model_key == 'DT':
        st.write("Training Decision Tree...")
        model = DecisionTreeClassifier(
            max_depth=8,
            min_samples_leaf=50,
            min_samples_split=100,
            class_weight='balanced',
            random_state=42
        )
        model.fit(X_train_tree, y_train)
        return model

    if model_key == 'RF':
        st.write("Training Random Forest...")
        model = RandomForestClassifier(
            n_estimators=300,
            max_depth=12,
            min_samples_leaf=30,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train_tree, y_train)
        return model

    if model_key == 'XGB':
        st.write("Training XGBoost...")
        model = XGBClassifier(
            n_estimators=500,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            gamma=1,
            reg_alpha=0.5,
            reg_lambda=1,
            eval_metric='logloss',
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train_tree, y_train)
        return model

    raise ValueError(f"Unknown model key: {model_key}")

# Train all models
@st.cache_resource
def train_all_models(X_train_lr, X_train_tree, y_train):
    models = {}
    for key in ['LR', 'DT', 'RF', 'XGB']:
        models[key] = train_single_model(key, X_train_lr, X_train_tree, y_train)
    return models

# Main content
if page == "Data Overview":
    st.header("📊 Data Overview")

    if st.button("Load and Process Data"):
        with st.spinner("Loading data..."):
            df = load_data()

        st.success(f"Loaded {len(df)} records")

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Target Distribution")
            target_counts = df['TARGET'].value_counts()
            fig, ax = plt.subplots()
            ax.pie(target_counts, labels=['Approved', 'Defaulted'], autopct='%1.1f%%')
            st.pyplot(fig)

        with col2:
            st.subheader("Data Info")
            st.write(f"Shape: {df.shape}")
            st.write("Columns:", list(df.columns[:10]) + ["..."])

        st.subheader("Sample Data")
        st.dataframe(df.head())

elif page == "Model Training":
    st.header("🤖 Model Training")

    # training includes dropdown for each model and all models
    model_options = ["All Models", "Logistic Regression", "Decision Tree", "Random Forest", "XGBoost"]
    selected_model_option = st.selectbox("Choose model(s) to train", model_options)

    if st.button("Train Selected Model(s)"):
        with st.spinner("Processing data..."):
            df = load_data()
            df = preprocess_data(df)
            df = feature_engineering(df)
            X, y, le_dict = prepare_data(df)

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, stratify=y, random_state=42
            )

            # Prepare scaled and tree data
            X_train_lr, X_train_tree, scaler = prepare_scaled_data(X_train, y_train)
            X_test_lr, X_test_tree, _ = prepare_scaled_data(X_test, y_test)

        st.success("Data processed successfully!")

        with st.spinner("Training model(s)..."):
            models = {}
            if selected_model_option == "All Models":
                models = train_all_models(X_train_lr, X_train_tree, y_train)
            else:
                single_map = {
                    "Logistic Regression": "LR",
                    "Decision Tree": "DT",
                    "Random Forest": "RF",
                    "XGBoost": "XGB"
                }
                model_key = single_map[selected_model_option]
                model = train_single_model(model_key, X_train_lr, X_train_tree, y_train)
                models[model_key] = model

            # Save all models and preprocessing objects
            joblib.dump(models, 'all_models.pkl')
            joblib.dump(le_dict, 'label_encoders.pkl')
            joblib.dump(scaler, 'scaler.pkl')
            joblib.dump((X_test_lr, X_test_tree, y_test), 'test_data.pkl')
            joblib.dump(X_test_tree.columns.tolist(), 'feature_columns.pkl')

        st.success("Model training and saving completed successfully!")

        # Quick evaluation summary
        st.subheader("Training Complete - Quick Evaluation")
        results = {}
        for name, model in models.items():
            if name == 'LR':
                proba = model.predict_proba(X_test_lr)[:, 1]
            else:
                proba = model.predict_proba(X_test_tree)[:, 1]
            auc = roc_auc_score(y_test, proba)
            results[name] = auc

        results_df = pd.DataFrame([{
            'Model': {'LR': 'Logistic Regression', 'DT': 'Decision Tree', 'RF': 'Random Forest', 'XGB': 'XGBoost'}[k],
            'ROC-AUC': v
        } for k, v in results.items()])

        st.dataframe(results_df)

elif page == "Prediction":
    st.header("🔮 Loan Default Prediction")

    # Check if models exist
    if not os.path.exists('all_models.pkl'):
        st.warning("Please train the models first in the Model Training tab.")
    else:
        models = joblib.load('all_models.pkl')
        le_dict = joblib.load('label_encoders.pkl')
        scaler = joblib.load('scaler.pkl')

        # Load expected feature order for inference
        feature_columns = []
        if os.path.exists('feature_columns.pkl'):
            feature_columns = joblib.load('feature_columns.pkl')

        # Model selection
        model_choice = st.selectbox("Select Model for Prediction",
                                   list(models.keys()),
                                   format_func=lambda x: {
                                       'LR': 'Logistic Regression',
                                       'DT': 'Decision Tree',
                                       'RF': 'Random Forest',
                                       'XGB': 'XGBoost'
                                   }[x])

        model = models[model_choice]
        st.subheader(f"Using {model_choice} Model")

        st.subheader("Enter Loan Application Details")

        col1, col2, col3 = st.columns(3)

        with col1:
            amt_income = st.number_input("Annual Income", min_value=0.0, value=50000.0)
            amt_credit = st.number_input("Credit Amount", min_value=0.0, value=200000.0)
            amt_annuity = st.number_input("Annuity Amount", min_value=0.0, value=10000.0)

        with col2:
            days_birth = st.number_input("Age (days since birth, negative)", min_value=-365*80, max_value=-365*18, value=-365*30)
            days_employed = st.number_input("Days Employed (negative)", min_value=-365*50, max_value=0, value=-365*5)
            cnt_fam_members = st.number_input("Family Members", min_value=1, value=2)
            cnt_children = st.number_input("Children", min_value=0, value=0)

        with col3:
            ext_source_1 = st.number_input("External Source 1", min_value=0.0, max_value=1.0, value=0.5)
            ext_source_2 = st.number_input("External Source 2", min_value=0.0, max_value=1.0, value=0.5)
            ext_source_3 = st.number_input("External Source 3", min_value=0.0, max_value=1.0, value=0.5)

        # Categorical inputs
        name_contract_type = st.selectbox("Contract Type", ["Cash loans", "Revolving loans"])
        name_income_type = st.selectbox("Income Type", ["Working", "Commercial associate", "Pensioner", "State servant", "Unemployed"])
        name_education_type = st.selectbox("Education", ["Secondary / secondary special", "Higher education", "Incomplete higher", "Lower secondary", "Academic degree"])
        name_family_status = st.selectbox("Family Status", ["Single / not married", "Married", "Civil marriage", "Widow", "Separated"])
        name_housing_type = st.selectbox("Housing Type", ["House / apartment", "Rented apartment", "With parents", "Municipal apartment", "Office apartment"])

        if st.button("Predict Default Risk"):
            # Create input dataframe
            input_data = {
                'AMT_INCOME_TOTAL': amt_income,
                'AMT_CREDIT': amt_credit,
                'AMT_ANNUITY': amt_annuity,
                'DAYS_BIRTH': days_birth,
                'DAYS_EMPLOYED': days_employed,
                'CNT_FAM_MEMBERS': cnt_fam_members,
                'CNT_CHILDREN': cnt_children,
                'EXT_SOURCE_1': ext_source_1,
                'EXT_SOURCE_2': ext_source_2,
                'EXT_SOURCE_3': ext_source_3,
                'NAME_CONTRACT_TYPE': name_contract_type,
                'NAME_INCOME_TYPE': name_income_type,
                'NAME_EDUCATION_TYPE': name_education_type,
                'NAME_FAMILY_STATUS': name_family_status,
                'NAME_HOUSING_TYPE': name_housing_type,
            }

            # Add default values for other features (simplified)
            # In a real app, you'd need all features
            df_input = pd.DataFrame([input_data])

            # Feature engineering (simplified version)
            df_input['AGE'] = df_input['DAYS_BIRTH'] / -365
            df_input['EMPLOYED_YEARS'] = df_input['DAYS_EMPLOYED'].clip(upper=0) / -365
            df_input['CREDIT_INCOME_RATIO'] = df_input['AMT_CREDIT'] / (df_input['AMT_INCOME_TOTAL'] + 1)
            df_input['EXT_MEAN'] = df_input[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].mean(axis=1)
            df_input['INCOME_PER_PERSON'] = df_input['AMT_INCOME_TOTAL'] / (df_input['CNT_FAM_MEMBERS'] + 1)
            df_input['IS_HIGH_INCOME'] = (df_input['AMT_INCOME_TOTAL'] > 50000).astype(int)  # approximate

            # Encode categoricals
            for col in le_dict:
                if col in df_input.columns:
                    try:
                        df_input[col] = le_dict[col].transform(df_input[col].astype(str))
                    except:
                        df_input[col] = 0  # default for unknown categories

            # Fill missing columns with defaults
            # This is a simplification; in practice, need all features
            if feature_columns:
                expected_features = feature_columns
            elif hasattr(model, 'feature_names_in_'):
                expected_features = list(model.feature_names_in_)
            else:
                expected_features = list(df_input.columns)

            for feat in expected_features:
                if feat not in df_input.columns:
                    df_input[feat] = 0

            df_input = df_input[expected_features]

            # Prepare data based on model type
            if model_choice == 'LR':
                # Scale numeric features for LR
                numeric_cols = df_input.select_dtypes(include=['int64', 'float64']).columns
                df_input_scaled = df_input.copy()
                df_input_scaled[numeric_cols] = scaler.transform(df_input_scaled[numeric_cols].fillna(0))
                df_input_final = df_input_scaled.fillna(0)
            else:
                # Use unscaled for tree models
                df_input_final = df_input.fillna(-999)

            # Predict
            proba = model.predict_proba(df_input_final)[0, 1]

            # Decision zones (approximate)
            approve_zone = 0.15
            reject_zone = 0.35

            if proba < approve_zone:
                decision = "✅ Approve"
                color = "green"
            elif proba < reject_zone:
                decision = "⚠️ Review"
                color = "orange"
            else:
                decision = "❌ Reject"
                color = "red"

            st.subheader("Prediction Results")
            col1, col2 = st.columns(2)

            with col1:
                st.metric("Default Probability", f"{proba:.3f}")

            with col2:
                st.markdown(f"**Decision:** <span style='color:{color}; font-size:24px'>{decision}</span>", unsafe_allow_html=True)

            # Gauge chart
            fig, ax = plt.subplots(figsize=(6, 2))
            ax.barh([0], [1], color='lightgray', height=0.3)
            ax.barh([0], [proba], color='red', height=0.3)
            ax.set_xlim(0, 1)
            ax.set_yticks([])
            ax.set_xlabel("Default Risk")
            ax.axvline(approve_zone, color='green', linestyle='--', alpha=0.7)
            ax.axvline(reject_zone, color='orange', linestyle='--', alpha=0.7)
            st.pyplot(fig)

elif page == "Model Evaluation":
    st.header("📈 Model Evaluation")

    if not os.path.exists('all_models.pkl'):
        st.warning("Please train the models first.")
    else:
        models = joblib.load('all_models.pkl')
        X_test_lr, X_test_tree, y_test = joblib.load('test_data.pkl')

        # Model selection dropdown
        model_options = ["All Models", "Logistic Regression", "Decision Tree", "Random Forest", "XGBoost"]
        selected_option = st.selectbox("Select Model to Evaluate", model_options)

        if selected_option == "All Models":
            if st.button("Evaluate All Models"):
                with st.spinner("Evaluating all models..."):
                    results = {}

                    for name, model in models.items():
                        if name == 'LR':
                            y_proba = model.predict_proba(X_test_lr)[:, 1]
                        else:
                            y_proba = model.predict_proba(X_test_tree)[:, 1]

                        # Find optimal threshold
                        precision, recall, thresholds = precision_recall_curve(y_test, y_proba)
                        f1 = 2 * precision * recall / (precision + recall + 1e-8)
                        best_idx = np.argmax(f1[:-1])
                        best_thresh = thresholds[best_idx]

                        y_pred = (y_proba >= best_thresh).astype(int)
                        auc = roc_auc_score(y_test, y_proba)

                        results[name] = {
                            'auc': auc,
                            'threshold': best_thresh,
                            'y_pred': y_pred,
                            'y_proba': y_proba
                        }

                # Summary table
                st.subheader("Model Performance Summary")
                summary_df = pd.DataFrame([{
                    'Model': {
                        'LR': 'Logistic Regression',
                        'DT': 'Decision Tree',
                        'RF': 'Random Forest',
                        'XGB': 'XGBoost'
                    }[name],
                    'ROC-AUC': res['auc'],
                    'Optimal Threshold': res['threshold']
                } for name, res in results.items()])

                st.dataframe(summary_df.style.highlight_max(axis=0, subset=['ROC-AUC']))

        else:
            # Map option to model key
            model_key_map = {
                "Logistic Regression": "LR",
                "Decision Tree": "DT",
                "Random Forest": "RF",
                "XGBoost": "XGB"
            }
            selected_model = model_key_map[selected_option]
            model = models[selected_model]

            if st.button(f"Evaluate {selected_option}"):
                with st.spinner(f"Evaluating {selected_option}..."):
                    if selected_model == 'LR':
                        y_proba = model.predict_proba(X_test_lr)[:, 1]
                    else:
                        y_proba = model.predict_proba(X_test_tree)[:, 1]

                    # Find optimal threshold
                    precision, recall, thresholds = precision_recall_curve(y_test, y_proba)
                    f1 = 2 * precision * recall / (precision + recall + 1e-8)
                    best_idx = np.argmax(f1[:-1])
                    best_thresh = thresholds[best_idx]

                    y_pred = (y_proba >= best_thresh).astype(int)
                    auc = roc_auc_score(y_test, y_proba)

                st.subheader(f"Detailed Results - {selected_option}")

                col1, col2 = st.columns(2)

                with col1:
                    st.metric("ROC-AUC Score", f"{auc:.4f}")
                    st.metric("Optimal Threshold", f"{best_thresh:.3f}")

                with col2:
                    # Classification report
                    st.subheader("Classification Report")
                    report = classification_report(y_test, y_pred,
                                                 target_names=['Approved', 'Defaulted'],
                                                 output_dict=True)
                    st.dataframe(pd.DataFrame(report).transpose().round(3))

                # Confusion matrix
                st.subheader("Confusion Matrix")
                fig, ax = plt.subplots(figsize=(6, 4))
                ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred),
                                       display_labels=['Approved', 'Defaulted']).plot(ax=ax, colorbar=False)
                st.pyplot(fig)

                # Feature importance (for tree models)
                if selected_model in ['DT', 'RF', 'XGB']:
                    st.subheader("Feature Importance")
                    feat_imp = pd.Series(model.feature_importances_, index=X_test_tree.columns)
                    top10 = feat_imp.nlargest(10)
                    fig, ax = plt.subplots(figsize=(8, 5))
                    top10.sort_values().plot(kind='barh', ax=ax, color='steelblue')
                    ax.set_title(f"{selected_option} - Top 10 Feature Importances")
                    st.pyplot(fig)

# Footer
st.markdown("---")
st.markdown("Built with Streamlit | Loan Default Prediction with Bias Handling")