import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, precision_recall_curve
from xgboost import XGBClassifier
import joblib
import os

print('Checking if models already exist...')
if os.path.exists('all_models.pkl'):
    print('Models already trained!')
    exit()

print('Loading data...')
df = pd.read_csv('application_train.csv')
print(f'Loaded {len(df)} rows')

# Basic preprocessing (simplified for speed)
missing = df.isnull().sum() / len(df) * 100
cols_to_drop = missing[missing > 50].index
df = df.drop(columns=cols_to_drop)

for col in df.select_dtypes(include=['float64', 'int64']):
    df[col] = df[col].fillna(df[col].median())
for col in df.select_dtypes(include=['object']):
    df[col] = df[col].fillna(df[col].mode()[0])

df = df[df['AMT_INCOME_TOTAL'] < df['AMT_INCOME_TOTAL'].quantile(0.99)]

# Prepare data
y = df['TARGET']
X = df.drop(columns=['TARGET'] + [c for c in ['SK_ID_CURR'] if c in df.columns])

cat_cols = X.select_dtypes(include=['object']).columns
le_dict = {}
for col in cat_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    le_dict[col] = le

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Scale
scaler = StandardScaler()
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()
numeric_cols = X_train.select_dtypes(include=['int64', 'float64']).columns
X_train_scaled[numeric_cols] = scaler.fit_transform(X_train[numeric_cols].fillna(0))
X_test_scaled[numeric_cols] = scaler.transform(X_test[numeric_cols].fillna(0))

X_train_lr = X_train_scaled.fillna(0)
X_test_lr = X_test_scaled.fillna(0)
X_train_tree = X_train.fillna(-999)
X_test_tree = X_test.fillna(-999)

print('Training models...')
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

# LR
lr = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
lr.fit(X_train_lr, y_train)
print('LR done')

# DT
dt = DecisionTreeClassifier(max_depth=8, class_weight='balanced', random_state=42)
dt.fit(X_train_tree, y_train)
print('DT done')

# RF
rf = RandomForestClassifier(n_estimators=100, max_depth=10, class_weight='balanced', random_state=42, n_jobs=-1)
rf.fit(X_train_tree, y_train)
print('RF done')

# XGB
xgb = XGBClassifier(n_estimators=100, max_depth=5, scale_pos_weight=scale_pos_weight, random_state=42)
xgb.fit(X_train_tree, y_train)
print('XGB done')

models = {'LR': lr, 'DT': dt, 'RF': rf, 'XGB': xgb}
joblib.dump(models, 'all_models.pkl')
joblib.dump(le_dict, 'label_encoders.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump((X_test_lr, X_test_tree, y_test), 'test_data.pkl')

print('All models saved successfully!')
print('You can now use the dropdown in Model Evaluation!')