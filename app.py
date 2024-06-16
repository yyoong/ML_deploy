import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
from sklearn.feature_selection import RFE

# Load the dataset
df = pd.read_excel('marketing_campaign.xlsx')

# Data preprocessing
imp_mean = SimpleImputer(strategy="mean")
df['Income'] = imp_mean.fit_transform(df[['Income']])
df = pd.get_dummies(df, drop_first=True)

# Define features and target variable
features = [col for col in df.columns if col != 'Response']
target = 'Response'
X = df[features]
y = df[target]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize models
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
gbm_model = GradientBoostingClassifier(n_estimators=100, random_state=42)

# Apply RFE with Random Forest
rfe_rf = RFE(estimator=rf_model, n_features_to_select=10)
rfe_rf.fit(X_train, y_train)
selected_features_rf = X_train.columns[rfe_rf.support_].tolist()

# Apply RFE with GBM
rfe_gbm = RFE(estimator=gbm_model, n_features_to_select=10)
rfe_gbm.fit(X_train, y_train)
selected_features_gbm = X_train.columns[rfe_gbm.support_].tolist()

# Combine selected features
top_features = list(set(selected_features_rf).union(set(selected_features_gbm)))

# Filter data with selected features
X_train_selected = X_train[top_features]
X_test_selected = X_test[top_features]

# Retrain models with selected features
rf_model_selected = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model_selected.fit(X_train_selected, y_train)
gbm_model_selected = GradientBoostingClassifier(n_estimators=100, random_state=42)
gbm_model_selected.fit(X_train_selected, y_train)

def evaluate_model(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    TN, FP, FN, TP = cm.ravel()
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    return TN, FP, FN, TP, precision, recall, f1, accuracy

# Streamlit app
st.title('Marketing Campaign Response Prediction')

st.write("""
This application predicts the likelihood of a customer responding to a marketing campaign.
""")

# Input features for prediction
input_data = {}
for feature in top_features:
    input_data[feature] = st.text_input(f'Enter {feature}', '0')

# Convert input data to DataFrame
input_df = pd.DataFrame([input_data], columns=top_features, dtype=float)

if st.button('Predict'):
    # Predict using Random Forest model
    prediction_rf = rf_model_selected.predict(input_df)
    st.write(f'Random Forest Prediction: {prediction_rf[0]}')

    # Predict using GBM model
    prediction_gbm = gbm_model_selected.predict(input_df)
    st.write(f'GBM Prediction: {prediction_gbm[0]}')

# Display evaluation metrics
if st.button('Show Evaluation Metrics'):
    y_pred_rf_selected = rf_model_selected.predict(X_test_selected)
    y_pred_gbm_selected = gbm_model_selected.predict(X_test_selected)

    TN_rf, FP_rf, FN_rf, TP_rf, precision_rf, recall_rf, f1_rf, accuracy_rf = evaluate_model(y_test, y_pred_rf_selected)
    TN_gbm, FP_gbm, FN_gbm, TP_gbm, precision_gbm, recall_gbm, f1_gbm, accuracy_gbm = evaluate_model(y_test, y_pred_gbm_selected)

    st.write("Random Forest Evaluation Metrics:")
    st.write(f"Confusion Matrix: TN={TN_rf}, FP={FP_rf}, FN={FN_rf}, TP={TP_rf}")
    st.write(f"Precision: {precision_rf:.2f}")
    st.write(f"Recall: {recall_rf:.2f}")
    st.write(f"F1 Score: {f1_rf:.2f}")
    st.write(f"Accuracy: {accuracy_rf:.2f}\n")

    st.write("GBM Evaluation Metrics:")
    st.write(f"Confusion Matrix: TN={TN_gbm}, FP={FP_gbm}, FN={FN_gbm}, TP={TP_gbm}")
    st.write(f"Precision: {precision_gbm:.2f}")
    st.write(f"Recall: {recall_gbm:.2f}")
    st.write(f"F1 Score: {f1_gbm:.2f}")
    st.write(f"Accuracy: {accuracy_gbm:.2f}")
