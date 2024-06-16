import streamlit as st
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

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

# Initialize GBM model
gbm_model = GradientBoostingClassifier(n_estimators=100, random_state=42)

# Fit GBM model
gbm_model.fit(X_train, y_train)

# Streamlit app
st.title('Marketing Campaign Response Prediction')

st.write("""
This application predicts the likelihood of a customer responding to a marketing campaign.
""")

# Input features for prediction
input_data = {}
for feature in features:
    if feature == 'Date':  # Use a date picker for 'Date' feature
        input_data[feature] = st.date_input(f'Select {feature}')
    else:  # Assume other features are numeric and expect integer input
        input_data[feature] = st.text_input(f'Enter {feature} (integer)', '0')

# Convert input data to DataFrame
input_df = pd.DataFrame([input_data], columns=features, dtype=float)

if st.button('Predict'):
    # Predict using GBM model
    prediction_gbm = gbm_model.predict(input_df)
    st.write(f'GBM Prediction: {prediction_gbm[0]}')
