import streamlit as st
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from datetime import datetime

# Load the dataset
df = pd.read_excel('marketing_campaign.xlsx')

# Data preprocessing
imp_mean = SimpleImputer(strategy="mean")
df['Income'] = imp_mean.fit_transform(df[['Income']])

# Dropdown options for categorical columns
education_options = ['Basic', '2n Cycle', 'Graduation', 'Master', 'PhD']
marital_status_options = ['Married', 'Together', 'Single', 'Divorced', 'Widow', 'Alone', 'YOLO', 'Absurd']
kidhome_options = [0, 1, 2]
teenhome_options = [0, 1, 2]
response_options = [0, 1]
complain_options = [0, 1]

# Streamlit app
st.title('Marketing Campaign Response Prediction')

st.write("""
This application predicts the likelihood of a customer responding to a marketing campaign.
""")

# Function to display comments and input fields
def display_input_field(column_name):
    if column_name == 'AcceptedCmp1':
        st.write('Accepted Campaign 1: 1 if accepted, 0 if not accepted')
        return st.selectbox('Enter AcceptedCmp1', options=response_options)
    elif column_name == 'AcceptedCmp2':
        st.write('Accepted Campaign 2: 1 if accepted, 0 if not accepted')
        return st.selectbox('Enter AcceptedCmp2', options=response_options)
    elif column_name == 'AcceptedCmp3':
        st.write('Accepted Campaign 3: 1 if accepted, 0 if not accepted')
        return st.selectbox('Enter AcceptedCmp3', options=response_options)
    elif column_name == 'AcceptedCmp4':
        st.write('Accepted Campaign 4: 1 if accepted, 0 if not accepted')
        return st.selectbox('Enter AcceptedCmp4', options=response_options)
    elif column_name == 'AcceptedCmp5':
        st.write('Accepted Campaign 5: 1 if accepted, 0 if not accepted')
        return st.selectbox('Enter AcceptedCmp5', options=response_options)
    elif column_name == 'Response':
        st.write('Response (Target): 1 if accepted, 0 if not accepted')
        return st.selectbox('Enter Response', options=response_options)
    elif column_name == 'Complain':
        st.write('Complained in last 2 years: 1 if complained, 0 if not complained')
        return st.selectbox('Enter Complain', options=complain_options)
    elif column_name == 'DtCustomer':
        st.write('Date of customer’s enrolment with the company')
        return st.date_input('Enter DtCustomer')
    elif column_name == 'Education':
        st.write('Education level')
        return st.selectbox('Enter Education', options=education_options)
    elif column_name == 'Marital':
        st.write('Marital status')
        return st.selectbox('Enter Marital', options=marital_status_options)
    elif column_name == 'Kidhome':
        st.write('Number of small children in household')
        return st.selectbox('Enter Kidhome', options=kidhome_options)
    elif column_name == 'Teenhome':
        st.write('Number of teenagers in household')
        return st.selectbox('Enter Teenhome', options=teenhome_options)
    elif column_name in ['MntFruits', 'MntSweetProducts', 'MntWines', 'MntGoldProds']:
        st.write(f'Amount spent on {column_name} (in RM)')
        return st.number_input(f'Enter {column_name}', min_value=0, format="%.2f")
    else:
        st.write(f'{column_name} (Integer data)')
        return st.number_input(f'Enter {column_name}', min_value=0)

# Input features for prediction
input_data = {}
for feature in df.columns:
    if feature != 'Response':
        input_data[feature] = display_input_field(feature)

# Convert input data to DataFrame
input_df = pd.DataFrame([input_data])

# Convert categorical variables into dummy/indicator variables
input_df = pd.get_dummies(input_df, drop_first=True)

# Initialize GBM model
gbm_model = GradientBoostingClassifier(n_estimators=100, random_state=42)

# Train-test split (using entire dataset for demonstration purposes)
X = df[features]
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Fit GBM model on entire dataset
gbm_model.fit(X_train, y_train)

if st.button('Predict'):
    # Predict using GBM model
    prediction_gbm = gbm_model.predict(input_df)
    st.write(f'GBM Prediction: {prediction_gbm[0]}')
