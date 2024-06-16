import streamlit as st
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from datetime import datetime

# Load the dataset
df = pd.read_excel('marketing_campaign.xlsx')

# Remove columns not needed for input and prediction
df.drop(['ID', 'Year_Birth', 'Z_CostContact', 'Z_Revenue'], axis=1, inplace=True)

# Dropdown options for Marital status, Education, Kidhome, Teenhome
marital_status_options = ['Married', 'Together', 'Single', 'Divorced', 'Widow', 'Alone', 'YOLO', 'Absurd']
education_options = ['Graduation', 'Master', 'PhD', '2n Cycle', 'Basic']
kidteen_options = [0, 1, 2]

# Streamlit app
st.title('Marketing Campaign Response Prediction')

st.write("""
This application predicts the likelihood of a customer responding to a marketing campaign.
""")

# Function to display input fields
def display_input_field(column_name):
    if column_name in ['AcceptedCmp1', 'AcceptedCmp2', 'AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5', 'Response', 'Complain']:
        if column_name.startswith('AcceptedCmp'):
            st.write(f'{column_name}: 1 if accepted, 0 if not accepted')
        elif column_name == 'Response':
            st.write('Response (Target): 1 if accepted, 0 if not accepted')
        elif column_name == 'Complain':
            st.write('Complained in last 2 years: 1 if complained, 0 if not complained')
        return st.selectbox(f'Enter {column_name}', options=[0, 1])
    elif column_name == 'Dt_Customer':
        st.write('Date of customer’s enrolment with the company')
        return st.date_input(f'Enter {column_name}', datetime.now())
    elif column_name == 'Marital_Status':
        st.write('Marital status')
        return st.selectbox(f'Enter {column_name}', options=marital_status_options)
    elif column_name == 'Education':
        st.write('Customer’s level of education')
        return st.selectbox(f'Enter {column_name}', options=education_options)
    elif column_name in ['Kidhome', 'Teenhome']:
        st.write(f'Number of {column_name.lower()}')
        return st.selectbox(f'Enter {column_name}', options=kidteen_options)
    elif column_name in ['MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds', 'Income']:
        st.write(f'Enter Amount (RM) for {column_name}')
        return st.text_input(f'Enter {column_name}')
    elif column_name == 'NumDealsPurchases':
        st.write(f'Number of Deals Purchases, Enter Number (Integer)')
        return st.number_input(f'Enter {column_name}', step=1, value=0)
    elif column_name == 'NumWebPurchases':
        st.write(f'Number of Web Purchases, Enter Number (Integer)')
        return st.number_input(f'Enter {column_name}', step=1, value=0)
    elif column_name == 'NumCatalogPurchases':
        st.write(f'Number of Catalogue Purchases, Enter Number (Integer)')
        return st.number_input(f'Enter {column_name}', step=1, value=0)
    elif column_name == 'NumStorePurchases':
        st.write(f'Number of Store Purchases, Enter Number (Integer)')
        return st.number_input(f'Enter {column_name}', step=1, value=0)
    elif column_name == 'NumWebVisitsMonth':
        st.write(f'Number of Web Visit this Month, Enter Number (Integer)')
        return st.number_input(f'Enter {column_name}', step=1, value=0)
    elif column_name == 'Recency':
        st.write(f'Number of Days Since Last Purchase')
        return st.number_input(f'Enter {column_name}', step=1, value=0)
    else:
        st.write(f'Enter {column_name}')
        return st.text_input(f'Enter {column_name}')

# Input features for prediction
input_data = {}
for feature in df.columns:
    input_data[feature] = display_input_field(feature)

# Convert input data to DataFrame
input_df = pd.DataFrame([input_data])

# Encode categorical variables (Education and Marital_Status)
input_df_encoded = pd.get_dummies(input_df, columns=['Education', 'Marital_Status'])

# Initialize GBM model
gbm_model = GradientBoostingClassifier(n_estimators=100, random_state=42)

# Train-test split (using entire dataset for demonstration purposes)
features = [col for col in df.columns if col != 'Response']
target = 'Response'
X = df[features]
y = df[target]

# Fit GBM model on entire dataset
try:
    gbm_model.fit(X, y)
except Exception as e:
    st.write(f"Error occurred during model training: {e}")

if st.button('Predict'):
    # Ensure columns in input_df_encoded match the columns in X
    missing_columns = set(X.columns) - set(input_df_encoded.columns)
    for col in missing_columns:
        input_df_encoded[col] = 0  # Add missing columns with default value 0

    # Reorder columns to match X
    input_df_encoded = input_df_encoded[X.columns]

    # Predict using GBM model
    try:
        # Convert date input to days since the earliest date in the dataset
        input_df_encoded['Dt_Customer'] = (input_df_encoded['Dt_Customer'] - df['Dt_Customer'].min()).dt.days

        prediction_gbm = gbm_model.predict(input_df_encoded)
        st.write(f'GBM Prediction: {prediction_gbm[0]}')
    except Exception as e:
        st.write(f"Error occurred during prediction: {e}")
