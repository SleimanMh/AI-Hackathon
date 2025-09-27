import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="IBM HR Attrition Prediction", page_icon="💼", layout="centered")
st.title("💼 IBM HR Analytics Attrition Prediction")
st.caption("Enter employee details to predict attrition risk.")

# ----------------------------
# 1) Load model and scaler
# ----------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load("model.joblib")
    scaler = joblib.load("scaler.joblib")
    return model, scaler

clf, min_max_scaler = load_artifacts()

# ----------------------------
# 2) Define input UI
# ----------------------------
categorical_options = {
    'BusinessTravel': ['Travel_Rarely', 'Travel_Frequently', 'Non-Travel'],
    'Department': ['Sales', 'Research & Development', 'Human Resources'],
    'EducationField': ['Life Sciences', 'Other', 'Medical', 'Marketing', 'Technical Degree', 'Human Resources'],
    'Gender': ['Female', 'Male'],
    'JobRole': ['Sales Executive', 'Research Scientist', 'Laboratory Technician',
                'Manufacturing Director', 'Healthcare Representative', 'Manager',
                'Sales Representative', 'Research Director', 'Human Resources'],
    'MaritalStatus': ['Single', 'Married', 'Divorced'],
    'OverTime': ['Yes', 'No']
}

# Label encoding as in your notebook
categorical_mappings = {
    'BusinessTravel': {
        'Travel_Rarely': 0,
        'Non-Travel': 1,
        'Travel_Frequently': 2
    },
    'Department': {
        'Human Resources': 0,
        'Research & Development': 1,
        'Sales': 2
    },
    'EducationField': {
        'Human Resources': 0,
        'Marketing': 1,
        'Medical': 2,
        'Life Sciences': 3,
        'Other': 4,
        'Technical Degree': 5
    },
    'Gender': {
        'Female': 0,
        'Male': 1
    },
    'JobRole': {
        'Human Resources': 0,
        'Manufacturing Director': 1,
        'Laboratory Technician': 2,
        'Healthcare Representative': 3,
        'Manager': 4,
        'Research Director': 5,
        'Research Scientist': 6,
        'Sales Executive': 7,
        'Sales Representative': 8
    },
    'MaritalStatus': {
        'Divorced': 0,
        'Married': 1,
        'Single': 2
    },
    'OverTime': {
        'No': 0,
        'Yes': 1
    }
}

# The features used for prediction (order must match training)
all_features = [
    'BusinessTravel',
    'Department',
    'EducationField',
    'Gender',
    'JobRole',
    'MaritalStatus',
    'OverTime',
    'Age',
    'DailyRate',
    'DistanceFromHome',
    'Education',
    'EnvironmentSatisfaction',
    'HourlyRate',
    'JobInvolvement',
    'JobLevel',
    'JobSatisfaction',
    'MonthlyIncome',
    'MonthlyRate',
    'NumCompaniesWorked',
    'PercentSalaryHike',
    'PerformanceRating',
    'RelationshipSatisfaction',
    'TotalWorkingYears',
    'TrainingTimesLastYear',
    'WorkLifeBalance',
    'YearsAtCompany',
    'YearsInCurrentRole',
    'YearsSinceLastPromotion',
    'YearsWithCurrManager'
]

# Remove 'Attrition' if present
if 'Attrition' in all_features:
    all_features.remove('Attrition')

# Build the input form
user_input = {}
for feature in all_features:
    if feature in categorical_options:
        user_input[feature] = st.selectbox(f"{feature}", categorical_options[feature])
    else:
        user_input[feature] = st.number_input(f"{feature}", value=0)

# Map categorical values to integer codes
for feature, mapping in categorical_mappings.items():
    if feature in user_input:
        user_input[feature] = mapping[user_input[feature]]

sample = pd.DataFrame([user_input])
sample = sample[all_features]  # Ensure correct order as specified

# ----------------------------
# 3) Predict
# ----------------------------
if st.button('Predict'):
    try:
        # Ensure columns are in the exact order as during fit
        sample = sample[all_features]
        sample_scaled = min_max_scaler.transform(sample)
        prediction = clf.predict(sample_scaled)[0]
        probability = clf.predict_proba(sample_scaled)[0][1]
        st.subheader(f"Prediction: {'Will Leave' if prediction == 1 else 'Will Stay'}")
        st.markdown(f"Probability of Leaving: **{probability*100:.2f}%**")
        st.progress(float(min(max(probability, 0.0), 1.0)))
    except Exception as e:
        print(f"Prediction failed: {e}")

with st.expander("See model inputs"):
    st.dataframe(sample)