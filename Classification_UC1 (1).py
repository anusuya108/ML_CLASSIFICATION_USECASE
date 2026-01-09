import os
import pickle
import numpy as np
import streamlit as st
st.set_page_config(page_title="UC-1 Customer Churn", layout="centered")
@st.cache_resource
def load_uc1():
    base_path = os.path.dirname(__file__)
    model_path = os.path.join(base_path, "uc1_model.pkl")
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model
uc1_model = load_uc1()
st.title("Use Case 1 – Customer Churn Prediction")
st.write("Predict whether a customer is likely to churn based on behavior and activity.")
st.subheader("Enter Customer Details")
age = st.number_input("Age", 18, 100, 30)
membership_years = st.number_input("Membership Years", 0.0, 20.0, 2.0)
login_frequency = st.number_input("Login Frequency (per month)", 0, 100, 10)
session_duration = st.number_input("Avg Session Duration (minutes)", 0.0, 300.0, 15.0)
total_purchases = st.number_input("Total Purchases", 0, 1000, 5)
avg_order_value = st.number_input("Average Order Value", 0.0, 10000.0, 100.0)
days_since_last_purchase = st.number_input("Days Since Last Purchase", 0, 365, 20)
email_open_rate = st.slider("Email Open Rate (%)", 0, 100, 30)
customer_service_calls = st.number_input("Customer Service Calls", 0, 50, 1)
if st.button("Predict Churn"):
    input_data = np.array([[age, membership_years, login_frequency, session_duration,
                             total_purchases, avg_order_value, days_since_last_purchase,
                             email_open_rate, customer_service_calls]])
    prediction = uc1_model.predict(input_data)[0]
    prob = uc1_model.predict_proba(input_data)[0][1]
    if prediction == 1:
        st.error(f" Customer is likely to churn (Risk Score: {prob:.2f})")
    else:
        st.success(f" Customer is NOT likely to churn (Risk Score: {prob:.2f})")
