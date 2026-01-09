import os
import pickle
import streamlit as st
import numpy as np

st.set_page_config(page_title="UC-1: Customer Churn Prediction", layout="centered")

st.title(" Use Case 1 – Customer Churn Prediction")
st.write("Predict whether a customer is likely to churn based on behavioral and transactional features.")
@st.cache_resource
def load_uc1():
    base_path = os.path.dirname(__file__)
    model_path = os.path.join(base_path, "uc1_model.pkl")

    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model
uc1_model = load_uc1()
st.subheader("Enter Customer Details")

age = st.number_input("Age", min_value=18, max_value=100, value=30)
membership_years = st.number_input("Membership Years", min_value=0.0, max_value=20.0, value=2.0)
login_frequency = st.number_input("Login Frequency", min_value=0, max_value=100, value=10)
session_duration = st.number_input("Avg Session Duration", min_value=0.0, max_value=500.0, value=50.0)
pages_per_session = st.number_input("Pages Per Session", min_value=0.0, max_value=50.0, value=5.0)
cart_abandonment_rate = st.number_input("Cart Abandonment Rate", min_value=0.0, max_value=100.0, value=40.0)
total_purchases = st.number_input("Total Purchases", min_value=0, max_value=500, value=20)
avg_order_value = st.number_input("Average Order Value", min_value=0.0, max_value=1000.0, value=150.0)
days_since_last_purchase = st.number_input("Days Since Last Purchase", min_value=0, max_value=365, value=30)
discount_usage_rate = st.number_input("Discount Usage Rate", min_value=0.0, max_value=100.0, value=20.0)
returns_rate = st.number_input("Returns Rate", min_value=0.0, max_value=100.0, value=5.0)
email_open_rate = st.number_input("Email Open Rate", min_value=0.0, max_value=100.0, value=40.0)
customer_service_calls = st.number_input("Customer Service Calls", min_value=0, max_value=50, value=1)
product_reviews_written = st.number_input("Product Reviews Written", min_value=0, max_value=100, value=2)
social_media_score = st.number_input("Social Media Engagement Score", min_value=0.0, max_value=100.0, value=10.0)
mobile_app_usage = st.number_input("Mobile App Usage", min_value=0, max_value=100, value=20)
payment_method_diversity = st.number_input("Payment Method Diversity", min_value=1, max_value=10, value=2)
lifetime_value = st.number_input("Customer Lifetime Value", min_value=0.0, max_value=100000.0, value=5000.0)
credit_balance = st.number_input("Credit Balance", min_value=0.0, max_value=50000.0, value=2000.0)

gender = st.selectbox("Gender", ["Male", "Female"])
country = st.selectbox("Country", ["India", "USA", "Germany", "Canada", "Australia"])
signup_quarter = st.selectbox("Signup Quarter", ["Q1", "Q2", "Q3", "Q4"])
gender_male = 1 if gender == "Male" else 0
country_india = 1 if country == "India" else 0
signup_q1 = 1 if signup_quarter == "Q1" else 0
signup_q2 = 1 if signup_quarter == "Q2" else 0
input_data = np.array([[
    age,
    membership_years,
    login_frequency,
    session_duration,
    pages_per_session,
    cart_abandonment_rate,
    total_purchases,
    avg_order_value,
    days_since_last_purchase,
    discount_usage_rate,
    returns_rate,
    email_open_rate,
    customer_service_calls,
    product_reviews_written,
    social_media_score,
    mobile_app_usage,
    payment_method_diversity,
    lifetime_value,
    credit_balance,
    gender_male,
    country_india,
    signup_q1,
    signup_q2
]])

if st.button(" Predict Churn"):
    try:
        prediction = uc1_model.predict(input_data)[0]
        probability = uc1_model.predict_proba(input_data)[0][1]

        if prediction == 1:
            st.error(f"Customer is **Likely to Churn** (Probability: {probability:.2f})")
        else:
            st.success(f"Customer is **Not Likely to Churn** (Probability: {probability:.2f})")

    except Exception as e:
        st.error(" Prediction failed due to feature mismatch.")
        st.write("Model expects:", uc1_model.n_features_in_)
        st.write("You provided:", input_data.shape[1])
        st.exception(e)
