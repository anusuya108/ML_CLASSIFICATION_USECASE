import streamlit as st
import pandas as pd
import numpy as np
import pickle
st.set_page_config(page_title="ML Capstone", layout="wide")
with open("models/churn_model.pkl", "rb") as f:
    churn_model = pickle.load(f)
with open("models/abandonment_model.pkl", "rb") as f:
    abandon_model = pickle.load(f)
with open("models/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)
st.sidebar.title(" Use Case Selection")
usecase = st.sidebar.radio("Choose Use Case:", ["UC-1: Customer Churn Prediction", "UC-2: Cart Abandonment Risk"])
if usecase == "UC-1: Customer Churn Prediction":
    st.title("Customer Churn Prediction")
    st.write("Enter customer details below:")
    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.number_input("Age", 18, 80, 30)
        membership_years = st.number_input("Membership Years", 0, 10, 2)
        login_freq = st.number_input("Login Frequency", 0, 100, 10)
    with col2:
        session_duration = st.number_input("Avg Session Duration", 0.0, 100.0, 10.0)
        pages_per_session = st.number_input("Pages per Session", 0.0, 50.0, 5.0)
        cart_abandon_rate = st.number_input("Cart Abandonment Rate", 0.0, 1.0, 0.3)
    with col3:
        total_purchases = st.number_input("Total Purchases", 0, 500, 5)
        avg_order_value = st.number_input("Average Order Value", 0.0, 10000.0, 500.0)
        days_last_purchase = st.number_input("Days Since Last Purchase", 0, 365, 30)
    if st.button("Predict Churn"):
        input_data = np.array([[age, membership_years, login_freq, session_duration,
                                 pages_per_session, cart_abandon_rate,
                                 total_purchases, avg_order_value, days_last_purchase]])
        input_scaled = scaler.transform(input_data)
        pred = churn_model.predict(input_scaled)[0]
        prob = churn_model.predict_proba(input_scaled)[0][1]
        if pred == 1:
            st.error(f"Customer is likely to churn (Risk: {prob:.2%})")
        else:
            st.success(f"Customer is not likely to churn (Risk: {prob:.2%})")
elif usecase == "UC-2: Cart Abandonment Risk":
    st.title(" Cart Abandonment Risk Prediction")
    st.write("Enter customer shopping behavior:")
    col1, col2 = st.columns(2)
    with col1:
        total_purchases = st.number_input("Total Purchases", 0, 100, 2)
        cart_abandon_rate = st.number_input("Cart Abandonment Rate", 0.0, 1.0, 0.6)
        wishlist_items = st.number_input("Wishlist Items", 0, 50, 5)
    with col2:
        discount_usage = st.number_input("Discount Usage Rate", 0.0, 1.0, 0.4)
        days_last_purchase = st.number_input("Days Since Last Purchase", 0, 365, 45)
        avg_order_value = st.number_input("Average Order Value", 0.0, 10000.0, 400.0)

    if st.button("🔮 Predict Abandonment"):
        input_data = np.array([[total_purchases, cart_abandon_rate, wishlist_items,
                                 discount_usage, days_last_purchase, avg_order_value]])

        input_scaled = scaler.transform(input_data)
        pred = abandon_model.predict(input_scaled)[0]
        prob = abandon_model.predict_proba(input_scaled)[0][1]

        if pred == 1:
            st.error(f"⚠ High Risk of Cart Abandonment (Risk: {prob:.2%})")
        else:
            st.success(f"✅ Low Risk of Cart Abandonment (Risk: {prob:.2%})")
