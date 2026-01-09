import streamlit as st
import pandas as pd
import numpy as np
import pickle

# ---------------- Page Config ----------------
st.set_page_config(page_title="UC-1: Customer Churn Prediction", layout="wide")
st.title("📊 Use Case 1 – Customer Churn Prediction")

st.markdown("Predict whether a customer is likely to churn based on behavioral and transactional features.")

# ---------------- Load Model & Scaler ----------------
@st.cache_resource
def load_uc1():
    with open("models/uc1_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("models/uc1_scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    return model, scaler

uc1_model, uc1_scaler = load_uc1()

# ---------------- Sidebar Inputs ----------------
st.sidebar.header("🔧 Enter Customer Details")

def user_inputs():
    data = {
        "Age": st.sidebar.number_input("Age", 18, 100, 30),
        "Membership_Years": st.sidebar.number_input("Membership Years", 0, 20, 2),
        "Login_Frequency": st.sidebar.slider("Login Frequency", 0, 50, 10),
        "Session_Duration_Avg": st.sidebar.slider("Session Duration Avg", 1, 120, 20),
        "Pages_Per_Session": st.sidebar.slider("Pages Per Session", 1, 50, 8),
        "Wishlist_Items": st.sidebar.slider("Wishlist Items", 0, 50, 5),
        "Average_Order_Value": st.sidebar.number_input("Average Order Value", 100.0, 10000.0, 1500.0),
        "Days_Since_Last_Purchase": st.sidebar.slider("Days Since Last Purchase", 0, 365, 30),
        "Discount_Usage_Rate": st.sidebar.slider("Discount Usage Rate (%)", 0, 100, 20),
        "Returns_Rate": st.sidebar.slider("Returns Rate (%)", 0, 100, 5),
        "Email_Open_Rate": st.sidebar.slider("Email Open Rate (%)", 0, 100, 40),
        "Customer_Service_Calls": st.sidebar.slider("Customer Service Calls", 0, 50, 1),
        "Mobile_App_Usage": st.sidebar.slider("Mobile App Usage", 0, 100, 40),
        "Lifetime_Value": st.sidebar.number_input("Lifetime Value", 1000.0, 1000000.0, 50000.0),
        "Credit_Balance": st.sidebar.number_input("Credit Balance", 0.0, 100000.0, 2000.0),
    }
    return pd.DataFrame([data])

input_df = user_inputs()

# ---------------- Show Input ----------------
st.subheader("📝 Input Data")
st.dataframe(input_df)

# ---------------- Prediction ----------------
if st.button("🔮 Predict Churn"):
    try:
        X_scaled = uc1_scaler.transform(input_df)
        pred = uc1_model.predict(X_scaled)
        prob = uc1_model.predict_proba(X_scaled)[:, 1] if hasattr(uc1_model, "predict_proba") else None

        st.subheader("📈 Prediction Result")
        if pred[0] == 1:
            st.error("⚠️ Customer is likely to CHURN")
        else:
            st.success("✅ Customer is likely to STAY")

        if prob is not None:
            st.info(f"Churn Probability: **{round(prob[0]*100, 2)}%**")

    except Exception as e:
        st.error("❌ Prediction failed")
        st.exception(e)

st.markdown("---")
st.markdown("👩‍💻 Capstone Project – UC-1")
