import streamlit as st
import pandas as pd
import numpy as np
import pickle
st.set_page_config(page_title="UC-2: Cart Abandonment Risk", layout="wide")
st.title(" Use Case 2 – Cart Abandonment Risk Classification")

st.markdown("Predict whether a customer is at **high risk of cart abandonment** based on behavior and engagement data.")
@st.cache_resource
def load_uc2():
    with open("models/uc2_model.pkl", "rb") as f:
        pipeline = pickle.load(f)
    return pipeline

uc2_model = load_uc2()
st.sidebar.header("Enter Customer Behavior")

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
        "Total_Purchases": st.sidebar.number_input("Total Purchases", 0, 500, 5),
        "Cart_Abandonment_Rate": st.sidebar.slider("Cart Abandonment Rate (%)", 0, 100, 50),
    }
    return pd.DataFrame([data])

input_df = user_inputs()
st.subheader("Input Data")
st.dataframe(input_df)
if st.button("Predict Abandonment Risk"):
    try:
        pred = uc2_model.predict(input_df)
        prob = uc2_model.predict_proba(input_df)[:, 1]

        st.subheader(" Prediction Result")
        if pred[0] == 1:
            st.error("High Risk of Cart Abandonment")
        else:
            st.success("Low Risk of Cart Abandonment")

        st.info(f"Abandonment Probability: **{round(prob[0]*100, 2)}%**")

    except Exception as e:
        st.error(" Prediction failed")
        st.exception(e)

st.markdown("---")
st.markdown(" Capstone Project – UC-2")
