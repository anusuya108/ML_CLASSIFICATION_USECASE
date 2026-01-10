import pandas as pd
import numpy as np
import pickle
import streamlit as st
import os

st.set_page_config(page_title="Customer Churn Prediction", layout="centered")

# ============================
# LOAD MODEL
# ============================
@st.cache_resource
def load_model():
    model_path = "uc1_model.pkl"   # model in root folder
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model

uc1_model = load_model()

st.title(" Customer Churn Prediction (UC1)")
st.write("Enter customer details below to predict churn risk.")

# ============================
# USER INPUT UI
# ============================
Age = st.number_input("Age", min_value=10, max_value=100, value=32)
Membership_Years = st.number_input("Membership Years", min_value=0, max_value=20, value=3)
Login_Frequency = st.number_input("Login Frequency", min_value=0, max_value=100, value=12)
Session_Duration_Avg = st.number_input("Session Duration (Avg)", min_value=0.0, max_value=100.0, value=8.5)
Pages_Per_Session = st.number_input("Pages Per Session", min_value=0, max_value=100, value=5)
Cart_Abandonment_Rate = st.number_input("Cart Abandonment Rate (%)", min_value=0.0, max_value=100.0, value=65.0)
Total_Purchases = st.number_input("Total Purchases", min_value=0, max_value=1000, value=4)
Average_Order_Value = st.number_input("Average Order Value", min_value=0.0, max_value=10000.0, value=120.0)
Days_Since_Last_Purchase = st.number_input("Days Since Last Purchase", min_value=0, max_value=365, value=18)
Discount_Usage_Rate = st.number_input("Discount Usage Rate", min_value=0.0, max_value=1.0, value=0.35)
Returns_Rate = st.number_input("Returns Rate (%)", min_value=0.0, max_value=100.0, value=3.0)
Email_Open_Rate = st.number_input("Email Open Rate (%)", min_value=0.0, max_value=100.0, value=40.0)
Customer_Service_Calls = st.number_input("Customer Service Calls", min_value=0, max_value=100, value=2)
Mobile_App_Usage = st.number_input("Mobile App Usage (0 = No, 1 = Yes)", min_value=0, max_value=1, value=1)
Lifetime_Value = st.number_input("Lifetime Value", min_value=0.0, max_value=100000.0, value=1450.0)

# ============================
# CREATE INPUT DATAFRAME
# ============================
input_dict = {
    "Age": Age,
    "Membership_Years": Membership_Years,
    "Login_Frequency": Login_Frequency,
    "Session_Duration_Avg": Session_Duration_Avg,
    "Pages_Per_Session": Pages_Per_Session,
    "Cart_Abandonment_Rate": Cart_Abandonment_Rate,
    "Total_Purchases": Total_Purchases,
    "Average_Order_Value": Average_Order_Value,
    "Days_Since_Last_Purchase": Days_Since_Last_Purchase,
    "Discount_Usage_Rate": Discount_Usage_Rate,
    "Returns_Rate": Returns_Rate,
    "Email_Open_Rate": Email_Open_Rate,
    "Customer_Service_Calls": Customer_Service_Calls,
    "Mobile_App_Usage": Mobile_App_Usage,
    "Lifetime_Value": Lifetime_Value
}

df = pd.DataFrame([input_dict])

# ============================
# FEATURE ENGINEERING (SAME AS TRAINING)
# ============================
df['Engagement_Score'] = (
    df['Login_Frequency'] +
    df['Session_Duration_Avg'] +
    df['Pages_Per_Session']
)

df['Inactivity_Score'] = df['Days_Since_Last_Purchase']
df['Discount_Dependency'] = df['Discount_Usage_Rate'] * df['Total_Purchases']
df['Revenue_Per_Purchase'] = df['Average_Order_Value']
df['High_Abandonment_Flag'] = (df['Cart_Abandonment_Rate'] > 0.7).astype(int)

# Final feature set (same as training)
input_data = df.drop(columns=['Churned'], errors='ignore')

# ============================
# PREDICTION
# ============================
if st.button(" Predict Churn"):
    prediction = uc1_model.predict(input_data)[0]
    probability = uc1_model.predict_proba(input_data)[0][1]

    st.markdown("Prediction Result")

    if prediction == 1:
        st.error(" Customer is likely to **CHURN**")
    else:
        st.success(" Customer is likely to **STAY**")

    st.write(f"**Churn Probability:** {probability:.2f}")
