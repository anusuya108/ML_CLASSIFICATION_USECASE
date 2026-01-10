import pandas as pd
import numpy as np
import pickle
import streamlit as st

# ============================
# LOAD UC2 MODEL (SAFE)
# ============================
@st.cache_resource
def load_uc2():
    model_path = "models/uc2_pipeline.pkl"   # ⚠ update if path is different
    with open(model_path, "rb") as f:
        pipeline = pickle.load(f)
    return pipeline

uc2_model = load_uc2()

st.success("UC2 Model Loaded Successfully")
st.write("Model expects:", uc2_model.n_features_in_)

# ============================
# USER INPUT (EXAMPLE)
# ============================
# These should match your dataset columns
input_dict = {
    "Age": 35,
    "Membership_Years": 4,
    "Login_Frequency": 10,
    "Session_Duration_Avg": 7.5,
    "Pages_Per_Session": 6,
    "Cart_Abandonment_Rate": 60,
    "Total_Purchases": 5,
    "Average_Order_Value": 130,
    "Days_Since_Last_Purchase": 20,
    "Discount_Usage_Rate": 0.4,
    "Returns_Rate": 2,
    "Email_Open_Rate": 45,
    "Customer_Service_Calls": 1,
    "Mobile_App_Usage": 1,
    "Lifetime_Value": 1600
}

df = pd.DataFrame([input_dict])

# ============================
# FEATURE ENGINEERING (SAME AS NOTEBOOK UC2)
# ============================

# Engagement score
df['Engagement_Score'] = (
    df['Login_Frequency'] +
    df['Session_Duration_Avg'] +
    df['Pages_Per_Session']
)

# Purchase behavior
df['Purchase_Frequency'] = df['Total_Purchases'] / (df['Membership_Years'] + 1)

# Discount dependency
df['Discount_Dependency'] = df['Discount_Usage_Rate'] * df['Total_Purchases']

# Risk flag
df['High_Abandonment_Flag'] = (df['Cart_Abandonment_Rate'] > 0.7).astype(int)

# ============================
# FINAL FEATURE SET USED IN UC2 TRAINING
# (MATCH THIS TO YOUR NOTEBOOK)
# ============================
uc2_features = [
    'Lifetime_Value',
    'Age',
    'Days_Since_Last_Purchase',
    'Cart_Abandonment_Rate',
    'Discount_Usage_Rate',
    'Total_Purchases',
    'Average_Order_Value',
    'Returns_Rate',
    'Email_Open_Rate',
    'Customer_Service_Calls',
    'Mobile_App_Usage',
    'Engagement_Score',
    'Purchase_Frequency',
    'Discount_Dependency',
    'High_Abandonment_Flag'
]

# Align input exactly like training
input_data = df.reindex(columns=uc2_features, fill_value=0)

# ============================
# DEBUG CHECK
# ============================
st.write("Input shape:", input_data.shape)
st.write("Input columns:", input_data.columns.tolist())

# ============================
# PREDICTION
# ============================
prediction = uc2_model.predict(input_data)[0]
probability = uc2_model.predict_proba(input_data)[0][1]

st.markdown("## UC2 Prediction Result")

if prediction == 1:
    st.error("Customer classified as HIGH RISK")
else:
    st.success(" Customer classified as LOW RISK")

st.write(f"Risk Probability: {probability:.2f}")
