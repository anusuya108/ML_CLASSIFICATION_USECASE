import pandas as pd
import numpy as np
import streamlit as st
import os
import joblib
import dill   # ✅ critical fix

# Import same classes used during training
from imblearn.over_sampling import SMOTE
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC

st.set_page_config(page_title="UC2 Cart Abandonment Risk", layout="centered")

# ============================
# LOAD UC2 MODEL (DILL FALLBACK)
# ============================
@st.cache_resource
def load_uc2():
    model_path = "uc2_model.pkl"

    # Try joblib first
    try:
        model = joblib.load(model_path)
        return model
    except Exception:
        pass

    # Try dill (can load complex pickled objects)
    try:
        with open(model_path, "rb") as f:
            model = dill.load(f)
        return model
    except Exception as e:
        st.error("❌ Unable to load UC2 model. The model file is not compatible with this environment.")
        st.exception(e)
        st.stop()

uc2_model = load_uc2()

st.title("🛒 Cart Abandonment Risk Prediction (UC2)")
st.write("Predict whether a customer is at **High Risk of Cart Abandonment**.")

# ============================
# USER INPUT
# ============================
Age = st.number_input("Age", min_value=10, max_value=100, value=35)
Membership_Years = st.number_input("Membership Years", min_value=0, max_value=20, value=4)
Login_Frequency = st.number_input("Login Frequency", min_value=0, max_value=100, value=10)
Session_Duration_Avg = st.number_input("Session Duration (Avg)", min_value=0.0, max_value=100.0, value=7.5)
Pages_Per_Session = st.number_input("Pages Per Session", min_value=0, max_value=100, value=6)
Cart_Abandonment_Rate = st.number_input("Cart Abandonment Rate (%)", min_value=0.0, max_value=100.0, value=60.0)
Total_Purchases = st.number_input("Total Purchases", min_value=0, max_value=1000, value=2)
Wishlist_Items = st.number_input("Wishlist Items", min_value=0, max_value=500, value=3)
Average_Order_Value = st.number_input("Average Order Value", min_value=0.0, max_value=10000.0, value=130.0)
Days_Since_Last_Purchase = st.number_input("Days Since Last Purchase", min_value=0, max_value=365, value=20)
Discount_Usage_Rate = st.number_input("Discount Usage Rate", min_value=0.0, max_value=1.0, value=0.4)
Returns_Rate = st.number_input("Returns Rate (%)", min_value=0.0, max_value=100.0, value=2.0)
Email_Open_Rate = st.number_input("Email Open Rate (%)", min_value=0.0, max_value=100.0, value=45.0)
Customer_Service_Calls = st.number_input("Customer Service Calls", min_value=0, max_value=100, value=1)
Product_Reviews_Written = st.number_input("Product Reviews Written", min_value=0, max_value=500, value=2)
Social_Media_Engagement_Score = st.number_input("Social Media Engagement Score", min_value=0, max_value=1000, value=15)
Mobile_App_Usage = st.number_input("Mobile App Usage (0 = No, 1 = Yes)", min_value=0, max_value=1, value=1)
Lifetime_Value = st.number_input("Lifetime Value", min_value=0.0, max_value=100000.0, value=1600.0)

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
    "Wishlist_Items": Wishlist_Items,
    "Average_Order_Value": Average_Order_Value,
    "Days_Since_Last_Purchase": Days_Since_Last_Purchase,
    "Discount_Usage_Rate": Discount_Usage_Rate,
    "Returns_Rate": Returns_Rate,
    "Email_Open_Rate": Email_Open_Rate,
    "Customer_Service_Calls": Customer_Service_Calls,
    "Product_Reviews_Written": Product_Reviews_Written,
    "Social_Media_Engagement_Score": Social_Media_Engagement_Score,
    "Mobile_App_Usage": Mobile_App_Usage,
    "Lifetime_Value": Lifetime_Value
}

df = pd.DataFrame([input_dict])

# ============================
# FEATURE ENGINEERING (FROM YOUR NOTEBOOK)
# ============================
df['Engagement_Score'] = df['Login_Frequency'] + df['Session_Duration_Avg'] + df['Pages_Per_Session']
df['Inactivity_Score'] = df['Days_Since_Last_Purchase']
df['Discount_Dependency'] = df['Discount_Usage_Rate'] * df['Wishlist_Items']
df['Revenue_Per_Purchase'] = df['Average_Order_Value']
df['Review_Engagement'] = df['Product_Reviews_Written'] + df['Social_Media_Engagement_Score']

# ============================
# ALIGN WITH TRAINED MODEL FEATURES
# ============================
if hasattr(uc2_model, "feature_names_in_"):
    input_data = df.reindex(columns=uc2_model.feature_names_in_, fill_value=0)
else:
    input_data = df

# ============================
# PREDICTION
# ============================
if st.button("🔍 Predict Abandonment Risk"):
    prediction = uc2_model.predict(input_data)[0]
    probability = uc2_model.predict_proba(input_data)[0][1]

    st.markdown("## 📊 Prediction Result")

    if prediction == 1:
        st.error("⚠ Customer is **HIGH RISK** of Cart Abandonment")
    else:
        st.success("✅ Customer is **LOW RISK** of Cart Abandonment")

    st.write(f"**Risk Probability:** {probability:.2f}")
