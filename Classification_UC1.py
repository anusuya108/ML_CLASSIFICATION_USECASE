import streamlit as st
import pandas as pd
import numpy as np
import pickle
st.set_page_config(page_title="UC-1 | Sales Prediction", layout="wide")
@st.cache_resource
def load_uc1_artifacts():
    with open("models/uc1_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("models/uc1_preprocessor.pkl", "rb") as f:
        preprocessor = pickle.load(f)
    return model, preprocessor

model, preprocessor = load_uc1_artifacts()
st.title(" UC-1: Sales Prediction")
st.markdown("Predict **Sales / Revenue** based on customer behavior and transaction features.")
st.sidebar.header("Enter Input Features")
def user_input():
    Units_Sold = st.sidebar.number_input("Units Sold", min_value=0, max_value=10000, value=10)
    Price = st.sidebar.number_input("Unit Price", min_value=0.0, max_value=100000.0, value=150.0)
    Discount_Rate = st.sidebar.slider("Discount Rate", 0.0, 1.0, 0.1)
    Inventory_Units = st.sidebar.number_input("Inventory Units", min_value=0, max_value=100000, value=500)
    Demand = st.sidebar.number_input("Demand", min_value=0, max_value=100000, value=300)
    Store_Type = st.sidebar.selectbox("Store Type", ["Online", "Retail", "Hybrid"])
    Region = st.sidebar.selectbox("Region", ["North", "South", "East", "West"])
    Promotion = st.sidebar.selectbox("Promotion Applied", ["Yes", "No"])
    data = {
        "Units_Sold": Units_Sold,
        "Price": Price,
        "Discount_Rate": Discount_Rate,
        "Inventory_Units": Inventory_Units,
        "Demand": Demand,
        "Store_Type": Store_Type,
        "Region": Region,
        "Promotion": Promotion
    }
    return pd.DataFrame([data])
input_df = user_input()
input_df["Discounted_Price"] = input_df["Price"] * (1 - input_df["Discount_Rate"])
input_df["Stock_Gap"] = input_df["Demand"] - input_df["Inventory_Units"]
if st.button(" Predict Sales"):
    try:
        X_transformed = preprocessor.transform(input_df)
        prediction = model.predict(X_transformed)[0]
        st.subheader("Prediction Result")
        st.success(f" Predicted Sales / Revenue: **{prediction:,.2f}**")
    except Exception as e:
        st.error(" Input mismatch with trained model.")
        st.text(str(e))
st.markdown("---")
st.markdown("**UC-1: Sales Prediction (Regression)** | Built with Scikit-learn & Streamlit")
