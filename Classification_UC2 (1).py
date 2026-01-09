import os
import pickle
import numpy as np
import streamlit as st

st.set_page_config(page_title="UC-2 Cart Abandonment", layout="centered")

# -------------------------
# Load Pipeline
# -------------------------
@st.cache_resource
def load_uc2():
    base_path = os.path.dirname(__file__)
    model_path = os.path.join(base_path, "uc2_model.pkl")
    with open(model_path, "rb") as f:
        pipeline = pickle.load(f)
    return pipeline
uc2_model = load_uc2()
st.title(" Use Case 2 – Cart Abandonment Prediction")
st.write("Predict whether a customer will abandon their cart.")
st.subheader("Enter Session Details")
pages_per_session = st.number_input("Pages per Session", 1, 100, 5)
cart_abandonment_rate = st.slider("Cart Abandonment Rate", 0.0, 1.0, 0.3)
discount_usage_rate = st.slider("Discount Usage Rate", 0.0, 1.0, 0.4)
returns_rate = st.slider("Returns Rate", 0.0, 1.0, 0.1)
email_open_rate = st.slider("Email Open Rate (%)", 0, 100, 40)
mobile_app_usage = st.number_input("Mobile App Usage (hrs/week)", 0.0, 100.0, 2.5)
if st.button("Predict Abandonment"):
    input_data = np.array([[pages_per_session, cart_abandonment_rate, discount_usage_rate,
                             returns_rate, email_open_rate, mobile_app_usage]])

    prediction = uc2_model.predict(input_data)[0]
    prob = uc2_model.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.error(f"⚠️ High chance of cart abandonment (Risk: {prob:.2f})")
    else:
        st.success(f"✅ Low chance of cart abandonment (Risk: {prob:.2f})")
