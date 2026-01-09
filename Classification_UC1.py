import os
import pickle
import streamlit as st
@st.cache_resource
def load_uc1():
    base_path = os.path.dirname(__file__)
    model_path = os.path.join(base_path, "models", "uc1_model.pkl")
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model
uc1_model = load_uc1()