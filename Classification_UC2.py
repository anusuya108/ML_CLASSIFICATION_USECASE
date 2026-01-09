import os
import pickle
import streamlit as st
@st.cache_resource
def load_uc2():
    base_path = os.path.dirname(__file__)
    model_path = os.path.join(base_path, "models", "uc2_model.pkl")
    with open(model_path, "rb") as f:
        pipeline = pickle.load(f)
    return pipeline
uc2_model = load_uc2()
