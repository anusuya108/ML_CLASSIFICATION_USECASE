import pandas as pd
import numpy as np
import pickle
import os

# ============================
# LOAD MODEL
# ============================
print("Current working directory:", os.getcwd())
print("Files in directory:", os.listdir())

model_path = "uc1_model.pkl"   # <-- FIXED PATH

with open(model_path, "rb") as f:
    uc1_model = pickle.load(f)

print("Model loaded successfully")
print("Model expects features:", uc1_model.n_features_in_)
