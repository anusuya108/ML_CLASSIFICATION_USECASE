import pandas as pd
import numpy as np
import pickle

# ============================
# LOAD MODEL
# ============================
model_path = "models/uc1_model.pkl"   # Update path if needed
with open(model_path, "rb") as f:
    uc1_model = pickle.load(f)

print("Model loaded successfully")
print("Model expects features:", uc1_model.n_features_in_)

# ============================
# USER INPUT (EXAMPLE)
# ============================
# Must match dataset column names
input_dict = {
    "Age": 32,
    "Membership_Years": 3,
    "Login_Frequency": 12,
    "Session_Duration_Avg": 8.5,
    "Pages_Per_Session": 5,
    "Cart_Abandonment_Rate": 65,
    "Total_Purchases": 4,
    "Average_Order_Value": 120,
    "Days_Since_Last_Purchase": 18,
    "Discount_Usage_Rate": 0.35,
    "Returns_Rate": 3,
    "Email_Open_Rate": 40,
    "Customer_Service_Calls": 2,
    "Mobile_App_Usage": 1,
