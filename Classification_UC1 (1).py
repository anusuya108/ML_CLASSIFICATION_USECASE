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
    "Lifetime_Value": 1450
}   #  THIS WAS MISSING IN YOUR FILE

# Convert to DataFrame
df = pd.DataFrame([input_dict])

# ============================
# FEATURE ENGINEERING (EXACT NOTEBOOK LOGIC)
# ============================

# Engagement behavior
df['Engagement_Score'] = (
    df['Login_Frequency'] +
    df['Session_Duration_Avg'] +
    df['Pages_Per_Session']
)

# Inactivity risk
df['Inactivity_Score'] = df['Days_Since_Last_Purchase']

# Discount dependency
df['Discount_Dependency'] = df['Discount_Usage_Rate'] * df['Total_Purchases']

# Revenue efficiency
df['Revenue_Per_Purchase'] = df['Average_Order_Value']

# Abandonment behavior
df['High_Abandonment_Flag'] = (df['Cart_Abandonment_Rate'] > 0.7).astype(int)

# ============================
# FINAL FEATURE SET (MATCH TRAINING)
# ============================
# Model was trained using: X1 = df.drop(columns=['Churned'])
input_data = df.drop(columns=['Churned'], errors='ignore')

# ============================
# DEBUG CHECK
# ============================
print("Input shape:", input_data.shape)
print("Input columns:", input_data.columns.tolist())

# ============================
# PREDICTION
# ============================
prediction = uc1_model.predict(input_data)[0]
probability = uc1_model.predict_proba(input_data)[0][1]

print("\n============================")
print("CHURN PREDICTION RESULT")
print("============================")

if prediction == 1:
    print(" Customer is likely to CHURN")
else:
    print("Customer is likely to STAY")

print(f"Churn Probability: {probability:.2f}")
