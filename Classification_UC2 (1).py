import streamlit as st

st.set_page_config(page_title="UC2 - Cart Abandonment Risk", layout="centered")

st.title("🛒 UC2: Cart Abandonment Risk")

st.warning("⚠ This model cannot be loaded in the current deployment environment.")

st.markdown("""
### Why?

This model was trained using **scikit-learn 1.1.3** with a `ColumnTransformer`.  
Streamlit Cloud runs **Python 3.13**, which does not support this sklearn version.

### What was done in UC2?

- Defined target: **Abandonment_Risk**
- Engineered features:
  - Engagement_Score
  - Inactivity_Score
  - Discount_Dependency
  - Revenue_Per_Purchase
  - Review_Engagement
- Trained multiple models:
  - Logistic Regression
  - Random Forest
  - Gradient Boosting
  - AdaBoost
  - SVM
- Selected best model based on **ROC-AUC**

### Outcome

UC2 performs well in the development environment, but cannot be deployed here due to
**Python–scikit-learn incompatibility**.

This is a **deployment environment limitation**, not a modeling error.
""")
