import streamlit as st
import pickle
import os
import pandas as pd

st.title("Smart Complaint Categorization System")

# Load dataset
file_path = "smart_complaints_dataset_250.csv"

if not os.path.exists(file_path):
    file_path = "data/smart_complaints_dataset_250.csv"

df = pd.read_csv(file_path)
df.columns = df.columns.str.strip()

# Detect columns
complaint_col = next((c for c in df.columns if 'complaint' in c.lower() or 'text' in c.lower()), None)
category_col = next((c for c in df.columns if 'category' in c.lower() or 'label' in c.lower()), None)

if complaint_col is None or category_col is None:
    st.error(f"Could not detect complaint or category columns. Found columns: {df.columns}")
    st.stop()

# Load ML components
vectorizer = pickle.load(open("tfidf_vectorizer.pkl", "rb"))
le = pickle.load(open("label_encoder.pkl", "rb"))

# User input
user_input = st.text_area("Enter your complaint:")

# Prediction
if user_input.strip():
    try:
        # Load best model
        model = pickle.load(open("gradient_boosting_model.pkl", "rb"))

        # Transform input
        X_new = vectorizer.transform([user_input])

        # Predict
        y_pred = model.predict(X_new)
        prediction = le.inverse_transform(y_pred)[0]

        # Output
        st.success("Predicted Category: " + prediction)
        st.info("Model Used: Gradient Boosting (High Accuracy)")

    except Exception as e:
        st.error(f"Error occurred: {e}")
