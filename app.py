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

# Convert complaint column to string
df[complaint_col] = df[complaint_col].astype(str)

# Load ML components
vectorizer = pickle.load(open("tfidf_vectorizer.pkl", "rb"))
le = pickle.load(open("label_encoder.pkl", "rb"))

# User input
user_input = st.text_area("Enter your complaint:")

if user_input.strip():
    try:
        # Load best model
        model = pickle.load(open("gradient_boosting_model.pkl", "rb"))

        # Transform input
        X_new = vectorizer.transform([user_input])

        # Predict category
        y_pred = model.predict(X_new)
        prediction = le.inverse_transform(y_pred)[0]

        # Calculate accuracy
        X_all = vectorizer.transform(df[complaint_col])
        y_all = le.transform(df[category_col])
        accuracy = model.score(X_all, y_all)

        # Get related rows from dataset
        matched_df = df[df[category_col] == prediction]

        # Get cities (if exists)
        if "city" in df.columns:
            cities = matched_df["city"].dropna().unique().tolist()
            cities_display = ", ".join(cities[:3]) if cities else "Not Available"
        else:
            cities_display = "City column not found"

        # Create result table
        result_df = pd.DataFrame({
            "Model Used": ["Gradient Boosting"],
            "Accuracy": [round(accuracy, 3)],
            "Predicted Category": [prediction],
            "Related Cities": [cities_display]
        })

        # Add sample context rows (first 2 matching rows)
        if not matched_df.empty:
            sample_rows = matched_df.head(2).reset_index(drop=True)
            result_df = pd.concat([result_df, sample_rows], axis=1)

        # Display result
        st.markdown("### 📊 Prediction Result (Detailed View)")
        st.table(result_df)

    except Exception as e:
        st.error(f"Error occurred: {e}")
