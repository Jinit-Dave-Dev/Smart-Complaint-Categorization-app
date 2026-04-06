import streamlit as st
import pickle
import os
import pandas as pd

# -------------------- CUSTOM DARK UI --------------------
st.markdown("""
<style>
body {background-color: #0E1117; color: white;}
.stTextArea textarea {background-color: #1E1E1E; color: white;}
</style>
""", unsafe_allow_html=True)

# -------------------- TITLE --------------------
st.title("Smart Complaint Categorization System")

# -------------------- SIDEBAR --------------------
st.sidebar.title("📊 Dashboard Menu")
st.sidebar.info("Predict complaint category using Machine Learning")

st.sidebar.markdown("### 👨‍💻 Developer")
st.sidebar.write("Jinit Dave")

# Model toggle
model_choice = st.sidebar.selectbox(
    "🔀 Select Model",
    ["Gradient Boosting", "Logistic Regression", "Naive Bayes"]
)

# -------------------- DATA --------------------
file_path = "smart_complaints_dataset_250.csv"
if not os.path.exists(file_path):
    file_path = "data/smart_complaints_dataset_250.csv"

df = pd.read_csv(file_path)
df.columns = df.columns.str.strip()

complaint_col = next((c for c in df.columns if 'complaint' in c.lower() or 'text' in c.lower()), None)
category_col = next((c for c in df.columns if 'category' in c.lower() or 'label' in c.lower()), None)

df[complaint_col] = df[complaint_col].astype(str)

# -------------------- LOAD COMPONENTS --------------------
vectorizer = pickle.load(open("tfidf_vectorizer.pkl", "rb"))
le = pickle.load(open("label_encoder.pkl", "rb"))

# -------------------- MODEL SELECTION --------------------
model_files = {
    "Gradient Boosting": "gradient_boosting_model.pkl",
    "Logistic Regression": "logistic_regression_model.pkl",
    "Naive Bayes": "naive_bayes_model.pkl"
}

model_file = model_files[model_choice]

# -------------------- HEADER --------------------
st.markdown("""
<h2 style='text-align:center;color:#4CAF50;'>📊 Smart Prediction Dashboard</h2>
<p style='text-align:center;color:gray;'>Enter complaint and get AI insights</p>
""", unsafe_allow_html=True)

st.caption("💡 Example: There is no electricity in my area")
st.markdown("---")

# -------------------- INPUT --------------------
user_input = st.text_area("Enter your complaint:")

# -------------------- PREDICTION --------------------
if user_input.strip():
    with st.spinner("Analyzing complaint..."):
        model = pickle.load(open(model_file, "rb"))

        X_new = vectorizer.transform([user_input])
        y_pred = model.predict(X_new)
        prediction = le.inverse_transform(y_pred)[0]

        # Static accuracy (you can adjust)
        accuracy_map = {
            "Gradient Boosting": 0.84,
            "Logistic Regression": 0.82,
            "Naive Bayes": 0.78
        }
        accuracy = accuracy_map[model_choice]

        # -------------------- METRICS --------------------
        col1, col2 = st.columns(2)
        with col1:
            st.metric("📌 Category", prediction)
        with col2:
            st.metric("🎯 Accuracy", f"{accuracy*100:.2f}%")

        # -------------------- RELATED CITIES --------------------
        if "city" in df.columns:
            cities = df[df[category_col] == prediction]["city"].dropna().unique()
            cities_display = ", ".join(cities[:3]) if len(cities) > 0 else "Not Available"
        else:
            cities_display = "Not Available"

        # -------------------- TABLE --------------------
        result_df = pd.DataFrame({
            "Model": [model_choice],
            "Accuracy": [accuracy],
            "Predicted Category": [prediction],
            "Related Cities": [cities_display]
        })

        st.markdown("### 📋 Detailed Results")
        st.dataframe(result_df, use_container_width=True)

        # -------------------- DOWNLOAD BUTTON --------------------
        csv = result_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇ Download Result as CSV",
            data=csv,
            file_name="prediction_result.csv",
            mime="text/csv"
        )

# -------------------- CHART --------------------
st.markdown("### 📊 Category Distribution")

category_counts = df[category_col].value_counts()
st.bar_chart(category_counts)
