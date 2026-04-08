import streamlit as st
import pickle
import os

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(page_title="Smart Complaint System", layout="wide")

st.title("🏛️ Smart Municipal Complaint Categorization System")
st.markdown("AI-powered system for classifying public complaints")

# ===============================
# LOAD MODEL & VECTORIZER (CORRECT FILE NAMES)
# ===============================
try:
    model = pickle.load(open("logistic_regression_model.pkl", "rb"))
    vectorizer = pickle.load(open("tfidf_vectorizer.pkl", "rb"))
    le = pickle.load(open("label_encoder.pkl", "rb"))
except FileNotFoundError:
    st.error("❌ Model files not found! Check file names in your repository.")
    st.write("Available files:", os.listdir())
    st.stop()

# ===============================
# CATEGORY MAPPING FUNCTION
# ===============================
def map_category(text):
    text = str(text).lower()

    if any(word in text for word in ['water', 'supply', 'leak', 'pipeline']):
        return "Water Supply"
    elif any(word in text for word in ['road', 'pothole', 'street', 'bridge']):
        return "Road & Infrastructure"
    elif any(word in text for word in ['garbage', 'waste', 'clean', 'dustbin']):
        return "Sanitation & Waste"
    elif any(word in text for word in ['electric', 'power', 'light', 'streetlight']):
        return "Electricity"
    elif any(word in text for word in ['drain', 'sewage', 'drainage']):
        return "Drainage"
    elif any(word in text for word in ['traffic', 'parking', 'vehicle']):
        return "Traffic & Transport"
    elif any(word in text for word in ['health', 'hospital', 'medical']):
        return "Public Health"
    else:
        return "Other"

# ===============================
# MODE SELECTION
# ===============================
mode = st.radio(
    "Select Category Mode:",
    ["Original Model", "Enhanced Municipal Categories"]
)

# ===============================
# INPUT SECTION
# ===============================
st.subheader("📝 Enter Complaint")
user_input = st.text_area("Describe your issue here...", height=150)

# ===============================
# PREDICTION
# ===============================
if st.button("Predict Category"):

    if user_input.strip() == "":
        st.warning("⚠️ Please enter a complaint.")
    else:
        vectorized_input = vectorizer.transform([user_input])
        prediction = model.predict(vectorized_input)

        original_category = le.inverse_transform(prediction)[0]
        enhanced_category = map_category(user_input)

        if mode == "Original Model":
            st.success(f"📌 Predicted Category: {original_category}")
        else:
            st.success(f"🏛️ Enhanced Category: {enhanced_category}")

# ===============================
# OPTIONAL INFO SECTION
# ===============================
st.markdown("---")
st.subheader("📊 About System")

st.info("""
This system uses Machine Learning to classify public complaints.

- 📌 Original Model: Based on trained dataset categories
- 🏛️ Enhanced Categories: Maps complaints into real municipal service areas
""")
