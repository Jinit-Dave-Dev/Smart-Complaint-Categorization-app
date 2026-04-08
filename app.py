import streamlit as st
import pickle
import os
import pandas as pd

# -------------------- PAGE CONFIG --------------------
st.set_page_config(page_title="Municipal Complaint System", layout="wide")

# -------------------- CUSTOM UI --------------------
st.markdown("""
<style>
body {background-color: #0E1117; color: white;}
.stTextArea textarea {background-color: #1E1E1E; color: white;}
.big-title {text-align:center; font-size:30px; color:#4CAF50;}
.sub-text {text-align:center; color:gray;}
.card {
    background-color: #1E1E1E;
    padding: 20px;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

# -------------------- TITLE --------------------
st.markdown("<div class='big-title'>🏛️ Smart Municipal Complaint System</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-text'>AI-powered complaint classification dashboard</div>", unsafe_allow_html=True)

# -------------------- SIDEBAR --------------------
st.sidebar.title("📊 Dashboard")
st.sidebar.info("Predict complaint category using Machine Learning")

model_choice = st.sidebar.selectbox(
    "🔀 Select Model",
    ["Gradient Boosting", "Logistic Regression", "Naive Bayes"]
)

st.sidebar.markdown("### 👨‍💻 Developer")
st.sidebar.write("Jinit Dave")

# -------------------- LOAD DATA --------------------
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

model_files = {
    "Gradient Boosting": "gradient_boosting_model.pkl",
    "Logistic Regression": "logistic_regression_model.pkl",
    "Naive Bayes": "naive_bayes_model.pkl"
}

# -------------------- CATEGORY MAPPING --------------------
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

# -------------------- HEADER --------------------
st.markdown("---")
st.caption("💡 Example: There is no electricity in my area")

# -------------------- INPUT --------------------
user_input = st.text_area("📝 Enter your complaint:", height=150)

# -------------------- PREDICTION --------------------
if user_input.strip():
    with st.spinner("Analyzing complaint..."):

        model = pickle.load(open(model_files[model_choice], "rb"))

        X_new = vectorizer.transform([user_input])
        y_pred = model.predict(X_new)
        prediction = le.inverse_transform(y_pred)[0]

        # Confidence score
        try:
            prob = model.predict_proba(X_new).max()
            confidence = round(prob * 100, 2)
        except:
            confidence = "N/A"

        # Enhanced category
        enhanced_prediction = map_category(user_input)

        # -------------------- DISPLAY CARDS --------------------
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown(f"<div class='card'>📌 <b>Original Category</b><br><br>{prediction}</div>", unsafe_allow_html=True)

        with col2:
            st.markdown(f"<div class='card'>🏛️ <b>Municipal Category</b><br><br>{enhanced_prediction}</div>", unsafe_allow_html=True)

        with col3:
            st.markdown(f"<div class='card'>🎯 <b>Confidence</b><br><br>{confidence}</div>", unsafe_allow_html=True)

        # -------------------- SIMILAR DATA --------------------
        similar_df = df[df[category_col] == prediction].copy()
        similar_df = similar_df.head(5)

        similar_df["Predicted Category"] = prediction
        similar_df["Municipal Category"] = enhanced_prediction
        similar_df["Confidence"] = confidence

        st.markdown("### 📋 Similar Complaints from Dataset")

        # Column selector
        selected_columns = st.multiselect(
            "Select columns to display",
            options=similar_df.columns,
            default=similar_df.columns
        )

        st.dataframe(similar_df[selected_columns], use_container_width=True)

        # -------------------- DOWNLOAD --------------------
        csv = similar_df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇ Download Result", data=csv, file_name="result.csv")

# -------------------- CHART --------------------
st.markdown("### 📊 Category Distribution")
st.bar_chart(df[category_col].value_counts())
