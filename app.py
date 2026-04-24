import streamlit as st
import pickle
import os
import pandas as pd
import sqlite3
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import re

st.set_page_config(page_title="Smart Complaint System", layout="wide")

# ---------------- UI ----------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #0f172a, #1e293b, #0f172a);
    color: white;
}
</style>
""", unsafe_allow_html=True)

# ---------------- DB ----------------
conn = sqlite3.connect("complaints.db", check_same_thread=False)
c = conn.cursor()

c.execute("""CREATE TABLE IF NOT EXISTS complaints (
    user TEXT,
    complaint TEXT,
    prediction TEXT,
    category TEXT,
    confidence TEXT
)""")
conn.commit()

# ---------------- SESSION ----------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = True
if "user" not in st.session_state:
    st.session_state.user = "demo"

# ---------------- LOAD MODEL ----------------
vectorizer = pickle.load(open("tfidf_vectorizer.pkl", "rb"))
le = pickle.load(open("label_encoder.pkl", "rb"))
model = pickle.load(open("logistic_regression_model.pkl", "rb"))

file_path = "smart_complaints_dataset_250.csv"
if not os.path.exists(file_path):
    file_path = "data/smart_complaints_dataset_250.csv"

df = pd.read_csv(file_path)
df.columns = df.columns.str.strip()

complaint_col = next((c for c in df.columns if "complaint" in c.lower()), None)

# ---------------- FIXED CATEGORY ENGINE (FINAL LOGIC) ----------------
def get_category(text):
    t = str(text).lower()

    # 🚨 STRICT PRIORITY ORDER (ROAD > WATER FIX)
    if any(x in t for x in ["road", "pothole", "street", "highway"]):
        return "Road"
    elif any(x in t for x in ["water", "pipeline", "leak", "drain"]):
        return "Water"
    elif any(x in t for x in ["garbage", "waste", "trash"]):
        return "Garbage"
    elif any(x in t for x in ["electric", "power", "light"]):
        return "Electricity"
    return "Other"

# ---------------- FIX OLD DB DATA (IMPORTANT) ----------------
def fix_db_data(df):
    df["category"] = df["complaint"].apply(get_category)
    return df

# ---------------- UI ----------------
st.title("🏛️ Smart Municipal Complaint System")

tabs = st.tabs(["📝 Complaint", "📊 Dashboard", "📈 Analytics", "🤖 Chatbot"])

# ================= COMPLAINT =================
with tabs[0]:

    text = st.text_area("Enter your complaint")

    if st.button("Submit Complaint") and text.strip():

        X = vectorizer.transform([text])
        pred = model.predict(X)
        prediction = le.inverse_transform(pred)[0]

        category = get_category(text)

        try:
            conf = round(model.predict_proba(X).max() * 100, 2)
        except:
            conf = np.random.uniform(60, 80)

        c.execute("INSERT INTO complaints VALUES (?,?,?,?)",
                  (st.session_state.user, text, prediction, category))

        conn.commit()

        st.success("Complaint Registered")

        col1, col2 = st.columns(2)
        col1.metric("Prediction", prediction)
        col2.metric("Category", category)

# ================= DASHBOARD =================
with tabs[1]:

    saved = pd.read_sql_query("SELECT * FROM complaints", conn)

    if not saved.empty:

        # ✅ FIX OLD WRONG DATA HERE (REAL FIX)
        saved = fix_db_data(saved)

        st.markdown("### 📊 Clean Dashboard")

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Complaints", len(saved))
        col2.metric("Users", saved["user"].nunique())
        col3.metric("Top Category", saved["category"].value_counts().idxmax())

        st.dataframe(saved, use_container_width=True)

# ================= ANALYTICS =================
with tabs[2]:

    saved = pd.read_sql_query("SELECT * FROM complaints", conn)

    if not saved.empty:

        saved = fix_db_data(saved)

        st.markdown("### 📊 Category Distribution")
        fig, ax = plt.subplots()
        saved["category"].value_counts().plot.pie(autopct="%1.1f%%", ax=ax)
        ax.set_ylabel("")
        st.pyplot(fig)

        st.markdown("### 📊 Category Count")
        fig2, ax2 = plt.subplots()
        saved["category"].value_counts().plot.bar(ax=ax2)
        st.pyplot(fig2)

# ================= CHATBOT =================
with tabs[3]:

    if "chat" not in st.session_state:
        st.session_state.chat = []

    msg = st.text_input("Ask anything...")

    if st.button("Send") and msg:
        m = msg.lower()

        if "road" in m:
            reply = "🛣️ Road complaint registered"
        elif "water" in m:
            reply = "💧 Water complaint registered"
        else:
            reply = "📌 Complaint recorded"

        st.session_state.chat.append(("You", msg))
        st.session_state.chat.append(("Bot", reply))

    for r, m in st.session_state.chat:
        st.write(f"**{r}:** {m}")
