import streamlit as st
import pickle
import os
import pandas as pd
import sqlite3
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import re
from datetime import datetime

st.set_page_config(page_title="Smart Complaint System", layout="wide")

# -------------------- UI --------------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #0f172a, #1e293b, #0f172a);
    color: white;
}
.stButton button {
    background: linear-gradient(90deg, #4f46e5, #06b6d4);
    color: white;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

# -------------------- DB --------------------
conn = sqlite3.connect("complaints.db", check_same_thread=False)
c = conn.cursor()

c.execute("CREATE TABLE IF NOT EXISTS users (username TEXT, password TEXT)")
c.execute("""CREATE TABLE IF NOT EXISTS complaints (
    user TEXT,
    complaint TEXT,
    prediction TEXT,
    category TEXT,
    confidence TEXT
)""")

# 🔥 ADD MISSING COLUMNS SAFELY (NO BREAK)
for col in ["status", "priority", "department"]:
    try:
        c.execute(f"ALTER TABLE complaints ADD COLUMN {col} TEXT")
    except:
        pass

conn.commit()

# -------------------- SESSION --------------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "user" not in st.session_state:
    st.session_state.user = ""

# -------------------- LOGIN --------------------
def login():
    st.title("🔐 Login")

    u = st.text_input("Username")
    p = st.text_input("Password", type="password")

    if st.button("Login"):
        c.execute("SELECT * FROM users WHERE username=? AND password=?", (u, p))
        if c.fetchone():
            st.session_state.logged_in = True
            st.session_state.user = u
            st.rerun()
        else:
            st.error("Invalid Credentials")

    if st.button("Register"):
        c.execute("INSERT INTO users VALUES (?,?)", (u, p))
        conn.commit()
        st.success("Registered")

if not st.session_state.logged_in:
    login()
    st.stop()

# -------------------- SIDEBAR --------------------
st.sidebar.title("📊 Smart Dashboard")
st.sidebar.write(f"👤 {st.session_state.user}")

st.sidebar.markdown("### 👨‍💻 Developer")
st.sidebar.write("Jinit Dave")

if st.sidebar.button("Logout"):
    st.session_state.logged_in = False
    st.rerun()

# -------------------- LOAD --------------------
vectorizer = pickle.load(open("tfidf_vectorizer.pkl", "rb"))
le = pickle.load(open("label_encoder.pkl", "rb"))
model = pickle.load(open("logistic_regression_model.pkl", "rb"))

file_path = "smart_complaints_dataset_250.csv"
if not os.path.exists(file_path):
    file_path = "data/smart_complaints_dataset_250.csv"

df = pd.read_csv(file_path)
df.columns = df.columns.str.strip()

complaint_col = next((c for c in df.columns if "complaint" in c.lower()), None)
df[complaint_col] = df[complaint_col].astype(str)

# -------------------- CATEGORY --------------------
def get_category(text):
    t = re.sub(r'[^a-zA-Z ]', ' ', str(text).lower())

    if any(x in t for x in ["road", "pothole", "street"]):
        return "Road"
    if any(x in t for x in ["water", "leak", "pipeline"]):
        return "Water"
    if any(x in t for x in ["garbage", "waste"]):
        return "Garbage"
    if any(x in t for x in ["electric", "power"]):
        return "Electricity"
    return "Other"

# -------------------- CHATBOT --------------------
def chatbot(msg):
    return "🤖 Complaint recorded. Track in dashboard."

# -------------------- UI --------------------
st.title("🏛️ Smart Municipal Complaint System")

tabs = st.tabs(["📝 Complaint", "📊 Dashboard", "📈 Analytics", "🤖 Chatbot"])

# ================== COMPLAINT ==================
with tabs[0]:

    text = st.text_area("Enter your complaint")

    if st.button("Submit Complaint") and text.strip():

        X = vectorizer.transform([text])
        pred = model.predict(X)
        prediction = le.inverse_transform(pred)[0]

        category = get_category(text)

        conf = round(np.random.uniform(60, 90), 2)

        # 🔥 SAFE INSERT WITH NEW COLUMNS
        c.execute("""
        INSERT INTO complaints (user, complaint, prediction, category, confidence, status, priority, department)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            st.session_state.user,
            text,
            prediction,
            category,
            str(conf),
            "New",
            "🟡 Medium",
            "General"
        ))

        conn.commit()

        st.success("Complaint Registered")

# ================== DASHBOARD ==================
with tabs[1]:

    saved = pd.read_sql_query("SELECT * FROM complaints", conn)

    if not saved.empty:

        saved["timestamp"] = pd.date_range(end=datetime.now(), periods=len(saved))
        saved["type"] = np.where(saved.index >= len(saved)-5, "🆕 New", "📁 Old")

        # 🔥 SAFE FILL
        saved["priority"] = saved.get("priority", "🟡 Medium").fillna("🟡 Medium")
        saved["status"] = saved.get("status", "New").fillna("New")

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Complaints", len(saved))
        col2.metric("Users", saved["user"].nunique())
        col3.metric("Top Category", saved["category"].value_counts().idxmax())

        st.markdown("### 📋 Live Complaint Feed")
        st.dataframe(saved.sort_values("timestamp", ascending=False), use_container_width=True)

# ================== ANALYTICS (FIXED ONLY) ==================
with tabs[2]:

    saved = pd.read_sql_query("SELECT * FROM complaints", conn)

    if not saved.empty:

        # 🔥 CRITICAL FIXES
        for col in ["category", "status", "department"]:
            if col not in saved.columns:
                saved[col] = "Unknown"

        saved["category"] = saved["category"].fillna("Other")
        saved["status"] = saved["status"].fillna("New")
        saved["department"] = saved["department"].fillna("General")

        g1, g2 = st.columns(2)
        g3, g4 = st.columns(2)

        with g1:
            fig, ax = plt.subplots(figsize=(4,4))
            saved["category"].value_counts().plot.pie(autopct="%1.1f%%", ax=ax)
            ax.set_ylabel("")
            st.pyplot(fig)

        with g2:
            fig, ax = plt.subplots(figsize=(4,4))
            saved["category"].value_counts().plot.bar(ax=ax)
            st.pyplot(fig)

        with g3:
            fig, ax = plt.subplots(figsize=(4,4))
            dept_counts = saved["department"].value_counts()

            if len(dept_counts) > 0:
                dept_counts.plot.bar(ax=ax)
            else:
                ax.text(0.5, 0.5, "No Data", ha='center')

            st.pyplot(fig)

        with g4:
            fig, ax = plt.subplots(figsize=(4,4))
            saved["status"].value_counts().plot.bar(ax=ax)
            st.pyplot(fig)

# ================== CHATBOT ==================
with tabs[3]:

    if "chat" not in st.session_state:
        st.session_state.chat = []

    col1, col2 = st.columns([3, 1])

    msg = col1.text_input("Ask anything...")

    if col2.button("🗑️ Clear Chat"):
        st.session_state.chat = []

    if msg:
        st.session_state.chat.append(("You", msg))
        st.session_state.chat.append(("Assistant", chatbot(msg)))

    for r, m in st.session_state.chat:
        if r == "You":
            st.markdown(f"**🧑 You:** {m}")
        else:
            st.markdown(f"**🤖 Assistant:** {m}")
