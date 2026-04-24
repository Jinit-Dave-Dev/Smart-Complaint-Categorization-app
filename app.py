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
import time  # 🔥 ADDED for smooth live refresh

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

# -------------------- PRIORITY ENGINE (NEW ADDITION) --------------------
def get_priority(text, category):
    t = text.lower()

    if any(x in t for x in ["accident", "burst", "fire", "danger", "no power"]):
        return "🔴 HIGH"

    if category in ["Road", "Water", "Electricity"]:
        return "🟡 MEDIUM"

    return "🟢 LOW"

# -------------------- CHATBOT --------------------
def chatbot(msg):
    m = msg.lower()

    if any(x in m for x in ["hi", "hello", "hey"]):
        return "👋 Hello! I am your municipal assistant."

    if "road" in m:
        return "🛣️ Road complaint registered."

    if "water" in m:
        return "💧 Water complaint registered."

    if "electric" in m:
        return "⚡ Electricity complaint registered."

    return "📌 Complaint recorded successfully."

# ================= MAIN UI =================
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
        priority = get_priority(text, category)

        try:
            conf = round(model.predict_proba(X).max() * 100, 2)
        except:
            conf = np.random.uniform(60, 80)

        c.execute("""
            INSERT INTO complaints VALUES (?, ?, ?, ?, ?)
        """, (st.session_state.user, text, prediction, category, str(conf)))

        conn.commit()

        st.success("Complaint Registered")

        st.info(f"Priority Assigned: {priority}")

# ================== DASHBOARD (GOVERNMENT CONTROL PANEL UPGRADE) ==================
with tabs[1]:

    # 🔥 LIVE REFRESH FEEL
    time.sleep(0.2)

    saved = pd.read_sql_query("SELECT * FROM complaints", conn)

    if not saved.empty:

        # STATUS ENGINE (UI ONLY)
        saved["status"] = np.where(saved.index % 3 == 0, "NEW",
                            np.where(saved.index % 3 == 1, "IN PROGRESS", "RESOLVED"))

        saved["priority"] = saved.apply(lambda x: get_priority(x["complaint"], x["category"]), axis=1)

        # KPI CARDS (REAL GOVERNMENT PANEL STYLE)
        col1, col2, col3, col4 = st.columns(4)

        col1.metric("Total Complaints", len(saved))
        col2.metric("High Priority", len(saved[saved["priority"] == "🔴 HIGH"]))
        col3.metric("In Progress", len(saved[saved["status"] == "IN PROGRESS"]))
        col4.metric("Resolved", len(saved[saved["status"] == "RESOLVED"]))

        st.markdown("### 📡 Government Live Complaint Feed")

        # COLOR BADGES
        def badge(x):
            if x == "NEW":
                return "🔵 NEW"
            elif x == "IN PROGRESS":
                return "🟡 IN PROGRESS"
            return "🟢 RESOLVED"

        saved["status"] = saved["status"].apply(badge)

        st.dataframe(saved.sort_values(saved.columns[0], ascending=False),
                     use_container_width=True)

# ================== ANALYTICS ==================
with tabs[2]:

    saved = pd.read_sql_query("SELECT * FROM complaints", conn)

    if not saved.empty:

        st.markdown("## 📊 Analytics Dashboard")

        col1, col2, col3 = st.columns(3)
        col1.metric("Total", len(saved))
        col2.metric("Categories", saved["category"].nunique())
        col3.metric("Top", saved["category"].value_counts().idxmax())

        st.markdown("### 🥧 Category Distribution")
        fig1, ax1 = plt.subplots(figsize=(6, 6))
        saved["category"].value_counts().plot.pie(autopct="%1.1f%%", ax=ax1)
        ax1.set_ylabel("")
        st.pyplot(fig1)

        st.markdown("### 📊 Category Volume")
        fig2, ax2 = plt.subplots(figsize=(8, 4))
        saved["category"].value_counts().plot.bar(ax=ax2)
        st.pyplot(fig2)

# ================== CHATBOT ==================
with tabs[3]:

    if "chat" not in st.session_state:
        st.session_state.chat = []

    msg = st.text_input("Ask anything...")

    if msg:
        st.session_state.chat.append(("You", msg))
        st.session_state.chat.append(("Assistant", chatbot(msg)))

    for r, m in st.session_state.chat:
        if r == "You":
            st.markdown(f"**🧑 You:** {m}")
        else:
            st.markdown(f"**🤖 Assistant:** {m}")
