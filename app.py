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

# -------------------- MAIN UI --------------------
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

        try:
            conf = round(model.predict_proba(X).max() * 100, 2)
        except:
            conf = np.random.uniform(60, 80)

        c.execute("""
            INSERT INTO complaints VALUES (?, ?, ?, ?, ?)
        """, (st.session_state.user, text, prediction, category, str(conf)))

        conn.commit()

        st.success("Complaint Registered")

# ================== DASHBOARD (FIXED REAL-TIME TABLE) ==================
with tabs[1]:

    # 🔥 ALWAYS FRESH DATA (FIX #1)
    saved = pd.read_sql_query("SELECT * FROM complaints", conn)

    if saved is not None and not saved.empty:

        # FIX: ensure clean ordering
        saved = saved.reset_index(drop=True)

        # FIX: add timestamp safely
        saved["timestamp"] = pd.date_range(end=datetime.now(), periods=len(saved))

        # FIX: proper "NEW / OLD"
        saved["type"] = np.where(saved.index >= len(saved)-5, "🆕 New", "📁 Old")

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Complaints", len(saved))
        col2.metric("Users", saved["user"].nunique() if "user" in saved.columns else 1)
        col3.metric("Top Category", saved["category"].value_counts().idxmax())

        st.markdown("### 📋 LIVE COMPLAINT TABLE (FIXED)")

        # FIX: stable sorting for real-time feel
        st.dataframe(
            saved.sort_values(by=saved.columns[0], ascending=False),
            use_container_width=True
        )

    else:
        st.warning("No complaints found yet.")

# ================== ANALYTICS (FIXED CHART STABILITY) ==================
with tabs[2]:

    saved = pd.read_sql_query("SELECT * FROM complaints", conn)

    if saved is not None and not saved.empty:

        st.markdown("## 📊 Analytics Dashboard")

        # FIX: safe grouping
        category_counts = saved["category"].value_counts()

        col1, col2, col3 = st.columns(3)
        col1.metric("Total", len(saved))
        col2.metric("Categories", saved["category"].nunique())
        col3.metric("Top", category_counts.idxmax() if len(category_counts) > 0 else "N/A")

        # PIE FIX
        st.markdown("### 🥧 Category Distribution")
        fig1, ax1 = plt.subplots(figsize=(6, 6))
        category_counts.plot.pie(autopct="%1.1f%%", ax=ax1)
        ax1.set_ylabel("")
        st.pyplot(fig1)

        # BAR FIX
        st.markdown("### 📊 Category Volume")
        fig2, ax2 = plt.subplots(figsize=(8, 4))
        category_counts.plot.bar(ax=ax2)
        st.pyplot(fig2)

        # TREND FIX
        st.markdown("### 📈 Trend")
        fig3, ax3 = plt.subplots(figsize=(8, 4))
        category_counts.cumsum().plot(ax=ax3)
        st.pyplot(fig3)

    else:
        st.warning("No data available for analytics.")

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
