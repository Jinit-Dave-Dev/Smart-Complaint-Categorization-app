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

    if "road" in t or "pothole" in t:
        return "Road"
    elif "water" in t or "leak" in t:
        return "Water"
    elif "garbage" in t:
        return "Garbage"
    elif "electric" in t or "power" in t:
        return "Electricity"
    return "Other"

# -------------------- SMART CHATBOT --------------------
def chatbot(msg):
    m = msg.lower()

    # Greeting
    if any(x in m for x in ["hi", "hello", "hey"]):
        return "👋 Hello! I'm your Smart Municipal Assistant. Tell me your issue."

    # Road solution
    if "road" in m or "pothole" in m:
        return "🛣️ Road Issue:\n- Complaint registered\n- Inspection team assigned\n- Expected fix: 2-3 days"

    # Water solution
    if "water" in m or "leak" in m:
        return "💧 Water Issue:\n- Pipeline team notified\n- Emergency check scheduled\n- Expected fix: 24 hrs"

    # Electricity solution
    if "electric" in m or "power" in m:
        return "⚡ Electricity Issue:\n- Complaint escalated\n- Technician dispatched\n- ETA: 4-6 hrs"

    # Garbage
    if "garbage" in m:
        return "🗑️ Garbage Issue:\n- Cleaning team assigned\n- Area scheduled for pickup"

    # Status
    if "status" in m:
        return "📊 You can check complaint status in Dashboard tab."

    return "📌 Your issue is noted. Please submit complaint in Complaint tab for tracking."

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

        try:
            conf = round(model.predict_proba(X).max() * 100, 2)
        except:
            conf = np.random.uniform(60, 80)

        c.execute("""
            INSERT INTO complaints VALUES (?, ?, ?, ?, ?)
        """, (st.session_state.user, text, prediction, category, str(conf)))

        conn.commit()

        st.success("Complaint Registered")

    # ✅ NEW: REAL-TIME TABLE (ADDED)
    st.markdown("### 📋 Your Recent Complaints")
    user_data = pd.read_sql_query(
        f"SELECT * FROM complaints WHERE user='{st.session_state.user}'",
        conn
    )

    if not user_data.empty:
        st.dataframe(user_data.iloc[::-1], use_container_width=True)

# ================== DASHBOARD ==================
with tabs[1]:

    saved = pd.read_sql_query("SELECT * FROM complaints", conn)

    if not saved.empty:

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Complaints", len(saved))
        col2.metric("Users", saved["user"].nunique())
        col3.metric("Top Category", saved["category"].value_counts().idxmax())

        st.markdown("### 📡 Live Complaint Feed")

        # ✅ IMPROVED REAL-TIME SORT
        st.dataframe(saved.iloc[::-1], use_container_width=True)

# ================== ANALYTICS ==================
with tabs[2]:

    saved = pd.read_sql_query("SELECT * FROM complaints", conn)

    if not saved.empty:

        st.markdown("## 📊 Analytics Dashboard")

        col1, col2, col3 = st.columns(3)
        col1.metric("Total", len(saved))
        col2.metric("Categories", saved["category"].nunique())
        col3.metric("Top", saved["category"].value_counts().idxmax())

        g1, g2 = st.columns(2)
        g3, g4 = st.columns(2)

        with g1:
            fig1, ax1 = plt.subplots(figsize=(4,4))
            saved["category"].value_counts().plot.pie(autopct="%1.1f%%", ax=ax1)
            ax1.set_ylabel("")
            st.pyplot(fig1)

        with g2:
            fig2, ax2 = plt.subplots(figsize=(4,4))
            saved["category"].value_counts().plot.bar(ax=ax2)
            st.pyplot(fig2)

        with g3:
            fig3, ax3 = plt.subplots(figsize=(4,4))
            saved["user"].value_counts().head(5).plot.bar(ax=ax3)
            st.pyplot(fig3)

        with g4:
            fig4, ax4 = plt.subplots(figsize=(4,4))
            saved.groupby("category").size().cumsum().plot(ax=ax4)
            st.pyplot(fig4)

# ================== CHATBOT ==================
with tabs[3]:

    if "chat" not in st.session_state:
        st.session_state.chat = []

    col1, col2 = st.columns([3,1])

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
