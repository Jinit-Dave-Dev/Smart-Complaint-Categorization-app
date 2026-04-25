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
import uuid

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

# Base table
c.execute("""
CREATE TABLE IF NOT EXISTS complaints (
    user TEXT,
    complaint TEXT,
    prediction TEXT,
    category TEXT,
    confidence TEXT
)
""")

# Auto migration
def add_column(name, typ):
    try:
        c.execute(f"ALTER TABLE complaints ADD COLUMN {name} {typ}")
    except:
        pass

add_column("id", "TEXT")
add_column("status", "TEXT")
add_column("department", "TEXT")
add_column("timestamp", "TEXT")

c.execute("CREATE TABLE IF NOT EXISTS users (username TEXT, password TEXT)")
conn.commit()

# -------------------- SESSION --------------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "user" not in st.session_state:
    st.session_state.user = ""

# -------------------- LOGIN --------------------
def login():
    st.title("🔐 SMART COMPLAINT CATEGORIZATION GOVERNMENT PORTAL")

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

# -------------------- HELPERS --------------------
def get_category(text):
    t = re.sub(r'[^a-zA-Z ]', ' ', str(text).lower())
    if "road" in t or "pothole" in t: return "Road"
    if "water" in t or "leak" in t: return "Water"
    if "garbage" in t: return "Garbage"
    if "electric" in t: return "Electricity"
    return "Other"

def get_department(category):
    return {
        "Road": "Public Works",
        "Water": "Water Dept",
        "Garbage": "Sanitation",
        "Electricity": "Electric Dept"
    }.get(category, "General")

# -------------------- CHATBOT --------------------
def chatbot(msg):
    m = msg.lower()

    if any(x in m for x in ["hi", "hello", "hey"]):
        return "👋 Hello! I'm your Smart Municipal Assistant."

    if "road" in m:
        return "🛣️ Road issue → Fix expected in 2-3 days."

    if "water" in m:
        return "💧 Water issue → Team assigned (24 hrs)."

    if "electric" in m:
        return "⚡ Electricity → Technician dispatched."

    if "status" in m:
        return "📊 Track using Tracking ID in Track tab."

    return "📌 Submit complaint for proper tracking."

# -------------------- UI --------------------
st.title("🏛️ Smart Municipal Complaint System")

tabs = st.tabs(["📝 Complaint", "📊 Dashboard", "📈 Analytics", "🔎 Track Complaint", "🤖 Chatbot"])

# ================== COMPLAINT ==================
with tabs[0]:

    text = st.text_area("Enter your complaint")

    if st.button("Submit Complaint") and text.strip():

        X = vectorizer.transform([text])
        pred = model.predict(X)
        prediction = le.inverse_transform(pred)[0]

        category = get_category(text)
        department = get_department(category)
        conf = round(model.predict_proba(X).max() * 100, 2)

        tracking_id = str(uuid.uuid4())[:8]

        c.execute("""
        INSERT INTO complaints 
        (id, user, complaint, prediction, category, confidence, status, department, timestamp)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            tracking_id,
            st.session_state.user,
            text,
            prediction,
            category,
            str(conf),
            "Pending",
            department,
            str(datetime.now())
        ))

        conn.commit()

        st.success(f"Complaint Registered | ID: {tracking_id}")

    st.markdown("### 📋 Your Complaints")
    user_data = pd.read_sql_query(
        "SELECT * FROM complaints WHERE user=?",
        conn,
        params=(st.session_state.user,)
    )
    st.dataframe(user_data.iloc[::-1], use_container_width=True)

# ================== DASHBOARD ==================
with tabs[1]:

    saved = pd.read_sql_query("SELECT * FROM complaints", conn)

    if not saved.empty:

        saved.fillna("N/A", inplace=True)

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Complaints", len(saved))
        col2.metric("Users", saved["user"].nunique())
        col3.metric("Top Category", saved["category"].value_counts().idxmax())

        st.markdown("### 📡 Live Feed")
        st.dataframe(saved.iloc[::-1], use_container_width=True)

# ================== ANALYTICS ==================
with tabs[2]:

    saved = pd.read_sql_query("SELECT * FROM complaints", conn)

    if not saved.empty:

        saved["timestamp"] = pd.to_datetime(saved["timestamp"], errors="coerce")

        g1, g2 = st.columns(2)
        g3, g4 = st.columns(2)

        with g1:
            fig, ax = plt.subplots()
            saved["category"].value_counts().plot.pie(autopct="%1.1f%%", ax=ax)
            ax.set_ylabel("")
            st.pyplot(fig)

        with g2:
            fig, ax = plt.subplots()
            saved["category"].value_counts().plot.bar(ax=ax)
            st.pyplot(fig)

        with g3:
            fig, ax = plt.subplots()
            saved["department"].value_counts().plot.bar(ax=ax)
            st.pyplot(fig)

        with g4:
            fig, ax = plt.subplots()
            saved.groupby(saved["timestamp"].dt.date).size().plot(ax=ax)
            st.pyplot(fig)

# ================== TRACK COMPLAINT ==================
with tabs[3]:

    st.subheader("🔎 Track Your Complaint")

    track_id = st.text_input("Enter Tracking ID")

    if st.button("Track Complaint"):

        result = pd.read_sql_query(
            "SELECT * FROM complaints WHERE id=?",
            conn,
            params=(track_id,)
        )

        if not result.empty:
            st.success("Complaint Found")
            st.dataframe(result, use_container_width=True)
        else:
            st.error("Invalid Tracking ID")

# ================== CHATBOT ==================
with tabs[4]:

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
        st.write(f"**{r}:** {m}")
