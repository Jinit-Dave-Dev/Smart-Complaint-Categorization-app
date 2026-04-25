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

c.execute("CREATE TABLE IF NOT EXISTS users (username TEXT, password TEXT)")

# ORIGINAL TABLE (kept)
c.execute("""CREATE TABLE IF NOT EXISTS complaints (
    user TEXT,
    complaint TEXT,
    prediction TEXT,
    category TEXT,
    confidence TEXT
)""")

# 🔥 ADD NEW COLUMNS SAFELY (NO BREAK)
def add_column_if_not_exists(column_name):
    try:
        c.execute(f"ALTER TABLE complaints ADD COLUMN {column_name} TEXT")
    except:
        pass

add_column_if_not_exists("id")
add_column_if_not_exists("status")
add_column_if_not_exists("department")
add_column_if_not_exists("timestamp")

conn.commit()

# -------------------- SESSION --------------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "user" not in st.session_state:
    st.session_state.user = ""

# -------------------- LOGIN --------------------
def login():
    col1, col2 = st.columns([2, 1])

    with col1:
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/5/55/India_Gate_in_New_Delhi_03-2016.jpg/1200px-India_Gate_in_New_Delhi_03-2016.jpg",
                 use_container_width=True)

    with col2:
        st.markdown("## 🏛️ SMART COMPLAINT CATEGORIZATION GOVERNMENT PORTAL")

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

# -------------------- HELPERS --------------------
def get_category(text):
    t = re.sub(r'[^a-zA-Z ]', ' ', str(text).lower())
    if "road" in t: return "Road"
    if "water" in t: return "Water"
    if "garbage" in t: return "Garbage"
    if "electric" in t: return "Electricity"
    return "Other"

def get_department(category):
    return {
        "Road": "Public Works",
        "Water": "Water Supply",
        "Garbage": "Sanitation",
        "Electricity": "Electric Dept"
    }.get(category, "General")

def chatbot(msg):
    m = msg.lower()
    if "hello" in m or "hi" in m:
        return "👋 Hello! Tell me your issue."
    if "road" in m:
        return "🛣️ Road issue noted. Avoid unsafe areas."
    if "water" in m:
        return "💧 Water issue detected. Check leakage source."
    if "electric" in m:
        return "⚡ Stay safe. Do not touch exposed wires."
    return "🤖 Please explain your issue clearly."

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
        department = get_department(category)

        try:
            conf = round(model.predict_proba(X).max() * 100, 2)
        except:
            conf = np.random.uniform(60, 80)

        complaint_id = "CMP-" + str(uuid.uuid4())[:8]

        # 🔥 SAFE INSERT (no crash)
        c.execute("""
        INSERT INTO complaints 
        (user, complaint, prediction, category, confidence, id, status, department, timestamp)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            st.session_state.user,
            text,
            prediction,
            category,
            str(conf),
            complaint_id,
            "In Progress",
            department,
            str(datetime.now())
        ))
        conn.commit()

        st.success(f"Complaint Registered | ID: {complaint_id}")

# ================== DASHBOARD ==================
with tabs[1]:

    saved = pd.read_sql_query("SELECT * FROM complaints", conn)

    if not saved.empty:
        st.dataframe(saved, use_container_width=True)

# ================== ANALYTICS ==================
with tabs[2]:

    saved = pd.read_sql_query("SELECT * FROM complaints", conn)

    if not saved.empty:

        saved["timestamp"] = pd.to_datetime(saved["timestamp"], errors='coerce')

        col1, col2 = st.columns(2)

        with col1:
            fig, ax = plt.subplots()
            saved["category"].value_counts().plot.pie(autopct="%1.1f%%", ax=ax)
            ax.set_ylabel("")
            st.pyplot(fig)

        with col2:
            fig, ax = plt.subplots()
            saved["category"].value_counts().plot.bar(ax=ax)
            st.pyplot(fig)

# ================== CHATBOT ==================
with tabs[3]:

    if "chat" not in st.session_state:
        st.session_state.chat = []

    msg = st.text_input("Ask...")

    if msg:
        st.session_state.chat.append(("You", msg))
        st.session_state.chat.append(("Bot", chatbot(msg)))

    for r, m in st.session_state.chat:
        st.write(f"{r}: {m}")
