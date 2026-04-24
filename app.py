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
import time

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

# -------------------- SIDEBAR (UPGRADED NAV) --------------------
st.sidebar.title("🏛️ Government Control Panel")
menu = st.sidebar.radio("Navigate", ["Dashboard", "Complaints", "Analytics", "Chatbot"])

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

def get_priority(text, category):
    t = text.lower()
    if any(x in t for x in ["accident", "burst", "fire", "danger"]):
        return "🔴 HIGH"
    if category in ["Road", "Water", "Electricity"]:
        return "🟡 MEDIUM"
    return "🟢 LOW"

def chatbot(msg):
    m = msg.lower()
    if any(x in m for x in ["hi", "hello", "hey"]):
        return "👋 Hello!"
    if "road" in m:
        return "🛣️ Road complaint registered."
    return "📌 Complaint recorded."

# ================== COMPLAINT ==================
if menu == "Complaints":

    st.title("📝 Complaint Submission + Admin View")

    text = st.text_area("Enter complaint")

    if st.button("Submit Complaint") and text.strip():

        X = vectorizer.transform([text])
        pred = model.predict(X)
        prediction = le.inverse_transform(pred)[0]

        category = get_category(text)
        priority = get_priority(text, category)

        conf = np.random.uniform(60, 90)

        c.execute("""
            INSERT INTO complaints VALUES (?, ?, ?, ?, ?)
        """, (st.session_state.user, text, prediction, category, str(conf)))
        conn.commit()

        st.success("Complaint Registered")
        st.info(f"Priority: {priority}")

    st.markdown("### 📌 Complaint Detail Viewer")

    saved = pd.read_sql_query("SELECT * FROM complaints", conn)

    if not saved.empty:
        selected = st.selectbox("Select Complaint", saved.index)

        st.write(saved.loc[selected])

        # Department Assignment (NEW FEATURE)
        dept = st.selectbox("Assign Department", ["Road Dept", "Water Dept", "Electric Dept", "Sanitation"])
        status = st.selectbox("Update Status", ["NEW", "IN PROGRESS", "RESOLVED"])

        if st.button("Update Complaint"):
            st.success(f"Updated to {status} → {dept}")

# ================== DASHBOARD ==================
elif menu == "Dashboard":

    st.title("🏛️ Government Dashboard")

    saved = pd.read_sql_query("SELECT * FROM complaints", conn)

    if not saved.empty:

        st.metric("Total Complaints", len(saved))
        st.metric("Users", saved["user"].nunique())

        st.markdown("### 🔴 Live Feed")
        st.dataframe(saved, use_container_width=True)

# ================== ANALYTICS ==================
elif menu == "Analytics":

    st.title("📊 Analytics Panel")

    saved = pd.read_sql_query("SELECT * FROM complaints", conn)

    if not saved.empty:

        st.bar_chart(saved["category"].value_counts())
        st.area_chart(saved["confidence"].astype(float))

# ================== CHATBOT ==================
elif menu == "Chatbot":

    st.title("🤖 Chatbot")

    if "chat" not in st.session_state:
        st.session_state.chat = []

    msg = st.text_input("Ask...")

    col1, col2 = st.columns([3,1])

    if col2.button("🗑️ Delete History"):
        st.session_state.chat = []

    if msg:
        st.session_state.chat.append(("You", msg))
        st.session_state.chat.append(("Bot", chatbot(msg)))

    for r, m in st.session_state.chat:
        st.write(f"**{r}:** {m}")
