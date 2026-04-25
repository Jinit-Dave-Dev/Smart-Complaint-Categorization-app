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
import random
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

# 🔥 ADD NEW COLUMNS WITHOUT BREAKING OLD DB
try:
    c.execute("ALTER TABLE complaints ADD COLUMN status TEXT")
except:
    pass
try:
    c.execute("ALTER TABLE complaints ADD COLUMN priority TEXT")
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

    if "road" in t: return "Road"
    if "water" in t: return "Water"
    if "garbage" in t: return "Garbage"
    if "electric" in t: return "Electricity"
    return "Other"

# 🔥 PRIORITY ENGINE
def get_priority(text):
    if any(x in text.lower() for x in ["fire", "accident", "danger"]):
        return "🔴 High"
    return "🟡 Medium"

# 🔥 SIMULATED DATA
def simulate_data(n=120):
    cats = ["Road", "Water", "Garbage", "Electricity"]
    fake = []
    for _ in range(n):
        cat = random.choice(cats)
        fake.append({
            "user": "demo",
            "complaint": f"{cat} issue sample",
            "prediction": cat,
            "category": cat,
            "confidence": str(random.randint(70, 95)),
            "status": random.choice(["New", "In Progress", "Resolved"]),
            "priority": random.choice(["🔴 High", "🟡 Medium", "🟢 Low"])
        })
    return pd.DataFrame(fake)

# -------------------- CHATBOT --------------------
def chatbot(msg):
    return "🤖 Complaint noted. You can track status in dashboard."

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
        priority = get_priority(text)

        conf = round(np.random.uniform(60, 90), 2)

        c.execute("""
        INSERT INTO complaints (user, complaint, prediction, category, confidence, status, priority)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (st.session_state.user, text, prediction, category, str(conf), "New", priority))

        conn.commit()

        st.success("Complaint Registered")

# ================== DASHBOARD ==================
with tabs[1]:

    time.sleep(0.5)  # smooth refresh feel

    saved = pd.read_sql_query("SELECT * FROM complaints", conn)

    if len(saved) < 5:
        saved = simulate_data()

    saved["time"] = pd.date_range(end=datetime.now(), periods=len(saved))

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total", len(saved))
    col2.metric("High Priority", len(saved[saved["priority"].str.contains("High")]))
    col3.metric("In Progress", len(saved[saved["status"]=="In Progress"]))
    col4.metric("Resolved", len(saved[saved["status"]=="Resolved"]))

    st.markdown("### 🔧 Admin Panel")

    if not saved.empty:
        selected = st.selectbox("Select Complaint", saved.index)
        new_status = st.selectbox("Update Status", ["New", "In Progress", "Resolved"])

        if st.button("Update Status"):
            c.execute("UPDATE complaints SET status=? WHERE rowid=?",
                      (new_status, int(selected)+1))
            conn.commit()
            st.success("Updated")

    st.dataframe(saved, use_container_width=True)

# ================== ANALYTICS ==================
with tabs[2]:

    saved = pd.read_sql_query("SELECT * FROM complaints", conn)

    if len(saved) < 5:
        saved = simulate_data()

    saved["time"] = pd.date_range(end=datetime.now(), periods=len(saved))

    g1, g2 = st.columns(2)
    g3, g4 = st.columns(2)

    with g1:
        fig, ax = plt.subplots()
        saved["category"].value_counts().plot.pie(autopct="%1.1f%%", ax=ax)
        st.pyplot(fig)

    with g2:
        fig, ax = plt.subplots()
        saved["category"].value_counts().plot.bar(ax=ax)
        st.pyplot(fig)

    with g3:
        fig, ax = plt.subplots()
        saved["user"].value_counts().plot.bar(ax=ax)
        st.pyplot(fig)

    with g4:
        fig, ax = plt.subplots()
        saved.groupby(pd.Grouper(key="time", freq="D")).size().plot(ax=ax)
        st.pyplot(fig)

# ================== CHATBOT ==================
with tabs[3]:

    if "chat" not in st.session_state:
        st.session_state.chat = []

    msg = st.text_input("Ask anything...")

    if st.button("🗑️ Delete History"):
        st.session_state.chat = []

    if msg:
        st.session_state.chat.append(("You", msg))
        st.session_state.chat.append(("Assistant", chatbot(msg)))

    for r, m in st.session_state.chat:
        st.write(f"{r}: {m}")
