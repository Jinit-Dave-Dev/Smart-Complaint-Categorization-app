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
    background: linear-gradient(135deg, #0f172a, #1e293b);
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

def add_column(col):
    try:
        c.execute(f"ALTER TABLE complaints ADD COLUMN {col} TEXT")
    except:
        pass

for col in ["id", "status", "department", "timestamp"]:
    add_column(col)

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
        st.image("https://images.unsplash.com/photo-1581092918056-0c4c3acd3789", use_container_width=True)

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

is_admin = st.session_state.user == "admin"

if is_admin:
    st.sidebar.success("🧑‍💼 Admin Mode")

st.sidebar.markdown("### 👨‍💻 Developer")
st.sidebar.write("Jinit Dave")

if st.sidebar.button("Logout"):
    st.session_state.logged_in = False
    st.rerun()

# -------------------- LOAD --------------------
vectorizer = pickle.load(open("tfidf_vectorizer.pkl", "rb"))
le = pickle.load(open("label_encoder.pkl", "rb"))
model = pickle.load(open("logistic_regression_model.pkl", "rb"))

# -------------------- HELPERS --------------------
def get_category(text):
    t = re.sub(r'[^a-zA-Z ]', ' ', text.lower())
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

# -------------------- CHATBOT (UPGRADED) --------------------
def chatbot(msg):
    m = msg.lower()

    if any(x in m for x in ["hi", "hello"]):
        return "👋 Welcome! Describe your issue and I’ll guide you."

    if "road" in m:
        return "🛣️ Road issue detected. Avoid damaged areas. Municipal team will repair it soon."

    if "water" in m:
        return "💧 Water leakage issue. Turn off valves if possible and report location."

    if "electric" in m:
        return "⚡ Stay away from exposed wires. Do not touch. Emergency team will respond."

    if "track" in m:
        return "🔎 Use your Complaint ID in dashboard to track status."

    return "🤖 Please explain your issue in detail."

# -------------------- UI --------------------
st.title("🏛️ Smart Municipal Complaint System")

tabs = st.tabs(["📝 Complaint", "📊 Dashboard", "📈 Analytics", "🤖 Chatbot", "🔎 Track Complaint"])

# ================== COMPLAINT ==================
with tabs[0]:

    text = st.text_area("Enter your complaint")

    if st.button("Submit Complaint") and text.strip():

        X = vectorizer.transform([text])
        pred = model.predict(X)
        prediction = le.inverse_transform(pred)[0]

        category = get_category(text)
        department = get_department(category)

        conf = round(np.random.uniform(70, 95), 2)

        cid = "CMP-" + str(uuid.uuid4())[:8]

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
            cid,
            "Pending",
            department,
            str(datetime.now())
        ))
        conn.commit()

        st.success(f"Complaint Registered | ID: {cid}")

# ================== DASHBOARD ==================
with tabs[1]:

    df = pd.read_sql_query("SELECT * FROM complaints", conn)

    if not df.empty:
        df.fillna("N/A", inplace=True)

        st.dataframe(df, use_container_width=True)

        if is_admin:
            st.markdown("### 🛠️ Admin Controls")

            cid = st.text_input("Enter Complaint ID")

            if st.button("Mark Resolved"):
                c.execute("UPDATE complaints SET status='Resolved' WHERE id=?", (cid,))
                conn.commit()

            if st.button("Mark In Progress"):
                c.execute("UPDATE complaints SET status='In Progress' WHERE id=?", (cid,))
                conn.commit()

# ================== ANALYTICS ==================
with tabs[2]:

    df = pd.read_sql_query("SELECT * FROM complaints", conn)

    if not df.empty:
        col1, col2 = st.columns(2)

        with col1:
            fig, ax = plt.subplots()
            df["category"].value_counts().plot.pie(autopct="%1.1f%%", ax=ax)
            ax.set_ylabel("")
            st.pyplot(fig)

        with col2:
            fig, ax = plt.subplots()
            df["category"].value_counts().plot.bar(ax=ax)
            st.pyplot(fig)

# ================== CHATBOT ==================
with tabs[3]:

    if "chat" not in st.session_state:
        st.session_state.chat = []

    col1, col2 = st.columns([3, 1])

    msg = col1.text_input("Ask your issue...")

    if col2.button("🗑️ Clear Chat"):
        st.session_state.chat = []

    if msg:
        st.session_state.chat.append(("You", msg))
        st.session_state.chat.append(("Assistant", chatbot(msg)))

    for r, m in st.session_state.chat:
        st.markdown(f"**{r}:** {m}")

# ================== TRACK ==================
with tabs[4]:

    cid = st.text_input("Enter Complaint ID to track")

    if st.button("Track"):
        result = pd.read_sql_query("SELECT * FROM complaints WHERE id=?", conn, params=(cid,))
        if not result.empty:
            st.dataframe(result)
        else:
            st.warning("No complaint found")
