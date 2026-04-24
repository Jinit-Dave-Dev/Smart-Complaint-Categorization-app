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

# -------------------- SIDEBAR --------------------
st.sidebar.title("🏛️ Govt Complaint Portal")
page = st.sidebar.radio("Navigate", ["Submit Complaint", "All Complaints", "Analytics", "Chatbot"])

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

# -------------------- FUNCTIONS --------------------
def get_category(text):
    t = re.sub(r'[^a-zA-Z ]', ' ', str(text).lower())

    if "road" in t:
        return "Road"
    if "water" in t:
        return "Water"
    if "garbage" in t:
        return "Garbage"
    if "electric" in t:
        return "Electricity"
    return "Other"

def get_priority(text, category):
    t = text.lower()
    if any(x in t for x in ["fire", "accident", "burst"]):
        return "HIGH"
    if category in ["Road", "Water", "Electricity"]:
        return "MEDIUM"
    return "LOW"

def chatbot(msg):
    m = msg.lower()
    if "hello" in m:
        return "👋 Hello! How can I help?"
    if "road" in m:
        return "🛣️ Road complaint registered."
    return "📌 Your complaint is recorded."

# ================== PAGE 1: SUBMIT ==================
if page == "Submit Complaint":

    st.title("📝 Submit Complaint")

    text = st.text_area("Enter complaint")

    if st.button("Submit") and text.strip():

        X = vectorizer.transform([text])
        pred = model.predict(X)
        prediction = le.inverse_transform(pred)[0]

        category = get_category(text)
        priority = get_priority(text, category)

        try:
            conf = round(model.predict_proba(X).max() * 100, 2)
        except:
            conf = 70

        c.execute("""
            INSERT INTO complaints VALUES (?, ?, ?, ?, ?)
        """, (st.session_state.user, text, prediction, category, str(conf)))
        conn.commit()

        st.success("Complaint Submitted Successfully")
        st.info(f"Priority: {priority}")

# ================== PAGE 2: ALL COMPLAINTS ==================
elif page == "All Complaints":

    st.title("📋 Complaint Management Panel")

    saved = pd.read_sql_query("SELECT * FROM complaints", conn)

    if not saved.empty:

        saved["Status"] = "NEW"
        saved["Priority"] = saved.apply(lambda x: get_priority(x["complaint"], x["category"]), axis=1)

        st.metric("Total Complaints", len(saved))

        # clickable selection
        idx = st.selectbox("Select Complaint ID", saved.index)

        st.write("### Complaint Details")
        st.write(saved.loc[idx])

        # workflow system
        new_status = st.selectbox("Update Status", ["NEW", "IN PROGRESS", "RESOLVED"])
        dept = st.selectbox("Assign Department", ["Road", "Water", "Electricity", "Sanitation"])

        if st.button("Update Complaint"):
            st.success(f"Updated → {new_status} | Dept → {dept}")

        st.markdown("### 📊 All Complaints Table")
        st.dataframe(saved, use_container_width=True)

# ================== PAGE 3: ANALYTICS ==================
elif page == "Analytics":

    st.title("📊 Analytics Dashboard")

    saved = pd.read_sql_query("SELECT * FROM complaints", conn)

    if not saved.empty:

        col1, col2, col3 = st.columns(3)
        col1.metric("Total", len(saved))
        col2.metric("Categories", saved["category"].nunique())
        col3.metric("Top", saved["category"].value_counts().idxmax())

        st.bar_chart(saved["category"].value_counts())

# ================== PAGE 4: CHATBOT ==================
elif page == "Chatbot":

    st.title("🤖 Chatbot Assistant")

    if "chat" not in st.session_state:
        st.session_state.chat = []

    msg = st.text_input("Ask something...")

    col1, col2 = st.columns([3,1])

    if col2.button("🗑️ Delete History"):
        st.session_state.chat = []

    if msg:
        st.session_state.chat.append(("You", msg))
        st.session_state.chat.append(("Bot", chatbot(msg)))

    for r, m in st.session_state.chat:
        st.write(f"**{r}:** {m}")
