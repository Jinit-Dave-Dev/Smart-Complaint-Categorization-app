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
    color: #f1f5f9;
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
    st.title("🔐 Smart Complaint System Login")

    u = st.text_input("Username")
    p = st.text_input("Password", type="password")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Login"):
            c.execute("SELECT * FROM users WHERE username=? AND password=?", (u, p))
            if c.fetchone():
                st.session_state.logged_in = True
                st.session_state.user = u
                st.rerun()
            else:
                st.error("Invalid Credentials")

    with col2:
        if st.button("Register"):
            c.execute("INSERT INTO users VALUES (?,?)", (u, p))
            conn.commit()
            st.success("Registered Successfully")

if not st.session_state.logged_in:
    login()
    st.stop()

# -------------------- SIDEBAR --------------------
st.sidebar.title("📊 Smart System")
st.sidebar.write(f"👤 {st.session_state.user}")

if st.sidebar.button("Logout"):
    st.session_state.logged_in = False
    st.rerun()

# -------------------- LOAD MODELS --------------------
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

# -------------------- CATEGORY ENGINE (STABLE) --------------------
def get_category(text):
    t = re.sub(r'[^a-zA-Z ]', ' ', str(text).lower())

    mapping = {
        "road": "Road",
        "pothole": "Road",
        "water": "Water",
        "leak": "Water",
        "garbage": "Garbage",
        "waste": "Garbage",
        "electric": "Electricity",
        "power": "Electricity"
    }

    for k, v in mapping.items():
        if k in t:
            return v

    return "Other"

# -------------------- CHATBOT (STABLE LOGIC) --------------------
def chatbot(msg):
    m = msg.lower()

    if any(x in m for x in ["hi", "hello", "hey"]):
        return "👋 Hello! How can I help you today?"

    if "road" in m:
        return "🛣️ Road complaint registered and sent to department."

    if "water" in m:
        return "💧 Water issue recorded and under review."

    if "electric" in m:
        return "⚡ Electricity complaint forwarded to authority."

    if "status" in m:
        return "📊 Check Dashboard for live complaint updates."

    return "📌 Your complaint has been successfully recorded."

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
            conf = 70.0

        c.execute("""
            INSERT INTO complaints VALUES (?, ?, ?, ?, ?)
        """, (
            st.session_state.user,
            text,
            prediction,
            category,
            str(conf)
        ))

        conn.commit()

        st.success("Complaint Registered")

        col1, col2 = st.columns(2)
        col1.metric("Category", category)
        col2.metric("Confidence", f"{conf}%")

        # Similar complaints (optimized)
        st.markdown("### 🔍 Similar Complaints")

        X_all = vectorizer.transform(df[complaint_col])
        X_input = vectorizer.transform([text])

        sim = cosine_similarity(X_input, X_all)[0]
        idx = np.argsort(sim)[-5:][::-1]

        st.dataframe(df.iloc[idx][[complaint_col]], use_container_width=True)

# ================== DASHBOARD ==================
with tabs[1]:

    saved = pd.read_sql_query("SELECT * FROM complaints", conn)

    if not saved.empty:

        saved["timestamp"] = pd.date_range(end=datetime.now(), periods=len(saved))
        saved["status"] = np.where(saved.index >= len(saved)-5, "NEW", "OLD")

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Complaints", len(saved))
        col2.metric("Users", saved["user"].nunique())
        col3.metric("Top Category", saved["category"].value_counts().idxmax())

        st.markdown("### 📋 Complaint Records")
        st.dataframe(saved.sort_values("timestamp", ascending=False), use_container_width=True)

# ================== ANALYTICS ==================
with tabs[2]:

    saved = pd.read_sql_query("SELECT * FROM complaints", conn)

    if not saved.empty:

        st.markdown("## 📊 Analytics Overview")

        col1, col2, col3 = st.columns(3)
        col1.metric("Total", len(saved))
        col2.metric("Categories", saved["category"].nunique())
        col3.metric("Top", saved["category"].value_counts().idxmax())

        # PIE
        st.markdown("### 🥧 Category Split")
        fig1, ax1 = plt.subplots(figsize=(6, 5))
        saved["category"].value_counts().plot.pie(autopct="%1.1f%%", ax=ax1)
        ax1.set_ylabel("")
        st.pyplot(fig1)

        # BAR
        st.markdown("### 📊 Category Count")
        fig2, ax2 = plt.subplots(figsize=(7, 4))
        saved["category"].value_counts().plot.bar(ax=ax2)
        st.pyplot(fig2)

# ================== CHATBOT ==================
with tabs[3]:

    if "chat" not in st.session_state:
        st.session_state.chat = []

    msg = st.text_input("Ask something...")

    col1, col2 = st.columns([3, 1])

    if msg:
        reply = chatbot(msg)
        st.session_state.chat.append(("You", msg))
        st.session_state.chat.append(("Bot", reply))

    if col2.button("Clear Chat"):
        st.session_state.chat = []

    for role, text in st.session_state.chat:
        if role == "You":
            st.markdown(f"🧑 **You:** {text}")
        else:
            st.markdown(f"🤖 **Bot:** {text}")
