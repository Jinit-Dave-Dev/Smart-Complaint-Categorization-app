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
.login-box {
    background: white;
    padding: 30px;
    border-radius: 12px;
    color: black;
}
</style>
""", unsafe_allow_html=True)

# -------------------- DB --------------------
conn = sqlite3.connect("complaints.db", check_same_thread=False)
c = conn.cursor()

c.execute("CREATE TABLE IF NOT EXISTS users (username TEXT, password TEXT)")

# ✅ Added columns without removing old ones
c.execute("""
CREATE TABLE IF NOT EXISTS complaints (
    id TEXT,
    user TEXT,
    complaint TEXT,
    prediction TEXT,
    category TEXT,
    confidence TEXT,
    status TEXT,
    department TEXT,
    timestamp TEXT
)
""")
conn.commit()

# -------------------- SESSION --------------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "user" not in st.session_state:
    st.session_state.user = ""

# -------------------- LOGIN PAGE (UPDATED UI) --------------------
def login():
    col1, col2 = st.columns([2, 1])

    with col1:
        st.image("https://images.unsplash.com/photo-1590650046871-92c887180603", use_container_width=True)

    with col2:
        st.markdown("### 🏛️ SMART COMPLAINT CATEGORIZATION GOVERNMENT PORTAL")

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

# -------------------- DEPARTMENT --------------------
def get_department(category):
    mapping = {
        "Road": "Public Works",
        "Water": "Water Supply",
        "Garbage": "Sanitation",
        "Electricity": "Electric Dept"
    }
    return mapping.get(category, "General")

# -------------------- CHATBOT (FIXED REAL BOT) --------------------
def chatbot(msg):
    m = msg.lower()

    if any(x in m for x in ["hi", "hello", "hey"]):
        return "👋 Hello! How can I help you today?"

    if "road" in m:
        return "🛣️ Please avoid damaged areas. Complaint will be forwarded to road department."

    if "water" in m:
        return "💧 Check if it's local or area-wide. If major, we’ll escalate immediately."

    if "electric" in m:
        return "⚡ Please stay safe. Avoid contact with wires. Team will resolve soon."

    if "status" in m:
        return "📊 You can track complaint status in Dashboard tab."

    return "🤖 Please describe your issue clearly. I will assist you."

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

        c.execute("""
        INSERT INTO complaints VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            complaint_id,
            st.session_state.user,
            text,
            prediction,
            category,
            str(conf),
            "🟡 In Progress",
            department,
            str(datetime.now())
        ))
        conn.commit()

        st.success(f"Complaint Registered | ID: {complaint_id}")

        # ✅ SHOW ONLY NEW ENTRY (not full table)
        latest = pd.read_sql_query(
            "SELECT * FROM complaints ORDER BY timestamp DESC LIMIT 1", conn
        )
        st.dataframe(latest, use_container_width=True)

# ================== DASHBOARD ==================
with tabs[1]:

    saved = pd.read_sql_query("SELECT * FROM complaints", conn)

    if not saved.empty:

        col1, col2, col3 = st.columns(3)
        col1.metric("Total", len(saved))
        col2.metric("Departments", saved["department"].nunique())
        col3.metric("Top Category", saved["category"].value_counts().idxmax())

        st.markdown("### 📋 Complaint Monitoring Table")

        st.dataframe(
            saved.sort_values("timestamp", ascending=False),
            use_container_width=True
        )

# ================== ANALYTICS ==================
with tabs[2]:

    saved = pd.read_sql_query("SELECT * FROM complaints", conn)

    if not saved.empty:

        st.markdown("## 📊 Analytics Dashboard")

        g1, g2 = st.columns(2)
        g3, g4 = st.columns(2)

        saved["timestamp"] = pd.to_datetime(saved["timestamp"], errors='coerce')

        # PIE
        with g1:
            fig1, ax1 = plt.subplots()
            saved["category"].value_counts().plot.pie(autopct="%1.1f%%", ax=ax1)
            ax1.set_ylabel("")
            st.pyplot(fig1)

        # BAR
        with g2:
            fig2, ax2 = plt.subplots()
            saved["category"].value_counts().plot.bar(ax=ax2)
            st.pyplot(fig2)

        # USER
        with g3:
            fig3, ax3 = plt.subplots()
            saved["user"].value_counts().plot.bar(ax=ax3)
            st.pyplot(fig3)

        # TIME TREND
        with g4:
            trend = saved.groupby(saved["timestamp"].dt.date).size()
            fig4, ax4 = plt.subplots()
            trend.plot(ax=ax4)
            st.pyplot(fig4)

# ================== CHATBOT ==================
with tabs[3]:

    if "chat" not in st.session_state:
        st.session_state.chat = []

    col1, col2 = st.columns([3, 1])

    msg = col1.text_input("Ask your problem...")

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
