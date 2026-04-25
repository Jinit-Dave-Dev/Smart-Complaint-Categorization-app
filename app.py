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

# OLD TABLE (unchanged)
c.execute("CREATE TABLE IF NOT EXISTS users (username TEXT, password TEXT)")
c.execute("""CREATE TABLE IF NOT EXISTS complaints (
    user TEXT,
    complaint TEXT,
    prediction TEXT,
    category TEXT,
    confidence TEXT
)""")

# 🔥 ADD NEW COLUMNS SAFELY (NO BREAK)
def safe_add_column(col, col_type):
    try:
        c.execute(f"ALTER TABLE complaints ADD COLUMN {col} {col_type}")
    except:
        pass

safe_add_column("id", "TEXT")
safe_add_column("status", "TEXT")
safe_add_column("department", "TEXT")
safe_add_column("timestamp", "TEXT")

conn.commit()

# -------------------- HELPERS --------------------
def generate_id():
    return "CMP-" + str(uuid.uuid4())[:8].upper()

def get_department(category):
    return {
        "Road": "Public Works",
        "Water": "Water Supply",
        "Garbage": "Sanitation",
        "Electricity": "Electric Dept"
    }.get(category, "General")

def get_status():
    return "NEW"

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

# -------------------- CHATBOT --------------------
def chatbot(msg):
    m = msg.lower()

    if any(x in m for x in ["hi", "hello"]):
        return "👋 Hello! Tell me your issue."

    if "road" in m:
        return "🛣️ Road repair team will fix it in 2-3 days."

    if "water" in m:
        return "💧 Water team will resolve within 24 hrs."

    if "electric" in m:
        return "⚡ Electricity issue will be fixed in 4-6 hrs."

    return "📌 Please submit complaint for tracking."

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

        conf = np.random.uniform(60, 90)

        cid = generate_id()
        dept = get_department(category)
        status = get_status()
        time_now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # 🔥 INSERT WITH NEW FIELDS
        c.execute("""
        INSERT INTO complaints 
        (user, complaint, prediction, category, confidence, id, status, department, timestamp)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (st.session_state.user, text, prediction, category, str(conf),
              cid, status, dept, time_now))

        conn.commit()

        st.success(f"Complaint Registered | ID: {cid}")

    # 🔥 USER VIEW ONLY
    user_df = pd.read_sql_query(
        f"SELECT id, complaint, category, status, department, timestamp FROM complaints WHERE user='{st.session_state.user}'",
        conn
    )

    if not user_df.empty:
        st.dataframe(user_df.iloc[::-1], use_container_width=True)

# ================== DASHBOARD ==================
with tabs[1]:

    saved = pd.read_sql_query("SELECT * FROM complaints", conn)

    if not saved.empty:

        st.markdown("### 🔍 Filter by Status")
        status_filter = st.selectbox("Select Status", ["All", "NEW", "IN PROGRESS", "RESOLVED"])

        if status_filter != "All":
            saved = saved[saved["status"] == status_filter]

        col1, col2, col3 = st.columns(3)
        col1.metric("Total", len(saved))
        col2.metric("Departments", saved["department"].nunique())
        col3.metric("Top Category", saved["category"].value_counts().idxmax())

        st.markdown("### 📡 Admin Complaint View")
        st.dataframe(saved.iloc[::-1], use_container_width=True)

# ================== ANALYTICS ==================
with tabs[2]:

    saved = pd.read_sql_query("SELECT * FROM complaints", conn)

    if not saved.empty:

        g1, g2 = st.columns(2)
        g3, g4 = st.columns(2)

        with g1:
            fig, ax = plt.subplots()
            saved["category"].value_counts().plot.pie(ax=ax, autopct="%1.1f%%")
            ax.set_ylabel("")
            st.pyplot(fig)

        with g2:
            fig, ax = plt.subplots()
            saved["department"].value_counts().plot.bar(ax=ax)
            st.pyplot(fig)

        with g3:
            fig, ax = plt.subplots()
            saved["status"].value_counts().plot.bar(ax=ax)
            st.pyplot(fig)

        with g4:
            fig, ax = plt.subplots()
            saved.groupby("category").size().cumsum().plot(ax=ax)
            st.pyplot(fig)

# ================== CHATBOT ==================
with tabs[3]:

    if "chat" not in st.session_state:
        st.session_state.chat = []

    msg = st.text_input("Ask anything...")

    if msg:
        st.session_state.chat.append(("You", msg))
        st.session_state.chat.append(("Bot", chatbot(msg)))

    for r, m in st.session_state.chat:
        st.markdown(f"**{r}:** {m}")
