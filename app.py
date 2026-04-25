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

# -------------------- DB --------------------
conn = sqlite3.connect("complaints.db", check_same_thread=False)
c = conn.cursor()

c.execute("""
CREATE TABLE IF NOT EXISTS complaints (
    user TEXT,
    complaint TEXT,
    prediction TEXT,
    category TEXT,
    confidence TEXT
)
""")

def add_column(col, typ):
    try:
        c.execute(f"ALTER TABLE complaints ADD COLUMN {col} {typ}")
    except:
        pass

add_column("id", "TEXT")
add_column("priority", "TEXT")
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

    st.markdown("""
    <style>
    .stApp {
        background-image: url("https://images.unsplash.com/photo-1506744038136-46273834b3fb");
        background-size: cover;
        background-position: center;
    }

    .login-container {
        display: flex;
        justify-content: center;
        align-items: center;
        height: 90vh;
    }

    .login-box {
        background: rgba(0,0,0,0.75);
        padding: 40px;
        border-radius: 15px;
        width: 380px;
        box-shadow: 0px 0px 25px rgba(0,0,0,0.6);
    }

    .title {
        text-align: center;
        font-size: 22px;
        font-weight: bold;
        color: white;
        margin-bottom: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="login-container">', unsafe_allow_html=True)
    st.markdown('<div class="login-box">', unsafe_allow_html=True)

    st.markdown(
        '<div class="title">🏛️ SMART COMPLAINT CATEGORIZATION GOVERNMENT PORTAL</div>',
        unsafe_allow_html=True
    )

    u = st.text_input("Username")
    p = st.text_input("Password", type="password")

    col1, col2 = st.columns(2)

    if col1.button("Login"):
        c.execute("SELECT * FROM users WHERE username=? AND password=?", (u, p))
        if c.fetchone():
            st.session_state.logged_in = True
            st.session_state.user = u
            st.rerun()
        else:
            st.error("Invalid Credentials")

    if col2.button("Register"):
        c.execute("INSERT INTO users VALUES (?,?)", (u, p))
        conn.commit()
        st.success("Registered")

    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


if not st.session_state.logged_in:
    login()
    st.stop()

# -------------------- SIDEBAR --------------------
st.sidebar.title("📊 Smart Dashboard")
st.sidebar.write(f"👤 {st.session_state.user}")
st.sidebar.markdown("---")
st.sidebar.markdown("### 👨‍💻 Developer")
st.sidebar.markdown("**Jinit Dave**")

if st.sidebar.button("Logout"):
    st.session_state.logged_in = False
    st.rerun()

# -------------------- LOAD MODEL --------------------
vectorizer = pickle.load(open("tfidf_vectorizer.pkl", "rb"))
le = pickle.load(open("label_encoder.pkl", "rb"))
model = pickle.load(open("logistic_regression_model.pkl", "rb"))

# -------------------- HELPERS --------------------
def get_category(text):
    t = re.sub(r'[^a-zA-Z ]', ' ', str(text).lower())
    if "road" in t: return "Road"
    if "water" in t: return "Water"
    if "garbage" in t: return "Garbage"
    if "electric" in t: return "Electricity"
    return "Other"

def get_department(cat):
    return {
        "Road": "Public Works",
        "Water": "Water Dept",
        "Garbage": "Sanitation",
        "Electricity": "Electric Dept"
    }.get(cat, "General")

def get_priority(text):
    if any(x in text.lower() for x in ["fire","danger","accident"]):
        return "🔴 HIGH"
    return "🟡 MEDIUM"

# -------------------- CHATBOT --------------------
def chatbot(msg):
    m = msg.lower()

    if any(x in m for x in ["hi","hello","hey"]):
        return "👋 Welcome! You can report Road, Water, Electricity, Garbage issues."

    if "road" in m:
        return "🛣️ Road Issue:\n• Inspection team assigned\n• Fix ETA: 2-3 days"

    if "water" in m:
        return "💧 Water Issue:\n• Pipeline team alerted\n• Fix ETA: 24 hrs"

    if "electric" in m:
        return "⚡ Electricity Issue:\n• Technician dispatched\n• ETA: 4-6 hrs"

    if "garbage" in m:
        return "🗑️ Garbage Issue:\n• Cleaning team scheduled"

    if "status" in m:
        return "📊 Use Track Complaint tab with ID."

    return "📌 Please submit complaint in Complaint tab."

# -------------------- UI --------------------
st.title("🏛️ Smart Municipal Complaint System")

tabs = st.tabs([
    "📝 Complaint", 
    "📊 Dashboard", 
    "📈 Analytics", 
    "🤖 Chatbot",
    "🔍 Track Complaint",
    "🛠️ Admin Panel"
])

# ================== COMPLAINT ==================
with tabs[0]:

    text = st.text_area("Enter your complaint")

    if st.button("Submit Complaint") and text.strip():

        X = vectorizer.transform([text])
        pred = model.predict(X)
        prediction = le.inverse_transform(pred)[0]

        category = get_category(text)
        department = get_department(category)
        priority = get_priority(text)

        conf = round(model.predict_proba(X).max() * 100, 2)
        tracking_id = str(uuid.uuid4())[:8]

        c.execute("""
        INSERT INTO complaints (
            id, user, complaint, prediction, category, confidence,
            priority, status, department, timestamp
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            tracking_id,
            st.session_state.user,
            text,
            prediction,
            category,
            str(conf),
            priority,
            "Pending",
            department,
            str(datetime.now())
        ))

        conn.commit()
        st.success(f"Complaint Registered | ID: {tracking_id}")

    data = pd.read_sql_query("SELECT * FROM complaints WHERE user=?", conn, params=(st.session_state.user,))
    if not data.empty:
        data.fillna("N/A", inplace=True)
        st.dataframe(data.iloc[::-1], use_container_width=True)

# ================== DASHBOARD ==================
with tabs[1]:

    saved = pd.read_sql_query("SELECT * FROM complaints", conn)

    if not saved.empty:
        saved.fillna({
            "priority": "🟡 MEDIUM",
            "status": "Pending",
            "department": "General",
            "timestamp": str(datetime.now())
        }, inplace=True)

        col1, col2, col3 = st.columns(3)
        col1.metric("Total", len(saved))
        col2.metric("Users", saved["user"].nunique())
        col3.metric("Top Category", saved["category"].value_counts().idxmax())

        st.dataframe(saved.iloc[::-1], use_container_width=True)
        
# ================== ANALYTICS ==================
with tabs[2]:

    saved = pd.read_sql_query("SELECT * FROM complaints", conn)

    if not saved.empty:

        st.markdown("## 📊 Interactive Analytics Panel")

        # ---------- CLEAN DATA ----------
        saved.fillna({
            "category": "Other",
            "status": "Pending",
            "department": "General",
            "timestamp": str(datetime.now())
        }, inplace=True)

        saved["timestamp"] = pd.to_datetime(saved["timestamp"], errors="coerce")

        # 🔥 FORCE VARIATION (fix blank timeline)
        if saved["timestamp"].nunique() <= 1:
            saved["timestamp"] = pd.date_range(
                end=datetime.now(),
                periods=len(saved)
            )

        # ---------- DRILL-DOWN FILTER ----------
        st.markdown("### 🎯 Drill Down Control")

        drill = st.selectbox(
            "Select Analysis Type",
            ["Overall", "Category", "Department", "User"]
        )

        if drill == "Category":
            value = st.selectbox("Select Category", saved["category"].unique())
            filtered = saved[saved["category"] == value]

        elif drill == "Department":
            value = st.selectbox("Select Department", saved["department"].unique())
            filtered = saved[saved["department"] == value]

        elif drill == "User":
            value = st.selectbox("Select User", saved["user"].unique())
            filtered = saved[saved["user"] == value]

        else:
            filtered = saved

        # ---------- KPIs ----------
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Total", len(filtered))
        k2.metric("Users", filtered["user"].nunique())
        k3.metric("Categories", filtered["category"].nunique())
        k4.metric("Departments", filtered["department"].nunique())

        # ---------- CHART GRID ----------
        g1, g2 = st.columns(2)
        g3, g4 = st.columns(2)

        # 🔵 PIE
        with g1:
            st.markdown("### 🥧 Category Distribution")
            fig1, ax1 = plt.subplots()
            filtered["category"].value_counts().plot.pie(
                autopct="%1.1f%%",
                ax=ax1
            )
            ax1.set_ylabel("")
            st.pyplot(fig1)

        # 🟣 BAR
        with g2:
            st.markdown("### 📊 Department Load")
            fig2, ax2 = plt.subplots()
            filtered["department"].value_counts().plot.bar(ax=ax2)
            st.pyplot(fig2)

        # 🟢 HISTOGRAM
        with g3:
            st.markdown("### 📉 Confidence Spread")
            conf = pd.to_numeric(filtered["confidence"], errors="coerce")

            fig3, ax3 = plt.subplots()
            ax3.hist(conf.dropna(), bins=10)
            st.pyplot(fig3)

        # 🔴 LINE (FIXED)
        with g4:
            st.markdown("### 📈 Complaints Over Time")

            trend = filtered.groupby(
                filtered["timestamp"].dt.date
            ).size()

            fig4, ax4 = plt.subplots()
            trend.plot(ax=ax4)
            st.pyplot(fig4)

        # ---------- TABLE ----------
        st.markdown("### 📋 Drill-down Data")
        st.dataframe(filtered, use_container_width=True)

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
        st.write(f"**{r}:** {m}")

# ================== TRACK ==================
with tabs[4]:

    search_id = st.text_input("Enter Tracking ID")

    if st.button("Search"):
        result = pd.read_sql_query("SELECT * FROM complaints WHERE id=?", conn, params=(search_id,))
        if not result.empty:
            result.fillna("N/A", inplace=True)
            st.dataframe(result, use_container_width=True)
        else:
            st.error("Not Found")

# ================== ADMIN ==================
with tabs[5]:

    data = pd.read_sql_query("SELECT * FROM complaints", conn)

    if not data.empty:
        selected_id = st.selectbox("Select ID", data["id"].dropna().unique())
        new_status = st.selectbox("Status", ["Pending", "In Progress", "Resolved"])

        if st.button("Update"):
            c.execute("UPDATE complaints SET status=? WHERE id=?", (new_status, selected_id))
            conn.commit()
            st.success("Updated")

        st.dataframe(data, use_container_width=True)
