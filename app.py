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

def login():

    # 🔥 BACKGROUND IMAGE + UI STYLE
    st.markdown(f"""
    <style>
    .stApp {{
        background: url("https://images.unsplash.com/photo-1581092918056-0c4c3acd3789");
        background-size: cover;
    }}

    .login-box {{
        background: rgba(0,0,0,0.7);
        padding: 40px;
        border-radius: 15px;
        width: 400px;
        margin: auto;
        margin-top: 100px;
    }}

    .title {{
        text-align: center;
        font-size: 24px;
        color: white;
        margin-bottom: 20px;
    }}
    </style>
    """, unsafe_allow_html=True)

    # 🔥 CENTER BOX
    st.markdown('<div class="login-box">', unsafe_allow_html=True)

    st.markdown('<div class="title">🏛️ SMART COMPLAINT CATEGORIZATION GOVERNMENT PORTAL</div>', unsafe_allow_html=True)

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

    st.markdown('</div>', unsafe_allow_html=True)

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

# -------------------- ADVANCED CHATBOT --------------------
def chatbot(msg):
    m = msg.lower()

    if any(x in m for x in ["hi","hello","hey"]):
        return "👋 Welcome to Smart Municipal System.\nYou can report issues like road, water, electricity, garbage."

    if "road" in m:
        return "🛣️ Road Issue:\n• Complaint registered\n• Inspection team assigned\n• Expected resolution: 2-3 days"

    if "water" in m:
        return "💧 Water Issue:\n• Pipeline team alerted\n• Emergency fix scheduled\n• ETA: 24 hours"

    if "electric" in m:
        return "⚡ Electricity Issue:\n• Technician dispatched\n• Expected fix: 4-6 hours"

    if "garbage" in m:
        return "🗑️ Garbage Issue:\n• Cleaning team scheduled\n• Pickup within 12 hours"

    if "status" in m:
        return "📊 Track your complaint using Tracking ID in Track Complaint tab."

    return "📌 Please submit your complaint in Complaint tab for full tracking."

# -------------------- UI --------------------
st.title("🏛️ Smart Municipal Complaint System")

tabs = st.tabs([
    "📝 Complaint", 
    "📊 Dashboard", 
    "📈 Analytics", 
    "🤖 Chatbot",
    "🔍 Track Complaint",
    "🛠️ Admin Panel"  # NEW
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

        conf = round(model.predict_proba(X).max() * 100, 2)
        tracking_id = str(uuid.uuid4())[:8]

        c.execute("""
        INSERT INTO complaints (
            id, user, complaint, prediction, category, confidence,
            status, department, timestamp
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
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

    data = pd.read_sql_query("SELECT * FROM complaints WHERE user=?", conn, params=(st.session_state.user,))
    if not data.empty:
        st.dataframe(data.iloc[::-1], use_container_width=True)

# ================== DASHBOARD ==================
with tabs[1]:

    saved = pd.read_sql_query("SELECT * FROM complaints", conn)

    if not saved.empty:
        saved.fillna("N/A", inplace=True)

        col1, col2, col3 = st.columns(3)
        col1.metric("Total", len(saved))
        col2.metric("Users", saved["user"].nunique())
        col3.metric("Top Category", saved["category"].value_counts().idxmax())

        st.dataframe(saved.iloc[::-1], use_container_width=True)

# ================== ANALYTICS (UPGRADED + CLEAN + NO EMPTY CHARTS) ==================
with tabs[2]:

    saved = pd.read_sql_query("SELECT * FROM complaints", conn)

    if not saved.empty:

        # 🔥 FIX NULL VALUES
        saved.fillna({
            "priority": "🟡 MEDIUM",
            "status": "Pending",
            "department": "General",
            "timestamp": str(datetime.now())
        }, inplace=True)

        # 🔥 FIX TIMESTAMP
        saved["timestamp"] = pd.to_datetime(saved["timestamp"], errors="coerce")

        st.markdown("## 📊 Smart Analytics Dashboard")

        # KPI ROW
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Total", len(saved))
        k2.metric("Users", saved["user"].nunique())
        k3.metric("Top Category", saved["category"].value_counts().idxmax())
        k4.metric("Departments", saved["department"].nunique())

        # GRID
        g1, g2 = st.columns(2)
        g3, g4 = st.columns(2)

        # 🔵 CATEGORY PIE
        with g1:
            st.markdown("### 🥧 Category Distribution")
            fig1, ax1 = plt.subplots()
            saved["category"].value_counts().plot.pie(
                autopct="%1.1f%%",
                ax=ax1
            )
            ax1.set_ylabel("")
            st.pyplot(fig1)

        # 🟣 STATUS DISTRIBUTION
        with g2:
            st.markdown("### 🚦 Status Overview")
            fig2, ax2 = plt.subplots()
            saved["status"].value_counts().plot.bar(ax=ax2)
            st.pyplot(fig2)

        # 🟢 DEPARTMENT LOAD
        with g3:
            st.markdown("### 🏢 Department Load")
            fig3, ax3 = plt.subplots()
            saved["department"].value_counts().plot.bar(ax=ax3)
            st.pyplot(fig3)

        # 🔴 TIME TREND (FIXED)
        with g4:
            st.markdown("### 📈 Complaints Over Time")

            trend = saved.dropna(subset=["timestamp"])
            trend = trend.groupby(trend["timestamp"].dt.date).size()

            fig4, ax4 = plt.subplots()
            trend.plot(ax=ax4)
            st.pyplot(fig4)

        # 🔥 BONUS: FILTER (makes it feel dynamic)
        st.markdown("### 🔍 Filter Data")

        selected_cat = st.selectbox(
            "Filter by Category",
            ["All"] + list(saved["category"].unique())
        )

        if selected_cat != "All":
            filtered = saved[saved["category"] == selected_cat]
        else:
            filtered = saved

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
            st.dataframe(result, use_container_width=True)
        else:
            st.error("Not Found")

# ================== ADMIN PANEL ==================
with tabs[5]:

    st.subheader("🛠️ Admin Panel - Update Status")

    data = pd.read_sql_query("SELECT * FROM complaints", conn)

    if not data.empty:

        selected_id = st.selectbox("Select Complaint ID", data["id"].dropna().unique())

        new_status = st.selectbox("Update Status", ["Pending", "In Progress", "Resolved"])

        if st.button("Update Status"):
            c.execute("UPDATE complaints SET status=? WHERE id=?", (new_status, selected_id))
            conn.commit()
            st.success("Status Updated")

        st.dataframe(data, use_container_width=True)
