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

        # -------- FIX DATA --------
        saved.fillna({
            "category": "Other",
            "department": "General",
            "status": "Pending",
            "confidence": "0"
        }, inplace=True)

        saved["confidence"] = pd.to_numeric(saved["confidence"], errors="coerce")

        st.markdown("## 📊 Smart Analytics Dashboard (Interactive)")

        # -------- SESSION STATE (for click simulation) --------
        if "selected_category" not in st.session_state:
            st.session_state.selected_category = "All"

        if "selected_department" not in st.session_state:
            st.session_state.selected_department = "All"

        # -------- FILTER UI (acts like click) --------
        f1, f2 = st.columns(2)

        with f1:
            st.session_state.selected_category = st.selectbox(
                "🎯 Click Category",
                ["All"] + list(saved["category"].unique()),
                index=0
            )

        with f2:
            st.session_state.selected_department = st.selectbox(
                "🏢 Click Department",
                ["All"] + list(saved["department"].unique()),
                index=0
            )

        # -------- APPLY FILTER --------
        filtered = saved.copy()

        if st.session_state.selected_category != "All":
            filtered = filtered[filtered["category"] == st.session_state.selected_category]

        if st.session_state.selected_department != "All":
            filtered = filtered[filtered["department"] == st.session_state.selected_department]

        # -------- KPI --------
        k1, k2, k3, k4 = st.columns(4)

        k1.metric("Total", len(filtered))
        k2.metric("Users", filtered["user"].nunique() if not filtered.empty else 0)

        top_cat = filtered["category"].value_counts().idxmax() if not filtered.empty else "N/A"
        top_dep = filtered["department"].value_counts().idxmax() if not filtered.empty else "N/A"

        k3.metric("Top Category", top_cat)
        k4.metric("Top Department", top_dep)

        # -------- GRID --------
        g1, g2 = st.columns(2)
        g3, g4 = st.columns(2)

        # ================= PIE =================
        with g1:
            st.markdown("### 🥧 Category Distribution")

            fig1, ax1 = plt.subplots()

            cat_counts = filtered["category"].value_counts()

            colors = [
                "orange" if c == st.session_state.selected_category else "skyblue"
                for c in cat_counts.index
            ]

            cat_counts.plot.pie(
                autopct="%1.1f%%",
                colors=colors,
                ax=ax1
            )
            ax1.set_ylabel("")
            st.pyplot(fig1)

        # ================= BAR (HIGHLIGHT) =================
        with g2:
            st.markdown("### 📊 Department Load")

            fig2, ax2 = plt.subplots()

            dep_counts = filtered["department"].value_counts()

            colors = [
                "red" if d == st.session_state.selected_department else "gray"
                for d in dep_counts.index
            ]

            dep_counts.plot.bar(ax=ax2, color=colors)
            st.pyplot(fig2)

        # ================= HIST =================
        with g3:
            st.markdown("### 📈 Confidence Distribution")

            fig3, ax3 = plt.subplots()

            filtered["confidence"].dropna().plot.hist(bins=10, ax=ax3)

            st.pyplot(fig3)

        # ================= STATUS =================
        with g4:
            st.markdown("### 🚦 Status Overview")

            fig4, ax4 = plt.subplots()

            status_counts = filtered["status"].value_counts()

            status_counts.plot.bar(ax=ax4)

            st.pyplot(fig4)

        # -------- DRILL DOWN --------
        st.markdown("### 📋 Drill-down Data")

        st.dataframe(filtered.iloc[::-1], use_container_width=True)

        # -------- RESET BUTTON --------
        if st.button("🔄 Reset Filters"):
            st.session_state.selected_category = "All"
            st.session_state.selected_department = "All"
            st.rerun()

    else:
        st.warning("No data available yet.")

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
