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
# -------------------- GLOBAL LOGIN UI STYLE --------------------
st.markdown("""
<style>

/* Remove top spacing */
.block-container {
    padding-top: 0rem;
}

/* Background Image */
.stApp {
    background: url("https://images.unsplash.com/photo-1605902711622-cfb43c44367f") no-repeat center center fixed;
    background-size: cover;
}

/* Dark Overlay */
.overlay {
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background: rgba(0, 40, 80, 0.75);
    z-index: 0;
}

/* Center Card */
.center-box {
    position: fixed;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    z-index: 1;
}

/* Card */
.card {
    background: rgba(255,255,255,0.96);
    padding: 35px;
    border-radius: 15px;
    width: 380px;
    box-shadow: 0px 10px 40px rgba(0,0,0,0.4);
}

/* Title */
.title {
    text-align: center;
    font-size: 22px;
    font-weight: bold;
    color: #1f4e79;
    margin-bottom: 20px;
}

/* Buttons */
.stButton button {
    width: 100%;
    border-radius: 8px;
    background-color: #1f4e79;
    color: white;
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

    st.markdown('<div class="overlay"></div>', unsafe_allow_html=True)
    st.markdown('<div class="center-box">', unsafe_allow_html=True)
    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.markdown(
        '<div class="title">🏛️ Government Complaint Portal</div>',
        unsafe_allow_html=True
    )

    # -------- TABS --------
    tab1, tab2 = st.tabs(["🔐 Login", "📝 Register"])

    # ===== LOGIN =====
    with tab1:
        u = st.text_input("Username", key="login_user")
        p = st.text_input("Password", type="password", key="login_pass")

        if st.button("Login"):
            c.execute(
                "SELECT * FROM users WHERE username=? AND password=?",
                (u, p)
            )
            if c.fetchone():
                st.session_state.logged_in = True
                st.session_state.user = u
                st.rerun()
            else:
                st.error("Invalid Credentials")

    # ===== REGISTER =====
    with tab2:
        ru = st.text_input("New Username", key="reg_user")
        rp = st.text_input("New Password", type="password", key="reg_pass")

        if st.button("Register"):
            if ru and rp:
                c.execute("INSERT INTO users VALUES (?,?)", (ru, rp))
                conn.commit()
                st.success("Registered Successfully")
            else:
                st.warning("Enter all fields")

    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # -------- INPUTS (IMPORTANT: keys added) --------
    u = st.text_input("Username", key="login_user")
    p = st.text_input("Password", type="password", key="login_pass")

    col1, col2 = st.columns(2)

    # -------- LOGIN --------
    if col1.button("Login"):
        c.execute("SELECT * FROM users WHERE username=? AND password=?", (u, p))
        if c.fetchone():
            st.session_state.logged_in = True
            st.session_state.user = u
            st.rerun()
        else:
            st.error("Invalid Credentials")

    # -------- REGISTER --------
    if col2.button("Register"):
        if u and p:
            c.execute("INSERT INTO users VALUES (?,?)", (u, p))
            conn.commit()
            st.success("Registered Successfully")
        else:
            st.warning("Enter Username & Password")

    st.markdown("</div></div>", unsafe_allow_html=True)


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

    if saved.empty:
        st.warning("No data available yet.")
    else:

        # -------- CLEAN DATA --------
        saved.fillna({
            "category": "Other",
            "department": "General",
            "status": "Pending",
            "confidence": "0"
        }, inplace=True)

        saved["confidence"] = pd.to_numeric(saved["confidence"], errors="coerce")

        st.markdown("## 📊 Smart Analytics Dashboard")

        # -------- FILTERS --------
        col1, col2 = st.columns(2)

        with col1:
            category_filter = st.selectbox(
                "Filter by Category",
                ["All"] + sorted(saved["category"].unique())
            )

        with col2:
            dept_filter = st.selectbox(
                "Filter by Department",
                ["All"] + sorted(saved["department"].unique())
            )

        # -------- APPLY FILTER --------
        filtered = saved.copy()

        if category_filter != "All":
            filtered = filtered[filtered["category"] == category_filter]

        if dept_filter != "All":
            filtered = filtered[filtered["department"] == dept_filter]

        # -------- SAFE KPIs --------
        total = len(filtered)

        k1, k2, k3, k4 = st.columns(4)

        k1.metric("Total Complaints", total)
        k2.metric("Users", filtered["user"].nunique() if total else 0)

        k3.metric(
            "Top Category",
            filtered["category"].value_counts().idxmax()
            if total else "N/A"
        )

        k4.metric(
            "Top Department",
            filtered["department"].value_counts().idxmax()
            if total else "N/A"
        )

        # -------- GRID --------
        g1, g2 = st.columns(2)
        g3, g4 = st.columns(2)

        # ================= PIE =================
        with g1:
            st.markdown("### 🥧 Category Distribution")

            fig1, ax1 = plt.subplots(figsize=(4,4))

            if total:
                filtered["category"].value_counts().plot.pie(
                    autopct="%1.1f%%",
                    ax=ax1
                )
                ax1.set_ylabel("")
            else:
                ax1.text(0.5, 0.5, "No Data", ha='center')

            st.pyplot(fig1)

        # ================= BAR =================
        with g2:
            st.markdown("### 📊 Department Load")

            fig2, ax2 = plt.subplots(figsize=(4,4))

            if total:
                filtered["department"].value_counts().plot.bar(ax=ax2)
            else:
                ax2.text(0.5, 0.5, "No Data", ha='center')

            st.pyplot(fig2)

        # ================= HIST =================
        with g3:
            st.markdown("### 📈 Confidence Distribution")

            fig3, ax3 = plt.subplots(figsize=(4,4))

            if total and filtered["confidence"].notna().sum() > 0:
                filtered["confidence"].dropna().plot.hist(bins=10, ax=ax3)
            else:
                ax3.text(0.5, 0.5, "No Data", ha='center')

            st.pyplot(fig3)

        # ================= STATUS =================
        with g4:
            st.markdown("### 🚦 Complaint Status")

            fig4, ax4 = plt.subplots(figsize=(4,4))

            if total:
                filtered["status"].value_counts().plot.bar(ax=ax4)
            else:
                ax4.text(0.5, 0.5, "No Data", ha='center')

            st.pyplot(fig4)

        # -------- TABLE --------
        st.markdown("### 📋 Filtered Data")

        if total:
            st.dataframe(filtered.iloc[::-1], use_container_width=True)
        else:
            st.info("No data for selected filters.")

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
