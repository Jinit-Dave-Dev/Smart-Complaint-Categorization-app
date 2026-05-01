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

/* REMOVE DEFAULT TOP SPACE */
.block-container {
    padding-top: 1rem !important;
}

/* FORCE BACKGROUND ON ROOT */
html, body, .stApp {
    height: 100%;
}

.stApp {
    background: linear-gradient(rgba(10,35,70,0.55), rgba(10,35,70,0.55)),
                url("https://picsum.photos/1920/1080")
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}

/* CENTER COLUMN ONLY (THIS FIXES DOUBLE CARD) */
div[data-testid="stHorizontalBlock"] > div:nth-child(2) {
    display: flex;
    justify-content: center;
}

/* ACTUAL CARD */
div[data-testid="stHorizontalBlock"] > div:nth-child(2) > div {
    background: rgba(255,255,255,0.08);
    backdrop-filter: blur(20px);
    padding: 35px;
    border-radius: 18px;
    box-shadow: 0 10px 40px rgba(0,0,0,0.4);
    width: 420px;
    margin-top: 8vh;
}

/* TITLE */
.title {
    text-align: center;
    font-size: 26px;
    color: white;
    margin-bottom: 20px;
}

/* INPUT */
.stTextInput input {
    border-radius: 10px;
}

/* BUTTON */
.stButton button {
    width: 100%;
    border-radius: 10px;
}

/* REMOVE TAB WHITE BG */
[data-baseweb="tab-panel"] {
    background: transparent !important;
}

/* TAB ANIMATION */
button[role="tab"] {
    transition: all 0.3s ease !important;
    border-radius: 8px !important;
}

/* ACTIVE TAB */
button[aria-selected="true"] {
    background: linear-gradient(135deg, #1f4e79, #4da6ff) !important;
    color: white !important;
    transform: scale(1.05);
}

/* HOVER EFFECT */
button[role="tab"]:hover {
    transform: scale(1.05);
    background: rgba(255,255,255,0.1) !important;
}

/* INPUT ANIMATION */
.stTextInput input {
    border-radius: 10px;
    transition: all 0.3s ease;
}

/* INPUT FOCUS GLOW */
.stTextInput input:focus {
    border: 1px solid #4da6ff !important;
    box-shadow: 0 0 10px rgba(77,166,255,0.6);
    transform: scale(1.02);
}

/* BUTTON ANIMATION */
.stButton button {
    width: 100%;
    border-radius: 10px;
    transition: all 0.2s ease;
}

/* BUTTON CLICK EFFECT */
.stButton button:active {
    transform: scale(0.96);
}

/* BUTTON HOVER */
.stButton button:hover {
    background: linear-gradient(135deg, #1f4e79, #4da6ff);
    color: white;
    transform: scale(1.03);
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

    col1, col2, col3 = st.columns([1, 1.2, 1])

    with col2:

        st.markdown(
            '<div class="title">🏛️ Smart Government Complaint Portal</div>',
            unsafe_allow_html=True
        )

        tab1, tab2 = st.tabs(["🔐 Login", "📝 Register"])

        # LOGIN
        with tab1:
            u = st.text_input("Username", key="login_user")
            p = st.text_input("Password", type="password", key="login_pass")

            if st.button("Login", use_container_width=True):
                c.execute("SELECT * FROM users WHERE username=? AND password=?", (u, p))
                if c.fetchone():
                    st.session_state.logged_in = True
                    st.session_state.user = u
                    st.rerun()
                else:
                    st.error("Invalid Credentials")

        # REGISTER
        with tab2:
            ru = st.text_input("New Username", key="reg_user")
            rp = st.text_input("New Password", type="password", key="reg_pass")

            if st.button("Register", use_container_width=True):
                if ru and rp:
                    c.execute("INSERT INTO users VALUES (?,?)", (ru, rp))
                    conn.commit()
                    st.success("Registered Successfully")
                else:
                    st.warning("Enter all fields")

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
def seed_data():
    existing = pd.read_sql_query("SELECT COUNT(*) as cnt FROM complaints", conn)
    if existing["cnt"][0] >= 50:
     return

    sample_data = [
        # ROAD
        ("user1","There is a large pothole on the main road near my house causing accidents."),
        ("user2","Street roads in our area are completely broken and need repair."),
        ("user3","Road construction work has been incomplete for months."),
        ("user4","Waterlogging on roads during rain makes it difficult to travel."),
        ("user5","Street lights on the road are not working properly at night."),
        ("user6","Uneven road surface causing damage to vehicles."),
        ("user7","Road divider is broken and dangerous for traffic."),
        ("user8","No proper signage on newly constructed roads."),
        ("user9","Heavy traffic congestion due to poor road conditions."),
        ("user10","Sidewalks are damaged and unsafe for pedestrians."),

        # WATER
        ("user11","No water supply in our area for the last 2 days."),
        ("user12","Water coming from taps is dirty and smells bad."),
        ("user13","Leakage in water pipeline causing wastage."),
        ("user14","Low water pressure in residential area."),
        ("user15","Water tank overflow in public area."),
        ("user16","Drinking water supply is irregular."),
        ("user17","Broken water pipe flooding the street."),
        ("user18","Water contamination issue in locality."),
        ("user19","Water supply timing is not consistent."),
        ("user20","No water connection in newly developed area."),

        # ELECTRICITY
        ("user21","Frequent power cuts in our area."),
        ("user22","Street lights are not working at night."),
        ("user23","Transformer failure causing blackout."),
        ("user24","Electric wires are hanging dangerously."),
        ("user25","Sudden voltage fluctuation damaging appliances."),
        ("user26","No electricity for last 5 hours."),
        ("user27","Power outage during night time frequently."),
        ("user28","Electric pole is tilted and unsafe."),
        ("user29","Street light flickering continuously."),
        ("user30","Unauthorized power connections in area."),

        # GARBAGE
        ("user31","Garbage not collected regularly in our area."),
        ("user32","Overflowing dustbins causing bad smell."),
        ("user33","Waste is dumped on roadside without cleaning."),
        ("user34","Dead animal lying on street for days."),
        ("user35","Open drainage causing hygiene issues."),
        ("user36","Mosquito breeding due to garbage accumulation."),
        ("user37","Public area is very dirty and unhygienic."),
        ("user38","No garbage bins installed in locality."),
        ("user39","Sewage water overflowing on roads."),
        ("user40","Irregular cleaning of streets by municipal workers."),

        # OTHERS
        ("user41","Noise pollution due to construction work at night."),
        ("user42","Illegal parking blocking roads."),
        ("user43","Stray animals creating nuisance in area."),
        ("user44","Public park is not maintained properly."),
        ("user45","Street vendors blocking footpaths."),
        ("user46","Broken public benches in park."),
        ("user47","No proper street lighting in park area."),
        ("user48","Encroachment on public land."),
        ("user49","Lack of security in residential area."),
        ("user50","Government office staff not responding to complaints.")
    ]

    for user, text in sample_data:
        X = vectorizer.transform([text])
        pred = model.predict(X)
        prediction = le.inverse_transform(pred)[0]

        category = prediction
        confidence = str(round(model.predict_proba(X).max() * 100, 2))

        c.execute("""
        INSERT INTO complaints (
            user, complaint, prediction, category, confidence
        ) VALUES (?, ?, ?, ?, ?)
        """, (user, text, prediction, category, confidence))

    conn.commit()
    
seed_data()

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

    data = pd.read_sql_query("SELECT * FROM complaints", conn)
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
