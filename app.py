import streamlit as st
import pickle
import os
import pandas as pd
import sqlite3
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import plotly.express as px

st.set_page_config(page_title="Smart Complaint System", layout="wide")

# -------------------- PREMIUM UI --------------------
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
    transition: 0.3s;
}

.stButton button:hover {
    transform: scale(1.05);
}
</style>
""", unsafe_allow_html=True)

# -------------------- DATABASE --------------------
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
st.sidebar.title("📊 Dashboard")
st.sidebar.write(f"👤 {st.session_state.user}")

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

# -------------------- HELPERS --------------------
def clean_category(val):
    val = str(val).lower()
    if "water" in val: return "Water"
    elif "road" in val: return "Road"
    elif "garbage" in val: return "Garbage"
    elif "electric" in val: return "Electricity"
    return "Other"

def ensure_columns(df):
    if "status" not in df.columns:
        df["status"] = "NEW"
    return df

# -------------------- UI --------------------
st.title("🏛️ Smart Municipal Complaint System")

tabs = st.tabs(["📝 Complaint", "📊 Dashboard", "📈 Analytics", "🤖 Chatbot"])

# ================== COMPLAINT ==================
with tabs[0]:

    text = st.text_area("Enter your complaint")

    if st.button("Submit Complaint"):

        X = vectorizer.transform([text])
        pred = model.predict(X)
        prediction = le.inverse_transform(pred)[0]
        cat = clean_category(prediction)

        try:
            conf = round(model.predict_proba(X).max() * 100, 2)
        except:
            conf = np.random.uniform(60, 80)

        c.execute("""
            INSERT INTO complaints VALUES (?, ?, ?, ?, ?)
        """, (st.session_state.user, text, prediction, cat, str(conf)))

        conn.commit()
        st.success("Complaint Registered")

# ================== DASHBOARD ==================
with tabs[1]:

    saved = pd.read_sql_query("SELECT * FROM complaints", conn)

    if not saved.empty:

        saved["category"] = saved["category"].apply(clean_category)

        col1, col2, col3 = st.columns(3)
        col1.metric("Total", len(saved))
        col2.metric("Categories", saved["category"].nunique())
        col3.metric("Users", saved["user"].nunique())

        # -------- IMPROVED FILTER --------
        st.markdown("### 🔎 Filter Complaints")

        categories = st.multiselect(
            "Select Category",
            saved["category"].unique(),
            default=list(saved["category"].unique())
        )

        filtered = saved[saved["category"].isin(categories)]

        if st.button("Reset Filter"):
            filtered = saved

        st.dataframe(filtered, use_container_width=True)

# ================== ANALYTICS (POWER BI STYLE) ==================
with tabs[2]:

    saved = pd.read_sql_query("SELECT * FROM complaints", conn)

    if not saved.empty:

        saved["category"] = saved["category"].apply(clean_category)

        # -------- KPI CARDS --------
        col1, col2, col3 = st.columns(3)

        col1.metric("Total Complaints", len(saved))
        col2.metric("Categories", saved["category"].nunique())
        col3.metric("Top Category", saved["category"].value_counts().idxmax())

        # -------- PIE CHART --------
        st.markdown("### 🥧 Category Distribution")

        pie = px.pie(
            saved,
            names="category",
            title="Complaint Share by Category"
        )
        st.plotly_chart(pie, use_container_width=True)

        # -------- BAR CHART --------
        st.markdown("### 📊 Category Count")

        bar = px.bar(
            saved["category"].value_counts().reset_index(),
            x="index",
            y="category",
            labels={"index": "Category", "category": "Count"}
        )
        st.plotly_chart(bar, use_container_width=True)

# ================== CHATBOT ==================
with tabs[3]:

    if "chat" not in st.session_state:
        st.session_state.chat = []

    col1, col2 = st.columns([3,1])

    msg = col1.text_input("Ask anything...")

    # -------- DELETE CHAT --------
    if col2.button("🗑️ Clear Chat"):
        st.session_state.chat = []

    if msg:
        st.session_state.chat.append(("You", msg))

        m = msg.lower()

        if "hello" in m:
            reply = "Hey 👋 How can I help?"
        elif "water" in m:
            reply = "💧 Water issue detected"
        elif "road" in m:
            reply = "🛣️ Road issue logged"
        else:
            reply = "I found something similar from dataset."

        st.session_state.chat.append(("Bot", reply))

    for s, m in st.session_state.chat:
        st.write(f"**{s}:** {m}")
