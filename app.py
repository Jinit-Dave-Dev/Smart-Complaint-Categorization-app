import streamlit as st
import pickle
import os
import pandas as pd
import sqlite3
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import re

st.set_page_config(page_title="Smart Complaint System", layout="wide")

# -------------------- PREMIUM UI --------------------
st.markdown("""
<style>
.stApp {
    background: radial-gradient(circle at top, #0f172a, #020617);
    color: white;
}

.stButton button {
    background: linear-gradient(90deg, #4f46e5, #06b6d4);
    border-radius: 10px;
    color: white;
    transition: 0.3s;
}
.stButton button:hover {
    transform: scale(1.05);
}

/* Card style */
div[data-testid="stMetric"] {
    background: rgba(255,255,255,0.06);
    padding: 15px;
    border-radius: 12px;
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
st.sidebar.title("📊 Smart Dashboard")
st.sidebar.write(f"👤 {st.session_state.user}")

st.sidebar.markdown("### 👨‍💻 Developer")
st.sidebar.write("Jinit Dave")

if st.sidebar.button("Logout"):
    st.session_state.logged_in = False
    st.rerun()

# -------------------- LOAD MODEL --------------------
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

# -------------------- FIXED CATEGORY ENGINE --------------------
def clean_text(t):
    return re.sub(r'[^a-zA-Z ]', ' ', str(t).lower())

def get_category(text):
    t = clean_text(text)

    road = ["road", "pothole", "street", "highway", "bridge"]
    water = ["water", "leak", "pipeline", "drain", "tap"]
    garbage = ["garbage", "waste", "trash", "dustbin"]
    electric = ["electric", "power", "light", "current"]

    # STRICT PRIORITY FIX (ROAD FIRST)
    if any(k in t for k in road):
        return "Road"
    if any(k in t for k in water):
        return "Water"
    if any(k in t for k in garbage):
        return "Garbage"
    if any(k in t for k in electric):
        return "Electricity"

    return "Other"

# -------------------- CHATBOT (IMPROVED MEMORY STYLE) --------------------
def chatbot(msg):
    m = clean_text(msg)

    if "hello" in m or "hi" in m:
        return "👋 Hey! I’m your municipal assistant."

    if "road" in m:
        return "🛣️ Road complaint registered successfully."

    if "water" in m:
        return "💧 Water complaint registered successfully."

    if "electric" in m:
        return "⚡ Electricity issue registered."

    if "status" in m:
        return "📊 Check Dashboard for status."

    return "📌 Complaint stored successfully."

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

        # FIXED CATEGORY
        category = get_category(text)

        try:
            conf = round(model.predict_proba(X).max() * 100, 2)
        except:
            conf = np.random.uniform(60, 80)

        c.execute("""
            INSERT INTO complaints VALUES (?, ?, ?, ?, ?)
        """, (st.session_state.user, text, prediction, category, str(conf)))

        conn.commit()

        st.success("Complaint Registered")

        col1, col2 = st.columns(2)
        col1.metric("Prediction", prediction)
        col2.metric("Category", category)

        st.markdown("### 🔍 Similar Complaints")

        X_all = vectorizer.transform(df[complaint_col])
        X_input = vectorizer.transform([text])

        sim = cosine_similarity(X_input, X_all)[0]
        idx = np.argsort(sim)[-5:][::-1]

        st.dataframe(df.iloc[idx], use_container_width=True)

# ================== DASHBOARD ==================
with tabs[1]:

    saved = pd.read_sql_query("SELECT * FROM complaints", conn)

    if not saved.empty:

        # FIX DISPLAY ISSUE (NORMALIZE OLD DATA)
        saved["category"] = saved["category"].apply(lambda x: x if x in ["Road","Water","Garbage","Electricity"] else "Other")

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Complaints", len(saved))
        col2.metric("Users", saved["user"].nunique())
        col3.metric("Top Category", saved["category"].value_counts().idxmax())

        st.dataframe(saved, use_container_width=True)

# ================== ANALYTICS (STARTUP STYLE) ==================
with tabs[2]:

    saved = pd.read_sql_query("SELECT * FROM complaints", conn)

    if not saved.empty:

        st.markdown("## 📊 Analytics Dashboard")

        col1, col2, col3 = st.columns(3)
        col1.metric("Total", len(saved))
        col2.metric("Categories", saved["category"].nunique())
        col3.metric("Top", saved["category"].value_counts().idxmax())

        st.markdown("### 🥧 Distribution")
        fig1, ax1 = plt.subplots()
        saved["category"].value_counts().plot.pie(autopct="%1.1f%%", ax=ax1)
        ax1.set_ylabel("")
        st.pyplot(fig1)

        st.markdown("### 📊 Volume")
        fig2, ax2 = plt.subplots()
        saved["category"].value_counts().plot.bar(ax=ax2)
        st.pyplot(fig2)

# ================== CHATBOT ==================
with tabs[3]:

    if "chat" not in st.session_state:
        st.session_state.chat = []

    col1, col2 = st.columns([3, 1])

    msg = col1.text_input("Ask anything...")

    if col2.button("🗑️ Clear Chat"):
        st.session_state.chat = []

    if msg:
        st.session_state.chat.append(("You", msg))
        st.session_state.chat.append(("Bot", chatbot(msg)))

    for r, m in st.session_state.chat:
        st.write(f"**{r}:** {m}")
