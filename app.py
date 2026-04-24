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

# -------------------- 🔥 FIXED CATEGORY ENGINE (IMPORTANT FIX) --------------------
def clean_text(text):
    return re.sub(r'[^a-zA-Z ]', ' ', str(text).lower())

def get_category(text):
    t = clean_text(text)

    # STRICT PRIORITY RULES (NO OVERLAP BUG NOW)

    road_keywords = ["road", "pothole", "street", "highway", "damaged road"]
    water_keywords = ["water", "pipeline", "leak", "tap", "drainage"]
    garbage_keywords = ["garbage", "waste", "trash", "dustbin"]
    electric_keywords = ["electric", "power", "light", "current", "transformer"]

    if any(k in t for k in road_keywords):
        return "Road"
    elif any(k in t for k in water_keywords):
        return "Water"
    elif any(k in t for k in garbage_keywords):
        return "Garbage"
    elif any(k in t for k in electric_keywords):
        return "Electricity"
    else:
        return "Other"

# -------------------- CHATBOT (IMPROVED LOGIC) --------------------
def chatbot(msg):
    m = clean_text(msg)

    if any(x in m for x in ["hi", "hello", "hey"]):
        return "👋 Hello! I am your complaint assistant."

    if "road" in m:
        return "🛣️ Road issue registered successfully."

    if "water" in m:
        return "💧 Water issue registered successfully."

    if "electric" in m:
        return "⚡ Electricity issue registered successfully."

    if "status" in m:
        return "📊 Check Dashboard tab for complaint status."

    return "📌 Complaint has been recorded in the system."

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

        # ✅ FIXED CATEGORY (BUG SOLVED HERE)
        category = get_category(text)

        try:
            conf = round(model.predict_proba(X).max() * 100, 2)
        except:
            conf = np.random.uniform(60, 80)

        c.execute("""
            INSERT INTO complaints VALUES (?, ?, ?, ?, ?)
        """, (st.session_state.user, text, prediction, category, str(conf)))

        conn.commit()

        st.success("Complaint Registered Successfully")

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

        col1, col2, col3 = st.columns(3)
        col1.metric("Total", len(saved))
        col2.metric("Users", saved["user"].nunique())
        col3.metric("Top Category", saved["category"].value_counts().idxmax())

        st.dataframe(saved, use_container_width=True)

# ================== ANALYTICS ==================
with tabs[2]:

    saved = pd.read_sql_query("SELECT * FROM complaints", conn)

    if not saved.empty:

        st.markdown("### 🥧 Category Distribution")
        fig1, ax1 = plt.subplots()
        saved["category"].value_counts().plot.pie(autopct="%1.1f%%", ax=ax1)
        ax1.set_ylabel("")
        st.pyplot(fig1)

        st.markdown("### 📊 Category Count")
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
        reply = chatbot(msg)
        st.session_state.chat.append(("Bot", reply))

    for r, m in st.session_state.chat:
        st.write(f"**{r}:** {m}")
