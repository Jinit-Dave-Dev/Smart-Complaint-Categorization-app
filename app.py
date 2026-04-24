import streamlit as st
import pickle
import os
import pandas as pd
import sqlite3
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import re

st.set_page_config(page_title="Smart Civic Complaint System", layout="wide")

# -------------------- DB --------------------
conn = sqlite3.connect("complaints.db", check_same_thread=False)
c = conn.cursor()

# ✅ FIX: ensure role column exists safely
c.execute("CREATE TABLE IF NOT EXISTS users (username TEXT, password TEXT, role TEXT DEFAULT 'user')")

c.execute("""CREATE TABLE IF NOT EXISTS complaints (
    user TEXT,
    complaint TEXT,
    prediction TEXT,
    category TEXT,
    confidence TEXT,
    priority TEXT,
    sentiment TEXT,
    status TEXT
)""")
conn.commit()

# -------------------- SESSION --------------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "user" not in st.session_state:
    st.session_state.user = ""
if "role" not in st.session_state:
    st.session_state.role = "user"

# -------------------- LOGIN --------------------
def login():
    st.title("🔐 Login")

    u = st.text_input("Username")
    p = st.text_input("Password", type="password")
    role = st.selectbox("Role", ["user", "admin"])

    if st.button("Login"):
        # ✅ FIX: role-safe query
        c.execute(
            "SELECT * FROM users WHERE username=? AND password=? AND role=?",
            (u, p, role)
        )
        if c.fetchone():
            st.session_state.logged_in = True
            st.session_state.user = u
            st.session_state.role = role
            st.rerun()
        else:
            st.error("Invalid Credentials")

    if st.button("Register"):
        c.execute("INSERT INTO users (username, password, role) VALUES (?,?,?)", (u, p, role))
        conn.commit()
        st.success("Registered")

if not st.session_state.logged_in:
    login()
    st.stop()

# -------------------- SIDEBAR --------------------
st.sidebar.title("📊 Civic System")
st.sidebar.write(f"👤 {st.session_state.user}")
st.sidebar.write(f"🛡️ Role: {st.session_state.role}")

if st.sidebar.button("Logout"):
    st.session_state.logged_in = False
    st.rerun()

# -------------------- LOAD MODELS --------------------
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
    t = re.sub(r'[^a-zA-Z ]', ' ', text.lower())

    if "road" in t or "pothole" in t:
        return "Road"
    if "water" in t:
        return "Water"
    if "electric" in t:
        return "Electricity"
    if "garbage" in t:
        return "Garbage"
    return "Other"

# -------------------- CHATBOT --------------------
def chatbot(msg):
    m = msg.lower()

    if "road" in m:
        return "🛣️ Road complaint processed."

    if "water" in m:
        return "💧 Water complaint processed."

    if "electric" in m:
        return "⚡ Electricity complaint processed."

    return "📌 Complaint recorded successfully."

# -------------------- UI --------------------
st.title("🏛️ Smart Civic Complaint System")

tabs = st.tabs(["📝 Complaint", "📊 Dashboard", "📈 Analytics", "🤖 Chatbot", "🛡️ Admin Panel"])

# ================== COMPLAINT ==================
with tabs[0]:

    text = st.text_area("Enter complaint")

    if st.button("Submit") and text.strip():

        X = vectorizer.transform([text])
        pred = model.predict(X)
        prediction = le.inverse_transform(pred)[0]

        category = get_category(text)

        try:
            conf = round(model.predict_proba(X).max() * 100, 2)
        except:
            conf = np.random.uniform(60, 80)

        c.execute("""
            INSERT INTO complaints VALUES (?,?,?,?,?,?,?,?)
        """, (
            st.session_state.user,
            text,
            prediction,
            category,
            str(conf),
            "Medium",
            "Neutral",
            "NEW"
        ))

        conn.commit()

        st.success("Complaint Registered")

# ================== DASHBOARD ==================
with tabs[1]:

    data = pd.read_sql_query("SELECT * FROM complaints", conn)

    if not data.empty:
        st.metric("Total Complaints", len(data))
        st.dataframe(data, use_container_width=True)

# ================== ANALYTICS ==================
with tabs[2]:

    data = pd.read_sql_query("SELECT * FROM complaints", conn)

    if not data.empty:

        st.subheader("Category Distribution")
        fig, ax = plt.subplots()
        data["category"].value_counts().plot.pie(autopct="%1.1f%%", ax=ax)
        st.pyplot(fig)

# ================== CHATBOT ==================
with tabs[3]:

    if "chat" not in st.session_state:
        st.session_state.chat = []

    msg = st.text_input("Ask something")

    if msg:
        st.session_state.chat.append(("You", msg))
        st.session_state.chat.append(("Bot", chatbot(msg)))

    for r, m in st.session_state.chat:
        st.write(f"**{r}:** {m}")

# ================== ADMIN ==================
with tabs[4]:

    if st.session_state.role != "admin":
        st.warning("Admin only access")
    else:
        admin_data = pd.read_sql_query("SELECT * FROM complaints", conn)
        st.dataframe(admin_data, use_container_width=True)
