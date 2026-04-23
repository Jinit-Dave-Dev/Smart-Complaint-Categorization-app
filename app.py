import streamlit as st
import pickle
import os
import pandas as pd
import sqlite3
import time
import numpy as np

# -------------------- PAGE CONFIG --------------------
st.set_page_config(page_title="Municipal Complaint System", layout="wide")

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
    st.markdown("<h2 style='text-align:center;'>🔐 Login System</h2>", unsafe_allow_html=True)

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        c.execute("SELECT * FROM users WHERE username=? AND password=?", (username, password))
        if c.fetchone():
            st.session_state.logged_in = True
            st.session_state.user = username
            st.success("Login Successful")
            st.rerun()
        else:
            st.error("Invalid Credentials")

    if st.button("Register"):
        c.execute("INSERT INTO users VALUES (?,?)", (username, password))
        conn.commit()
        st.success("Registered! Now login.")

if not st.session_state.logged_in:
    login()
    st.stop()

# -------------------- UI STYLE --------------------
st.markdown("""
<style>
body {background: linear-gradient(135deg, #0f172a, #1e293b); color: white;}
.card {background: rgba(255,255,255,0.05); padding:20px; border-radius:15px;}
.chat-user {background:#4CAF50;padding:10px;border-radius:10px;margin:5px;text-align:right;color:black;}
.chat-bot {background:#1e293b;padding:10px;border-radius:10px;margin:5px;}
</style>
""", unsafe_allow_html=True)

# -------------------- SIDEBAR --------------------
st.sidebar.title("📊 Dashboard")
st.sidebar.write(f"👤 {st.session_state.user}")
st.sidebar.write("👨‍💻 Jinit Dave")

if st.sidebar.button("Logout"):
    st.session_state.logged_in = False
    st.rerun()

model_choice = st.sidebar.selectbox(
    "Model",
    ["Gradient Boosting", "Logistic Regression", "Naive Bayes"]
)

# -------------------- DATA --------------------
file_path = "smart_complaints_dataset_250.csv"
if not os.path.exists(file_path):
    file_path = "data/smart_complaints_dataset_250.csv"

df = pd.read_csv(file_path)
df.columns = df.columns.str.strip()

complaint_col = next((c for c in df.columns if 'complaint' in c.lower()), None)
category_col = next((c for c in df.columns if 'category' in c.lower()), None)

# 🔥 FIX ERROR HERE
df[complaint_col] = df[complaint_col].fillna("").astype(str)

# -------------------- LOAD ML --------------------
vectorizer = pickle.load(open("tfidf_vectorizer.pkl", "rb"))
le = pickle.load(open("label_encoder.pkl", "rb"))

model_files = {
    "Gradient Boosting": "gradient_boosting_model.pkl",
    "Logistic Regression": "logistic_regression_model.pkl",
    "Naive Bayes": "naive_bayes_model.pkl"
}

# -------------------- FUNCTIONS --------------------
def map_category(text):
    text = str(text).lower()
    if "water" in text: return "Water"
    if "road" in text: return "Road"
    if "garbage" in text: return "Garbage"
    if "electric" in text: return "Electricity"
    return "Other"

def chatbot_response(msg):
    msg = msg.lower()
    if "hi" in msg or "hello" in msg:
        return "Hello 👋 How can I help you?"
    elif "how are you" in msg:
        return "I'm working perfectly 🚀"
    elif "water" in msg:
        return "💧 Water issue detected. Try submitting complaint."
    else:
        return "🤖 I'm here to help with complaints or general questions."

# -------------------- TABS --------------------
tab1, tab2, tab3 = st.tabs(["📌 Complaint", "📊 Dashboard", "🤖 Chatbot"])

# ================= TAB 1 =================
with tab1:
    st.markdown("### Enter Complaint")
    user_input = st.text_area("Write complaint")

    if user_input:
        model = pickle.load(open(model_files[model_choice], "rb"))

        X_new = vectorizer.transform([str(user_input)])  # 🔥 FIX
        pred = model.predict(X_new)
        prediction = le.inverse_transform(pred)[0]

        category = map_category(user_input)

        st.markdown(f"<div class='card'>Prediction: {prediction}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='card'>Category: {category}</div>", unsafe_allow_html=True)

        # 🔥 DYNAMIC CHART
        filtered = df[df[category_col] == prediction]
        st.markdown("### 📊 Complaint Trends")
        st.bar_chart(filtered[category_col].value_counts())

# ================= TAB 2 =================
with tab2:
    st.markdown("### Dashboard")

    saved = pd.read_sql_query("SELECT * FROM complaints", conn)

    if not saved.empty:
        st.bar_chart(saved["category"].value_counts())

# ================= TAB 3 =================
with tab3:
    st.markdown("### Chatbot")

    if "chat" not in st.session_state:
        st.session_state.chat = []

    msg = st.text_input("Message")

    if msg:
        reply = chatbot_response(msg)
        st.session_state.chat.append(("You", msg))
        st.session_state.chat.append(("Bot", reply))

    for sender, text in st.session_state.chat:
        if sender == "You":
            st.markdown(f"<div class='chat-user'>{text}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='chat-bot'>{text}</div>", unsafe_allow_html=True)
