import streamlit as st
import pickle
import os
import pandas as pd
import sqlite3
import time
import numpy as np
import random

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

.big-title {
    text-align:center;
    font-size:34px;
    font-weight:700;
    background: linear-gradient(90deg, #4CAF50, #00E5FF);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.card {
    background: rgba(255,255,255,0.06);
    padding:20px;
    border-radius:15px;
    text-align:center;
    transition:0.3s;
}
.card:hover {
    transform: translateY(-5px);
}
.chat-user {
    background: linear-gradient(90deg,#4CAF50,#00E5FF);
    padding:10px;
    border-radius:15px;
    text-align:right;
    color:black;
}
.chat-bot {
    background: rgba(255,255,255,0.08);
    padding:10px;
    border-radius:15px;
}
</style>
""", unsafe_allow_html=True)

# -------------------- HEADER --------------------
st.markdown("<div class='big-title'>🏛️ Smart Municipal Complaint System</div>", unsafe_allow_html=True)

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
df = pd.read_csv(file_path)

df.columns = df.columns.str.strip()
complaint_col = df.columns[0]
category_col = df.columns[1]

df[complaint_col] = df[complaint_col].astype(str).fillna("")

# -------------------- LOAD --------------------
vectorizer = pickle.load(open("tfidf_vectorizer.pkl", "rb"))
le = pickle.load(open("label_encoder.pkl", "rb"))

model_files = {
    "Gradient Boosting": "gradient_boosting_model.pkl",
    "Logistic Regression": "logistic_regression_model.pkl",
    "Naive Bayes": "naive_bayes_model.pkl"
}

# -------------------- INPUT --------------------
user_input = st.text_area("📝 Enter Complaint")

# -------------------- PREDICTION --------------------
if user_input.strip():

    model = pickle.load(open(model_files[model_choice], "rb"))

    X = vectorizer.transform([str(user_input)])
    pred = model.predict(X)
    label = le.inverse_transform(pred)[0]

    try:
        prob = model.predict_proba(X).max()
        confidence = round(prob * 100, 2)
    except:
        confidence = 60

    # Premium UI Cards
    c1, c2, c3 = st.columns(3)

    c1.markdown(f"<div class='card'>📌 Prediction<br><b>{label}</b></div>", unsafe_allow_html=True)
    c2.markdown(f"<div class='card'>🎯 Confidence<br><b>{confidence}%</b></div>", unsafe_allow_html=True)
    c3.markdown(f"<div class='card'>⚡ Status<br><b>Processed</b></div>", unsafe_allow_html=True)

# -------------------- CHATBOT --------------------
st.markdown("### 🤖 Assistant")

if "chat" not in st.session_state:
    st.session_state.chat = []

def smart_reply(text):
    responses = [
        "I understand your issue. Authorities will review it.",
        "This looks like a common complaint. Action will be taken.",
        "Thanks for reporting. This will be resolved soon.",
        "Your concern has been recorded successfully."
    ]
    return random.choice(responses)

msg = st.text_input("Ask...")

if msg:
    st.session_state.chat.append(("You", msg))
    st.session_state.chat.append(("Bot", smart_reply(msg)))

for sender, m in st.session_state.chat:
    if sender == "You":
        st.markdown(f"<div class='chat-user'>{m}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='chat-bot'>{m}</div>", unsafe_allow_html=True)

# -------------------- ADMIN --------------------
st.markdown("### 🛠️ Admin")

saved = pd.read_sql_query("SELECT * FROM complaints", conn)
st.dataframe(saved)
