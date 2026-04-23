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

# -------------------- 🌈 STYLE --------------------
st.markdown("""
<style>
body {background: linear-gradient(135deg, #0f172a, #1e293b); color: white;}

.chat-container {
    max-height: 400px;
    overflow-y: auto;
    padding: 10px;
}

.chat-user {
    background: linear-gradient(90deg,#4CAF50,#00E5FF);
    padding:10px 15px;
    border-radius:15px;
    margin:8px 0;
    text-align:right;
    color:black;
    width: fit-content;
    margin-left:auto;
}

.chat-bot {
    background: rgba(255,255,255,0.08);
    padding:10px 15px;
    border-radius:15px;
    margin:8px 0;
    text-align:left;
    width: fit-content;
}

.input-box {
    position: sticky;
    bottom: 0;
}
</style>
""", unsafe_allow_html=True)

# -------------------- HEADER --------------------
st.markdown("<h2 style='text-align:center;'>🏛️ Smart Municipal Complaint System</h2>", unsafe_allow_html=True)

# -------------------- SIDEBAR --------------------
st.sidebar.title("📊 Dashboard")
st.sidebar.write(f"👤 {st.session_state.user}")
st.sidebar.write("👨‍💻 Jinit Dave")

if st.sidebar.button("Logout"):
    st.session_state.logged_in = False
    st.rerun()

model_choice = st.sidebar.selectbox(
    "Select Model",
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

df[complaint_col] = df[complaint_col].astype(str)

# -------------------- ML --------------------
vectorizer = pickle.load(open("tfidf_vectorizer.pkl", "rb"))
le = pickle.load(open("label_encoder.pkl", "rb"))

model_files = {
    "Gradient Boosting": "gradient_boosting_model.pkl",
    "Logistic Regression": "logistic_regression_model.pkl",
    "Naive Bayes": "naive_bayes_model.pkl"
}

# -------------------- TABS --------------------
tab1, tab2, tab3, tab4 = st.tabs(["📝 Complaint", "📊 Analytics", "🛠️ Admin", "🤖 Assistant"])

# ================= TAB 1 =================
with tab1:
    user_input = st.text_area("📝 Enter your complaint")

    if user_input:
        model = pickle.load(open(model_files[model_choice], "rb"))
        X = vectorizer.transform([str(user_input)])
        pred = le.inverse_transform(model.predict(X))[0]
        st.success(f"Prediction: {pred}")

# ================= TAB 2 =================
with tab2:
    saved = pd.read_sql_query("SELECT * FROM complaints", conn)
    st.dataframe(saved)

# ================= TAB 3 =================
with tab3:
    saved = pd.read_sql_query("SELECT rowid,* FROM complaints", conn)
    st.dataframe(saved)

# ================= TAB 4 (CHATGPT STYLE) =================
with tab4:

    st.markdown("### 🤖 AI Assistant")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    def chatbot_response(text):
        text = text.lower()

        if "hi" in text:
            return "👋 Hello! How can I help you?"
        elif "water" in text:
            return "💧 Water issue detected. You can submit complaint."
        elif "road" in text:
            return "🛣️ Road issue detected."
        elif "hello" in text:
            return "Hey there! 😊"
        else:
            try:
                sample = df.sample(1)
                return f"🤖 Similar complaint:\n{sample[complaint_col].values[0]}"
            except:
                return "Tell me more about your issue."

    # Chat display container
    chat_box = st.container()

    with chat_box:
        for sender, msg in st.session_state.chat_history:
            if sender == "You":
                st.markdown(f"<div class='chat-user'>{msg}</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='chat-bot'>{msg}</div>", unsafe_allow_html=True)

    # Input
    user_msg = st.text_input("💬 Type your message...")

    if user_msg:
        st.session_state.chat_history.append(("You", user_msg))
        reply = chatbot_response(user_msg)
        st.session_state.chat_history.append(("Bot", reply))
        st.rerun()
