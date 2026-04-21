import streamlit as st
import pickle
import os
import pandas as pd
import sqlite3
import time

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

# -------------------- 🌈 UI STYLE --------------------
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

.sub-text {text-align:center;color:#94a3b8;}

.card {
    background: rgba(255,255,255,0.05);
    padding:20px;
    border-radius:15px;
    backdrop-filter: blur(10px);
    text-align:center;
}

.kpi {
    background: rgba(255,255,255,0.08);
    padding:15px;
    border-radius:12px;
    text-align:center;
}

/* Chat UI */
.chat-user {
    background: linear-gradient(90deg,#4CAF50,#00E5FF);
    padding:10px;
    border-radius:15px;
    text-align:right;
    color:black;
    margin:5px 0;
}

.chat-bot {
    background: rgba(255,255,255,0.08);
    padding:10px;
    border-radius:15px;
    text-align:left;
    margin:5px 0;
}

.badge-user {
    background:black;
    color:#00E5FF;
    padding:2px 8px;
    border-radius:8px;
}

.badge-bot {
    background:#4CAF50;
    color:black;
    padding:2px 8px;
    border-radius:8px;
}
</style>
""", unsafe_allow_html=True)

# -------------------- TITLE --------------------
st.markdown("<div class='big-title'>🏛️ Smart Municipal Complaint System</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-text'>AI-powered complaint classification dashboard</div>", unsafe_allow_html=True)

# -------------------- SIDEBAR --------------------
st.sidebar.title("📊 Dashboard")
st.sidebar.write(f"👤 {st.session_state.user}")

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

complaint_col = next((c for c in df.columns if 'complaint' in c.lower() or 'text' in c.lower()), None)
category_col = next((c for c in df.columns if 'category' in c.lower() or 'label' in c.lower()), None)

df[complaint_col] = df[complaint_col].astype(str)

# -------------------- LOAD ML --------------------
vectorizer = pickle.load(open("tfidf_vectorizer.pkl", "rb"))
le = pickle.load(open("label_encoder.pkl", "rb"))

model_files = {
    "Gradient Boosting": "gradient_boosting_model.pkl",
    "Logistic Regression": "logistic_regression_model.pkl",
    "Naive Bayes": "naive_bayes_model.pkl"
}

# -------------------- INPUT --------------------
st.markdown("---")
user_input = st.text_area("Enter complaint")

# -------------------- PREDICTION --------------------
if user_input.strip():
    model = pickle.load(open(model_files[model_choice], "rb"))
    X_new = vectorizer.transform([user_input])
    y_pred = model.predict(X_new)
    prediction = le.inverse_transform(y_pred)[0]

    st.success(prediction)

# -------------------- 🤖 CHATBOT --------------------
st.markdown("### 🤖 AI Assistant")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

def chatbot_response(text):
    if "water" in text.lower():
        return "💧 Water issue detected"
    return "🤖 Try giving more details"

def typing_effect(text):
    placeholder = st.empty()
    out = ""
    for char in text:
        out += char
        placeholder.markdown(f"<div class='chat-bot'><span class='badge-bot'>AI</span> {out}</div>", unsafe_allow_html=True)
        time.sleep(0.01)

# 🔥 NEW: THINKING ANIMATION
def thinking_animation():
    placeholder = st.empty()
    for i in range(3):
        placeholder.markdown(f"<div class='chat-bot'><span class='badge-bot'>AI</span> Thinking{'.'*i}</div>", unsafe_allow_html=True)
        time.sleep(0.3)
    placeholder.empty()

msg = st.text_input("Ask something...")

if msg:
    st.session_state.chat_history.append(("user", msg))

    thinking_animation()   # ✅ added here

    reply = chatbot_response(msg)
    st.session_state.chat_history.append(("bot", reply))

for sender, message in st.session_state.chat_history:
    if sender == "user":
        st.markdown(f"<div class='chat-user'><span class='badge-user'>YOU</span> {message}</div>", unsafe_allow_html=True)
    else:
        typing_effect(message)
