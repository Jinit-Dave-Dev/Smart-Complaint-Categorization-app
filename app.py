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
st.markdown("""<style>
body {background: linear-gradient(135deg, #0f172a, #1e293b); color: white;}
.big-title {text-align:center;font-size:34px;font-weight:700;
background: linear-gradient(90deg,#4CAF50,#00E5FF);
-webkit-background-clip:text;-webkit-text-fill-color:transparent;}
.sub-text {text-align:center;color:#94a3b8;}
.card {background: rgba(255,255,255,0.05);padding:20px;border-radius:15px;text-align:center;}
.kpi {background: rgba(255,255,255,0.08);padding:15px;border-radius:12px;text-align:center;}
.chat-user {background: linear-gradient(90deg,#4CAF50,#00E5FF);
padding:10px;border-radius:15px;text-align:right;color:black;}
.chat-bot {background: rgba(255,255,255,0.08);padding:10px;border-radius:15px;}
</style>""", unsafe_allow_html=True)

# -------------------- SIDEBAR --------------------
st.sidebar.title("📊 Dashboard")
st.sidebar.write(f"👤 Logged in as: {st.session_state.user}")

if st.sidebar.button("Logout"):
    st.session_state.logged_in = False
    st.rerun()

model_choice = st.sidebar.selectbox(
    "🔀 Select Model",
    ["Gradient Boosting", "Logistic Regression", "Naive Bayes"]
)

# ✅ PAGE NAVIGATION (NEW - WITHOUT BREAKING UI)
page = st.sidebar.radio("📂 Navigate", [
    "🏠 Prediction",
    "📊 Analytics",
    "🛠️ Admin",
    "🤖 Chatbot",
    "📈 Evaluation"
])

# -------------------- DATA --------------------
file_path = "smart_complaints_dataset_250.csv"
if not os.path.exists(file_path):
    file_path = "data/smart_complaints_dataset_250.csv"

df = pd.read_csv(file_path)
df.columns = df.columns.str.strip()

complaint_col = next((c for c in df.columns if 'complaint' in c.lower()), None)
category_col = next((c for c in df.columns if 'category' in c.lower()), None)

vectorizer = pickle.load(open("tfidf_vectorizer.pkl", "rb"))
le = pickle.load(open("label_encoder.pkl", "rb"))

model_files = {
    "Gradient Boosting": "gradient_boosting_model.pkl",
    "Logistic Regression": "logistic_regression_model.pkl",
    "Naive Bayes": "naive_bayes_model.pkl"
}

# =========================================================
# 🏠 PAGE 1: PREDICTION (UNCHANGED UI)
# =========================================================
if page == "🏠 Prediction":

    st.markdown("<div class='big-title'>🏛️ Smart Municipal Complaint System</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-text'>AI-powered complaint classification dashboard</div>", unsafe_allow_html=True)

    user_input = st.text_area("📝 Enter your complaint:", height=150)

    if user_input.strip():
        model = pickle.load(open(model_files[model_choice], "rb"))

        X_new = vectorizer.transform([user_input])
        pred = model.predict(X_new)
        prediction = le.inverse_transform(pred)[0]

        prob = model.predict_proba(X_new).max()
        confidence = round(prob * 100, 2)

        st.markdown(f"<div class='card'>📌 {prediction}</div>", unsafe_allow_html=True)

# =========================================================
# 📊 ANALYTICS PAGE
# =========================================================
elif page == "📊 Analytics":

    saved = pd.read_sql_query("SELECT * FROM complaints", conn)

    if not saved.empty:
        st.markdown("### 📊 Analytics Dashboard")

        st.bar_chart(saved["category"].value_counts())

# =========================================================
# 🛠️ ADMIN PAGE
# =========================================================
elif page == "🛠️ Admin":

    saved = pd.read_sql_query("SELECT rowid, * FROM complaints", conn)
    st.dataframe(saved)

    delete_id = st.number_input("Delete ID")
    if st.button("Delete"):
        c.execute("DELETE FROM complaints WHERE rowid=?", (delete_id,))
        conn.commit()
        st.success("Deleted")

# =========================================================
# 🤖 CHATBOT PAGE
# =========================================================
elif page == "🤖 Chatbot":

    st.markdown("### 🤖 AI Assistant")

    if "chat" not in st.session_state:
        st.session_state.chat = []

    msg = st.text_input("Message")

    if msg:
        st.session_state.chat.append(("You", msg))
        st.session_state.chat.append(("Bot", "Processing..."))

    for s, m in st.session_state.chat:
        if s == "You":
            st.markdown(f"<div class='chat-user'>{m}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='chat-bot'>{m}</div>", unsafe_allow_html=True)

# =========================================================
# 📈 EVALUATION PAGE
# =========================================================
elif page == "📈 Evaluation":

    from sklearn.metrics import accuracy_score, confusion_matrix

    model = pickle.load(open(model_files[model_choice], "rb"))

    X_all = vectorizer.transform(df[complaint_col])
    y_true = le.transform(df[category_col])
    y_pred = model.predict(X_all)

    acc = accuracy_score(y_true, y_pred)
    st.success(f"Accuracy: {round(acc*100,2)}%")

    cm = confusion_matrix(y_true, y_pred)
    st.dataframe(pd.DataFrame(cm))
