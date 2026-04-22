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
.big-title {text-align:center;font-size:34px;font-weight:700;
background: linear-gradient(90deg, #4CAF50, #00E5FF);
-webkit-background-clip: text;-webkit-text-fill-color: transparent;}
.sub-text {text-align:center;color:#94a3b8;}
.card {background: rgba(255,255,255,0.05);padding:20px;border-radius:15px;text-align:center;}
.kpi {background: rgba(255,255,255,0.08);padding:15px;border-radius:12px;text-align:center;}
</style>
""", unsafe_allow_html=True)

# -------------------- SIDEBAR --------------------
st.sidebar.title("📊 Dashboard")
st.sidebar.write(f"👤 {st.session_state.user}")

page = st.sidebar.radio("📌 Navigate", [
    "🏠 Home",
    "🤖 Prediction",
    "📊 Analytics",
    "🛠️ Admin",
    "💬 Chatbot",
    "📈 Evaluation"
])

model_choice = st.sidebar.selectbox(
    "🔀 Select Model",
    ["Gradient Boosting", "Logistic Regression", "Naive Bayes"]
)

if st.sidebar.button("Logout"):
    st.session_state.logged_in = False
    st.rerun()

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

# -------------------- HOME --------------------
if page == "🏠 Home":
    st.markdown("<div class='big-title'>🏛️ Smart Complaint System</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-text'>AI Dashboard</div>", unsafe_allow_html=True)

# -------------------- PREDICTION --------------------
if page == "🤖 Prediction":

    user_input = st.text_area("Enter Complaint")

    if user_input:
        model = pickle.load(open(model_files[model_choice], "rb"))
        X = vectorizer.transform([user_input])
        pred = model.predict(X)
        result = le.inverse_transform(pred)[0]

        st.success(f"Prediction: {result}")

# -------------------- ANALYTICS --------------------
if page == "📊 Analytics":

    saved = pd.read_sql_query("SELECT * FROM complaints", conn)

    if not saved.empty:
        st.bar_chart(saved["category"].value_counts())

# -------------------- ADMIN --------------------
if page == "🛠️ Admin":

    saved = pd.read_sql_query("SELECT rowid,* FROM complaints", conn)
    st.dataframe(saved)

# -------------------- CHATBOT --------------------
if page == "💬 Chatbot":

    if "chat" not in st.session_state:
        st.session_state.chat = []

    msg = st.text_input("Ask")

    if msg:
        st.session_state.chat.append(msg)

    for m in st.session_state.chat:
        st.write("You:", m)

# -------------------- EVALUATION --------------------
if page == "📈 Evaluation":

    from sklearn.metrics import accuracy_score, confusion_matrix

    model = pickle.load(open(model_files[model_choice], "rb"))
    X = vectorizer.transform(df[complaint_col])
    y = le.transform(df[category_col])

    pred = model.predict(X)

    st.success(f"Accuracy: {accuracy_score(y, pred)*100:.2f}%")

    cm = confusion_matrix(y, pred)
    st.dataframe(pd.DataFrame(cm))
