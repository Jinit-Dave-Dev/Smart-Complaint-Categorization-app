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

.card {
    background: rgba(255,255,255,0.05);
    padding:20px;
    border-radius:15px;
    text-align:center;
}

.kpi {
    background: rgba(255,255,255,0.08);
    padding:15px;
    border-radius:12px;
    text-align:center;
}
</style>
""", unsafe_allow_html=True)

# -------------------- TITLE --------------------
st.markdown("<div class='big-title'>🏛️ Smart Municipal Complaint System</div>", unsafe_allow_html=True)

# -------------------- SIDEBAR --------------------
st.sidebar.title("📊 Dashboard")
st.sidebar.write(f"👤 {st.session_state.user}")

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

complaint_col = df.columns[0]
category_col = df.columns[1]

# -------------------- LOAD ML --------------------
vectorizer = pickle.load(open("tfidf_vectorizer.pkl", "rb"))
le = pickle.load(open("label_encoder.pkl", "rb"))

model_files = {
    "Gradient Boosting": "gradient_boosting_model.pkl",
    "Logistic Regression": "logistic_regression_model.pkl",
    "Naive Bayes": "naive_bayes_model.pkl"
}

# -------------------- TABS (SAFE) --------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "🧠 Prediction",
    "📊 Analytics",
    "🛠 Admin",
    "🤖 Chatbot"
])

# ================== TAB 1 ==================
with tab1:

    user_input = st.text_area("Enter complaint")

    if user_input:
        model = pickle.load(open(model_files[model_choice], "rb"))

        X = vectorizer.transform([user_input])
        pred = model.predict(X)
        label = le.inverse_transform(pred)[0]

        st.success(label)

        c.execute("INSERT INTO complaints VALUES (?,?,?,?,?)",
                  (st.session_state.user, user_input, label, label, "90"))
        conn.commit()

# ================== TAB 2 ==================
with tab2:

    saved = pd.read_sql_query("SELECT * FROM complaints", conn)

    if not saved.empty:
        st.bar_chart(saved["category"].value_counts())

# ================== TAB 3 ==================
with tab3:

    saved = pd.read_sql_query("SELECT rowid,* FROM complaints", conn)
    st.dataframe(saved)

    delete_id = st.number_input("Delete ID", step=1)

    if st.button("Delete"):
        c.execute("DELETE FROM complaints WHERE rowid=?", (delete_id,))
        conn.commit()
        st.success("Deleted")

# ================== TAB 4 ==================
with tab4:

    if "chat" not in st.session_state:
        st.session_state.chat = []

    msg = st.text_input("Ask")

    if msg:
        st.session_state.chat.append(("You", msg))
        st.session_state.chat.append(("Bot", "Response"))

    for s, m in st.session_state.chat:
        st.write(f"{s}: {m}")

# ================== EVALUATION ==================
st.markdown("## 📊 Evaluation")

from sklearn.metrics import accuracy_score

model_eval = pickle.load(open(model_files[model_choice], "rb"))

X_all = vectorizer.transform(df[complaint_col])
y = le.transform(df[category_col])

preds = model_eval.predict(X_all)

acc = accuracy_score(y, preds)

st.success(f"Accuracy: {round(acc*100,2)}%")
