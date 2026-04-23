# ===================== IMPORTS =====================
import streamlit as st
import pickle
import os
import pandas as pd
import sqlite3
import time
import numpy as np

# ===================== PAGE CONFIG =====================
st.set_page_config(page_title="Municipal Complaint System", layout="wide")

# ===================== DATABASE =====================
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

# ===================== SESSION =====================
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "user" not in st.session_state:
    st.session_state.user = ""

# ===================== LOGIN =====================
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

# ===================== STYLE =====================
st.markdown("""
<style>
.card {padding:20px;border-radius:15px;background:rgba(255,255,255,0.05);}
.kpi {padding:15px;border-radius:12px;background:rgba(255,255,255,0.08);}
</style>
""", unsafe_allow_html=True)

# ===================== SIDEBAR =====================
st.sidebar.title("📊 Dashboard")
st.sidebar.write(f"👤 {st.session_state.user}")

model_choice = st.sidebar.selectbox(
    "Model",
    ["Gradient Boosting", "Logistic Regression", "Naive Bayes"]
)

# ===================== LOAD DATA =====================
df = pd.read_csv("smart_complaints_dataset_250.csv")

df.columns = df.columns.str.strip()

complaint_col = next((c for c in df.columns if 'complaint' in c.lower() or 'text' in c.lower()), df.columns[0])
category_col = next((c for c in df.columns if 'category' in c.lower() or 'label' in c.lower()), df.columns[1])

# ✅ CRITICAL FIX (THIS SOLVES YOUR ERROR)
df[complaint_col] = df[complaint_col].astype(str).fillna("")

vectorizer = pickle.load(open("tfidf_vectorizer.pkl", "rb"))
le = pickle.load(open("label_encoder.pkl", "rb"))

model_files = {
    "Gradient Boosting": "gradient_boosting_model.pkl",
    "Logistic Regression": "logistic_regression_model.pkl",
    "Naive Bayes": "naive_bayes_model.pkl"
}

# ===================== SAFE MODEL LOAD =====================
@st.cache_resource
def load_model(name):
    return pickle.load(open(model_files[name], "rb"))

model = load_model(model_choice)

# ===================== TABS =====================
tab1, tab2, tab3, tab4 = st.tabs(["🏠 Main", "📊 Analytics", "🤖 Chatbot", "⚙ Admin"])

# ===================== TAB 1 =====================
with tab1:
    user_input = st.text_area("Enter complaint")

    if user_input:
        X = vectorizer.transform([str(user_input)])
        pred = model.predict(X)[0]
        label = le.inverse_transform([pred])[0]

        try:
            conf = model.predict_proba(X).max() * 100
        except:
            conf = 0

        st.markdown(f"<div class='card'>Prediction: {label}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='card'>Confidence: {round(conf,2)}%</div>", unsafe_allow_html=True)

# ===================== TAB 2 =====================
with tab2:
    st.subheader("Analytics")

    saved = pd.read_sql_query("SELECT * FROM complaints", conn)

    if not saved.empty:
        st.bar_chart(saved["category"].value_counts())

        csv = saved.to_csv(index=False)
        st.download_button("Download CSV", csv, "analytics.csv")

# ===================== TAB 3 =====================
with tab3:
    st.subheader("Chatbot")

    if "chat" not in st.session_state:
        st.session_state.chat = []

    msg = st.text_input("Ask")

    if msg:
        st.session_state.chat.append(("You", msg))
        st.session_state.chat.append(("Bot", "Processing..."))

    for s, m in st.session_state.chat:
        st.write(f"{s}: {m}")

# ===================== TAB 4 =====================
with tab4:
    st.subheader("Admin Panel")

    saved = pd.read_sql_query("SELECT rowid,* FROM complaints", conn)
    st.dataframe(saved)

    delete_id = st.number_input("Delete ID", 0)

    if st.button("Delete"):
        c.execute("DELETE FROM complaints WHERE rowid=?", (delete_id,))
        conn.commit()
        st.success("Deleted")

# ===================== EVALUATION =====================
st.markdown("## Model Evaluation")

from sklearn.metrics import accuracy_score

X_all = vectorizer.transform(df[complaint_col])
y_true = le.transform(df[category_col])
y_pred = model.predict(X_all)

acc = accuracy_score(y_true, y_pred)

st.success(f"Accuracy: {round(acc*100,2)}%")
