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

# -------------------- STYLE --------------------
st.markdown("""
<style>
.card {padding:20px;border-radius:15px;background:rgba(255,255,255,0.05);}
.kpi {padding:15px;border-radius:12px;background:rgba(255,255,255,0.08);}
</style>
""", unsafe_allow_html=True)

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

# -------------------- DATA --------------------
file_path = "smart_complaints_dataset_250.csv"
if not os.path.exists(file_path):
    file_path = "data/smart_complaints_dataset_250.csv"

df = pd.read_csv(file_path)
df.columns = df.columns.str.strip()

complaint_col = next((c for c in df.columns if 'complaint' in c.lower() or 'text' in c.lower()), None)
category_col = next((c for c in df.columns if 'category' in c.lower() or 'label' in c.lower()), None)

# ✅ CRITICAL FIX
df[complaint_col] = df[complaint_col].astype(str).fillna("")

# -------------------- LOAD ML --------------------
vectorizer = pickle.load(open("tfidf_vectorizer.pkl", "rb"))
le = pickle.load(open("label_encoder.pkl", "rb"))

model_files = {
    "Gradient Boosting": "gradient_boosting_model.pkl",
    "Logistic Regression": "logistic_regression_model.pkl",
    "Naive Bayes": "naive_bayes_model.pkl"
}

# ✅ NEW: Cached model loading
@st.cache_resource
def load_model(name):
    return pickle.load(open(model_files[name], "rb"))

# -------------------- CONFIDENCE LABEL --------------------
def get_confidence_label(conf):
    try:
        conf = float(conf)
        if conf >= 75:
            return f"{conf}% 🟢 High"
        elif conf >= 50:
            return f"{conf}% 🟡 Medium"
        else:
            return f"{conf}% 🔴 Low"
    except:
        return "N/A"

# -------------------- INPUT --------------------
st.markdown("---")
user_input = st.text_area("📝 Enter your complaint:", height=150)

# -------------------- PREDICTION --------------------
if user_input.strip():
    with st.spinner("Analyzing complaint..."):

        model = load_model(model_choice)

        X_new = vectorizer.transform([user_input])
        y_pred = model.predict(X_new)
        prediction = le.inverse_transform(y_pred)[0]

        try:
            prob = model.predict_proba(X_new).max()
            confidence = round(prob * 100, 2)
        except:
            confidence = "N/A"

        c.execute("INSERT INTO complaints VALUES (?,?,?,?,?)",
                  (st.session_state.user, user_input, prediction, prediction, str(confidence)))
        conn.commit()

        col1, col2, col3 = st.columns(3)
        col1.markdown(f"<div class='card'>📌 {prediction}</div>", unsafe_allow_html=True)
        col2.markdown(f"<div class='card'>🏛️ {prediction}</div>", unsafe_allow_html=True)
        col3.markdown(f"<div class='card'>🎯 {get_confidence_label(confidence)}</div>", unsafe_allow_html=True)

# -------------------- ADMIN PANEL --------------------
st.markdown("### 🛠️ Admin Panel")
saved = pd.read_sql_query("SELECT rowid, * FROM complaints", conn)
st.dataframe(saved, use_container_width=True)

# ✅ NEW: Export full data
if not saved.empty:
    st.download_button("⬇ Export All Complaints", saved.to_csv(index=False), "complaints.csv")

delete_id = st.number_input("Enter Record ID to Delete", min_value=0, step=1)
if st.button("Delete Record"):
    c.execute("DELETE FROM complaints WHERE rowid=?", (delete_id,))
    conn.commit()
    st.success("Record Deleted")
    st.rerun()

# -------------------- ANALYTICS --------------------
st.markdown("### 📊 Analytics Dashboard")

if not saved.empty:
    total = len(saved)
    top_category = saved["category"].value_counts().idxmax()
    avg_conf = saved["confidence"].astype(float).mean()

    # ✅ NEW KPI
    top_user = saved["user"].value_counts().idxmax()

    k1, k2, k3, k4 = st.columns(4)
    k1.markdown(f"<div class='kpi'>📌 Total<br><b>{total}</b></div>", unsafe_allow_html=True)
    k2.markdown(f"<div class='kpi'>🏆 Top<br><b>{top_category}</b></div>", unsafe_allow_html=True)
    k3.markdown(f"<div class='kpi'>🎯 Confidence<br><b>{round(avg_conf,2)}</b></div>", unsafe_allow_html=True)
    k4.markdown(f"<div class='kpi'>👤 Top User<br><b>{top_user}</b></div>", unsafe_allow_html=True)

    st.bar_chart(saved["category"].value_counts())

# -------------------- CHATBOT --------------------
st.markdown("### 🤖 AI Assistant")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_msg = st.text_input("💬 Ask something...")

if user_msg:
    st.session_state.chat_history.append(("You", user_msg))
    st.session_state.chat_history.append(("Bot", "🤖 Processing your query..."))

for sender, msg in st.session_state.chat_history:
    st.write(f"{sender}: {msg}")

# ================== EVALUATION ==================
st.markdown("## 📊 Model Evaluation")

from sklearn.metrics import accuracy_score

model_eval = load_model(model_choice)

X_all = vectorizer.transform(df[complaint_col])
y_true = le.transform(df[category_col])
y_pred = model_eval.predict(X_all)

acc = accuracy_score(y_true, y_pred)

st.success(f"Accuracy: {round(acc*100,2)}%")
