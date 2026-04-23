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
body {background: linear-gradient(135deg, #0f172a, #1e293b); color: white;}
.big-title {text-align:center;font-size:34px;font-weight:700;}
.sub-text {text-align:center;color:#94a3b8;}
.card {background: rgba(255,255,255,0.05);padding:20px;border-radius:15px;text-align:center;}
.kpi {background: rgba(255,255,255,0.08);padding:15px;border-radius:12px;text-align:center;}
</style>
""", unsafe_allow_html=True)

# -------------------- TITLE --------------------
st.markdown("<div class='big-title'>🏛️ Smart Municipal Complaint System</div>", unsafe_allow_html=True)

# -------------------- SIDEBAR --------------------
st.sidebar.title("📊 Dashboard")
st.sidebar.write(f"👤 Logged in as: {st.session_state.user}")

page = st.sidebar.radio(
    "📂 Navigate",
    ["🏠 Main", "📊 Analytics", "🤖 Chatbot", "🛠️ Admin", "📈 Evaluation"]
)

model_choice = st.sidebar.selectbox(
    "🔀 Select Model",
    ["Gradient Boosting", "Logistic Regression", "Naive Bayes"]
)

if st.sidebar.button("Logout"):
    st.session_state.logged_in = False
    st.session_state.user = ""
    st.rerun()

# -------------------- DATA --------------------
file_path = "smart_complaints_dataset_250.csv"
if not os.path.exists(file_path):
    file_path = "data/smart_complaints_dataset_250.csv"

df = pd.read_csv(file_path)
df.columns = df.columns.str.strip()

complaint_col = df.columns[0]
category_col = df.columns[1]

vectorizer = pickle.load(open("tfidf_vectorizer.pkl", "rb"))
le = pickle.load(open("label_encoder.pkl", "rb"))

model_files = {
    "Gradient Boosting": "gradient_boosting_model.pkl",
    "Logistic Regression": "logistic_regression_model.pkl",
    "Naive Bayes": "naive_bayes_model.pkl"
}

# -------------------- MAIN PAGE --------------------
if page == "🏠 Main":

    st.markdown("---")
    user_input = st.text_area("📝 Enter your complaint:", height=150)

    if user_input.strip():
        with st.spinner("Analyzing complaint..."):

            model = pickle.load(open(model_files[model_choice], "rb"))

            X_new = vectorizer.transform([user_input])
            y_pred = model.predict(X_new)
            prediction = le.inverse_transform(y_pred)[0]

            st.success(f"📌 Prediction: {prediction}")

# -------------------- ANALYTICS --------------------
elif page == "📊 Analytics":

    saved = pd.read_sql_query("SELECT * FROM complaints", conn)

    if not saved.empty:
        st.bar_chart(saved["category"].value_counts())

# -------------------- CHATBOT --------------------
elif page == "🤖 Chatbot":

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    user_msg = st.text_input("💬 Ask something...")

    if user_msg:
        st.session_state.chat_history.append(("You", user_msg))
        st.session_state.chat_history.append(("Bot", "Try describing your complaint clearly."))

    for sender, msg in st.session_state.chat_history:
        st.write(f"{sender}: {msg}")

# -------------------- ADMIN --------------------
elif page == "🛠️ Admin":

    saved = pd.read_sql_query("SELECT rowid, * FROM complaints", conn)
    st.dataframe(saved)

# -------------------- EVALUATION --------------------
elif page == "📈 Evaluation":

    from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

    model = pickle.load(open(model_files[model_choice], "rb"))

    X_all = vectorizer.transform(df[complaint_col])
    y_true = le.transform(df[category_col])
    y_pred = model.predict(X_all)

    st.success(f"Accuracy: {round(accuracy_score(y_true, y_pred)*100,2)}%")

    st.markdown("### Classification Report")
    report = classification_report(y_true, y_pred, output_dict=True)
    st.dataframe(pd.DataFrame(report).transpose())

    st.markdown("### Confusion Matrix")
    cm = confusion_matrix(y_true, y_pred)
    st.dataframe(pd.DataFrame(cm))
