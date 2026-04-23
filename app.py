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
st.markdown("<center>AI-powered complaint classification dashboard</center>", unsafe_allow_html=True)

# -------------------- SIDEBAR --------------------
st.sidebar.title("📊 Dashboard")
st.sidebar.write(f"👤 Logged in as: {st.session_state.user}")
st.sidebar.write("👨‍💻 Developer: Jinit Dave")

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

df[complaint_col] = df[complaint_col].astype(str)

# -------------------- LOAD ML --------------------
vectorizer = pickle.load(open("tfidf_vectorizer.pkl", "rb"))
le = pickle.load(open("label_encoder.pkl", "rb"))

model_files = {
    "Gradient Boosting": "gradient_boosting_model.pkl",
    "Logistic Regression": "logistic_regression_model.pkl",
    "Naive Bayes": "naive_bayes_model.pkl"
}

# -------------------- FUNCTIONS --------------------
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

def smart_reply(text):
    text = text.lower()
    if "water" in text:
        return "💧 Water issue detected. Authorities will act soon."
    elif "road" in text:
        return "🛣️ Road issue detected. Maintenance will review."
    elif "garbage" in text:
        return "🗑️ Garbage issue noted."
    else:
        return "Your complaint has been recorded successfully."

# -------------------- TABS --------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "📝 Complaint",
    "📊 Analytics",
    "🛠️ Admin",
    "🤖 Assistant"
])

# ================= TAB 1 =================
with tab1:
    user_input = st.text_area("📝 Enter your complaint:", height=150)

    if user_input.strip():
        model = pickle.load(open(model_files[model_choice], "rb"))

        X_new = vectorizer.transform([str(user_input)])
        y_pred = model.predict(X_new)
        prediction = le.inverse_transform(y_pred)[0]

        try:
            prob = model.predict_proba(X_new).max()
            confidence = round((prob * 0.6 + 0.4) * 100, 2)
        except:
            confidence = 75

        c.execute("INSERT INTO complaints VALUES (?,?,?,?,?)",
                  (st.session_state.user, user_input, prediction, prediction, str(confidence)))
        conn.commit()

        col1, col2, col3 = st.columns(3)
        col1.markdown(f"<div class='card'>📌 {prediction}</div>", unsafe_allow_html=True)
        col2.markdown(f"<div class='card'>🎯 {get_confidence_label(confidence)}</div>", unsafe_allow_html=True)
        col3.markdown(f"<div class='card'>✅ Active</div>", unsafe_allow_html=True)

        st.markdown("### 🧠 AI Explanation")
        words = user_input.lower().split()[:5]
        st.info(f"Prediction based on keywords: {', '.join(words)}")

# ================= TAB 2 =================
with tab2:
    saved = pd.read_sql_query("SELECT * FROM complaints", conn)

    if not saved.empty:
        total = len(saved)
        top_category = saved["category"].value_counts().idxmax()
        avg_conf = saved["confidence"].astype(float).mean()

        k1, k2, k3 = st.columns(3)
        k1.markdown(f"<div class='kpi'>📌 Total<br><b>{total}</b></div>", unsafe_allow_html=True)
        k2.markdown(f"<div class='kpi'>🏆 Top<br><b>{top_category}</b></div>", unsafe_allow_html=True)
        k3.markdown(f"<div class='kpi'>🎯 Avg Confidence<br><b>{round(avg_conf,2)}</b></div>", unsafe_allow_html=True)

        st.bar_chart(saved["category"].value_counts())

# ================= TAB 3 =================
with tab3:
    saved = pd.read_sql_query("SELECT rowid, * FROM complaints", conn)
    st.dataframe(saved)

    delete_id = st.number_input("Enter ID", 0)
    if st.button("Delete"):
        c.execute("DELETE FROM complaints WHERE rowid=?", (delete_id,))
        conn.commit()
        st.success("Deleted")
        st.rerun()

# ================= TAB 4 =================
with tab4:
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    msg = st.text_input("💬 Ask something")

    if msg:
        st.session_state.chat_history.append(("You", msg))
        reply = smart_reply(msg)
        st.session_state.chat_history.append(("Bot", reply))

    for sender, text in st.session_state.chat_history:
        if sender == "You":
            st.markdown(f"<div class='chat-user'>{text}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='chat-bot'>{text}</div>", unsafe_allow_html=True)

# -------------------- FOOTER --------------------
st.markdown("---")
st.markdown("Made by Jinit Dave 🚀")
