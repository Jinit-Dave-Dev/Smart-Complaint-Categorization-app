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
.big-title {text-align:center;font-size:34px;font-weight:700;background: linear-gradient(90deg, #4CAF50, #00E5FF);-webkit-background-clip: text;-webkit-text-fill-color: transparent;}
.sub-text {text-align:center;color:#94a3b8;}
.card {background: rgba(255,255,255,0.05);padding:20px;border-radius:15px;text-align:center;}
.chat-user {background: linear-gradient(90deg,#4CAF50,#00E5FF);padding:10px;border-radius:15px;text-align:right;color:black;}
.chat-bot {background: rgba(255,255,255,0.08);padding:10px;border-radius:15px;text-align:left;}
</style>
""", unsafe_allow_html=True)

# -------------------- TITLE --------------------
st.markdown("<div class='big-title'>🏛️ Smart Municipal Complaint System</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-text'>AI-powered complaint classification dashboard</div>", unsafe_allow_html=True)

# -------------------- SIDEBAR --------------------
st.sidebar.title("📊 Dashboard")
st.sidebar.write(f"👤 {st.session_state.user}")

st.sidebar.markdown("### 👨‍💻 Developer")
st.sidebar.write("Jinit Dave")

page = st.sidebar.radio("📂 Navigate", ["🏠 Main", "📊 Analytics", "🤖 Chatbot", "🛠 Admin", "📈 Evaluation"])

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

complaint_col = next((c for c in df.columns if 'complaint' in c.lower()), None)
category_col = next((c for c in df.columns if 'category' in c.lower()), None)

# ✅ CLEANING (BOOST ACCURACY)
df[complaint_col] = df[complaint_col].fillna("").astype(str).str.lower().str.strip()
df[category_col] = df[category_col].fillna("").astype(str)

# -------------------- LOAD ML --------------------
vectorizer = pickle.load(open("tfidf_vectorizer.pkl", "rb"))
le = pickle.load(open("label_encoder.pkl", "rb"))

model_files = [
    "gradient_boosting_model.pkl",
    "logistic_regression_model.pkl",
    "naive_bayes_model.pkl"
]

# -------------------- ENSEMBLE --------------------
def ensemble_predict(X):
    preds = []
    for path in model_files:
        model = pickle.load(open(path, "rb"))
        preds.append(model.predict(X))
    preds = np.array(preds)
    return np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=preds)

# ================= MAIN =================
if page == "🏠 Main":

    user_input = st.text_area("📝 Enter your complaint:", height=150)

    if user_input.strip():
        clean_input = user_input.lower().strip()

        X_new = vectorizer.transform([clean_input])
        pred_encoded = ensemble_predict(X_new)[0]
        prediction = le.inverse_transform([pred_encoded])[0]

        st.success(f"Prediction: {prediction}")

# ================= ANALYTICS =================
if page == "📊 Analytics":
    saved = pd.read_sql_query("SELECT * FROM complaints", conn)
    if not saved.empty:
        st.bar_chart(saved["category"].value_counts())

# ================= CHATBOT =================
if page == "🤖 Chatbot":

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    def smart_bot(text):
        text = text.lower()
        if "water" in text:
            return "💧 Water issue detected."
        if "road" in text:
            return "🛣️ Road issue detected."
        sample = df.sample(1)
        return f"🤖 Similar complaint:\n{sample[complaint_col].values[0]}"

    msg = st.text_input("Ask something")

    if msg:
        st.session_state.chat_history.append(("You", msg))
        st.session_state.chat_history.append(("Bot", smart_bot(msg)))

    for s, m in st.session_state.chat_history:
        if s == "You":
            st.markdown(f"<div class='chat-user'>{m}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='chat-bot'>{m}</div>", unsafe_allow_html=True)

# ================= ADMIN =================
if page == "🛠 Admin":
    saved = pd.read_sql_query("SELECT rowid,* FROM complaints", conn)
    st.dataframe(saved)

# ================= EVALUATION =================
if page == "📈 Evaluation":

    st.markdown("## 📊 Model Evaluation")

    from sklearn.metrics import accuracy_score

    # ✅ FIX LABEL MISMATCH
    df_eval = df[df[category_col].isin(le.classes_)]

    X_all = vectorizer.transform(df_eval[complaint_col])
    y_true = le.transform(df_eval[category_col])

    y_pred = ensemble_predict(X_all)

    acc = accuracy_score(y_true, y_pred)

    st.success(f"🚀 Ensemble Accuracy: {round(acc*100,2)}%")
