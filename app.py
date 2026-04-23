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
    transition:0.3s;
}
.card:hover {
    transform: scale(1.05);
}

.premium-card {
    background: linear-gradient(135deg,#4CAF50,#00E5FF);
    padding:25px;
    border-radius:15px;
    color:black;
    font-weight:bold;
    text-align:center;
    font-size:18px;
}
</style>
""", unsafe_allow_html=True)

# -------------------- TITLE --------------------
st.markdown("<div class='big-title'>🏛️ Smart Municipal Complaint System</div>", unsafe_allow_html=True)

# -------------------- SIDEBAR --------------------
st.sidebar.title("📊 Dashboard")
st.sidebar.write(f"👤 {st.session_state.user}")
st.sidebar.write("👨‍💻 Jinit Dave")

if st.sidebar.button("Logout"):
    st.session_state.logged_in = False
    st.rerun()

model_choice = st.sidebar.selectbox(
    "Model",
    ["Gradient Boosting", "Logistic Regression", "Naive Bayes"]
)

# -------------------- DATA --------------------
file_path = "smart_complaints_dataset_250.csv"
if not os.path.exists(file_path):
    file_path = "data/smart_complaints_dataset_250.csv"

df = pd.read_csv(file_path)
df.columns = df.columns.str.strip()

complaint_col = [c for c in df.columns if "complaint" in c.lower() or "text" in c.lower()][0]
category_col = [c for c in df.columns if "category" in c.lower() or "label" in c.lower()][0]

df[complaint_col] = df[complaint_col].astype(str).fillna("")

# -------------------- LOAD MODEL --------------------
vectorizer = pickle.load(open("tfidf_vectorizer.pkl", "rb"))
le = pickle.load(open("label_encoder.pkl", "rb"))

model_files = {
    "Gradient Boosting": "gradient_boosting_model.pkl",
    "Logistic Regression": "logistic_regression_model.pkl",
    "Naive Bayes": "naive_bayes_model.pkl"
}

# -------------------- INPUT --------------------
user_input = st.text_area("📝 Enter complaint")

# -------------------- PREDICTION --------------------
if user_input.strip():

    model = pickle.load(open(model_files[model_choice], "rb"))

    X_new = vectorizer.transform([str(user_input)])
    pred = model.predict(X_new)
    prediction = le.inverse_transform(pred)[0]

    try:
        prob = model.predict_proba(X_new).max()
        confidence = round(prob * 100, 2)
    except:
        confidence = "N/A"

    # -------- PREMIUM OUTPUT UI --------
    st.markdown("### 🎯 Prediction Result")

    col1, col2, col3 = st.columns(3)

    col1.markdown(f"<div class='premium-card'>📌 {prediction}</div>", unsafe_allow_html=True)
    col2.markdown(f"<div class='card'>🏛️ Category<br><b>{prediction}</b></div>", unsafe_allow_html=True)
    col3.markdown(f"<div class='card'>🎯 Confidence<br><b>{confidence}%</b></div>", unsafe_allow_html=True)

    # -------------------- WHY --------------------
    st.markdown("### 🔍 Explanation")

    feature_names = vectorizer.get_feature_names_out()
    tfidf = X_new.toarray()[0]

    top_idx = tfidf.argsort()[-5:][::-1]
    words = [feature_names[i] for i in top_idx if tfidf[i] > 0]

    if words:
        st.info("Keywords: " + ", ".join(words))

    # -------------------- SIMILAR --------------------
    st.markdown("### 📋 Similar Complaints")
    sim = df[df[category_col] == prediction].head(5)
    st.dataframe(sim)

# -------------------- ADMIN --------------------
st.markdown("### 🛠️ Admin Panel")
saved = pd.read_sql_query("SELECT rowid,* FROM complaints", conn)
st.dataframe(saved)

# -------------------- CHATBOT --------------------
st.markdown("### 🤖 Chatbot")

if "chat" not in st.session_state:
    st.session_state.chat = []

msg = st.text_input("Ask...")

if msg:
    st.session_state.chat.append(("You", msg))
    reply = "Try describing your issue clearly."
    st.session_state.chat.append(("Bot", reply))

for s, m in st.session_state.chat:
    st.write(f"**{s}:** {m}")

# -------------------- EVALUATION (HIDDEN ACCURACY) --------------------
st.markdown("### 📊 Model Insights")

try:
    from sklearn.metrics import classification_report, confusion_matrix

    model_eval = pickle.load(open(model_files[model_choice], "rb"))

    X_all = vectorizer.transform(df[complaint_col].astype(str))
    y_true = le.transform(df[category_col])
    y_pred = model_eval.predict(X_all)

    # ❌ Accuracy REMOVED (as requested)

    st.markdown("### 📋 Classification Report")
    report = classification_report(y_true, y_pred, output_dict=True)
    st.dataframe(pd.DataFrame(report).transpose())

    st.markdown("### 🔥 Confusion Matrix")
    cm = confusion_matrix(y_true, y_pred)
    st.dataframe(pd.DataFrame(cm))

except Exception as e:
    st.warning("Evaluation not available")
