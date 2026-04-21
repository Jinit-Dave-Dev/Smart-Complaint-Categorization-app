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

# -------------------- UI STYLE --------------------
st.markdown("""
<style>
body {background: linear-gradient(135deg,#0f172a,#1e293b); color:white;}

.big-title {
    text-align:center;
    font-size:34px;
    font-weight:700;
    background: linear-gradient(90deg,#4CAF50,#00E5FF);
    -webkit-background-clip:text;
    -webkit-text-fill-color:transparent;
}

.sub-text {text-align:center;color:#94a3b8;}

.card {
    background:rgba(255,255,255,0.05);
    padding:20px;border-radius:15px;
    backdrop-filter:blur(10px);
    text-align:center;
}

.kpi {
    background:rgba(255,255,255,0.08);
    padding:15px;border-radius:12px;text-align:center;
}

.user-bubble {
    background:#4CAF50;
    padding:10px 15px;
    border-radius:15px;
    margin:5px 0;
    text-align:right;
}

.bot-bubble {
    background:#1E293B;
    padding:10px 15px;
    border-radius:15px;
    margin:5px 0;
    text-align:left;
}
</style>
""", unsafe_allow_html=True)

# -------------------- TITLE --------------------
st.markdown("<div class='big-title'>🏛️ Smart Municipal Complaint System</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-text'>AI-powered complaint classification dashboard</div>", unsafe_allow_html=True)

# -------------------- SIDEBAR --------------------
st.sidebar.title("📊 Dashboard")
st.sidebar.write(f"👤 Logged in as: {st.session_state.user}")

st.sidebar.markdown("### 👨‍💻 Developer")
st.sidebar.write("Jinit Dave")

if st.sidebar.button("Logout"):
    st.session_state.logged_in = False
    st.session_state.user = ""
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

# -------------------- CATEGORY MAP --------------------
def map_category(text):
    text = str(text).lower()
    if "water" in text: return "Water Supply"
    if "road" in text: return "Road & Infrastructure"
    if "garbage" in text: return "Sanitation & Waste"
    if "electric" in text: return "Electricity"
    return "Other"

# -------------------- INPUT --------------------
st.markdown("---")
user_input = st.text_area("📝 Enter your complaint:", height=150)

# -------------------- PREDICTION --------------------
if user_input.strip():
    with st.spinner("Analyzing complaint..."):

        model = pickle.load(open(model_files[model_choice], "rb"))
        X_new = vectorizer.transform([user_input])
        y_pred = model.predict(X_new)
        prediction = le.inverse_transform(y_pred)[0]

        try:
            prob = model.predict_proba(X_new).max()
            confidence = round(prob * 100, 2)
        except:
            confidence = "N/A"

        enhanced = map_category(user_input)

        c.execute("INSERT INTO complaints VALUES (?,?,?,?,?)",
                  (st.session_state.user, user_input, prediction, enhanced, str(confidence)))
        conn.commit()

        col1, col2, col3 = st.columns(3)
        col1.markdown(f"<div class='card'>📌 {prediction}</div>", unsafe_allow_html=True)
        col2.markdown(f"<div class='card'>🏛️ {enhanced}</div>", unsafe_allow_html=True)
        col3.markdown(f"<div class='card'>🎯 {confidence}</div>", unsafe_allow_html=True)

        st.markdown("### 📋 Similar Complaints")
        sim = df[df[category_col] == prediction].head(5)
        st.dataframe(sim, use_container_width=True)

# -------------------- ADMIN --------------------
st.markdown("### 🛠️ Admin Panel")
saved = pd.read_sql_query("SELECT rowid, * FROM complaints", conn)
st.dataframe(saved, use_container_width=True)

# -------------------- ANALYTICS --------------------
st.markdown("### 📊 Analytics Dashboard")
if not saved.empty:
    st.bar_chart(saved["category"].value_counts())

# -------------------- CHATBOT --------------------
st.markdown("### 🤖 AI Assistant")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

def chatbot_response(text):
    t = text.lower()
    if "water" in t: return "💧 Water Supply issue detected"
    if "road" in t: return "🛣️ Road issue detected"
    if "electric" in t: return "⚡ Electricity issue detected"
    return "🤖 Please provide more details"

def typing_effect(text):
    placeholder = st.empty()
    full = ""
    for char in text:
        full += char
        placeholder.markdown(f"<div class='bot-bubble'>{full}</div>", unsafe_allow_html=True)
        time.sleep(0.01)

msg = st.text_input("Ask something...")

if msg:
    st.session_state.chat_history.append(("user", msg))
    reply = chatbot_response(msg)
    st.session_state.chat_history.append(("bot", reply))

# display chat
for sender, message in st.session_state.chat_history:
    if sender == "user":
        st.markdown(f"<div class='user-bubble'>{message}</div>", unsafe_allow_html=True)
    else:
        typing_effect(message)
