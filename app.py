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
.sub-text {text-align:center;color:#94a3b8;}
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
    padding:10px;border-radius:10px;margin:5px;
}
.chat-bot {
    background: rgba(255,255,255,0.08);
    padding:10px;border-radius:10px;margin:5px;
}
</style>
""", unsafe_allow_html=True)

# -------------------- SIDEBAR --------------------
st.sidebar.title("📊 Dashboard")
st.sidebar.write(f"👤 {st.session_state.user}")
st.sidebar.write("👨‍💻 Jinit Dave")

if st.sidebar.button("Logout"):
    st.session_state.logged_in = False
    st.rerun()

# ✅ PAGE NAVIGATION (NEW)
page = st.sidebar.radio("📂 Navigate", [
    "🏠 Home",
    "📊 Analytics",
    "🛠️ Admin Panel",
    "🤖 AI Assistant"
])

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

complaint_col = [c for c in df.columns if 'complaint' in c.lower()][0]
category_col = [c for c in df.columns if 'category' in c.lower()][0]

vectorizer = pickle.load(open("tfidf_vectorizer.pkl", "rb"))
le = pickle.load(open("label_encoder.pkl", "rb"))

model_files = {
    "Gradient Boosting": "gradient_boosting_model.pkl",
    "Logistic Regression": "logistic_regression_model.pkl",
    "Naive Bayes": "naive_bayes_model.pkl"
}

# -------------------- FUNCTIONS --------------------
def map_category(text):
    text = text.lower()
    if "water" in text: return "Water Supply"
    if "road" in text: return "Road & Infrastructure"
    if "garbage" in text: return "Sanitation & Waste"
    if "electric" in text: return "Electricity"
    return "Other"

def get_conf(conf):
    try:
        conf = float(conf)
        if conf > 75: return f"{conf}% 🟢"
        elif conf > 50: return f"{conf}% 🟡"
        else: return f"{conf}% 🔴"
    except:
        return "N/A"

# ================== 🏠 HOME ==================
if page == "🏠 Home":

    st.markdown("<div class='big-title'>🏛️ Smart Municipal Complaint System</div>", unsafe_allow_html=True)

    user_input = st.text_area("Enter complaint")

    if user_input:
        model = pickle.load(open(model_files[model_choice], "rb"))
        X = vectorizer.transform([user_input])
        pred = le.inverse_transform(model.predict(X))[0]

        try:
            conf = round(model.predict_proba(X).max()*100,2)
        except:
            conf = "N/A"

        cat = map_category(user_input)

        c.execute("INSERT INTO complaints VALUES (?,?,?,?,?)",
                  (st.session_state.user, user_input, pred, cat, str(conf)))
        conn.commit()

        c1,c2,c3 = st.columns(3)
        c1.markdown(f"<div class='card'>{pred}</div>", unsafe_allow_html=True)
        c2.markdown(f"<div class='card'>{cat}</div>", unsafe_allow_html=True)
        c3.markdown(f"<div class='card'>{get_conf(conf)}</div>", unsafe_allow_html=True)

        # WHY
        st.subheader("Why Prediction?")
        feat = vectorizer.get_feature_names_out()
        arr = X.toarray()[0]
        idx = arr.argsort()[-5:]
        words = [feat[i] for i in idx if arr[i]>0]
        st.info(", ".join(words) if words else "No keywords")

        # FEATURE IMPORTANCE
        st.subheader("Feature Importance")
        if hasattr(model, "feature_importances_"):
            imp = model.feature_importances_
            top = np.argsort(imp)[-10:]
            st.bar_chart(pd.DataFrame({"imp":imp[top]}, index=[feat[i] for i in top]))

# ================== 📊 ANALYTICS ==================
if page == "📊 Analytics":

    saved = pd.read_sql("SELECT * FROM complaints", conn)

    if not saved.empty:
        total = len(saved)
        top = saved["category"].value_counts().idxmax()

        c1,c2 = st.columns(2)
        c1.metric("Total", total)
        c2.metric("Top", top)

        st.bar_chart(saved["category"].value_counts())

    st.subheader("Top 5 Categories")
    top5 = df[category_col].value_counts().head(5)
    for i,(k,v) in enumerate(top5.items(),1):
        st.write(f"{i}. {k} ({v})")

# ================== 🛠️ ADMIN ==================
if page == "🛠️ Admin Panel":

    saved = pd.read_sql("SELECT rowid,* FROM complaints", conn)
    st.dataframe(saved)

    delete_id = st.number_input("ID")
    if st.button("Delete"):
        c.execute("DELETE FROM complaints WHERE rowid=?", (delete_id,))
        conn.commit()
        st.success("Deleted")

# ================== 🤖 CHATBOT ==================
if page == "🤖 AI Assistant":

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    def bot(msg):
        if "hi" in msg: return "Hello 👋"
        if "water" in msg: return "Water issue 💧"
        return "Try describing complaint"

    msg = st.text_input("Chat")

    if msg:
        st.session_state.chat_history.append(("You", msg))
        time.sleep(0.3)
        st.session_state.chat_history.append(("Bot", bot(msg)))

    for s,m in st.session_state.chat_history:
        if s=="You":
            st.markdown(f"<div class='chat-user'>{m}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='chat-bot'>{m}</div>", unsafe_allow_html=True)
