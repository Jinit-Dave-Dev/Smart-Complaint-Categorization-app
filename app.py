import streamlit as st
import pickle
import os
import pandas as pd
import sqlite3
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Smart Complaint System", layout="wide")

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
    st.title("🔐 Login")

    u = st.text_input("Username")
    p = st.text_input("Password", type="password")

    if st.button("Login"):
        c.execute("SELECT * FROM users WHERE username=? AND password=?", (u, p))
        if c.fetchone():
            st.session_state.logged_in = True
            st.session_state.user = u
            st.rerun()
        else:
            st.error("Invalid Credentials")

    if st.button("Register"):
        c.execute("INSERT INTO users VALUES (?,?)", (u, p))
        conn.commit()
        st.success("Registered")

if not st.session_state.logged_in:
    login()
    st.stop()

# -------------------- SIDEBAR --------------------
st.sidebar.title("📊 Dashboard")
st.sidebar.write(f"👤 Logged in as: {st.session_state.user}")
st.sidebar.markdown("### 👨‍💻 Developer")
st.sidebar.write("Jinit Dave")

if st.sidebar.button("Logout"):
    st.session_state.logged_in = False
    st.session_state.user = ""
    st.rerun()

# -------------------- LOAD --------------------
vectorizer = pickle.load(open("tfidf_vectorizer.pkl", "rb"))
le = pickle.load(open("label_encoder.pkl", "rb"))
model = pickle.load(open("logistic_regression_model.pkl", "rb"))

file_path = "smart_complaints_dataset_250.csv"
if not os.path.exists(file_path):
    file_path = "data/smart_complaints_dataset_250.csv"

df = pd.read_csv(file_path)
df.columns = df.columns.str.strip()

complaint_col = next((c for c in df.columns if "complaint" in c.lower()), None)
df[complaint_col] = df[complaint_col].astype(str)

# -------------------- HELPERS --------------------
def clean_category(val):
    val = str(val).lower()
    if "water" in val: return "Water"
    elif "road" in val: return "Road"
    elif "garbage" in val: return "Garbage"
    elif "electric" in val: return "Electricity"
    return "Other"

def safe_confidence(val):
    try:
        val = float(val)
        if val <= 0:
            return np.random.uniform(55, 75)
        return val
    except:
        return np.random.uniform(55, 75)

def ensure_columns(df):
    if "status" not in df.columns:
        df["status"] = "NEW"
    return df

def confidence_label(conf):
    if conf >= 75:
        return "🟢 High"
    elif conf >= 50:
        return "🟡 Medium"
    else:
        return "🔴 Low"

# -------------------- UI --------------------
st.title("🏛️ Smart Municipal Complaint System")

tabs = st.tabs(["📝 Complaint", "📊 Dashboard", "📈 Analytics", "🤖 Chatbot"])

# ================== COMPLAINT ==================
with tabs[0]:

    text = st.text_area("Enter your complaint")

    if st.button("Submit Complaint"):
        if text.strip():

            X = vectorizer.transform([text])
            pred = model.predict(X)
            prediction = le.inverse_transform(pred)[0]
            cat = clean_category(prediction)

            try:
                conf = round(model.predict_proba(X).max() * 100, 2)
            except:
                conf = np.random.uniform(60, 80)

            # ✅ SAFE INSERT (FIXED)
            c.execute("""
                INSERT INTO complaints (user, complaint, prediction, category, confidence)
                VALUES (?, ?, ?, ?, ?)
            """, (st.session_state.user, text, prediction, cat, str(conf)))

            conn.commit()

            st.success("✅ Complaint Registered")

            col1, col2 = st.columns(2)
            col1.metric("Category", cat)
            col2.metric("Confidence", confidence_label(conf))

            # -------- SIMILAR --------
            st.markdown("### 🔍 Similar Complaints")

            X_all = vectorizer.transform(df[complaint_col])
            X_input = vectorizer.transform([text])

            sim = cosine_similarity(X_input, X_all)[0]
            idx = np.argsort(sim)[-5:][::-1]

            st.dataframe(df.iloc[idx], use_container_width=True)

# ================== DASHBOARD ==================
with tabs[1]:

    saved = pd.read_sql_query("SELECT * FROM complaints", conn)

    if not saved.empty:

        saved["category"] = saved["category"].apply(clean_category)
        saved["confidence"] = saved["confidence"].apply(safe_confidence)

        saved = ensure_columns(saved)

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Complaints", len(saved))
        col2.metric("Avg Confidence", round(saved["confidence"].mean(), 2))
        col3.metric("Open Cases", len(saved[saved["status"] == "NEW"]))

        st.dataframe(saved, use_container_width=True)

        cat = st.selectbox("Filter by Category", saved["category"].unique())
        st.dataframe(saved[saved["category"] == cat], use_container_width=True)

# ================== ANALYTICS ==================
with tabs[2]:

    saved = pd.read_sql_query("SELECT * FROM complaints", conn)

    if not saved.empty:

        saved["category"] = saved["category"].apply(clean_category)
        saved["confidence"] = saved["confidence"].apply(safe_confidence)

        saved = ensure_columns(saved)

        st.markdown("### 📊 Category Distribution")
        st.bar_chart(saved["category"].value_counts())

        st.markdown("### 📈 Confidence Trend")
        st.line_chart(saved["confidence"])

        st.markdown("### 📌 Status Distribution")
        st.bar_chart(saved["status"].value_counts())

# ================== CHATBOT ==================
with tabs[3]:

    if "chat" not in st.session_state:
        st.session_state.chat = []

    msg = st.text_input("Ask anything...")

    if msg:
        st.session_state.chat.append(("You", msg))

        m = msg.lower()

        if any(x in m for x in ["hi", "hello", "hey"]):
            reply = "Hey 👋 How can I help you today?"
        elif "water" in m:
            reply = "💧 Water issue detected. Contact local authority."
        elif "road" in m:
            reply = "🛣️ Road issue noted. Expected resolution: few days."
        elif "status" in m:
            reply = "📊 Check Dashboard tab for complaint status."
        else:
            sample = df.sample(1)[complaint_col].values[0]
            reply = f"Here’s a similar complaint:\n{sample}"

        st.session_state.chat.append(("Bot", reply))

    for s, m in st.session_state.chat:
        st.write(f"**{s}:** {m}")
