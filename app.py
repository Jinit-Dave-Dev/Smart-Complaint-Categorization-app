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
    category TEXT,
    confidence TEXT,
    status TEXT
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
category_col = next((c for c in df.columns if "category" in c.lower()), None)

df[complaint_col] = df[complaint_col].astype(str)

# -------------------- HELPERS --------------------
def clean_category(val):
    val = str(val).lower()
    if "water" in val: return "Water"
    elif "road" in val: return "Road"
    elif "garbage" in val: return "Garbage"
    elif "electric" in val: return "Electricity"
    return "Other"

def confidence_label(conf):
    if conf >= 75:
        return f"{conf}% 🟢 High"
    elif conf >= 50:
        return f"{conf}% 🟡 Medium"
    else:
        return f"{conf}% 🔴 Low"

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
            cat = clean_category(le.inverse_transform(pred)[0])

            try:
                conf = round(model.predict_proba(X).max() * 100, 2)
            except:
                conf = 0

            c.execute("INSERT INTO complaints VALUES (?,?,?,?,?)",
                      (st.session_state.user, text, cat, str(conf), "NEW"))
            conn.commit()

            st.success("✅ Complaint Registered Successfully")

            col1, col2 = st.columns(2)
            col1.metric("Category", cat)
            col2.metric("Confidence", confidence_label(conf))

            # -------- SMART SIMILAR --------
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
        saved["confidence"] = pd.to_numeric(saved["confidence"], errors="coerce").fillna(0)

        st.dataframe(saved, use_container_width=True)

        cat = st.selectbox("Filter", saved["category"].unique())
        st.dataframe(saved[saved["category"] == cat])

# ================== ANALYTICS ==================
with tabs[2]:
    saved = pd.read_sql_query("SELECT * FROM complaints", conn)

    if not saved.empty:
        saved["category"] = saved["category"].apply(clean_category)

        st.markdown("### 📊 Category Distribution")
        st.bar_chart(saved["category"].value_counts())

        st.markdown("### 📈 Confidence Distribution")
        saved["confidence"] = pd.to_numeric(saved["confidence"], errors="coerce").fillna(0)
        st.line_chart(saved["confidence"])

# ================== CHATBOT ==================
with tabs[3]:

    if "chat" not in st.session_state:
        st.session_state.chat = []

    msg = st.text_input("Ask anything...")

    if msg:
        st.session_state.chat.append(("You", msg))
        m = msg.lower()

        if "hello" in m or "hi" in m:
            reply = "👋 Hello! I’m here to help with your complaints."
        elif "water" in m:
            reply = "💧 Water issues are handled by local supply department."
        elif "road" in m:
            reply = "🛣️ Road issues usually get fixed within a few days."
        elif "status" in m:
            reply = "📌 You can check your complaint status in Dashboard tab."
        else:
            sample = df.sample(1)[complaint_col].values[0]
            reply = f"🤖 Here's a similar case:\n{sample}"

        st.session_state.chat.append(("Bot", reply))

    for s, m in st.session_state.chat:
        st.write(f"**{s}:** {m}")
