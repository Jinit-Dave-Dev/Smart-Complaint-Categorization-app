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

# -------------------- LOAD FILES --------------------
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

# -------------------- CLEAN FUNCTION --------------------
def clean_category(val):
    val = str(val).lower()

    if "water" in val:
        return "Water"
    elif "road" in val:
        return "Road"
    elif "garbage" in val:
        return "Garbage"
    elif "electric" in val:
        return "Electricity"
    else:
        return "Other"

# -------------------- UI --------------------
st.title("🏛️ Smart Municipal Complaint System")

tabs = st.tabs(["📝 Complaint", "📊 Dashboard", "🤖 Chatbot"])

# ================== TAB 1 ==================
with tabs[0]:

    text = st.text_area("Enter your complaint")

    if st.button("Submit Complaint"):
        if text.strip():

            X = vectorizer.transform([text])
            pred = model.predict(X)
            cat = le.inverse_transform(pred)[0]

            cat = clean_category(cat)

            try:
                conf = round(model.predict_proba(X).max() * 100, 2)
            except:
                conf = 0

            c.execute("INSERT INTO complaints VALUES (?,?,?,?,?)",
                      (st.session_state.user, text, cat, str(conf), "NEW"))
            conn.commit()

            st.success("✅ Complaint submitted successfully")

            st.write("### 📌 Prediction")
            st.write(f"Category: {cat}")
            st.write(f"Confidence: {conf}%")

            # -------- SEMANTIC SIMILARITY --------
            st.write("### 🔍 Similar Complaints (Smart Search)")

            X_all = vectorizer.transform(df[complaint_col])
            X_input = vectorizer.transform([text])

            sim_scores = cosine_similarity(X_input, X_all)[0]
            top_idx = np.argsort(sim_scores)[-5:][::-1]

            sim_df = df.iloc[top_idx]

            st.dataframe(sim_df, use_container_width=True)

# ================== TAB 2 ==================
with tabs[1]:

    saved = pd.read_sql_query("SELECT * FROM complaints", conn)

    if not saved.empty:

        # -------- CLEAN CATEGORY --------
        saved["category"] = saved["category"].apply(clean_category)

        # -------- CLEAN CONFIDENCE --------
        saved["confidence"] = pd.to_numeric(saved["confidence"], errors="coerce")
        saved["confidence"] = saved["confidence"].fillna(0)

        st.write("### 📋 All Complaints")
        st.dataframe(saved, use_container_width=True)

        # -------- FILTER --------
        cat = st.selectbox("Filter by Category", sorted(saved["category"].unique()))
        filtered = saved[saved["category"] == cat]

        st.dataframe(filtered, use_container_width=True)

        # -------- CHART --------
        st.write("### 📊 Category Distribution")
        st.bar_chart(saved["category"].value_counts())

# ================== TAB 3 ==================
with tabs[2]:

    if "chat" not in st.session_state:
        st.session_state.chat = []

    msg = st.text_input("Ask anything...")

    if msg:
        st.session_state.chat.append(("You", msg))

        m = msg.lower()

        if "hi" in m or "hello" in m:
            reply = "Hey! 👋 How can I help you today?"
        elif "water" in m:
            reply = "💧 Water issue detected. You should contact municipal water dept."
        elif "road" in m:
            reply = "🛣️ Road issue noted. It usually takes 3–5 days to resolve."
        else:
            sample = df.sample(1)[complaint_col].values[0]
            reply = f"🤖 Here's a similar case:\n{sample}"

        st.session_state.chat.append(("Bot", reply))

    for s, m in st.session_state.chat:
        st.write(f"**{s}:** {m}")
