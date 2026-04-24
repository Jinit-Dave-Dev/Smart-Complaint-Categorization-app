import streamlit as st
import pickle
import os
import pandas as pd
import sqlite3
import time
import numpy as np

st.set_page_config(page_title="Municipal Complaint System", layout="wide")

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
            st.error("Invalid")

    if st.button("Register"):
        c.execute("INSERT INTO users VALUES (?,?)", (u, p))
        conn.commit()
        st.success("Registered")

if not st.session_state.logged_in:
    login()
    st.stop()

# -------------------- LOAD --------------------
vectorizer = pickle.load(open("tfidf_vectorizer.pkl", "rb"))
le = pickle.load(open("label_encoder.pkl", "rb"))

model = pickle.load(open("logistic_regression_model.pkl", "rb"))

file_path = "smart_complaints_dataset_250.csv"
if not os.path.exists(file_path):
    file_path = "data/smart_complaints_dataset_250.csv"

df = pd.read_csv(file_path)
df.columns = df.columns.str.strip()

complaint_col = next((c for c in df.columns if 'complaint' in c.lower()), None)
category_col = next((c for c in df.columns if 'category' in c.lower()), None)

df[complaint_col] = df[complaint_col].astype(str)

# -------------------- UI --------------------
st.title("🏛️ Smart Complaint System")

tabs = st.tabs(["📝 Complaint", "📊 Dashboard", "🤖 Chatbot"])

# ================== TAB 1 ==================
with tabs[0]:
    text = st.text_area("Enter complaint")

    if st.button("Submit"):
        if text.strip():
            X = vectorizer.transform([text])
            pred = model.predict(X)
            cat = le.inverse_transform(pred)[0]

            try:
                conf = round(model.predict_proba(X).max() * 100, 2)
            except:
                conf = 0

            # SAVE FIXED ORDER
            c.execute("INSERT INTO complaints VALUES (?,?,?,?,?)",
                      (st.session_state.user, text, cat, str(conf), "NEW"))
            conn.commit()

            st.success("Complaint submitted")

            st.write("### Result")
            st.write(cat, conf)

            # -------- FIXED SIMILAR --------
            st.write("### Similar Complaints")

            keywords = text.lower().split()

            sim = df[df[complaint_col].str.lower().apply(
                lambda x: any(k in x for k in keywords)
            )].head(5)

            if sim.empty:
                st.warning("No similar found")
            else:
                st.dataframe(sim)

# ================== TAB 2 ==================
with tabs[1]:
    saved = pd.read_sql_query("SELECT * FROM complaints", conn)

    if not saved.empty:

        # -------- FIX CATEGORY --------
        saved["category"] = saved["category"].astype(str)

        valid = ["Water", "Road", "Garbage", "Electricity", "Other"]
        saved["category"] = saved["category"].apply(
            lambda x: x if x in valid else "Other"
        )

        # -------- FIX CONFIDENCE --------
        saved["confidence"] = pd.to_numeric(saved["confidence"], errors="coerce")
        saved["confidence"] = saved["confidence"].fillna(0)

        st.write("### All Complaints")
        st.dataframe(saved)

        # -------- FILTER FIX --------
        cat = st.selectbox("Filter", sorted(saved["category"].unique()))
        st.dataframe(saved[saved["category"] == cat])

        # -------- CHART --------
        st.bar_chart(saved["category"].value_counts())

# ================== TAB 3 ==================
with tabs[2]:

    if "chat" not in st.session_state:
        st.session_state.chat = []

    msg = st.text_input("Ask")

    if msg:
        st.session_state.chat.append(("You", msg))

        if "hello" in msg.lower():
            reply = "Hi! How can I help?"
        elif "water" in msg.lower():
            reply = "Water issue → please contact local authority."
        else:
            sample = df.sample(1)[complaint_col].values[0]
            reply = f"Similar issue: {sample}"

        st.session_state.chat.append(("Bot", reply))

    for s, m in st.session_state.chat:
        st.write(f"**{s}:** {m}")
