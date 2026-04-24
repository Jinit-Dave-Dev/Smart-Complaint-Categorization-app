import streamlit as st
import pickle
import os
import pandas as pd
import sqlite3
import numpy as np

# -------------------- CONFIG --------------------
st.set_page_config(page_title="Smart Complaint System", layout="wide")

# -------------------- DB --------------------
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
if "logged" not in st.session_state:
    st.session_state.logged = False
if "user" not in st.session_state:
    st.session_state.user = ""

# -------------------- LOGIN --------------------
def login():
    st.title("🔐 Login")

    user = st.text_input("Username")
    pwd = st.text_input("Password", type="password")

    if st.button("Login"):
        c.execute("SELECT * FROM users WHERE username=? AND password=?", (user, pwd))
        if c.fetchone():
            st.session_state.logged = True
            st.session_state.user = user
            st.success("Login successful")
            st.rerun()
        else:
            st.error("Invalid credentials")

    if st.button("Register"):
        c.execute("INSERT INTO users VALUES (?,?)", (user, pwd))
        conn.commit()
        st.success("Registered successfully")

if not st.session_state.logged:
    login()
    st.stop()

# -------------------- LOAD DATA --------------------
file_path = "smart_complaints_dataset_250.csv"
if not os.path.exists(file_path):
    file_path = "data/smart_complaints_dataset_250.csv"

df = pd.read_csv(file_path)
df.columns = df.columns.str.strip()

complaint_col = [c for c in df.columns if "complaint" in c.lower() or "text" in c.lower()][0]
category_col = [c for c in df.columns if "category" in c.lower() or "label" in c.lower()][0]

df[complaint_col] = df[complaint_col].astype(str)

# -------------------- LOAD MODEL --------------------
vectorizer = pickle.load(open("tfidf_vectorizer.pkl", "rb"))
model = pickle.load(open("logistic_regression_model.pkl", "rb"))
le = pickle.load(open("label_encoder.pkl", "rb"))

# -------------------- HELPERS --------------------
def map_category(text):
    text = text.lower()
    if "water" in text: return "Water"
    if "road" in text: return "Road"
    if "garbage" in text: return "Garbage"
    if "electric" in text: return "Electricity"
    return "Other"

def predict(text):
    X = vectorizer.transform([text])
    pred = model.predict(X)
    label = le.inverse_transform(pred)[0]

    try:
        prob = model.predict_proba(X).max()
        conf = round(prob * 100, 2)
    except:
        conf = 0

    return label, conf

# -------------------- UI --------------------
st.title("🏛️ Smart Complaint System")
st.sidebar.write(f"👤 {st.session_state.user}")

if st.sidebar.button("Logout"):
    st.session_state.logged = False
    st.rerun()

tab1, tab2, tab3, tab4 = st.tabs(["Complaint", "Dashboard", "Chatbot", "Analytics"])

# -------------------- TAB 1 --------------------
with tab1:
    st.subheader("Register Complaint")

    text = st.text_area("Enter complaint")

    if st.button("Submit"):
        if text.strip():
            pred, conf = predict(text)
            cat = map_category(text)

            c.execute("INSERT INTO complaints VALUES (?,?,?,?,?)",
                      (st.session_state.user, text, cat, str(conf), "NEW"))
            conn.commit()

            st.success("Complaint registered")

            st.dataframe(pd.DataFrame({
                "Complaint": [text],
                "Category": [cat],
                "Confidence": [conf],
                "Status": ["NEW"]
            }))

# -------------------- TAB 2 --------------------
with tab2:
    st.subheader("Dashboard")

    saved = pd.read_sql_query("SELECT * FROM complaints", conn)

    if not saved.empty:
        st.dataframe(saved)

        cat = st.selectbox("Filter", saved["category"].unique())
        sim = df[df[category_col] == cat]

        st.write("Similar complaints")
        st.dataframe(sim.head(5))

# -------------------- TAB 3 --------------------
with tab3:
    st.subheader("Chatbot")

    if "chat" not in st.session_state:
        st.session_state.chat = []

    def reply(t):
        t = t.lower()
        if "hi" in t:
            return "Hello 👋"
        if "water" in t:
            return "Water issue noted 💧"
        if "road" in t:
            return "Road issue noted 🛣️"
        return "Please explain more"

    msg = st.text_input("Message")

    if msg:
        st.session_state.chat.append(("You", msg))
        st.session_state.chat.append(("Bot", reply(msg)))

    for s, m in st.session_state.chat:
        st.write(f"{s}: {m}")

# -------------------- TAB 4 --------------------
with tab4:
    st.subheader("Analytics")

    saved = pd.read_sql_query("SELECT * FROM complaints", conn)

    if not saved.empty:

        saved["confidence"] = pd.to_numeric(saved["confidence"], errors="coerce")
        saved["confidence"] = saved["confidence"].fillna(0)

        st.write("Category Distribution")
        st.bar_chart(saved["category"].value_counts())

        st.write("Confidence")
        st.bar_chart(saved["confidence"])
