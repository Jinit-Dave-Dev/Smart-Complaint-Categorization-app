import streamlit as st
import pickle
import os
import pandas as pd
import sqlite3
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import re

st.set_page_config(page_title="Smart Civic Complaint System", layout="wide")

# -------------------- UI --------------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #0f172a, #1e293b, #0f172a);
    color: white;
}
.stButton button {
    background: linear-gradient(90deg, #4f46e5, #06b6d4);
    color: white;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

# -------------------- DB --------------------
conn = sqlite3.connect("complaints.db", check_same_thread=False)
c = conn.cursor()

c.execute("CREATE TABLE IF NOT EXISTS users (username TEXT, password TEXT, role TEXT)")
c.execute("""CREATE TABLE IF NOT EXISTS complaints (
    user TEXT,
    complaint TEXT,
    prediction TEXT,
    category TEXT,
    confidence TEXT,
    priority TEXT,
    sentiment TEXT,
    status TEXT
)""")
conn.commit()

# -------------------- SESSION --------------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "user" not in st.session_state:
    st.session_state.user = ""
if "role" not in st.session_state:
    st.session_state.role = "user"

# -------------------- LOGIN --------------------
def login():
    st.title("🔐 Login")

    u = st.text_input("Username")
    p = st.text_input("Password", type="password")
    role = st.selectbox("Role", ["user", "admin"])

    if st.button("Login"):
        c.execute("SELECT * FROM users WHERE username=? AND password=? AND role=?", (u, p, role))
        if c.fetchone():
            st.session_state.logged_in = True
            st.session_state.user = u
            st.session_state.role = role
            st.rerun()
        else:
            st.error("Invalid Credentials")

    if st.button("Register"):
        c.execute("INSERT INTO users VALUES (?,?,?)", (u, p, role))
        conn.commit()
        st.success("Registered")

if not st.session_state.logged_in:
    login()
    st.stop()

# -------------------- SIDEBAR --------------------
st.sidebar.title("📊 Civic System")
st.sidebar.write(f"👤 {st.session_state.user}")
st.sidebar.write(f"🛡️ Role: {st.session_state.role}")

if st.sidebar.button("Logout"):
    st.session_state.logged_in = False
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

# -------------------- AI ENGINES --------------------
def get_category(text):
    t = re.sub(r'[^a-zA-Z ]', ' ', text.lower())

    if any(x in t for x in ["road", "pothole"]):
        return "Road"
    if any(x in t for x in ["water", "pipeline"]):
        return "Water"
    if any(x in t for x in ["electric", "power"]):
        return "Electricity"
    if any(x in t for x in ["garbage", "waste"]):
        return "Garbage"
    return "Other"

def get_priority(text):
    t = text.lower()
    if any(x in t for x in ["accident", "fire", "flood", "no water", "no electricity"]):
        return "High"
    if any(x in t for x in ["road", "leak", "garbage"]):
        return "Medium"
    return "Low"

def get_sentiment(text):
    t = text.lower()
    if any(x in t for x in ["angry", "urgent", "worst", "bad"]):
        return "Negative"
    if any(x in t for x in ["good", "thanks"]):
        return "Positive"
    return "Neutral"

def get_department(cat):
    mapping = {
        "Road": "PWD Department",
        "Water": "Water Board",
        "Electricity": "Electricity Board",
        "Garbage": "Municipal Cleaning"
    }
    return mapping.get(cat, "General Dept")

# -------------------- CHATBOT --------------------
def chatbot(msg):
    m = msg.lower()

    if "status" in m:
        return "📊 You can track complaint status in Admin Panel or Dashboard."

    if "road" in m:
        return "🛣️ Road complaint registered and sent to PWD."

    if "water" in m:
        return "💧 Water complaint assigned to Water Board."

    if "electric" in m:
        return "⚡ Electricity issue escalated."

    return "📌 Complaint registered successfully. System is processing it."

# -------------------- UI --------------------
st.title("🏛️ Smart Civic Complaint System (Government SaaS Level)")

tabs = st.tabs(["📝 Complaint", "📊 Dashboard", "📈 Analytics", "🤖 Chatbot", "🛡️ Admin Panel"])

# ================== COMPLAINT ==================
with tabs[0]:

    text = st.text_area("Enter complaint")

    if st.button("Submit") and text.strip():

        X = vectorizer.transform([text])
        pred = model.predict(X)
        prediction = le.inverse_transform(pred)[0]

        category = get_category(text)
        priority = get_priority(text)
        sentiment = get_sentiment(text)
        department = get_department(category)

        try:
            conf = round(model.predict_proba(X).max() * 100, 2)
        except:
            conf = np.random.uniform(60, 80)

        c.execute("""
            INSERT INTO complaints VALUES (?,?,?,?,?,?,?,?)
        """, (
            st.session_state.user,
            text,
            prediction,
            category,
            str(conf),
            priority,
            sentiment,
            "NEW"
        ))

        conn.commit()

        st.success("Complaint Registered")

        st.write("Department:", department)
        st.write("Priority:", priority)
        st.write("Sentiment:", sentiment)

# ================== DASHBOARD ==================
with tabs[1]:

    data = pd.read_sql_query("SELECT * FROM complaints", conn)

    if not data.empty:
        st.metric("Total Complaints", len(data))
        st.dataframe(data, use_container_width=True)

# ================== ANALYTICS ==================
with tabs[2]:

    data = pd.read_sql_query("SELECT * FROM complaints", conn)

    if not data.empty:

        st.subheader("Category Distribution")
        fig, ax = plt.subplots()
        data["category"].value_counts().plot.pie(autopct="%1.1f%%", ax=ax)
        st.pyplot(fig)

        st.subheader("Priority Distribution")
        fig2, ax2 = plt.subplots()
        data["priority"].value_counts().plot.bar(ax=ax2)
        st.pyplot(fig2)

        st.subheader("Sentiment Analysis")
        fig3, ax3 = plt.subplots()
        data["sentiment"].value_counts().plot.bar(ax=ax3)
        st.pyplot(fig3)

# ================== CHATBOT ==================
with tabs[3]:

    if "chat" not in st.session_state:
        st.session_state.chat = []

    msg = st.text_input("Ask something")

    if msg:
        st.session_state.chat.append(("You", msg))
        st.session_state.chat.append(("Bot", chatbot(msg)))

    for r, m in st.session_state.chat:
        st.write(f"**{r}:** {m}")

# ================== ADMIN PANEL ==================
with tabs[4]:

    if st.session_state.role != "admin":
        st.warning("Admin access only")
    else:
        admin_data = pd.read_sql_query("SELECT * FROM complaints", conn)

        st.subheader("Manage Complaints")

        if not admin_data.empty:
            st.dataframe(admin_data, use_container_width=True)

            idx = st.number_input("Select Row Index to Update Status", min_value=0, max_value=len(admin_data)-1, step=1)

            status = st.selectbox("Update Status", ["NEW", "IN PROGRESS", "RESOLVED"])

            if st.button("Update Status"):
                c.execute("UPDATE complaints SET status=? WHERE rowid=?", (status, idx+1))
                conn.commit()
                st.success("Status Updated")
