import streamlit as st
import pickle
import os
import pandas as pd
import sqlite3

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
    st.subheader("🔐 Login / Register")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Login"):
            c.execute("SELECT * FROM users WHERE username=? AND password=?", (username, password))
            result = c.fetchone()

            if result:
                st.session_state.logged_in = True
                st.session_state.user = username
                st.success("Login Successful")
                st.rerun()   # 🔥 FIX
            else:
                st.error("Invalid Credentials")

    with col2:
        if st.button("Register"):
            c.execute("INSERT INTO users VALUES (?,?)", (username, password))
            conn.commit()
            st.success("Registered! Now login.")

# STOP IF NOT LOGGED IN
if not st.session_state.logged_in:
    login()
    st.stop()

# -------------------- UI --------------------
st.markdown("""
<style>
body {background-color: #0E1117; color: white;}
.stTextArea textarea {background-color: #1E1E1E; color: white;}
.big-title {text-align:center; font-size:30px; color:#4CAF50;}
.sub-text {text-align:center; color:gray;}
.card {background-color:#1E1E1E;padding:20px;border-radius:10px;}
</style>
""", unsafe_allow_html=True)

# -------------------- TITLE --------------------
st.markdown("<div class='big-title'>🏛️ Smart Municipal Complaint System</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-text'>AI-powered complaint classification dashboard</div>", unsafe_allow_html=True)

# -------------------- SIDEBAR --------------------
st.sidebar.title("📊 Dashboard")
st.sidebar.write(f"👤 Logged in as: {st.session_state.user}")

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

        # SAVE
        c.execute("INSERT INTO complaints VALUES (?,?,?,?,?)",
                  (st.session_state.user, user_input, prediction, enhanced, str(confidence)))
        conn.commit()

        # CARDS
        col1, col2, col3 = st.columns(3)
        col1.markdown(f"<div class='card'>📌 {prediction}</div>", unsafe_allow_html=True)
        col2.markdown(f"<div class='card'>🏛️ {enhanced}</div>", unsafe_allow_html=True)
        col3.markdown(f"<div class='card'>🎯 {confidence}</div>", unsafe_allow_html=True)

        # TABLE
        st.markdown("### 📋 Similar Complaints")
        sim = df[df[category_col] == prediction].head(5)
        st.dataframe(sim, use_container_width=True)

        # DOWNLOAD
        st.download_button("⬇ Download", sim.to_csv(index=False), "result.csv")

# -------------------- DATABASE VIEW --------------------
st.markdown("### 📁 Saved Complaints")
saved = pd.read_sql_query("SELECT * FROM complaints", conn)
st.dataframe(saved, use_container_width=True)

# -------------------- CHART --------------------
st.markdown("### 📊 Category Distribution")
st.bar_chart(df[category_col].value_counts())
