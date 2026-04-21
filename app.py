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

# -------------------- 🌈 SAAS UI STYLE --------------------
st.markdown("""
<style>
/* Gradient Background */
body {
    background: linear-gradient(135deg, #0f172a, #1e293b);
    color: white;
}

/* Title */
.big-title {
    text-align:center;
    font-size:34px;
    font-weight:700;
    background: linear-gradient(90deg, #4CAF50, #00E5FF);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

/* Subtitle */
.sub-text {
    text-align:center;
    color:#94a3b8;
}

/* Glass Card */
.card {
    background: rgba(255,255,255,0.05);
    padding:20px;
    border-radius:15px;
    backdrop-filter: blur(10px);
    text-align:center;
    transition:0.3s;
}
.card:hover {
    transform: translateY(-5px);
    box-shadow: 0px 10px 25px rgba(0,0,0,0.3);
}

/* KPI */
.kpi {
    background: rgba(255,255,255,0.08);
    padding:15px;
    border-radius:12px;
    text-align:center;
}

/* Input */
.stTextArea textarea {
    background-color: rgba(255,255,255,0.05);
    color: white;
    border-radius:10px;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #020617, #0f172a);
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

        st.download_button("⬇ Download", sim.to_csv(index=False), "result.csv")

# -------------------- ADMIN PANEL --------------------
st.markdown("### 🛠️ Admin Panel")
saved = pd.read_sql_query("SELECT rowid, * FROM complaints", conn)
st.dataframe(saved, use_container_width=True)

delete_id = st.number_input("Enter Record ID to Delete", min_value=0, step=1)
if st.button("Delete Record"):
    c.execute("DELETE FROM complaints WHERE rowid=?", (delete_id,))
    conn.commit()
    st.success("Record Deleted")
    st.rerun()

# -------------------- ANALYTICS --------------------
st.markdown("### 📊 Analytics Dashboard")

if not saved.empty:
    total = len(saved)
    top_category = saved["category"].value_counts().idxmax()
    avg_conf = saved["confidence"].astype(float).mean()

    k1, k2, k3 = st.columns(3)
    k1.markdown(f"<div class='kpi'>📌 Total<br><b>{total}</b></div>", unsafe_allow_html=True)
    k2.markdown(f"<div class='kpi'>🏆 Top<br><b>{top_category}</b></div>", unsafe_allow_html=True)
    k3.markdown(f"<div class='kpi'>🎯 Confidence<br><b>{round(avg_conf,2)}</b></div>", unsafe_allow_html=True)

    st.bar_chart(saved["category"].value_counts())

# -------------------- DATASET --------------------
st.markdown("### 📊 Dataset Category Distribution")
st.bar_chart(df[category_col].value_counts())

# -------------------- 🤖 AI CHAT ASSISTANT --------------------
st.markdown("### 🤖 AI Assistant")

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

def chatbot_response(user_text):
    text = user_text.lower()

    # Rule-based + dataset-based intelligence
    if "water" in text:
        return "💧 This seems related to Water Supply issues."
    elif "road" in text or "pothole" in text:
        return "🛣️ This looks like a Road & Infrastructure complaint."
    elif "garbage" in text or "waste" in text:
        return "🗑️ This falls under Sanitation & Waste."
    elif "electric" in text or "power" in text:
        return "⚡ This is related to Electricity issues."
    elif "hello" in text or "hi" in text:
        return "👋 Hello! I can help you categorize complaints or answer basic queries."
    else:
        # Smart fallback using dataset
        sample = df.sample(1)
        return f"🤖 I suggest checking similar complaint: '{sample[complaint_col].values[0]}'"

# Chat input
user_msg = st.text_input("Ask something...")

if user_msg:
    bot_reply = chatbot_response(user_msg)

    st.session_state.chat_history.append(("You", user_msg))
    st.session_state.chat_history.append(("Bot", bot_reply))

# Display chat
for sender, msg in st.session_state.chat_history:
    if sender == "You":
        st.markdown(f"**🧑 {msg}**")
    else:
        st.markdown(f"**🤖 {msg}**")
