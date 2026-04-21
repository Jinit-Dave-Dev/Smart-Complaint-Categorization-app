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
    st.subheader("🔐 Login System")

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

# -------------------- UI --------------------
st.markdown("""
<style>
body {background-color: #0E1117; color: white;}
.stTextArea textarea {background-color: #1E1E1E; color: white;}
.big-title {text-align:center; font-size:30px; color:#4CAF50;}
.sub-text {text-align:center; color:gray;}
.card {
    background-color:#1E1E1E;
    padding:20px;
    border-radius:12px;
    text-align:center;
}
.kpi {
    background-color:#262730;
    padding:15px;
    border-radius:10px;
    text-align:center;
}
</style>
""", unsafe_allow_html=True)

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

# -------------------- 🔥 PREMIUM ANALYTICS --------------------
st.markdown("### 📊 Analytics Dashboard")

if not saved.empty:
    total = len(saved)
    top_category = saved["category"].value_counts().idxmax()
    avg_conf = saved["confidence"].astype(float).mean()

    # KPI CARDS
    k1, k2, k3 = st.columns(3)
    k1.markdown(f"<div class='kpi'>📌 Total Complaints<br><b>{total}</b></div>", unsafe_allow_html=True)
    k2.markdown(f"<div class='kpi'>🏆 Top Category<br><b>{top_category}</b></div>", unsafe_allow_html=True)
    k3.markdown(f"<div class='kpi'>🎯 Avg Confidence<br><b>{round(avg_conf,2)}</b></div>", unsafe_allow_html=True)

    st.markdown("---")

    cat_counts = saved["category"].value_counts()
    st.bar_chart(cat_counts)

    analytics_df = cat_counts.reset_index()
    analytics_df.columns = ["Category", "Count"]
    st.dataframe(analytics_df, use_container_width=True)

else:
    st.info("No data available yet.")

# -------------------- 🔥 DATASET DISTRIBUTION --------------------
st.markdown("### 📊 Dataset Category Distribution")

data_counts = df[category_col].value_counts()

d1, d2, d3 = st.columns(3)
d1.metric("Total Records", len(df))
d2.metric("Unique Categories", df[category_col].nunique())
d3.metric("Top Category", data_counts.idxmax())

st.bar_chart(data_counts)

dist_df = data_counts.reset_index()
dist_df.columns = ["Category", "Count"]

st.dataframe(dist_df, use_container_width=True)
