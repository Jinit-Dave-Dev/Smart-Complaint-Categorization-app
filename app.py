import streamlit as st
import pickle
import os
import pandas as pd
import sqlite3
import numpy as np
import time

# -------------------- CONFIG --------------------
st.set_page_config(page_title="Smart Complaint System", layout="wide")

# -------------------- DB --------------------
conn = sqlite3.connect("complaints.db", check_same_thread=False)
c = conn.cursor()

c.execute("""CREATE TABLE IF NOT EXISTS complaints (
    user TEXT,
    complaint TEXT,
    category TEXT,
    confidence TEXT,
    status TEXT
)""")
conn.commit()

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
        conf = 50
    return label, conf

# -------------------- STYLE --------------------
st.markdown("""
<style>
.big-title {text-align:center;font-size:32px;font-weight:700;}
.card {background:#111;padding:15px;border-radius:10px;margin:5px;}
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='big-title'>🏛️ Smart Complaint Categorization System</div>", unsafe_allow_html=True)

# -------------------- TABS --------------------
tab1, tab2, tab3, tab4 = st.tabs(["📌 Complaint", "📊 Dashboard", "🤖 Chatbot", "📈 Analytics"])

# =========================================================
# 📌 TAB 1: COMPLAINT
# =========================================================
with tab1:
    st.subheader("Register Complaint")

    user_input = st.text_area("Enter your complaint")

    if st.button("Submit Complaint"):
        if user_input.strip():

            pred, conf = predict(user_input)
            category = map_category(user_input)

            c.execute("INSERT INTO complaints VALUES (?,?,?,?,?)",
                      ("User", user_input, category, str(conf), "NEW"))
            conn.commit()

            st.success("✅ Complaint registered successfully")

            new_df = pd.DataFrame({
                "Complaint": [user_input],
                "Category": [category],
                "Confidence": [conf],
                "Status": ["NEW"]
            })

            st.dataframe(new_df)

# =========================================================
# 📊 TAB 2: DASHBOARD
# =========================================================
with tab2:
    st.subheader("All Complaints")

    saved = pd.read_sql_query("SELECT * FROM complaints", conn)

    if not saved.empty:
        st.dataframe(saved, use_container_width=True)

        st.subheader("Similar Complaints")

        selected = st.selectbox("Filter by category", saved["category"].unique())

        sim = df[df[category_col] == selected]

        st.dataframe(sim.head(5))

# =========================================================
# 🤖 TAB 3: CHATBOT
# =========================================================
with tab3:
    st.subheader("AI Assistant")

    if "chat" not in st.session_state:
        st.session_state.chat = []

    def bot_reply(text):
        t = text.lower()

        if "hi" in t or "hello" in t:
            return "Hey 👋 How can I help you today?"

        elif "water" in t:
            return "💧 Please check your local supply office or report leakage."

        elif "road" in t:
            return "🛣️ Road issues are forwarded to municipality engineers."

        elif "garbage" in t:
            return "🗑️ Garbage collection team will be notified."

        else:
            return "I understand your concern. Please provide more details."

    msg = st.text_input("Type message")

    if msg:
        st.session_state.chat.append(("You", msg))
        reply = bot_reply(msg)
        st.session_state.chat.append(("Bot", reply))

    for sender, text in st.session_state.chat:
        if sender == "You":
            st.markdown(f"**You:** {text}")
        else:
            st.markdown(f"**Bot:** {text}")

# =========================================================
# 📈 TAB 4: ANALYTICS
# =========================================================
with tab4:
    st.subheader("Analytics Dashboard")

    saved = pd.read_sql_query("SELECT * FROM complaints", conn)

    if not saved.empty:

        category_filter = st.selectbox("Select Category", ["All"] + list(saved["category"].unique()))

        if category_filter != "All":
            saved = saved[saved["category"] == category_filter]

        st.write("### Complaint Distribution")
        st.bar_chart(saved["category"].value_counts())

        st.write("### Confidence Distribution")
        saved["confidence"] = saved["confidence"].astype(float)
        st.bar_chart(saved["confidence"])
