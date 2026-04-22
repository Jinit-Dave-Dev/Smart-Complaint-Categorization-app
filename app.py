# ================== NEW PAGE NAVIGATION (ONLY ADDITION) ==================
page = st.sidebar.radio(
    "📌 Navigate",
    [
        "🏠 Home",
        "📊 Analytics",
        "🛠️ Admin Panel",
        "🤖 AI Assistant"
    ]
)

# ================== HOME PAGE ==================
if page == "🏠 Home":

    st.markdown("---")
    user_input = st.text_area("📝 Enter your complaint:", height=150)

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
            col3.markdown(f"<div class='card'>🎯 {get_confidence_label(confidence)}</div>", unsafe_allow_html=True)

            # WHY
            st.markdown("### 🔍 Why this prediction?")
            feature_names = vectorizer.get_feature_names_out()
            tfidf_array = X_new.toarray()[0]
            top_indices = tfidf_array.argsort()[-5:][::-1]
            top_words = [feature_names[i] for i in top_indices if tfidf_array[i] > 0]

            if top_words:
                st.info("Top keywords influencing prediction: " + ", ".join(top_words))
            else:
                st.info("No strong keywords detected.")

            # FEATURE IMPORTANCE
            st.markdown("### 📈 Feature Importance")
            if hasattr(model, "feature_importances_"):
                importances = model.feature_importances_
                top_idx = np.argsort(importances)[-10:]
                words = [feature_names[i] for i in top_idx]
                values = importances[top_idx]

                imp_df = pd.DataFrame({
                    "Feature": words,
                    "Importance": values
                })

                st.bar_chart(imp_df.set_index("Feature"))
            else:
                st.info("Feature importance not available.")

            # SIMILAR
            st.markdown("### 📋 Similar Complaints")
            sim = df[df[category_col] == prediction].head(5)

            if sim.empty:
                st.warning("⚠️ No similar complaints found.")
            else:
                st.dataframe(sim, use_container_width=True)

            st.download_button("⬇ Download", sim.to_csv(index=False), "result.csv")


# ================== ANALYTICS PAGE ==================
elif page == "📊 Analytics":

    st.markdown("### 📊 Analytics Dashboard")

    saved = pd.read_sql_query("SELECT rowid, * FROM complaints", conn)

    if not saved.empty:
        total = len(saved)
        top_category = saved["category"].value_counts().idxmax()
        avg_conf = saved["confidence"].astype(float).mean()

        k1, k2, k3 = st.columns(3)
        k1.markdown(f"<div class='kpi'>📌 Total<br><b>{total}</b></div>", unsafe_allow_html=True)
        k2.markdown(f"<div class='kpi'>🏆 Top<br><b>{top_category}</b></div>", unsafe_allow_html=True)
        k3.markdown(f"<div class='kpi'>🎯 Confidence<br><b>{round(avg_conf,2)}</b></div>", unsafe_allow_html=True)

        st.bar_chart(saved["category"].value_counts())

    # TOP 5
    st.markdown("### 🏆 Top 5 Categories")
    top5 = df[category_col].value_counts().head(5)
    for i, (cat, val) in enumerate(top5.items(), start=1):
        st.write(f"{i}. {cat} ({val})")

    # DATASET
    st.markdown("### 📊 Dataset Category Distribution")
    st.bar_chart(df[category_col].value_counts())


# ================== ADMIN PANEL ==================
elif page == "🛠️ Admin Panel":

    st.markdown("### 🛠️ Admin Panel")

    saved = pd.read_sql_query("SELECT rowid, * FROM complaints", conn)
    st.dataframe(saved, use_container_width=True)

    delete_id = st.number_input("Enter Record ID to Delete", min_value=0, step=1)

    if st.button("Delete Record"):
        c.execute("DELETE FROM complaints WHERE rowid=?", (delete_id,))
        conn.commit()
        st.success("Record Deleted")
        st.rerun()


# ================== CHATBOT PAGE ==================
elif page == "🤖 AI Assistant":

    st.markdown("### 🤖 AI Assistant")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if "suggestions" not in st.session_state:
        st.session_state.suggestions = ["Hi", "Water issue", "Road problem"]

    user_msg = st.text_input("💬 Ask something...")

    if user_msg:
        handle_message(user_msg)

    st.markdown("#### ⚡ Quick Suggestions")
    cols = st.columns(len(st.session_state.suggestions))

    for i, s in enumerate(st.session_state.suggestions):
        if cols[i].button(s):
            handle_message(s)

    for sender, msg in st.session_state.chat_history:
        if sender == "You":
            st.markdown(f"<div class='chat-user'><span class='badge-user'>YOU</span> {msg}</div>", unsafe_allow_html=True)
        else:
            typing_effect(msg)
