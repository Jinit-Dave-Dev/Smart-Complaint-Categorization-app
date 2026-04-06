import streamlit as st
import pickle
import os
import pandas as pd
from sklearn.metrics import accuracy_score

project_folder = r"C:\Users\davej\6 months internship work\Dave Jinit Jigeshkumar_GIT_Smart Complaint Categorization for Government Portals"

data_folder = os.path.join(project_folder, "data")
search_folders = [project_folder, data_folder]

csv_files = []
for folder in search_folders:
    if os.path.exists(folder):
        csv_files += [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(".csv")]

if not csv_files:
    st.error(f"No CSV file found in project or data folder: {search_folders}")
    st.stop()

dataset_path = csv_files[0]
st.info(f"Using dataset: {dataset_path}")

df = pd.read_csv(dataset_path)
df.columns = df.columns.str.strip()

complaint_col = next((c for c in df.columns if 'complaint' in c.lower() or 'text' in c.lower()), None)
category_col = next((c for c in df.columns if 'category' in c.lower() or 'label' in c.lower()), None)

if complaint_col is None or category_col is None:
    st.error(f"Could not detect complaint or category columns. Found columns: {df.columns}")
    st.stop()

models_folder = os.path.join(project_folder, "models")
if not os.path.exists(models_folder):
    st.error(f"Models folder not found: {models_folder}")
    st.stop()

with open(os.path.join(models_folder, "tfidf_vectorizer.pkl"), "rb") as f:
    vectorizer = pickle.load(f)
with open(os.path.join(models_folder, "label_encoder.pkl"), "rb") as f:
    le = pickle.load(f)

model_files = [f for f in os.listdir(models_folder) if f.endswith("_model.pkl")]
model_map = {f.replace("_model.pkl","").replace("_"," ").title(): f for f in model_files}

if not model_files:
    st.error("No trained model files found in the models folder!")
    st.stop()

df[complaint_col] = df[complaint_col].astype(str)

X = vectorizer.transform(df[complaint_col])
y_true = le.transform(df[category_col])

model_accuracies = {}
for model_name, file_name in model_map.items():
    with open(os.path.join(models_folder, file_name), "rb") as f:
        model = pickle.load(f)
    y_pred = model.predict(X)
    model_accuracies[model_name] = round(accuracy_score(y_true, y_pred), 3)

st.title("Smart Complaint Categorization (Professional Version)")

user_input = st.text_area("Enter your complaint:")

if user_input.strip():
    X_new = vectorizer.transform([str(user_input)])
    

    predictions = {}
    for model_name, file_name in model_map.items():
        with open(os.path.join(models_folder, file_name), "rb") as f:
            model = pickle.load(f)
        y_pred = model.predict(X_new)
        predictions[model_name] = le.inverse_transform(y_pred)[0]
    

    df_pred = pd.DataFrame({
        "Model": list(predictions.keys()),
        "Accuracy": [model_accuracies[m] for m in predictions.keys()],
        "Predicted Category": list(predictions.values())
    })
    

    for col in df.columns:
        df_pred[col] = [df[col].iloc[0]] * len(df_pred)
    
    st.markdown("### Predictions from all models with accuracy & context")
    st.table(df_pred)