import streamlit as st
import pickle
import re
import os

# PATH CONFIGURATION

APP_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(APP_DIR)

model_path = os.path.join(ROOT_DIR, "models", "language_model_lr.pkl")
vectorizer_path = os.path.join(ROOT_DIR, "models", "vectorizer.pkl")

# LOAD MODELS

try:
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    with open(vectorizer_path, "rb") as f:
        vectorizer = pickle.load(f)

except Exception as e:
    st.error(f"Model loading failed: {e}")
    st.stop()

# PREPROCESSING

def preprocess(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    return text

# PAGE CONFIG

st.set_page_config(
    page_title="Language Detector",
    page_icon="🌐",
    layout="centered"
)

# CUSTOM STYLING

st.markdown("""
    <style>
        body {
            background-color: #0e1117;
            color: #ffffff;
        }
        .main {
            text-align: center;
        }
        .stTextInput > div > div > input {
            background-color: #1c1f26;
            color: white;
            border-radius: 8px;
            border: 1px solid #333;
            padding: 10px;
        }
        .stButton > button {
            background-color: #4CAF50;
            color: white;
            border-radius: 8px;
            padding: 10px 20px;
            border: none;
        }
        .stButton > button:hover {
            background-color: #45a049;
        }
    </style>
""", unsafe_allow_html=True)

# UI CONTENT

st.markdown("## 🌐 Language Detector")
st.caption("Simple • Fast • Minimal")

text = st.text_input("Enter text")


# PREDICTION

if st.button("Detect"):
    if not text.strip():
        st.warning("Type something first.")
    else:
        clean = preprocess(text)
        vector = vectorizer.transform([clean])
        prediction = model.predict(vector)

        st.markdown(f"### ✅ {prediction[0]}")