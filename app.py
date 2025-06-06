import streamlit as st
import joblib

# Load model and vectorizer
vectorizer = joblib.load("vectorizer.jb")
model = joblib.load("lr_model.jb")

# Inject custom CSS for dark theme and stylish UI
st.markdown("""
    <style>
    body {
        background-color: #0e1117;
        color: #FAFAFA;
    }
    .stApp {
        background-color: #0e1117;
    }
    h1 {
        color: #1f77b4;
        text-align: center;
        font-family: 'Segoe UI', sans-serif;
    }
    .stTextArea textarea {
        background-color: #262730;
        color: white;
        font-size: 16px;
    }
    .stButton button {
        background-color: #1f77b4;
        color: white;
        font-weight: bold;
        border-radius: 10px;
        padding: 0.5em 1em;
    }
    .stButton button:hover {
        background-color: #125e94;
    }
    </style>
""", unsafe_allow_html=True)

# App title and instructions
st.title("🕵️‍♀️ Fake News Detector")
st.write("🔎 **Enter a news article below** to check whether it's **fake or real.**")

# Input field
news_input = st.text_area("📰 News Article", "", height=200)

# Button and prediction
if st.button("🚀 Check News"):
    if news_input.strip():
        transform_input = vectorizer.transform([news_input])
        prediction = model.predict(transform_input)

        if prediction[0] == 1:
            st.success("✅ The news is **real**!")
        else:
            st.error("⚠️ The news is **fake**!")
    else:
        st.warning("⚠️ Please enter some text to analyze.")
