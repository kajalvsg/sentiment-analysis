import streamlit as st

from src.predict import load_model, predict_sentiment


st.set_page_config(page_title="Sentiment Analysis NLP App", page_icon="💬", layout="centered")

st.title("Sentiment Analysis & NLP Text Classifier")
st.write(
    "Enter a customer review/tweet and get sentiment prediction using a "
    "TF-IDF + machine learning pipeline."
)

try:
    model = load_model()
except Exception as exc:
    st.error(f"Failed to initialize model: {exc}")
    st.stop()

text_input = st.text_area("Text", placeholder="Type review text here...")

if st.button("Predict Sentiment"):
    if not text_input.strip():
        st.error("Please enter text before predicting.")
    else:
        result = predict_sentiment(text_input, model=model)
        emoji = "😊" if result["label"] == "positive" else "😞"
        st.subheader(f"Prediction: {result['label'].upper()} {emoji}")
        st.write(f"Confidence: **{result['confidence'] * 100:.2f}%**")
        with st.expander("Preprocessed Text"):
            st.code(result["processed_text"])
